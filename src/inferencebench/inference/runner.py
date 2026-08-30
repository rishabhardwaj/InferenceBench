from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from inferencebench.artifacts import canonical_sha256, sha256_file
from inferencebench.config import Settings
from inferencebench.domain import (
    AttemptEvidence,
    AttemptPurpose,
    CorpusManifest,
    CustomerLabel,
    Issue,
    ParseStatus,
    ProviderOutcome,
    RunManifest,
    RunStatus,
    ScoredOutcome,
)
from inferencebench.evaluation.cost import calculate_request_cost
from inferencebench.inference.contract import (
    load_shared_inference_contract,
    prepare_classification_request,
)
from inferencebench.inference.digitalocean import (
    DIGITALOCEAN_CHAT_COMPLETIONS_URL,
    DigitalOceanChatCompletionsAdapter,
)
from inferencebench.inference.domain import (
    ModelEvaluationConfiguration,
    ModelEvaluationExecution,
    ModelEvaluationProgress,
    PreparedClassificationRequest,
    SharedInferenceContract,
    SingleIssueClassificationResult,
)
from inferencebench.models.artifacts import ModelCatalogArtifacts, PricingArtifacts
from inferencebench.models.domain import ModelPricing
from inferencebench.persistence.repository import EvidenceRepository


ProgressCallback = Callable[[ModelEvaluationProgress], None]
CancelRequested = Callable[[], bool]


async def execute_model_evaluation(
    settings: Settings,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    configuration: ModelEvaluationConfiguration,
    *,
    api_key: str,
    client: httpx.AsyncClient,
    repository: EvidenceRepository | None = None,
    contract: SharedInferenceContract | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_requested: CancelRequested | None = None,
    evaluation_stage: str = "prompt_development",
) -> ModelEvaluationExecution:
    """Run one model over one frozen issue population and persist every outcome.

    This is intentionally a single-model operation. Comparing two models means
    calling it twice, so each model has its own concurrency pool and Run Manifest.
    """

    if not api_key:
        raise ValueError("DigitalOcean API key must not be empty")
    _validate_frozen_population(corpus_manifest, issues, configuration)

    active_contract = contract or load_shared_inference_contract()
    _require_frozen_contract_for_stage(active_contract, evaluation_stage)
    prepared_requests = tuple(
        prepare_classification_request(issue, configuration.model_id, active_contract)
        for issue in issues
    )
    catalog_manifest, _, _ = ModelCatalogArtifacts(
        settings.model_catalog_directory
    ).load()
    pricing_manifest, pricing_entries = PricingArtifacts(
        settings.pricing_directory
    ).load_for_catalog(catalog_manifest)
    pricing_by_model = {entry.model_id: entry for entry in pricing_entries}
    model_pricing = pricing_by_model[configuration.model_id]

    active_repository = repository or EvidenceRepository(settings.database_path)
    active_repository.initialize()
    running_manifest = _new_running_manifest(
        settings,
        corpus_manifest,
        issues,
        configuration,
        active_contract,
        catalog_version=catalog_manifest.catalog_version,
        catalog_sha256=catalog_manifest.content_sha256,
        pricing_snapshot_id=pricing_manifest.pricing_snapshot_id,
        pricing_snapshot_sha256=pricing_manifest.content_sha256,
    )

    # The manifest commit must finish before any worker—and therefore any
    # provider request—is created.
    active_repository.insert_run(running_manifest)
    try:
        _report_progress(
            progress_callback,
            running_manifest,
            (),
            time.perf_counter(),
            latest_issue_number=None,
        )
    except BaseException:
        run_started_monotonic = time.perf_counter()
        _finalize_run(
            active_repository,
            running_manifest,
            RunStatus.INCOMPLETE,
            run_started_monotonic,
        )
        raise

    # Run Wall-Clock Time starts immediately before the frozen workload becomes
    # eligible for dispatch. Configuration, manifest persistence, and UI callback
    # work above are deliberately outside this clock.
    run_started_monotonic = time.perf_counter()
    semaphore = asyncio.Semaphore(configuration.concurrency)
    adapter = DigitalOceanChatCompletionsAdapter(client)
    tasks = []
    for dispatch_order, (issue, request) in enumerate(
        zip(issues, prepared_requests, strict=True)
    ):
        eligible_at = datetime.now(UTC)
        eligible_monotonic = time.perf_counter()
        tasks.append(
            asyncio.create_task(
                _execute_attempt(
                    issue,
                    request,
                    dispatch_order,
                    running_manifest,
                    configuration,
                    adapter,
                    semaphore,
                    api_key,
                    model_pricing,
                    eligible_at=eligible_at,
                    eligible_monotonic=eligible_monotonic,
                )
            )
        )

    try:
        for completed in asyncio.as_completed(tasks):
            attempt = await completed
            # This loop is the only writer. Provider work remains concurrent in
            # worker tasks, while each terminal row gets its own short commit.
            active_repository.insert_attempt(attempt)
            persisted = active_repository.get_attempts(running_manifest.run_id)
            if len(persisted) < running_manifest.expected_count:
                _report_progress(
                    progress_callback,
                    running_manifest,
                    persisted,
                    run_started_monotonic,
                    latest_issue_number=attempt.issue_number,
                )
                if cancel_requested is not None and cancel_requested():
                    _cancel_tasks(tasks)
                    await asyncio.gather(*tasks, return_exceptions=True)
                    terminal = _finalize_run(
                        active_repository,
                        running_manifest,
                        RunStatus.INCOMPLETE,
                        run_started_monotonic,
                    )
                    attempts = active_repository.get_attempts(running_manifest.run_id)
                    _report_progress(
                        progress_callback,
                        terminal,
                        attempts,
                        run_started_monotonic,
                        latest_issue_number=attempt.issue_number,
                    )
                    return ModelEvaluationExecution(
                        schema_version="model_evaluation_execution.v1",
                        manifest=terminal,
                        attempts=attempts,
                    )

        terminal = _finalize_run(
            active_repository,
            running_manifest,
            RunStatus.COMPLETE,
            run_started_monotonic,
        )
        attempts = active_repository.get_attempts(running_manifest.run_id)
        _report_progress(
            progress_callback,
            terminal,
            attempts,
            run_started_monotonic,
            latest_issue_number=attempts[-1].issue_number if attempts else None,
        )
        return ModelEvaluationExecution(
            schema_version="model_evaluation_execution.v1",
            manifest=terminal,
            attempts=attempts,
        )
    except BaseException:
        _cancel_tasks(tasks)
        await asyncio.gather(*tasks, return_exceptions=True)
        current = active_repository.get_run(running_manifest.run_id)
        if current.status is RunStatus.RUNNING:
            _finalize_run(
                active_repository,
                running_manifest,
                RunStatus.INCOMPLETE,
                run_started_monotonic,
            )
        raise


async def _execute_attempt(
    issue: Issue,
    request: PreparedClassificationRequest,
    dispatch_order: int,
    run: RunManifest,
    configuration: ModelEvaluationConfiguration,
    adapter: DigitalOceanChatCompletionsAdapter,
    semaphore: asyncio.Semaphore,
    api_key: str,
    pricing: ModelPricing,
    *,
    eligible_at: datetime,
    eligible_monotonic: float,
) -> AttemptEvidence:
    async with semaphore:
        queue_wait_ms = (time.perf_counter() - eligible_monotonic) * 1000
        result = await adapter.classify(
            request,
            api_key=api_key,
            timeout_seconds=configuration.timeout_seconds,
        )
    return _to_attempt_evidence(
        issue,
        dispatch_order,
        run,
        configuration,
        result,
        pricing,
        eligible_at=eligible_at,
        queue_wait_ms=queue_wait_ms,
    )


def _to_attempt_evidence(
    issue: Issue,
    dispatch_order: int,
    run: RunManifest,
    configuration: ModelEvaluationConfiguration,
    result: SingleIssueClassificationResult,
    pricing: ModelPricing,
    *,
    eligible_at: datetime,
    queue_wait_ms: float,
) -> AttemptEvidence:
    parsed_label = result.parse_result.parsed_label
    cost = calculate_request_cost(result.usage, pricing)
    return AttemptEvidence(
        schema_version="attempt_evidence.v3",
        attempt_id=str(uuid4()),
        run_id=run.run_id,
        issue_number=issue.issue_number,
        dispatch_order=dispatch_order,
        attempt_purpose=AttemptPurpose.BENCHMARK,
        request_messages=result.request_messages,
        request_messages_sha256=result.request_messages_sha256,
        request_parameters={
            "model": configuration.model_id,
            **result.effective_settings.api_parameters(),
        },
        eligible_at=eligible_at,
        request_started_at=result.request_started_at,
        request_ended_at=result.request_ended_at,
        queue_wait_ms=queue_wait_ms,
        request_latency_ms=result.request_latency_ms,
        configured_timeout_seconds=result.configured_timeout_seconds,
        provider_outcome=result.provider_outcome,
        http_status=result.http_status,
        provider_request_id=result.provider_request_id,
        finish_reason=result.finish_reason,
        response_headers=result.response_headers,
        raw_response=result.raw_response,
        raw_error=result.raw_error,
        raw_model_output=result.raw_model_output,
        parsed_label=parsed_label,
        parse_status=result.parse_result.parse_status,
        normalizations=tuple(
            normalization.value for normalization in result.parse_result.normalizations
        ),
        scored_outcome=_score_outcome(
            result.provider_outcome,
            parsed_label,
            configuration.ground_truth_labels.get(issue.issue_number),
        ),
        usable=parsed_label is not None,
        usage=result.usage,
        pricing_snapshot_id=run.pricing_snapshot_id,
        cost_formula_version=cost.formula_version,
        calculated_request_cost_usd=cost.calculated_cost_usd,
        cost_completeness=cost.completeness,
        cost_calculation_terms=cost.terms,
        cost_unknown_reasons=cost.unknown_reasons,
    )


def _score_outcome(
    provider_outcome: ProviderOutcome,
    parsed_label: CustomerLabel | None,
    ground_truth_label: CustomerLabel | None,
) -> ScoredOutcome | None:
    if ground_truth_label is None:
        return None
    if provider_outcome is not ProviderOutcome.SUCCESS:
        return ScoredOutcome.REQUEST_ERROR
    if parsed_label is None:
        return ScoredOutcome.INVALID_OUTPUT
    if parsed_label == ground_truth_label:
        return ScoredOutcome.CORRECT
    return ScoredOutcome.INCORRECT_LABEL


def _new_running_manifest(
    settings: Settings,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    configuration: ModelEvaluationConfiguration,
    contract: SharedInferenceContract,
    *,
    catalog_version: str,
    catalog_sha256: str,
    pricing_snapshot_id: str,
    pricing_snapshot_sha256: str,
) -> RunManifest:
    issue_numbers = tuple(issue.issue_number for issue in issues)
    contract_manifest = contract.manifest
    return RunManifest(
        schema_version="run_manifest.v2",
        run_id=str(uuid4()),
        run_type="model_evaluation",
        model_id=configuration.model_id,
        provider_endpoint_id=DIGITALOCEAN_CHAT_COMPLETIONS_URL,
        status=RunStatus.RUNNING,
        corpus_version=corpus_manifest.corpus_version,
        corpus_sha256=corpus_manifest.artifact_sha256,
        ordered_issue_numbers=issue_numbers,
        issue_order_sha256=canonical_sha256(issue_numbers),
        ground_truth_version=configuration.ground_truth_version,
        ground_truth_sha256=configuration.ground_truth_sha256,
        prompt_version=contract_manifest.prompt_version,
        prompt_sha256=contract_manifest.system_message_sha256,
        parser_version=contract_manifest.parser_version,
        rubric_version=contract_manifest.rubric_version,
        rubric_sha256=contract_manifest.rubric_sha256,
        generation_configuration=contract.generation_configuration.model_dump(
            mode="json"
        ),
        generation_configuration_sha256=(
            contract_manifest.generation_configuration_sha256
        ),
        model_catalog_version=catalog_version,
        model_catalog_sha256=catalog_sha256,
        pricing_snapshot_id=pricing_snapshot_id,
        pricing_snapshot_sha256=pricing_snapshot_sha256,
        timeout_seconds=configuration.timeout_seconds,
        concurrency=configuration.concurrency,
        retries=0,
        stream=False,
        application_version=_application_version(),
        schema_revision=2,
        metric_version="classification-metrics-v1",
        dependency_lock_sha256=sha256_file(settings.project_root / "uv.lock"),
        source_sha256=_source_sha256(settings.project_root),
        started_at=datetime.now(UTC),
        ended_at=None,
        wall_clock_ms=None,
        expected_count=len(issues),
        persisted_count=0,
        usable_count=0,
        normalized_count=0,
        invalid_output_count=0,
        request_error_count=0,
    )


def _finalize_run(
    repository: EvidenceRepository,
    running_manifest: RunManifest,
    requested_status: RunStatus,
    run_started_monotonic: float,
) -> RunManifest:
    attempts = repository.get_attempts(running_manifest.run_id)
    complete_population = (
        len(attempts) == running_manifest.expected_count
        and tuple(attempt.issue_number for attempt in attempts)
        == running_manifest.ordered_issue_numbers
    )
    status = (
        RunStatus.COMPLETE
        if requested_status is RunStatus.COMPLETE and complete_population
        else RunStatus.INCOMPLETE
    )
    terminal = running_manifest.model_copy(
        update={
            "status": status,
            "ended_at": datetime.now(UTC),
            "wall_clock_ms": (time.perf_counter() - run_started_monotonic) * 1000,
            "persisted_count": len(attempts),
            "usable_count": sum(attempt.usable for attempt in attempts),
            "normalized_count": sum(
                attempt.parse_status is ParseStatus.NORMALIZED for attempt in attempts
            ),
            "invalid_output_count": sum(
                attempt.provider_outcome is ProviderOutcome.SUCCESS
                and not attempt.usable
                for attempt in attempts
            ),
            "request_error_count": sum(
                attempt.provider_outcome is not ProviderOutcome.SUCCESS
                for attempt in attempts
            ),
        }
    )
    # model_copy does not revalidate updates in Pydantic, so force the lifecycle
    # and count invariants before committing the terminal transition.
    terminal = RunManifest.model_validate(terminal.model_dump(mode="python"))
    repository.finalize_run(terminal)
    return terminal


def _report_progress(
    callback: ProgressCallback | None,
    manifest: RunManifest,
    attempts: tuple[AttemptEvidence, ...],
    run_started_monotonic: float,
    *,
    latest_issue_number: int | None,
) -> None:
    if callback is None:
        return
    callback(
        ModelEvaluationProgress(
            schema_version="model_evaluation_progress.v1",
            run_id=manifest.run_id,
            model_id=manifest.model_id,
            status=manifest.status,
            expected_count=manifest.expected_count,
            persisted_count=len(attempts),
            usable_count=sum(attempt.usable for attempt in attempts),
            normalized_count=sum(
                attempt.parse_status is ParseStatus.NORMALIZED for attempt in attempts
            ),
            invalid_output_count=sum(
                attempt.provider_outcome is ProviderOutcome.SUCCESS
                and not attempt.usable
                for attempt in attempts
            ),
            request_error_count=sum(
                attempt.provider_outcome is not ProviderOutcome.SUCCESS
                for attempt in attempts
            ),
            elapsed_wall_clock_ms=(time.perf_counter() - run_started_monotonic)
            * 1000,
            latest_issue_number=latest_issue_number,
        )
    )


def _validate_frozen_population(
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    configuration: ModelEvaluationConfiguration,
) -> None:
    if not issues:
        raise ValueError("Model Evaluation Run requires at least one issue")
    issue_numbers = tuple(issue.issue_number for issue in issues)
    if len(set(issue_numbers)) != len(issue_numbers):
        raise ValueError("Model Evaluation Run issue population must be unique")
    corpus_issue_numbers = set(corpus_manifest.ordered_issue_numbers)
    if any(issue_number not in corpus_issue_numbers for issue_number in issue_numbers):
        raise ValueError("Run issue population is not contained in the Corpus Manifest")
    if any(
        issue.corpus_version != corpus_manifest.corpus_version
        or issue.repository != corpus_manifest.repository
        for issue in issues
    ):
        raise ValueError("Run issues must reference the supplied frozen Corpus")
    if not set(configuration.ground_truth_labels).issubset(issue_numbers):
        raise ValueError("Ground Truth labels must be limited to the run population")


def _require_frozen_contract_for_stage(
    contract: SharedInferenceContract, evaluation_stage: str
) -> None:
    if (
        evaluation_stage != "prompt_development"
        and contract.manifest.contract_status != "frozen"
    ):
        raise ValueError(
            "Primary Scored Holdout and later evaluations require a frozen "
            "Shared Inference Contract"
        )


def _source_sha256(project_root: Path) -> str:
    paths = [project_root / "pyproject.toml", project_root / "app.py"]
    paths.extend(sorted((project_root / "src").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _application_version() -> str:
    try:
        return importlib.metadata.version("inferencebench")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0+uninstalled"


def _cancel_tasks(tasks: list[asyncio.Task[AttemptEvidence]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
