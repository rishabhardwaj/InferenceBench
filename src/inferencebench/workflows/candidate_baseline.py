from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from pydantic import Field, JsonValue, model_validator

from inferencebench.artifacts import CorpusArtifacts, canonical_sha256
from inferencebench.config import Settings
from inferencebench.domain import (
    AttemptEvidence,
    CorpusManifest,
    CustomerLabel,
    GroundTruthAnnotation,
    GroundTruthManifest,
    Issue,
    ParseStatus,
    ProviderOutcome,
    RunManifest,
    RunStatus,
    Sha256,
    StrictModel,
)
from inferencebench.evaluation.bootstrap import (
    BOOTSTRAP_ALGORITHM_VERSION,
    BOOTSTRAP_INTERVAL_VERSION,
    BOOTSTRAP_METRIC_VERSION,
    SHARED_BOOTSTRAP_RESAMPLE_COUNT,
    SHARED_BOOTSTRAP_SEED,
    BootstrapMetricIntervals,
    BootstrapHoldoutRow,
    BootstrapResample,
    CandidateBootstrapSummary,
    ObservedBootstrapRow,
    PairBootstrapSummary,
    SharedBootstrapPlanManifest,
    assert_shared_bootstrap_resamples,
    calculate_candidate_bootstrap_summary,
    calculate_pair_bootstrap_summary,
    canonical_holdout_rows,
    unavailable_candidate_bootstrap_summary,
    unavailable_pair_bootstrap_summary,
)
from inferencebench.evaluation.operational import (
    LatencySummary,
    PERCENTILE_METHOD_VERSION,
    RunOperationalSummary,
    build_run_operational_summary,
    linear_percentile,
)
from inferencebench.evaluation.scored import (
    ScoredGroundTruthSet,
    ScoredModelSummary,
    ScoredPopulation,
    build_scored_model_summary,
    scored_ground_truth_from_evaluation_corpus,
)
from inferencebench.ground_truth.artifacts import EvaluationCorpusArtifacts
from inferencebench.inference.contract import load_shared_inference_contract
from inferencebench.inference.digitalocean import redact_text
from inferencebench.inference.domain import (
    ModelEvaluationConfiguration,
    ModelEvaluationProgress,
    SharedInferenceContract,
)
from inferencebench.inference.runner import execute_model_evaluation
from inferencebench.models.artifacts import (
    ModelCatalogArtifacts,
    PricingArtifacts,
    assert_run_uses_model_sources,
)
from inferencebench.models.domain import (
    APPROVED_ELIGIBLE_MODEL_IDS,
    ModelCatalogManifest,
    ModelPricing,
    PricingSnapshotManifest,
)
from inferencebench.persistence.repository import EvidenceRepository


CandidateClientFactory = Callable[[], httpx.AsyncClient]
CandidateProgressCallback = Callable[["CandidateBaselineProgress"], None]
CANDIDATE_SCREENING_SIZE = 40
CANDIDATE_SCREENING_SEED = 20_260_830
CANDIDATE_SCREENING_SELECTION_VERSION = "sha256-stratified-primary-holdout-v1"
CANDIDATE_SCREENING_CONCURRENCY = 4


class CandidateBaselinePlan(StrictModel):
    schema_version: Literal["candidate_baseline_plan.v1"]
    baseline_version: str
    corpus_version: str
    corpus_sha256: Sha256
    repository: str
    ground_truth_version: str
    ground_truth_sha256: Sha256
    model_catalog_version: str
    model_catalog_sha256: Sha256
    pricing_snapshot_id: str
    pricing_snapshot_sha256: Sha256
    contract_version: str
    contract_status: Literal["development", "frozen"]
    prompt_version: str
    prompt_sha256: Sha256
    parser_version: str
    rubric_version: str
    generation_configuration_version: str
    generation_configuration_sha256: Sha256
    generation_configuration: dict[str, JsonValue]
    shared_benchmark_timeout_seconds: float = Field(gt=0)
    concurrency: Literal[4]
    retries: Literal[0]
    eligible_model_ids: tuple[str, ...]
    random_sample_issue_numbers: tuple[int, ...]
    prompt_development_issue_numbers: tuple[int, ...]
    primary_holdout_issue_numbers: tuple[int, ...]
    diagnostic_issue_numbers: tuple[int, ...]
    candidate_screening_issue_numbers: tuple[int, ...]
    candidate_screening_selection_seed: int
    candidate_screening_selection_version: Literal[
        "sha256-stratified-primary-holdout-v1"
    ]
    ordered_baseline_issue_numbers: tuple[int, ...]
    expected_attempts_per_model: int = Field(gt=0)
    expected_provider_requests: int = Field(gt=0)
    shared_bootstrap_seed: int
    shared_bootstrap_resample_count: Literal[10_000]
    can_authorize: bool
    blockers: tuple[str, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> "CandidateBaselinePlan":
        if PurePosixPath(self.baseline_version).name != self.baseline_version:
            raise ValueError("Candidate Baseline version must be a directory name")
        if self.eligible_model_ids != APPROVED_ELIGIBLE_MODEL_IDS:
            raise ValueError("Candidate Baseline must contain the exact 25-model active pool")
        if len(self.random_sample_issue_numbers) != 100:
            raise ValueError("Candidate Baseline requires the 100 Random Human-Reviewed Sample")
        if len(self.prompt_development_issue_numbers) != 20:
            raise ValueError("Candidate Baseline requires 20 Prompt Development issues")
        if len(self.primary_holdout_issue_numbers) != 80:
            raise ValueError("Candidate Baseline requires 80 Primary Scored Holdout issues")
        if not 0 <= len(self.diagnostic_issue_numbers) <= 20:
            raise ValueError("Diagnostic Scored Supplement must contain at most 20 issues")
        if set(self.prompt_development_issue_numbers) | set(
            self.primary_holdout_issue_numbers
        ) != set(self.random_sample_issue_numbers):
            raise ValueError("Prompt Development and Holdout must partition the random sample")
        if set(self.diagnostic_issue_numbers) & set(self.random_sample_issue_numbers):
            raise ValueError("Diagnostic issues must be disjoint from the random sample")
        if len(self.candidate_screening_issue_numbers) != CANDIDATE_SCREENING_SIZE:
            raise ValueError("Candidate screening requires exactly 40 holdout issues")
        if not set(self.candidate_screening_issue_numbers) <= set(
            self.primary_holdout_issue_numbers
        ):
            raise ValueError("Candidate screening must use only untouched holdout issues")
        if self.ordered_baseline_issue_numbers != self.candidate_screening_issue_numbers:
            raise ValueError("Candidate Baseline must use only the screening holdout")
        if len(set(self.ordered_baseline_issue_numbers)) != len(
            self.ordered_baseline_issue_numbers
        ):
            raise ValueError("Candidate Baseline issue population must be unique")
        if self.expected_attempts_per_model != len(self.ordered_baseline_issue_numbers):
            raise ValueError("Expected attempts per model disagree with baseline population")
        if self.expected_provider_requests != (
            len(self.eligible_model_ids) * self.expected_attempts_per_model
        ):
            raise ValueError("Expected provider requests disagree with plan dimensions")
        if self.can_authorize == bool(self.blockers):
            raise ValueError("Candidate Baseline readiness disagrees with blockers")
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(payload) != self.content_sha256:
            raise ValueError("Candidate Baseline Plan content hash is invalid")
        return self


class CandidateBaselineAuthorization(StrictModel):
    schema_version: Literal["candidate_baseline_authorization.v1"]
    baseline_version: str
    plan_sha256: Sha256
    authorized_by: str
    authorized_at: datetime
    expected_provider_requests: int = Field(gt=0)
    authorization_statement: str

    @model_validator(mode="after")
    def validate_statement(self) -> "CandidateBaselineAuthorization":
        if not self.authorized_by.strip():
            raise ValueError("Paid baseline authorization requires a human identity")
        if self.authorized_at.tzinfo is None:
            raise ValueError("Paid baseline authorization time must include a timezone")
        expected = (
            f"I authorize {self.expected_provider_requests} paid DigitalOcean "
            f"inference requests for {self.baseline_version} under plan "
            f"{self.plan_sha256}."
        )
        if self.authorization_statement != expected:
            raise ValueError("Paid baseline authorization statement is not exact")
        return self


class CandidateBaselineProgress(StrictModel):
    candidate_position: int = Field(gt=0)
    candidate_count: Literal[25]
    model_id: str
    run_id: str
    status: RunStatus
    expected_count: int = Field(gt=0)
    persisted_count: int = Field(ge=0)
    usable_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    request_error_count: int = Field(ge=0)
    elapsed_wall_clock_ms: float = Field(ge=0)


class CandidateBaselineRunRecord(StrictModel):
    candidate_position: int = Field(gt=0)
    model_id: str
    run_id: str | None
    status: Literal["complete", "incomplete", "runner_error"]
    expected_count: int = Field(gt=0)
    persisted_count: int = Field(ge=0)
    usable_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    request_error_count: int = Field(ge=0)
    error_type: str | None = None
    sanitized_error: str | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "CandidateBaselineRunRecord":
        if self.persisted_count > self.expected_count:
            raise ValueError("Persisted attempts cannot exceed the expected population")
        if (
            self.usable_count
            + self.invalid_output_count
            + self.request_error_count
            != self.persisted_count
        ):
            raise ValueError("Candidate result states must cover persisted attempts")
        if self.status == "complete" and (
            self.run_id is None or self.persisted_count != self.expected_count
        ):
            raise ValueError("Complete candidate evidence requires its exact attempt set")
        if self.status == "incomplete" and self.run_id is None:
            raise ValueError("Incomplete candidate evidence requires a durable Run Manifest")
        if self.status == "runner_error" and self.run_id is not None:
            raise ValueError("Runner errors without a durable run must not invent a run ID")
        return self


class CandidateBaselineExecution(StrictModel):
    schema_version: Literal["candidate_baseline_execution.v1"]
    baseline_version: str
    plan_sha256: Sha256
    authorization: CandidateBaselineAuthorization
    runs: tuple[CandidateBaselineRunRecord, ...]

    @model_validator(mode="after")
    def validate_candidate_coverage(self) -> "CandidateBaselineExecution":
        if tuple(row.model_id for row in self.runs) != APPROVED_ELIGIBLE_MODEL_IDS:
            raise ValueError("Candidate Baseline execution must retain all 25 active candidates")
        if tuple(row.candidate_position for row in self.runs) != tuple(range(1, 26)):
            raise ValueError("Candidate Baseline positions must follow catalog order")
        if self.authorization.plan_sha256 != self.plan_sha256:
            raise ValueError("Candidate Baseline authorization references another plan")
        if self.authorization.baseline_version != self.baseline_version:
            raise ValueError("Candidate Baseline authorization references another version")
        return self


class OutputAdherenceSummary(StrictModel):
    # Existing immutable timeboxed evidence used the former internal name.
    # Accept it on read; newly generated evidence uses Candidate Screening.
    population_name: Literal[
        "Candidate Baseline Evaluation", "Candidate Screening Evaluation"
    ]
    expected_count: int = Field(gt=0)
    observed_count: int = Field(ge=0)
    unobserved_count: int = Field(ge=0)
    exact_count: int = Field(ge=0)
    normalized_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    request_error_count: int = Field(ge=0)
    exact_rate: float = Field(ge=0, le=1)
    normalized_rate: float = Field(ge=0, le=1)
    invalid_output_rate: float = Field(ge=0, le=1)
    request_error_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_partition(self) -> "OutputAdherenceSummary":
        if self.observed_count + self.unobserved_count != self.expected_count:
            raise ValueError("Output Adherence must expose unobserved attempts")
        if (
            self.exact_count
            + self.normalized_count
            + self.invalid_output_count
            + self.request_error_count
            != self.observed_count
        ):
            raise ValueError("Output Adherence states must cover observed attempts")
        expected_rates = (
            self.exact_count / self.expected_count,
            self.normalized_count / self.expected_count,
            self.invalid_output_count / self.expected_count,
            self.request_error_count / self.expected_count,
        )
        if expected_rates != (
            self.exact_rate,
            self.normalized_rate,
            self.invalid_output_rate,
            self.request_error_rate,
        ):
            raise ValueError("Output Adherence rates must use every expected attempt")
        return self


class CandidateScreeningLatencySummary(StrictModel):
    population_name: Literal["Candidate Screening Holdout"]
    issue_count: Literal[40]
    latency: LatencySummary


class CandidateEvidenceRow(StrictModel):
    candidate_position: int = Field(gt=0)
    model_id: str
    run_id: str | None
    source_attempts_sha256: Sha256 | None
    run_status: Literal["complete", "incomplete", "runner_error"]
    comparable_headlines: bool
    exclusion_reason: str | None
    candidate_screening_holdout: ScoredModelSummary | None
    candidate_screening_cost_per_correct_usd: Decimal | None
    candidate_screening_cost_status: str | None
    candidate_screening_cost_completeness: str | None
    candidate_screening_latency: CandidateScreeningLatencySummary | None
    output_adherence: OutputAdherenceSummary | None
    operational: RunOperationalSummary | None


class CandidateEvidenceTable(StrictModel):
    schema_version: Literal["candidate_evidence_table.v1"]
    baseline_version: str
    plan_sha256: Sha256
    calculated_at: datetime
    row_count: Literal[25]
    rows: tuple[CandidateEvidenceRow, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_table(self) -> "CandidateEvidenceTable":
        if tuple(row.model_id for row in self.rows) != APPROVED_ELIGIBLE_MODEL_IDS:
            raise ValueError("Candidate evidence table must retain all 25 active candidates")
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(payload) != self.content_sha256:
            raise ValueError("Candidate evidence table content hash is invalid")
        return self


class CandidateBootstrapAnalysis(StrictModel):
    schema_version: Literal["candidate_bootstrap_analysis.v1"]
    baseline_version: str
    baseline_plan_sha256: Sha256
    bootstrap_plan_sha256: Sha256
    calculated_at: datetime
    candidate_count: Literal[25]
    pair_count: Literal[300]
    candidates: tuple[CandidateBootstrapSummary, ...]
    pairs: tuple[PairBootstrapSummary, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_analysis(self) -> "CandidateBootstrapAnalysis":
        if tuple(row.model_id for row in self.candidates) != APPROVED_ELIGIBLE_MODEL_IDS:
            raise ValueError("Bootstrap analysis must retain all 25 active candidates")
        expected_pairs = tuple(combinations(APPROVED_ELIGIBLE_MODEL_IDS, 2))
        actual_pairs = tuple((row.model_a_id, row.model_b_id) for row in self.pairs)
        if actual_pairs != expected_pairs:
            raise ValueError("Bootstrap analysis must contain all 300 ordered model pairs")
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(payload) != self.content_sha256:
            raise ValueError("Bootstrap analysis content hash is invalid")
        return self


class _CandidateSources(StrictModel):
    corpus_manifest: CorpusManifest
    corpus_issues: tuple[Issue, ...]
    ground_truth_manifest: GroundTruthManifest
    annotations: tuple[GroundTruthAnnotation, ...]
    catalog_manifest: ModelCatalogManifest
    pricing_manifest: PricingSnapshotManifest
    pricing: tuple[ModelPricing, ...]
    contract: SharedInferenceContract


def prepare_candidate_baseline(
    settings: Settings,
    baseline_version: str = "candidate-baseline-v1",
) -> CandidateBaselinePlan:
    sources = _load_candidate_sources(settings)
    random_annotations = tuple(
        row for row in sources.annotations if row.sampling_stratum == "random"
    )
    prompt_annotations = tuple(
        row for row in random_annotations if row.evaluation_role == "prompt_development"
    )
    holdout_annotations = tuple(
        row for row in random_annotations if row.evaluation_role == "primary_holdout"
    )
    diagnostic_annotations = tuple(
        row for row in sources.annotations if row.evaluation_role == "diagnostic"
    )
    screening_annotations = _stratified_screening_annotations(holdout_annotations)
    blockers: list[str] = []
    if sources.contract.manifest.contract_status != "frozen":
        blockers.append(
            "The Shared Inference Contract is not frozen; Candidate Baseline "
            "Evaluation cannot inspect the Primary Scored Holdout yet."
        )
    manifest = sources.contract.manifest
    generation = sources.contract.generation_configuration
    payload = {
        "schema_version": "candidate_baseline_plan.v1",
        "baseline_version": baseline_version,
        "corpus_version": sources.corpus_manifest.corpus_version,
        "corpus_sha256": sources.corpus_manifest.artifact_sha256,
        "repository": sources.corpus_manifest.repository,
        "ground_truth_version": sources.ground_truth_manifest.ground_truth_version,
        "ground_truth_sha256": sources.ground_truth_manifest.artifact_sha256,
        "model_catalog_version": sources.catalog_manifest.catalog_version,
        "model_catalog_sha256": sources.catalog_manifest.content_sha256,
        "pricing_snapshot_id": sources.pricing_manifest.pricing_snapshot_id,
        "pricing_snapshot_sha256": sources.pricing_manifest.content_sha256,
        "contract_version": manifest.contract_version,
        "contract_status": manifest.contract_status,
        "prompt_version": manifest.prompt_version,
        "prompt_sha256": manifest.system_message_sha256,
        "parser_version": manifest.parser_version,
        "rubric_version": manifest.rubric_version,
        "generation_configuration_version": generation.configuration_version,
        "generation_configuration_sha256": manifest.generation_configuration_sha256,
        "generation_configuration": generation.model_dump(mode="json"),
        "shared_benchmark_timeout_seconds": settings.shared_timeout_seconds,
        "concurrency": CANDIDATE_SCREENING_CONCURRENCY,
        "retries": 0,
        "eligible_model_ids": APPROVED_ELIGIBLE_MODEL_IDS,
        "random_sample_issue_numbers": tuple(
            row.issue_number for row in random_annotations
        ),
        "prompt_development_issue_numbers": tuple(
            row.issue_number for row in prompt_annotations
        ),
        "primary_holdout_issue_numbers": tuple(
            row.issue_number for row in holdout_annotations
        ),
        "diagnostic_issue_numbers": tuple(
            row.issue_number for row in diagnostic_annotations
        ),
        "candidate_screening_issue_numbers": tuple(
            row.issue_number for row in screening_annotations
        ),
        "candidate_screening_selection_seed": CANDIDATE_SCREENING_SEED,
        "candidate_screening_selection_version": CANDIDATE_SCREENING_SELECTION_VERSION,
        "ordered_baseline_issue_numbers": tuple(
            row.issue_number for row in screening_annotations
        ),
        "expected_attempts_per_model": len(screening_annotations),
        "expected_provider_requests": len(APPROVED_ELIGIBLE_MODEL_IDS)
        * len(screening_annotations),
        "shared_bootstrap_seed": SHARED_BOOTSTRAP_SEED,
        "shared_bootstrap_resample_count": SHARED_BOOTSTRAP_RESAMPLE_COUNT,
        "can_authorize": not blockers,
        "blockers": tuple(blockers),
    }
    return CandidateBaselinePlan.model_validate(
        {**payload, "content_sha256": canonical_sha256(payload)}
    )


def create_candidate_baseline_authorization(
    plan: CandidateBaselinePlan,
    *,
    confirmed_plan_sha256: str,
    authorized_by: str,
    authorized_at: datetime,
) -> CandidateBaselineAuthorization:
    if not plan.can_authorize:
        raise ValueError("Candidate Baseline Plan is blocked and cannot be authorized")
    if confirmed_plan_sha256 != plan.content_sha256:
        raise ValueError("Confirmed Candidate Baseline Plan hash does not match")
    statement = (
        f"I authorize {plan.expected_provider_requests} paid DigitalOcean "
        f"inference requests for {plan.baseline_version} under plan "
        f"{plan.content_sha256}."
    )
    return CandidateBaselineAuthorization(
        schema_version="candidate_baseline_authorization.v1",
        baseline_version=plan.baseline_version,
        plan_sha256=plan.content_sha256,
        authorized_by=authorized_by,
        authorized_at=authorized_at,
        expected_provider_requests=plan.expected_provider_requests,
        authorization_statement=statement,
    )


async def execute_candidate_baseline(
    settings: Settings,
    plan: CandidateBaselinePlan,
    authorization: CandidateBaselineAuthorization,
    *,
    api_key: str,
    progress_callback: CandidateProgressCallback | None = None,
    client_factory: CandidateClientFactory | None = None,
) -> CandidateBaselineExecution:
    if not api_key:
        raise ValueError("DigitalOcean API key must not be empty")
    current = prepare_candidate_baseline(settings, plan.baseline_version)
    if current != plan:
        raise ValueError("Frozen Candidate Baseline identities changed after authorization")
    if authorization.plan_sha256 != plan.content_sha256:
        raise ValueError("Paid authorization references a different baseline plan")
    if (
        authorization.baseline_version != plan.baseline_version
        or authorization.expected_provider_requests != plan.expected_provider_requests
    ):
        raise ValueError("Paid authorization does not match the baseline dimensions")
    sources = _load_candidate_sources(settings)
    issue_by_number = {issue.issue_number: issue for issue in sources.corpus_issues}
    annotation_by_number = {
        annotation.issue_number: annotation for annotation in sources.annotations
    }
    issues = tuple(issue_by_number[number] for number in plan.ordered_baseline_issue_numbers)
    labels = {
        number: annotation_by_number[number].label
        for number in plan.ordered_baseline_issue_numbers
    }
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    make_client = client_factory or httpx.AsyncClient
    records: list[CandidateBaselineRunRecord] = []

    for position, model_id in enumerate(plan.eligible_model_ids, 1):
        latest_run_id: str | None = None

        def report(progress: ModelEvaluationProgress) -> None:
            nonlocal latest_run_id
            latest_run_id = progress.run_id
            if progress_callback is not None:
                progress_callback(
                    CandidateBaselineProgress(
                        candidate_position=position,
                        candidate_count=25,
                        model_id=model_id,
                        run_id=progress.run_id,
                        status=progress.status,
                        expected_count=progress.expected_count,
                        persisted_count=progress.persisted_count,
                        usable_count=progress.usable_count,
                        invalid_output_count=progress.invalid_output_count,
                        request_error_count=progress.request_error_count,
                        elapsed_wall_clock_ms=progress.elapsed_wall_clock_ms,
                    )
                )

        configuration = ModelEvaluationConfiguration(
            schema_version="model_evaluation_configuration.v1",
            model_id=model_id,
            timeout_seconds=plan.shared_benchmark_timeout_seconds,
            concurrency=plan.concurrency,
            ground_truth_version=plan.ground_truth_version,
            ground_truth_sha256=plan.ground_truth_sha256,
            ground_truth_labels=labels,
        )
        try:
            async with make_client() as client:
                execution = await execute_model_evaluation(
                    settings,
                    sources.corpus_manifest,
                    issues,
                    configuration,
                    api_key=api_key,
                    client=client,
                    repository=repository,
                    contract=sources.contract,
                    progress_callback=report,
                    evaluation_stage="candidate_baseline",
                )
            records.append(_record_from_run(position, execution.manifest))
        except Exception as error:
            run = repository.get_run(latest_run_id) if latest_run_id else None
            if run is not None and run.status is RunStatus.COMPLETE:
                records.append(_record_from_run(position, run))
            else:
                records.append(
                    CandidateBaselineRunRecord(
                        candidate_position=position,
                        model_id=model_id,
                        run_id=run.run_id if run else None,
                        status="incomplete" if run is not None else "runner_error",
                        expected_count=plan.expected_attempts_per_model,
                        persisted_count=run.persisted_count if run else 0,
                        usable_count=run.usable_count if run else 0,
                        invalid_output_count=run.invalid_output_count if run else 0,
                        request_error_count=run.request_error_count if run else 0,
                        error_type=type(error).__name__,
                        sanitized_error=redact_text(str(error), api_key),
                    )
                )

    return CandidateBaselineExecution(
        schema_version="candidate_baseline_execution.v1",
        baseline_version=plan.baseline_version,
        plan_sha256=plan.content_sha256,
        authorization=authorization,
        runs=tuple(records),
    )


def build_candidate_evidence_table(
    settings: Settings,
    plan: CandidateBaselinePlan,
    execution: CandidateBaselineExecution,
) -> CandidateEvidenceTable:
    _assert_execution_matches_plan(plan, execution)
    sources = _load_candidate_sources(settings)
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    pricing_by_model = {row.model_id: row for row in sources.pricing}
    ground_truth = scored_ground_truth_from_evaluation_corpus(
        sources.ground_truth_manifest, sources.annotations
    )
    screening_numbers = set(plan.candidate_screening_issue_numbers)
    screening_ground_truth = ground_truth.model_copy(
        update={
            "items": tuple(
                row for row in ground_truth.items if row.issue_number in screening_numbers
            )
        }
    )
    rows = tuple(
        _candidate_evidence_row(
            plan,
            record,
            repository,
            screening_ground_truth,
            pricing_by_model[record.model_id],
            sources.catalog_manifest,
            sources.pricing_manifest,
        )
        for record in execution.runs
    )
    payload = {
        "schema_version": "candidate_evidence_table.v1",
        "baseline_version": plan.baseline_version,
        "plan_sha256": plan.content_sha256,
        "calculated_at": _utc_timestamp(),
        "row_count": 25,
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    return CandidateEvidenceTable.model_validate(
        {**payload, "content_sha256": canonical_sha256(payload)}
    )


def build_candidate_bootstrap_analysis(
    settings: Settings,
    plan: CandidateBaselinePlan,
    execution: CandidateBaselineExecution,
    bootstrap_plan: SharedBootstrapPlanManifest,
    resamples: tuple[BootstrapResample, ...],
) -> CandidateBootstrapAnalysis:
    _assert_execution_matches_plan(plan, execution)
    if bootstrap_plan.baseline_plan_sha256 != plan.content_sha256:
        raise ValueError("Shared Bootstrap Plan references another Candidate Baseline")
    if (
        bootstrap_plan.seed != plan.shared_bootstrap_seed
        or bootstrap_plan.resample_count != plan.shared_bootstrap_resample_count
        or bootstrap_plan.algorithm_version != BOOTSTRAP_ALGORITHM_VERSION
        or bootstrap_plan.interval_version != BOOTSTRAP_INTERVAL_VERSION
        or bootstrap_plan.metric_version != BOOTSTRAP_METRIC_VERSION
    ):
        raise ValueError("Shared Bootstrap Plan does not match the frozen analysis plan")
    if (
        bootstrap_plan.ground_truth_version != plan.ground_truth_version
        or bootstrap_plan.ground_truth_sha256 != plan.ground_truth_sha256
        or tuple(
            row.issue_number for row in bootstrap_plan.canonical_holdout_rows
        )
        != plan.candidate_screening_issue_numbers
    ):
        raise ValueError("Shared Bootstrap Plan uses another Primary Scored Holdout")
    assert_shared_bootstrap_resamples(bootstrap_plan, resamples)
    sources = _load_candidate_sources(settings)
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    annotation_by_number = {
        annotation.issue_number: annotation for annotation in sources.annotations
    }
    summaries: list[CandidateBootstrapSummary] = []
    vectors_by_model: dict[str, tuple] = {}
    for record in execution.runs:
        if record.status != "complete" or record.run_id is None:
            summaries.append(
                unavailable_candidate_bootstrap_summary(
                    record.model_id,
                    record.run_id,
                    (
                        _attempts_sha256(repository.get_attempts(record.run_id))
                        if record.run_id is not None
                        else None
                    ),
                    bootstrap_plan.content_sha256,
                    "Candidate Baseline run is not complete",
                )
            )
            continue
        run = repository.get_run(record.run_id)
        if run.status is not RunStatus.COMPLETE or run.model_id != record.model_id:
            raise ValueError("Bootstrap candidate does not reference its complete run")
        assert_run_uses_model_sources(
            run, sources.catalog_manifest, sources.pricing_manifest
        )
        _assert_run_matches_plan(run, plan)
        attempts = repository.get_attempts(record.run_id)
        attempt_by_issue = {attempt.issue_number: attempt for attempt in attempts}
        observed = tuple(
            _observed_bootstrap_row(
                row.row_index,
                row.issue_number,
                annotation_by_number[row.issue_number].label,
                attempt_by_issue[row.issue_number],
            )
            for row in bootstrap_plan.canonical_holdout_rows
        )
        summary, vectors = calculate_candidate_bootstrap_summary(
            model_id=record.model_id,
            run_id=record.run_id,
            source_attempts_sha256=_attempts_sha256(attempts),
            plan=bootstrap_plan,
            resamples=resamples,
            observed_rows=observed,
        )
        summaries.append(summary)
        vectors_by_model[record.model_id] = vectors

    summary_by_model = {row.model_id: row for row in summaries}
    pairs: list[PairBootstrapSummary] = []
    for model_a, model_b in combinations(plan.eligible_model_ids, 2):
        summary_a = summary_by_model[model_a]
        summary_b = summary_by_model[model_b]
        if summary_a.intervals is None or summary_b.intervals is None:
            pairs.append(unavailable_pair_bootstrap_summary(summary_a, summary_b))
            continue
        pairs.append(
            calculate_pair_bootstrap_summary(
                model_a_id=model_a,
                model_b_id=model_b,
                run_a_id=summary_a.run_id or "",
                run_b_id=summary_b.run_id or "",
                source_a_attempts_sha256=summary_a.source_attempts_sha256 or "",
                source_b_attempts_sha256=summary_b.source_attempts_sha256 or "",
                plan_sha256=bootstrap_plan.content_sha256,
                candidate_a_vectors=vectors_by_model[model_a],
                candidate_b_vectors=vectors_by_model[model_b],
                candidate_a_point=summary_a.intervals,
                candidate_b_point=summary_b.intervals,
            )
        )
    payload = {
        "schema_version": "candidate_bootstrap_analysis.v1",
        "baseline_version": plan.baseline_version,
        "baseline_plan_sha256": plan.content_sha256,
        "bootstrap_plan_sha256": bootstrap_plan.content_sha256,
        "calculated_at": _utc_timestamp(),
        "candidate_count": 25,
        "pair_count": 300,
        "candidates": [row.model_dump(mode="json") for row in summaries],
        "pairs": [row.model_dump(mode="json") for row in pairs],
    }
    return CandidateBootstrapAnalysis.model_validate(
        {**payload, "content_sha256": canonical_sha256(payload)}
    )


def baseline_bootstrap_rows(
    settings: Settings,
) -> tuple[BootstrapHoldoutRow, ...]:
    sources = _load_candidate_sources(settings)
    holdout = tuple(
        annotation
        for annotation in sources.annotations
        if annotation.evaluation_role == "primary_holdout"
    )
    screening = _stratified_screening_annotations(holdout)
    return canonical_holdout_rows(
        (annotation.issue_number, annotation.label)
        for annotation in screening
    )


def write_json_artifact(path: Path, model: StrictModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_candidate_baseline_plan(path: Path) -> CandidateBaselinePlan:
    return CandidateBaselinePlan.model_validate_json(path.read_text(encoding="utf-8"))


def load_candidate_baseline_execution(path: Path) -> CandidateBaselineExecution:
    return CandidateBaselineExecution.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _candidate_evidence_row(
    plan: CandidateBaselinePlan,
    record: CandidateBaselineRunRecord,
    repository: EvidenceRepository,
    ground_truth: ScoredGroundTruthSet,
    pricing: ModelPricing,
    catalog_manifest: ModelCatalogManifest,
    pricing_manifest: PricingSnapshotManifest,
) -> CandidateEvidenceRow:
    if record.run_id is None:
        return CandidateEvidenceRow(
            candidate_position=record.candidate_position,
            model_id=record.model_id,
            run_id=None,
            source_attempts_sha256=None,
            run_status="runner_error",
            comparable_headlines=False,
            exclusion_reason=record.sanitized_error or "No Run Manifest was created",
            candidate_screening_holdout=None,
            candidate_screening_cost_per_correct_usd=None,
            candidate_screening_cost_status=None,
            candidate_screening_cost_completeness=None,
            candidate_screening_latency=None,
            output_adherence=None,
            operational=None,
        )
    run = repository.get_run(record.run_id)
    attempts = repository.get_attempts(record.run_id)
    attempts_sha256 = _attempts_sha256(attempts)
    expected_record_status = (
        "complete" if run.status is RunStatus.COMPLETE else "incomplete"
    )
    if record.status != expected_record_status:
        raise ValueError("Candidate execution status disagrees with its Run Manifest")
    assert_run_uses_model_sources(run, catalog_manifest, pricing_manifest)
    _assert_run_matches_plan(run, plan)
    output_adherence = _output_adherence(run, attempts)
    operational = build_run_operational_summary(
        run,
        attempts,
        pricing,
        plan.candidate_screening_issue_numbers
        if run.status is RunStatus.COMPLETE
        else (),
    )
    if run.status is not RunStatus.COMPLETE:
        return CandidateEvidenceRow(
            candidate_position=record.candidate_position,
            model_id=record.model_id,
            run_id=run.run_id,
            source_attempts_sha256=attempts_sha256,
            run_status="incomplete",
            comparable_headlines=False,
            exclusion_reason="Incomplete run is excluded from comparable headlines",
            candidate_screening_holdout=None,
            candidate_screening_cost_per_correct_usd=None,
            candidate_screening_cost_status=None,
            candidate_screening_cost_completeness=None,
            candidate_screening_latency=None,
            output_adherence=output_adherence,
            operational=operational,
        )
    primary = build_scored_model_summary(
        run, attempts, ground_truth, ScoredPopulation.PRIMARY_HOLDOUT
    )
    primary_cost = operational.primary_holdout_cost
    assert primary_cost is not None
    return CandidateEvidenceRow(
        candidate_position=record.candidate_position,
        model_id=record.model_id,
        run_id=run.run_id,
        source_attempts_sha256=attempts_sha256,
        run_status="complete",
        comparable_headlines=True,
        exclusion_reason=None,
        candidate_screening_holdout=primary,
        candidate_screening_cost_per_correct_usd=primary_cost.cost_per_correct_usd,
        candidate_screening_cost_status=primary_cost.cost_per_correct_status,
        candidate_screening_cost_completeness=primary_cost.completeness.value,
        candidate_screening_latency=_candidate_screening_latency(
            run, attempts, plan.candidate_screening_issue_numbers
        ),
        output_adherence=output_adherence,
        operational=operational,
    )


def _candidate_screening_latency(
    run: RunManifest,
    attempts: tuple[AttemptEvidence, ...],
    issue_numbers: tuple[int, ...],
) -> CandidateScreeningLatencySummary:
    selected = tuple(
        attempt for attempt in attempts if attempt.issue_number in set(issue_numbers)
    )
    if len(selected) != 40:
        raise ValueError("Candidate-screening latency requires exactly 40 attempts")
    usable = tuple(attempt.request_latency_ms for attempt in selected if attempt.usable)
    queue_wait = tuple(attempt.queue_wait_ms for attempt in selected)
    return CandidateScreeningLatencySummary(
        population_name="Candidate Screening Holdout",
        issue_count=40,
        latency=LatencySummary(
            percentile_method_version=PERCENTILE_METHOD_VERSION,
            is_comparable=run.status is RunStatus.COMPLETE,
            concurrency=run.concurrency,
            usable_count=len(usable),
            expected_count=40,
            p50_usable_request_latency_ms=linear_percentile(usable, 50),
            p95_usable_request_latency_ms=linear_percentile(usable, 95),
            p50_queue_wait_ms=linear_percentile(queue_wait, 50),
            p95_queue_wait_ms=linear_percentile(queue_wait, 95),
        ),
    )


def _output_adherence(
    run: RunManifest, attempts: tuple[AttemptEvidence, ...]
) -> OutputAdherenceSummary:
    exact = sum(
        attempt.usable and attempt.parse_status is ParseStatus.EXACT
        for attempt in attempts
    )
    normalized = sum(
        attempt.usable and attempt.parse_status is ParseStatus.NORMALIZED
        for attempt in attempts
    )
    expected = run.expected_count
    return OutputAdherenceSummary(
        population_name="Candidate Screening Evaluation",
        expected_count=expected,
        observed_count=len(attempts),
        unobserved_count=expected - len(attempts),
        exact_count=exact,
        normalized_count=normalized,
        invalid_output_count=run.invalid_output_count,
        request_error_count=run.request_error_count,
        exact_rate=exact / expected,
        normalized_rate=normalized / expected,
        invalid_output_rate=run.invalid_output_count / expected,
        request_error_rate=run.request_error_count / expected,
    )


def _observed_bootstrap_row(
    row_index: int,
    issue_number: int,
    label: CustomerLabel,
    attempt: AttemptEvidence,
) -> ObservedBootstrapRow:
    return ObservedBootstrapRow(
        row_index=row_index,
        issue_number=issue_number,
        ground_truth_label=label,
        predicted_label=attempt.parsed_label,
        is_invalid_output=(
            attempt.provider_outcome is ProviderOutcome.SUCCESS
            and attempt.parsed_label is None
        ),
        is_request_error=attempt.provider_outcome is not ProviderOutcome.SUCCESS,
        calculated_request_cost_usd=attempt.calculated_request_cost_usd,
    )


def _record_from_run(
    position: int, run: RunManifest
) -> CandidateBaselineRunRecord:
    return CandidateBaselineRunRecord(
        candidate_position=position,
        model_id=run.model_id,
        run_id=run.run_id,
        status="complete" if run.status is RunStatus.COMPLETE else "incomplete",
        expected_count=run.expected_count,
        persisted_count=run.persisted_count,
        usable_count=run.usable_count,
        invalid_output_count=run.invalid_output_count,
        request_error_count=run.request_error_count,
    )


def _assert_run_matches_plan(run: RunManifest, plan: CandidateBaselinePlan) -> None:
    expected = (
        run.corpus_version == plan.corpus_version
        and run.corpus_sha256 == plan.corpus_sha256
        and run.ordered_issue_numbers == plan.ordered_baseline_issue_numbers
        and run.ground_truth_version == plan.ground_truth_version
        and run.ground_truth_sha256 == plan.ground_truth_sha256
        and run.prompt_version == plan.prompt_version
        and run.prompt_sha256 == plan.prompt_sha256
        and run.parser_version == plan.parser_version
        and run.generation_configuration_sha256
        == plan.generation_configuration_sha256
        and run.model_catalog_version == plan.model_catalog_version
        and run.model_catalog_sha256 == plan.model_catalog_sha256
        and run.pricing_snapshot_id == plan.pricing_snapshot_id
        and run.pricing_snapshot_sha256 == plan.pricing_snapshot_sha256
        and run.timeout_seconds == plan.shared_benchmark_timeout_seconds
        and run.concurrency == plan.concurrency
        and run.retries == 0
    )
    if not expected:
        raise ValueError(f"Run {run.run_id} does not match Candidate Baseline Plan")


def _assert_execution_matches_plan(
    plan: CandidateBaselinePlan, execution: CandidateBaselineExecution
) -> None:
    if (
        execution.baseline_version != plan.baseline_version
        or execution.plan_sha256 != plan.content_sha256
        or execution.authorization.expected_provider_requests
        != plan.expected_provider_requests
        or any(
            row.expected_count != plan.expected_attempts_per_model
            for row in execution.runs
        )
    ):
        raise ValueError("Candidate Baseline execution does not match its plan")


def _attempts_sha256(attempts: tuple[AttemptEvidence, ...]) -> str:
    ordered = tuple(sorted(attempts, key=lambda row: row.dispatch_order))
    return canonical_sha256([row.model_dump(mode="json") for row in ordered])


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _stratified_screening_annotations(
    holdout: tuple[GroundTruthAnnotation, ...],
) -> tuple[GroundTruthAnnotation, ...]:
    if len(holdout) != 80:
        raise ValueError("Candidate screening selection requires the frozen 80-row holdout")
    by_label = {
        label: tuple(row for row in holdout if row.label is label)
        for label in CustomerLabel
    }
    targets = {
        label: max(1, len(rows) * CANDIDATE_SCREENING_SIZE // len(holdout))
        for label, rows in by_label.items()
        if rows
    }
    if sum(targets.values()) != CANDIDATE_SCREENING_SIZE:
        raise ValueError("Frozen holdout support cannot produce the agreed 40-row screen")
    selected_numbers: set[int] = set()
    for label, rows in by_label.items():
        if not rows:
            continue
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{CANDIDATE_SCREENING_SEED}:{label.value}:{row.issue_number}".encode(
                    "ascii"
                )
            ).digest(),
        )
        selected_numbers.update(
            row.issue_number for row in ranked[: targets[label]]
        )
    return tuple(row for row in holdout if row.issue_number in selected_numbers)


def _load_candidate_sources(settings: Settings) -> _CandidateSources:
    corpus_manifest, corpus_issues = CorpusArtifacts(settings.corpus_root).load_active()
    ground_truth_manifest, annotations = EvaluationCorpusArtifacts(
        settings.ground_truth_root
    ).load_version(
        settings.evaluation_ground_truth_version,
        corpus_manifest,
        corpus_issues,
    )
    catalog_manifest, _, _ = ModelCatalogArtifacts(
        settings.model_catalog_directory
    ).load()
    pricing_manifest, pricing = PricingArtifacts(
        settings.pricing_directory
    ).load_for_catalog(catalog_manifest)
    contract = load_shared_inference_contract(settings.shared_contract_directory)
    return _CandidateSources(
        corpus_manifest=corpus_manifest,
        corpus_issues=corpus_issues,
        ground_truth_manifest=ground_truth_manifest,
        annotations=annotations,
        catalog_manifest=catalog_manifest,
        pricing_manifest=pricing_manifest,
        pricing=pricing,
        contract=contract,
    )
