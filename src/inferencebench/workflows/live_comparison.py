from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Literal

import httpx
from pydantic import Field, model_validator

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.config import Settings
from inferencebench.domain import (
    CorpusManifest,
    CustomerLabel,
    GroundTruthAnnotation,
    GroundTruthManifest,
    Issue,
    RunStatus,
    StrictModel,
)
from inferencebench.ground_truth.artifacts import EvaluationCorpusArtifacts
from inferencebench.inference.contract import load_shared_inference_contract
from inferencebench.inference.domain import (
    ModelEvaluationConfiguration,
    ModelEvaluationExecution,
    ModelEvaluationProgress,
    SharedInferenceContract,
)
from inferencebench.inference.runner import execute_model_evaluation
from inferencebench.models.artifacts import ModelCatalogArtifacts, PricingArtifacts
from inferencebench.models.domain import (
    APPROVED_ELIGIBLE_MODEL_IDS,
    ModelCatalogManifest,
    PricingSnapshotManifest,
)
from inferencebench.persistence.repository import EvidenceRepository


LiveClientFactory = Callable[[], httpx.AsyncClient]
LiveProgressCallback = Callable[["LiveModelProgress"], None]


class LiveComparisonPreparation(StrictModel):
    corpus_version: str
    corpus_sha256: str
    repository: str
    corpus_issue_count: int = Field(gt=0)
    expected_provider_requests: int = Field(gt=0)
    ground_truth_version: str
    ground_truth_sha256: str
    scored_issue_count: int = Field(gt=0)
    unscored_issue_count: int = Field(ge=0)
    model_catalog_version: str
    model_catalog_sha256: str
    eligible_model_ids: tuple[str, ...]
    pricing_snapshot_id: str
    pricing_snapshot_sha256: str
    contract_version: str
    contract_status: Literal["development", "frozen"]
    prompt_version: str
    prompt_sha256: str
    parser_version: str
    generation_configuration_version: str
    generation_configuration_sha256: str
    max_completion_tokens: int = Field(gt=0)
    shared_timeout_seconds: float = Field(gt=0)
    default_concurrency: int = Field(gt=0)
    can_start: bool
    blockers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_readiness(self) -> "LiveComparisonPreparation":
        if self.expected_provider_requests != 2 * self.corpus_issue_count:
            raise ValueError("Expected provider requests must equal 2 x Corpus size")
        if self.scored_issue_count + self.unscored_issue_count != self.corpus_issue_count:
            raise ValueError("Scored and Unscored populations must cover the Corpus")
        if self.can_start == bool(self.blockers):
            raise ValueError("Live comparison readiness disagrees with its blockers")
        return self


class LiveComparisonConfiguration(StrictModel):
    model_a_id: str
    model_b_id: str
    concurrency: int = Field(gt=0)

    @model_validator(mode="after")
    def require_distinct_models(self) -> "LiveComparisonConfiguration":
        if self.model_a_id == self.model_b_id:
            raise ValueError("Model A and Model B must be distinct")
        return self


class LiveModelProgress(StrictModel):
    position: Literal["Model A", "Model B"]
    run_id: str
    model_id: str
    status: RunStatus
    expected_count: int = Field(gt=0)
    persisted_count: int = Field(ge=0)
    usable_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    request_error_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    elapsed_wall_clock_ms: float = Field(ge=0)
    known_cost_usd: Decimal
    known_cost_count: int = Field(ge=0)
    unknown_cost_count: int = Field(ge=0)


class LiveComparisonExecution(StrictModel):
    configuration: LiveComparisonConfiguration
    model_a: ModelEvaluationExecution
    model_b: ModelEvaluationExecution
    comparison_complete: bool

    @model_validator(mode="after")
    def validate_pair(self) -> "LiveComparisonExecution":
        if self.model_a.manifest.run_id == self.model_b.manifest.run_id:
            raise ValueError("Live Comparison requires two fresh run IDs")
        if self.model_a.manifest.model_id != self.configuration.model_a_id:
            raise ValueError("Model A execution disagrees with configuration")
        if self.model_b.manifest.model_id != self.configuration.model_b_id:
            raise ValueError("Model B execution disagrees with configuration")
        complete = (
            self.model_a.manifest.status is RunStatus.COMPLETE
            and self.model_b.manifest.status is RunStatus.COMPLETE
        )
        if self.comparison_complete != complete:
            raise ValueError("Live Comparison completeness disagrees with its runs")
        return self

    @property
    def run_ids(self) -> tuple[str, str]:
        return self.model_a.manifest.run_id, self.model_b.manifest.run_id


class _LiveSources(StrictModel):
    corpus_manifest: CorpusManifest
    issues: tuple[Issue, ...]
    ground_truth_manifest: GroundTruthManifest
    annotations: tuple[GroundTruthAnnotation, ...]
    catalog_manifest: ModelCatalogManifest
    pricing_manifest: PricingSnapshotManifest
    contract: SharedInferenceContract


def prepare_live_comparison(settings: Settings) -> LiveComparisonPreparation:
    sources = _load_live_sources(settings)
    contract_manifest = sources.contract.manifest
    blockers: list[str] = []
    if contract_manifest.contract_status != "frozen":
        blockers.append(
            "The Shared Inference Contract is still in prompt development; "
            "Issue 12 must freeze it before full-Corpus inference."
        )
    scored_count = sources.ground_truth_manifest.annotation_count
    return LiveComparisonPreparation(
        corpus_version=sources.corpus_manifest.corpus_version,
        corpus_sha256=sources.corpus_manifest.artifact_sha256,
        repository=sources.corpus_manifest.repository,
        corpus_issue_count=sources.corpus_manifest.issue_count,
        expected_provider_requests=2 * sources.corpus_manifest.issue_count,
        ground_truth_version=sources.ground_truth_manifest.ground_truth_version,
        ground_truth_sha256=sources.ground_truth_manifest.artifact_sha256,
        scored_issue_count=scored_count,
        unscored_issue_count=sources.corpus_manifest.issue_count - scored_count,
        model_catalog_version=sources.catalog_manifest.catalog_version,
        model_catalog_sha256=sources.catalog_manifest.content_sha256,
        eligible_model_ids=APPROVED_ELIGIBLE_MODEL_IDS,
        pricing_snapshot_id=sources.pricing_manifest.pricing_snapshot_id,
        pricing_snapshot_sha256=sources.pricing_manifest.content_sha256,
        contract_version=contract_manifest.contract_version,
        contract_status=contract_manifest.contract_status,
        prompt_version=contract_manifest.prompt_version,
        prompt_sha256=contract_manifest.system_message_sha256,
        parser_version=contract_manifest.parser_version,
        generation_configuration_version=(
            sources.contract.generation_configuration.configuration_version
        ),
        generation_configuration_sha256=(
            contract_manifest.generation_configuration_sha256
        ),
        max_completion_tokens=(
            sources.contract.generation_configuration.max_completion_tokens
        ),
        shared_timeout_seconds=settings.shared_timeout_seconds,
        default_concurrency=settings.live_default_concurrency,
        can_start=not blockers,
        blockers=tuple(blockers),
    )


async def execute_live_comparison(
    settings: Settings,
    configuration: LiveComparisonConfiguration,
    *,
    api_key: str,
    progress_callback: LiveProgressCallback | None = None,
    client_factory: LiveClientFactory | None = None,
) -> LiveComparisonExecution:
    """Create two fresh, sequential full-Corpus Model Evaluation Runs."""

    if not api_key:
        raise ValueError("DigitalOcean API key must not be empty")
    sources = _load_live_sources(settings)
    if sources.contract.manifest.contract_status != "frozen":
        raise ValueError(
            "Live Comparison requires a frozen Shared Inference Contract"
        )
    eligible = set(APPROVED_ELIGIBLE_MODEL_IDS)
    if configuration.model_a_id not in eligible or configuration.model_b_id not in eligible:
        raise ValueError("Live Comparison models must come from the Eligible Candidate Pool")

    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    labels = {
        annotation.issue_number: annotation.label
        for annotation in sources.annotations
    }
    make_client = client_factory or httpx.AsyncClient

    async def run_one(
        position: Literal["Model A", "Model B"], model_id: str
    ) -> ModelEvaluationExecution:
        model_configuration = ModelEvaluationConfiguration(
            schema_version="model_evaluation_configuration.v1",
            model_id=model_id,
            timeout_seconds=settings.shared_timeout_seconds,
            concurrency=configuration.concurrency,
            ground_truth_version=(
                sources.ground_truth_manifest.ground_truth_version
            ),
            ground_truth_sha256=sources.ground_truth_manifest.artifact_sha256,
            ground_truth_labels=labels,
        )

        def report(progress: ModelEvaluationProgress) -> None:
            if progress_callback is None:
                return
            attempts = repository.get_attempts(progress.run_id)
            known_costs = tuple(
                attempt.calculated_request_cost_usd
                for attempt in attempts
                if attempt.calculated_request_cost_usd is not None
            )
            progress_callback(
                LiveModelProgress(
                    position=position,
                    run_id=progress.run_id,
                    model_id=progress.model_id,
                    status=progress.status,
                    expected_count=progress.expected_count,
                    persisted_count=progress.persisted_count,
                    usable_count=progress.usable_count,
                    invalid_output_count=progress.invalid_output_count,
                    request_error_count=progress.request_error_count,
                    failure_count=(
                        progress.invalid_output_count + progress.request_error_count
                    ),
                    elapsed_wall_clock_ms=progress.elapsed_wall_clock_ms,
                    known_cost_usd=sum(known_costs, Decimal("0")),
                    known_cost_count=len(known_costs),
                    unknown_cost_count=len(attempts) - len(known_costs),
                )
            )

        async with make_client() as client:
            return await execute_model_evaluation(
                settings,
                sources.corpus_manifest,
                sources.issues,
                model_configuration,
                api_key=api_key,
                client=client,
                repository=repository,
                contract=sources.contract,
                progress_callback=report,
                evaluation_stage="live_comparison",
            )

    model_a = await run_one("Model A", configuration.model_a_id)
    # Model B starts only after Model A reaches a terminal state. Each model gets
    # the displayed concurrency without a hidden combined 2C request pool.
    model_b = await run_one("Model B", configuration.model_b_id)
    return LiveComparisonExecution(
        configuration=configuration,
        model_a=model_a,
        model_b=model_b,
        comparison_complete=(
            model_a.manifest.status is RunStatus.COMPLETE
            and model_b.manifest.status is RunStatus.COMPLETE
        ),
    )


def _load_live_sources(settings: Settings) -> _LiveSources:
    corpus_manifest, issues = CorpusArtifacts(settings.corpus_root).load_active()
    ground_truth_manifest, annotations = EvaluationCorpusArtifacts(
        settings.ground_truth_root
    ).load_version(
        settings.evaluation_ground_truth_version,
        corpus_manifest,
        issues,
    )
    catalog_manifest, _, _ = ModelCatalogArtifacts(
        settings.model_catalog_directory
    ).load()
    pricing_manifest, _ = PricingArtifacts(
        settings.pricing_directory
    ).load_for_catalog(catalog_manifest)
    contract = load_shared_inference_contract(settings.shared_contract_directory)
    return _LiveSources(
        corpus_manifest=corpus_manifest,
        issues=issues,
        ground_truth_manifest=ground_truth_manifest,
        annotations=annotations,
        catalog_manifest=catalog_manifest,
        pricing_manifest=pricing_manifest,
        contract=contract,
    )
