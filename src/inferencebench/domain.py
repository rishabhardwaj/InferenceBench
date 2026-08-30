from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveMilliseconds = Annotated[float, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CustomerLabel(StrEnum):
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    QUESTION = "question"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    OTHER = "other"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class AttemptPurpose(StrEnum):
    BENCHMARK = "benchmark"
    PREFLIGHT = "preflight"


class ParseStatus(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    INVALID = "invalid"


class ProviderOutcome(StrEnum):
    SUCCESS = "success"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN = "unknown"


class ScoredOutcome(StrEnum):
    CORRECT = "correct"
    INCORRECT_LABEL = "incorrect_label"
    INVALID_OUTPUT = "invalid_output"
    REQUEST_ERROR = "request_error"


class CostCompleteness(StrEnum):
    COMPLETE = "complete"
    UNKNOWN = "unknown"


class CostCategory(StrEnum):
    STANDARD_INPUT = "standard_input"
    OUTPUT = "output"
    CACHE_READ_INPUT = "cache_read_input"
    CACHE_WRITE_INPUT = "cache_write_input"


class CostCalculationTerm(StrictModel):
    category: CostCategory
    usage_path: str
    token_count: Annotated[int, Field(ge=0)]
    rate_usd: Annotated[Decimal, Field(gt=0)]
    unit_tokens: Annotated[int, Field(gt=0)]
    cost_usd: Annotated[Decimal, Field(ge=0)]

    @model_validator(mode="after")
    def validate_cost(self) -> "CostCalculationTerm":
        expected = Decimal(self.token_count) * self.rate_usd / self.unit_tokens
        if self.cost_usd != expected:
            raise ValueError("Cost term does not match its tokens, rate, and unit")
        return self


class GitHubLabel(StrictModel):
    label_id: int
    name: str
    description: str | None = None
    color: str


class Issue(StrictModel):
    schema_version: Literal["issue.v1"]
    corpus_version: str
    repository: str
    github_issue_id: int
    issue_number: Annotated[int, Field(gt=0)]
    node_id: str
    api_url: str
    html_url: str
    title: str
    body: str | None
    state: Literal["open", "closed"]
    state_reason: str | None = None
    labels: tuple[GitHubLabel, ...] = ()
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    content_sha256: Sha256


class CorpusManifest(StrictModel):
    schema_version: Literal["corpus_manifest.v1"]
    corpus_version: str
    repository: str
    endpoint_url: str
    query_parameters: dict[str, str]
    github_api_version: str
    retrieval_started_at: datetime
    retrieval_completed_at: datetime
    page_count: Annotated[int, Field(gt=0)]
    api_object_count: Annotated[int, Field(gt=0)]
    excluded_pull_request_count: Annotated[int, Field(ge=0)]
    artifact_file: str
    artifact_sha256: Sha256
    artifact_byte_count: Annotated[int, Field(gt=0)]
    issue_count: Annotated[int, Field(gt=0)]
    ordered_issue_numbers: tuple[Annotated[int, Field(gt=0)], ...]
    ordering: Literal["issue_number_ascending"]
    serialization: Literal[
        "utf-8-json-lines-sorted-keys-compact-lf",
        "utf-8-json-lines-field-order-compact-lf",
    ]
    pagination: Literal["github-link-header-until-exhaustion"]

    @model_validator(mode="after")
    def validate_order(self) -> "CorpusManifest":
        if len(self.ordered_issue_numbers) != self.issue_count:
            raise ValueError("issue_count must match ordered_issue_numbers")
        if tuple(sorted(self.ordered_issue_numbers)) != self.ordered_issue_numbers:
            raise ValueError("Corpus issue numbers must be ascending")
        if len(set(self.ordered_issue_numbers)) != self.issue_count:
            raise ValueError("Corpus issue numbers must be unique")
        if self.api_object_count != self.issue_count + self.excluded_pull_request_count:
            raise ValueError("API object count must equal retained issues plus excluded PRs")
        if self.retrieval_completed_at < self.retrieval_started_at:
            raise ValueError("Corpus retrieval interval cannot end before it starts")
        if self.query_parameters.get("state") != "all":
            raise ValueError("Corpus query must include open and closed issues")
        if self.query_parameters.get("per_page") != "100":
            raise ValueError("Corpus query must use GitHub's maximum page size")
        if PurePosixPath(self.artifact_file).name != self.artifact_file:
            raise ValueError("Corpus artifact_file must be a local filename")
        return self


class ActiveCorpusPointer(StrictModel):
    schema_version: Literal["active_corpus.v1"]
    corpus_version: str
    manifest_file: str
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest_path(self) -> "ActiveCorpusPointer":
        expected = f"{self.corpus_version}/manifest.json"
        if self.manifest_file != expected:
            raise ValueError(f"Active Corpus manifest_file must be {expected}")
        return self


class GroundTruthAnnotation(StrictModel):
    schema_version: Literal["ground_truth_annotation.v1"]
    annotation_id: str
    corpus_version: str
    issue_number: Annotated[int, Field(gt=0)]
    label: CustomerLabel
    ground_truth_source: Literal["human_review", "closed_issue_maintainer_evidence"]
    sampling_stratum: Literal["random", "diagnostic", "mapping_audit", "closed_maintainer"]
    evaluation_role: Literal[
        "prompt_development", "primary_holdout", "diagnostic", "mapping_audit", "sensitivity"
    ]
    rubric_version: str
    confidence: Literal["high", "medium", "low"]
    review_status: Literal["accepted"]
    review_pass_count: Annotated[int, Field(ge=1)]
    input_sufficiency: Literal["sufficient"]
    reviewed_at: datetime
    annotation_sha256: Sha256


class GroundTruthManifest(StrictModel):
    schema_version: Literal["ground_truth_manifest.v1"]
    ground_truth_version: str
    corpus_version: str
    artifact_file: str
    artifact_sha256: Sha256
    annotation_count: Annotated[int, Field(gt=0)]
    ordered_issue_numbers: tuple[Annotated[int, Field(gt=0)], ...]

    @model_validator(mode="after")
    def validate_count(self) -> "GroundTruthManifest":
        if len(self.ordered_issue_numbers) != self.annotation_count:
            raise ValueError("annotation_count must match ordered_issue_numbers")
        if len(set(self.ordered_issue_numbers)) != self.annotation_count:
            raise ValueError("Ground Truth issue numbers must be unique")
        return self


class HumanReviewAnnotation(StrictModel):
    """One auditable solo-review outcome, including reviewed exclusions."""

    schema_version: Literal["human_review_annotation.v1"]
    annotation_id: str
    corpus_version: str
    issue_number: Annotated[int, Field(gt=0)]
    random_order_position: Annotated[int, Field(gt=0)]
    sampling_stratum: Literal["random"]
    evaluation_role: Literal["prompt_development", "primary_holdout", "excluded"]
    initial_label: CustomerLabel | None
    final_label: CustomerLabel | None
    confidence: Literal["high", "medium", "low"] | None
    review_status: Literal["accepted", "unresolved", "excluded"]
    review_pass_count: Annotated[int, Field(ge=1)]
    requires_second_pass: bool = False
    quality_control_reviewed: bool = False
    input_sufficiency: Literal["sufficient", "insufficient"]
    exclusion_reason: str | None = None
    review_notes: str | None = None
    rubric_version: str
    rubric_sha256: Sha256
    reviewed_at: datetime
    second_pass_reviewed_at: datetime | None = None
    annotation_sha256: Sha256

    @model_validator(mode="after")
    def validate_review_outcome(self) -> "HumanReviewAnnotation":
        accepted = self.review_status == "accepted"
        if accepted:
            if (
                self.final_label is None
                or self.confidence is None
                or self.input_sufficiency != "sufficient"
                or self.evaluation_role == "excluded"
                or self.exclusion_reason is not None
            ):
                raise ValueError("Accepted review requires a sufficient final label and role")
        elif (
            self.final_label is not None
            or self.evaluation_role != "excluded"
            or self.exclusion_reason is None
        ):
            raise ValueError("Excluded review requires a reason and no final label")
        if self.requires_second_pass and self.review_pass_count < 2:
            raise ValueError("Second-pass review requires review_pass_count of at least 2")
        if self.quality_control_reviewed and self.review_pass_count < 2:
            raise ValueError("Quality-control review requires review_pass_count of at least 2")
        if self.review_pass_count >= 2 and self.second_pass_reviewed_at is None:
            raise ValueError("Second-pass timestamp is required when review_pass_count is at least 2")
        if self.initial_label != self.final_label and self.review_status == "accepted":
            if not self.requires_second_pass:
                raise ValueError("Changed accepted judgment requires a recorded second pass")
        return self


class HumanReviewManifest(StrictModel):
    """Identity, seeds, and counts for one completed random human-review artifact."""

    schema_version: Literal["human_review_manifest.v1"]
    ground_truth_version: str
    corpus_version: str
    corpus_sha256: Sha256
    rubric_version: str
    rubric_sha256: Sha256
    random_order_seed: int
    partition_seed: int
    quality_control_seed: int
    artifact_file: str
    artifact_sha256: Sha256
    reviewed_count: Annotated[int, Field(ge=100)]
    accepted_count: Literal[100]
    excluded_count: Annotated[int, Field(ge=0)]
    prompt_development_count: Literal[20]
    primary_holdout_count: Literal[80]
    quality_control_count: Literal[20]
    random_ordered_issue_numbers: tuple[Annotated[int, Field(gt=0)], ...]
    reviewed_issue_numbers: tuple[Annotated[int, Field(gt=0)], ...]

    @model_validator(mode="after")
    def validate_counts(self) -> "HumanReviewManifest":
        if self.reviewed_count != len(self.reviewed_issue_numbers):
            raise ValueError("reviewed_count must match reviewed_issue_numbers")
        if self.reviewed_count != self.accepted_count + self.excluded_count:
            raise ValueError("reviewed_count must equal accepted plus excluded")
        if len(set(self.reviewed_issue_numbers)) != self.reviewed_count:
            raise ValueError("Reviewed issue numbers must be unique")
        if len(set(self.random_ordered_issue_numbers)) != len(
            self.random_ordered_issue_numbers
        ):
            raise ValueError("Random order issue numbers must be unique")
        return self


class HumanReviewDraftEntry(StrictModel):
    """Mutable local work record; finalized annotations are immutable evidence."""

    issue_number: Annotated[int, Field(gt=0)]
    initial_label: CustomerLabel | None
    final_label: CustomerLabel | None
    confidence: Literal["high", "medium", "low"] | None
    review_status: Literal["accepted", "unresolved", "excluded"]
    review_pass_count: Annotated[int, Field(ge=1)]
    requires_second_pass: bool = False
    input_sufficiency: Literal["sufficient", "insufficient"]
    exclusion_reason: str | None = None
    review_notes: str | None = None
    reviewed_at: datetime
    second_pass_reviewed_at: datetime | None = None


class HumanReviewDraft(StrictModel):
    schema_version: Literal["human_review_draft.v1"]
    corpus_version: str
    corpus_sha256: Sha256
    rubric_version: str
    rubric_sha256: Sha256
    random_order_seed: int
    partition_seed: int
    quality_control_seed: int
    entries: tuple[HumanReviewDraftEntry, ...] = ()


class RunManifest(StrictModel):
    schema_version: Literal["run_manifest.v1", "run_manifest.v2"]
    run_id: str
    run_type: Literal["fixture_saved_comparison", "model_evaluation"]
    model_id: str
    provider_endpoint_id: str
    status: RunStatus
    corpus_version: str
    corpus_sha256: Sha256
    ordered_issue_numbers: tuple[Annotated[int, Field(gt=0)], ...]
    issue_order_sha256: Sha256
    ground_truth_version: str
    ground_truth_sha256: Sha256
    prompt_version: str
    parser_version: str
    rubric_version: str
    prompt_sha256: Sha256 | None = None
    rubric_sha256: Sha256 | None = None
    generation_configuration: dict[str, JsonValue]
    generation_configuration_sha256: Sha256
    model_catalog_version: str
    model_catalog_sha256: Sha256
    pricing_snapshot_id: str
    pricing_snapshot_sha256: Sha256
    timeout_seconds: Annotated[float, Field(gt=0)]
    concurrency: Annotated[int, Field(gt=0)]
    retries: Literal[0]
    stream: Literal[False]
    application_version: str
    schema_revision: Annotated[int, Field(gt=0)]
    metric_version: str
    dependency_lock_sha256: Sha256
    source_sha256: Sha256
    started_at: datetime
    ended_at: datetime | None
    wall_clock_ms: PositiveMilliseconds | None
    expected_count: Annotated[int, Field(gt=0)]
    persisted_count: Annotated[int, Field(ge=0)]
    usable_count: Annotated[int, Field(ge=0)]
    normalized_count: Annotated[int, Field(ge=0)]
    invalid_output_count: Annotated[int, Field(ge=0)]
    request_error_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_lifecycle_and_counts(self) -> "RunManifest":
        if len(self.ordered_issue_numbers) != self.expected_count:
            raise ValueError("expected_count must match ordered issue population")
        if len(set(self.ordered_issue_numbers)) != self.expected_count:
            raise ValueError("Run issue population must be unique")
        terminal_total = self.usable_count + self.invalid_output_count + self.request_error_count
        if terminal_total != self.persisted_count:
            raise ValueError("terminal outcome counts must match persisted_count")
        if self.normalized_count > self.usable_count:
            raise ValueError("normalized_count cannot exceed usable_count")
        if self.persisted_count > self.expected_count:
            raise ValueError("persisted_count cannot exceed expected_count")
        if self.schema_version == "run_manifest.v2" and self.run_type == "model_evaluation":
            if self.prompt_sha256 is None or self.rubric_sha256 is None:
                raise ValueError(
                    "Generated Model Evaluation Run requires prompt and rubric hashes"
                )
        if self.status is RunStatus.RUNNING:
            if self.ended_at is not None or self.wall_clock_ms is not None:
                raise ValueError("Running run cannot have terminal timing")
            if any(
                count != 0
                for count in (
                    self.persisted_count,
                    self.usable_count,
                    self.normalized_count,
                    self.invalid_output_count,
                    self.request_error_count,
                )
            ):
                raise ValueError("Running manifest counts remain zero until terminal")
        else:
            if self.ended_at is None or self.wall_clock_ms is None:
                raise ValueError("Terminal run requires end time and wall clock")
        if self.status is RunStatus.COMPLETE:
            if self.persisted_count != self.expected_count:
                raise ValueError("complete run must persist every expected attempt")
        return self

    def comparison_identity(self) -> tuple[object, ...]:
        return (
            self.corpus_version,
            self.corpus_sha256,
            self.ordered_issue_numbers,
            self.issue_order_sha256,
            self.ground_truth_version,
            self.ground_truth_sha256,
            self.prompt_version,
            self.prompt_sha256,
            self.parser_version,
            self.rubric_version,
            self.rubric_sha256,
            self.generation_configuration_sha256,
            self.model_catalog_version,
            self.model_catalog_sha256,
            self.pricing_snapshot_id,
            self.pricing_snapshot_sha256,
            self.timeout_seconds,
            self.concurrency,
            self.retries,
            self.stream,
            self.metric_version,
        )


class RequestMessage(StrictModel):
    role: Literal["system", "user"]
    content: str


class AttemptEvidence(StrictModel):
    schema_version: Literal[
        "attempt_evidence.v1", "attempt_evidence.v2", "attempt_evidence.v3"
    ]
    attempt_id: str
    run_id: str
    issue_number: Annotated[int, Field(gt=0)]
    dispatch_order: Annotated[int, Field(ge=0)]
    attempt_purpose: AttemptPurpose
    request_messages: tuple[RequestMessage, ...]
    request_messages_sha256: Sha256
    request_parameters: dict[str, JsonValue]
    eligible_at: datetime
    request_started_at: datetime
    request_ended_at: datetime
    queue_wait_ms: PositiveMilliseconds
    request_latency_ms: PositiveMilliseconds
    configured_timeout_seconds: Annotated[float, Field(gt=0)]
    provider_outcome: ProviderOutcome | None = None
    http_status: int | None
    provider_request_id: str | None
    finish_reason: str | None
    response_headers: dict[str, str]
    raw_response: JsonValue | None
    raw_error: dict[str, JsonValue] | None
    raw_model_output: str | None
    parsed_label: CustomerLabel | None
    parse_status: ParseStatus
    normalizations: tuple[str, ...]
    scored_outcome: ScoredOutcome | None
    usable: bool
    usage: dict[str, JsonValue]
    pricing_snapshot_id: str
    cost_formula_version: str
    calculated_request_cost_usd: Decimal | None
    cost_completeness: CostCompleteness
    cost_calculation_terms: tuple[CostCalculationTerm, ...] = ()
    cost_unknown_reasons: tuple[str, ...] = ()

    @field_validator("response_headers")
    @classmethod
    def reject_sensitive_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        sensitive = {"authorization", "proxy-authorization", "x-api-key"}
        if any(name.lower() in sensitive for name in headers):
            raise ValueError("sensitive headers are not evidence")
        return headers

    @model_validator(mode="after")
    def validate_terminal_outcome(self) -> "AttemptEvidence":
        if self.schema_version == "attempt_evidence.v1":
            if self.scored_outcome is None:
                raise ValueError("v1 fixture attempt requires a Scored Outcome")
        elif self.provider_outcome is None:
            raise ValueError("Generated Attempt Evidence requires a Provider Outcome")
        if self.usable != (self.parsed_label is not None):
            raise ValueError("usable must match presence of a parsed label")
        if self.parse_status is ParseStatus.INVALID and self.parsed_label is not None:
            raise ValueError("invalid parse cannot have a parsed label")
        if self.scored_outcome in {
            ScoredOutcome.CORRECT,
            ScoredOutcome.INCORRECT_LABEL,
        } and not self.usable:
            raise ValueError("label outcomes require a usable classification")
        if self.scored_outcome is ScoredOutcome.INVALID_OUTPUT and self.usable:
            raise ValueError("invalid output cannot be usable")
        if self.provider_outcome is not None:
            if self.provider_outcome is ProviderOutcome.SUCCESS:
                if self.scored_outcome is ScoredOutcome.REQUEST_ERROR:
                    raise ValueError("successful provider response is not a request error")
            elif self.usable or self.raw_model_output is not None:
                raise ValueError("provider request error cannot produce model output")
            if (
                self.scored_outcome is ScoredOutcome.INVALID_OUTPUT
                and self.provider_outcome is not ProviderOutcome.SUCCESS
            ):
                raise ValueError("invalid output requires a successful provider response")
        if self.cost_completeness is CostCompleteness.COMPLETE:
            if self.calculated_request_cost_usd is None:
                raise ValueError("complete cost requires a value")
            if self.cost_unknown_reasons:
                raise ValueError("complete cost cannot retain unknown reasons")
            if self.cost_calculation_terms and self.calculated_request_cost_usd != sum(
                (term.cost_usd for term in self.cost_calculation_terms), Decimal("0")
            ):
                raise ValueError("Calculated Request Cost must equal its exact terms")
        elif self.calculated_request_cost_usd is not None:
            raise ValueError("unknown cost cannot have a calculated value")
        if self.schema_version == "attempt_evidence.v3":
            if (
                self.cost_completeness is CostCompleteness.UNKNOWN
                and not self.cost_unknown_reasons
            ):
                raise ValueError("v3 unknown cost requires an explicit reason")
        return self


class FixtureRunBundle(StrictModel):
    schema_version: Literal["fixture_run_bundle.v1"]
    default_run_ids: tuple[str, str]
    runs: tuple[RunManifest, RunManifest]
    attempts: tuple[AttemptEvidence, ...]

    @model_validator(mode="after")
    def validate_references(self) -> "FixtureRunBundle":
        run_ids = {run.run_id for run in self.runs}
        if set(self.default_run_ids) != run_ids:
            raise ValueError("default_run_ids must identify both fixture runs")
        if any(attempt.run_id not in run_ids for attempt in self.attempts):
            raise ValueError("fixture attempt references an unknown run")
        return self
