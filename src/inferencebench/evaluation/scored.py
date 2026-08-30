from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from inferencebench.domain import (
    AttemptEvidence,
    AttemptPurpose,
    CustomerLabel,
    GroundTruthAnnotation,
    GroundTruthManifest,
    HumanReviewAnnotation,
    HumanReviewManifest,
    Issue,
    ProviderOutcome,
    RunManifest,
    RunStatus,
    ScoredOutcome,
    Sha256,
    StrictModel,
)
from inferencebench.evaluation.metrics import MetricIntegrityError, assert_comparable_runs


NO_VALID_PREDICTION = "No Valid Prediction"
CUSTOMER_LABEL_ORDER = tuple(CustomerLabel)


class ScoredEvidenceSource(StrEnum):
    HUMAN_REVIEW = "human_review"
    CLOSED_ISSUE_MAINTAINER = "closed_issue_maintainer_evidence"


class ScoredSamplingStratum(StrEnum):
    RANDOM = "random"
    DIAGNOSTIC = "diagnostic"
    MAPPING_AUDIT = "mapping_audit"
    CLOSED_ISSUE_MAINTAINER = "closed_issue_maintainer"


class ScoredEvaluationRole(StrEnum):
    PROMPT_DEVELOPMENT = "prompt_development"
    PRIMARY_HOLDOUT = "primary_holdout"
    DIAGNOSTIC = "diagnostic"
    MAPPING_AUDIT = "mapping_audit"
    CLOSED_ISSUE_MAINTAINER = "closed_issue_maintainer"


class ScoredPopulation(StrEnum):
    PRIMARY_HOLDOUT = "primary_holdout"
    PROMPT_DEVELOPMENT = "prompt_development"
    DIAGNOSTIC = "diagnostic"
    MAPPING_AUDIT = "mapping_audit"
    CLOSED_ISSUE_MAINTAINER = "closed_issue_maintainer"
    COMBINED_DESCRIPTIVE = "combined_descriptive"

    @property
    def display_name(self) -> str:
        return {
            self.PRIMARY_HOLDOUT: "Primary Scored Holdout",
            self.PROMPT_DEVELOPMENT: "Prompt Development Sample",
            self.DIAGNOSTIC: "Diagnostic Scored Supplement",
            self.MAPPING_AUDIT: "Mapping Audit Top-Up",
            self.CLOSED_ISSUE_MAINTAINER: "Closed-Issue Maintainer Evidence",
            self.COMBINED_DESCRIPTIVE: "Combined descriptive evidence",
        }[self]


class ScoredGroundTruth(StrictModel):
    issue_number: int = Field(gt=0)
    label: CustomerLabel
    ground_truth_source: ScoredEvidenceSource
    sampling_stratum: ScoredSamplingStratum
    evaluation_role: ScoredEvaluationRole
    provenance_id: str
    rubric_version: str
    confidence: Literal["high", "medium", "low"] | None = None
    review_pass_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> "ScoredGroundTruth":
        if self.ground_truth_source is ScoredEvidenceSource.CLOSED_ISSUE_MAINTAINER:
            if (
                self.sampling_stratum
                is not ScoredSamplingStratum.CLOSED_ISSUE_MAINTAINER
                or self.evaluation_role
                is not ScoredEvaluationRole.CLOSED_ISSUE_MAINTAINER
            ):
                raise ValueError(
                    "Closed-Issue Maintainer Evidence requires its own stratum and role"
                )
        return self


class ScoredGroundTruthSet(StrictModel):
    ground_truth_version: str
    artifact_sha256: Sha256
    items: tuple[ScoredGroundTruth, ...]

    @model_validator(mode="after")
    def validate_unique_items(self) -> "ScoredGroundTruthSet":
        issue_numbers = tuple(item.issue_number for item in self.items)
        if len(set(issue_numbers)) != len(issue_numbers):
            raise ValueError("Scored Ground Truth issue numbers must be unique")
        return self


class ScoredClassMetrics(StrictModel):
    label: CustomerLabel
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    support: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    precision: float | None
    recall: float | None
    f1: float | None


class ConfusionCell(StrictModel):
    ground_truth: CustomerLabel
    prediction: CustomerLabel | None
    issue_numbers: tuple[int, ...]

    @property
    def display_prediction(self) -> str:
        return self.prediction.value if self.prediction else NO_VALID_PREDICTION


class ScoredModelSummary(StrictModel):
    model_id: str
    run_id: str
    expected_count: int = Field(gt=0)
    correct_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    supported_class_macro_f1: float = Field(ge=0, le=1)
    supported_class_count: int = Field(ge=1, le=6)
    total_class_count: Literal[6]
    invalid_output_count: int = Field(ge=0)
    request_error_count: int = Field(ge=0)
    scored_outcome_counts: dict[ScoredOutcome, int]
    request_error_type_counts: dict[ProviderOutcome, int]
    per_class: tuple[ScoredClassMetrics, ...]
    confusion_cells: tuple[ConfusionCell, ...]

    @model_validator(mode="after")
    def validate_complete_summary(self) -> "ScoredModelSummary":
        if self.correct_count > self.expected_count:
            raise ValueError("Correct count cannot exceed expected count")
        if self.accuracy != self.correct_count / self.expected_count:
            raise ValueError("Accuracy must match correct and expected counts")
        if len(self.per_class) != len(CUSTOMER_LABEL_ORDER):
            raise ValueError("Per-class metrics must contain all six Customer Taxonomy labels")
        if tuple(row.label for row in self.per_class) != CUSTOMER_LABEL_ORDER:
            raise ValueError("Per-class metrics must use Customer Taxonomy order")
        if len(self.confusion_cells) != len(CUSTOMER_LABEL_ORDER) * 7:
            raise ValueError("Confusion evidence must contain six rows and seven columns")
        expected_cells = {
            (truth, prediction)
            for truth in CUSTOMER_LABEL_ORDER
            for prediction in (*CUSTOMER_LABEL_ORDER, None)
        }
        actual_cells = {
            (cell.ground_truth, cell.prediction) for cell in self.confusion_cells
        }
        if actual_cells != expected_cells:
            raise ValueError("Confusion evidence must contain every required cell once")
        if sum(len(cell.issue_numbers) for cell in self.confusion_cells) != self.expected_count:
            raise ValueError("Confusion evidence must cover every expected item")
        if sum(row.support for row in self.per_class) != self.expected_count:
            raise ValueError("Per-class support must cover every expected item")
        if self.supported_class_count != sum(row.support > 0 for row in self.per_class):
            raise ValueError("Supported-class coverage does not match per-class support")
        if sum(self.scored_outcome_counts.values()) != self.expected_count:
            raise ValueError("Scored Outcome counts must cover every expected item")
        if self.invalid_output_count != self.scored_outcome_counts.get(
            ScoredOutcome.INVALID_OUTPUT, 0
        ):
            raise ValueError("Invalid-output count disagrees with Scored Outcomes")
        if self.request_error_count != self.scored_outcome_counts.get(
            ScoredOutcome.REQUEST_ERROR, 0
        ):
            raise ValueError("Request-error count disagrees with Scored Outcomes")
        return self


class ScoredPairRow(StrictModel):
    issue: Issue
    ground_truth: ScoredGroundTruth
    model_a_attempt: AttemptEvidence
    model_b_attempt: AttemptEvidence
    model_a_outcome: ScoredOutcome
    model_b_outcome: ScoredOutcome
    predictions_agree: bool
    pair_outcome: Literal[
        "label_agreement",
        "label_disagreement",
        "model_a_no_valid_prediction",
        "model_b_no_valid_prediction",
        "joint_no_valid_prediction",
    ]
    correctness_pattern: Literal[
        "both_correct",
        "model_a_only_correct",
        "model_b_only_correct",
        "neither_correct",
    ]


class ScoredComparison(StrictModel):
    population: ScoredPopulation
    population_display_name: str
    is_primary_headline: bool
    interpretation: str
    ground_truth_version: str
    ground_truth_sha256: Sha256
    run_a: RunManifest
    run_b: RunManifest
    model_a: ScoredModelSummary
    model_b: ScoredModelSummary
    rows: tuple[ScoredPairRow, ...]

    @model_validator(mode="after")
    def validate_pair(self) -> "ScoredComparison":
        if self.model_a.run_id != self.run_a.run_id:
            raise ValueError("Model A summary references the wrong run")
        if self.model_b.run_id != self.run_b.run_id:
            raise ValueError("Model B summary references the wrong run")
        if self.model_a.expected_count != len(self.rows):
            raise ValueError("Model A summary does not cover every Scored View row")
        if self.model_b.expected_count != len(self.rows):
            raise ValueError("Model B summary does not cover every Scored View row")
        return self


class ScoredViewReview(StrictModel):
    default_population: ScoredPopulation
    comparisons: tuple[ScoredComparison, ...]

    @model_validator(mode="after")
    def validate_default(self) -> "ScoredViewReview":
        if not self.comparisons:
            raise ValueError("Scored View requires at least one population")
        populations = tuple(item.population for item in self.comparisons)
        if len(set(populations)) != len(populations):
            raise ValueError("Scored View populations must be unique")
        if self.default_population not in populations:
            raise ValueError("Default Scored View population is unavailable")
        return self


class ScoredRowFilter(StrictModel):
    ground_truth_labels: tuple[CustomerLabel, ...] = ()
    model_a_predictions: tuple[CustomerLabel | None, ...] = ()
    model_b_predictions: tuple[CustomerLabel | None, ...] = ()
    model_a_outcomes: tuple[ScoredOutcome, ...] = ()
    model_b_outcomes: tuple[ScoredOutcome, ...] = ()
    evidence_sources: tuple[ScoredEvidenceSource, ...] = ()
    sampling_strata: tuple[ScoredSamplingStratum, ...] = ()
    agreement: Literal["all", "agreement", "disagreement"] = "all"
    text_query: str = ""


def scored_ground_truth_from_fixture(
    manifest: GroundTruthManifest,
    annotations: tuple[GroundTruthAnnotation, ...],
) -> ScoredGroundTruthSet:
    return ScoredGroundTruthSet(
        ground_truth_version=manifest.ground_truth_version,
        artifact_sha256=manifest.artifact_sha256,
        items=tuple(
            ScoredGroundTruth(
                issue_number=annotation.issue_number,
                label=annotation.label,
                ground_truth_source=ScoredEvidenceSource.HUMAN_REVIEW,
                sampling_stratum=ScoredSamplingStratum.RANDOM,
                evaluation_role=ScoredEvaluationRole.PRIMARY_HOLDOUT,
                provenance_id=annotation.annotation_id,
                rubric_version=annotation.rubric_version,
                confidence=annotation.confidence,
                review_pass_count=annotation.review_pass_count,
            )
            for annotation in annotations
        ),
    )


def scored_ground_truth_from_evaluation_corpus(
    manifest: GroundTruthManifest,
    annotations: tuple[GroundTruthAnnotation, ...],
) -> ScoredGroundTruthSet:
    """Preserve each accepted Evaluation Corpus row's purpose and provenance."""

    return ScoredGroundTruthSet(
        ground_truth_version=manifest.ground_truth_version,
        artifact_sha256=manifest.artifact_sha256,
        items=tuple(
            ScoredGroundTruth(
                issue_number=annotation.issue_number,
                label=annotation.label,
                ground_truth_source=ScoredEvidenceSource(
                    annotation.ground_truth_source
                ),
                sampling_stratum=ScoredSamplingStratum(
                    annotation.sampling_stratum
                ),
                evaluation_role=ScoredEvaluationRole(
                    annotation.evaluation_role
                ),
                provenance_id=annotation.annotation_id,
                rubric_version=annotation.rubric_version,
                confidence=annotation.confidence,
                review_pass_count=annotation.review_pass_count,
            )
            for annotation in annotations
        ),
    )


def scored_ground_truth_from_human_review(
    manifest: HumanReviewManifest,
    annotations: tuple[HumanReviewAnnotation, ...],
) -> ScoredGroundTruthSet:
    accepted = tuple(
        annotation for annotation in annotations if annotation.review_status == "accepted"
    )
    return ScoredGroundTruthSet(
        ground_truth_version=manifest.ground_truth_version,
        artifact_sha256=manifest.artifact_sha256,
        items=tuple(
            ScoredGroundTruth(
                issue_number=annotation.issue_number,
                label=annotation.final_label,
                ground_truth_source=ScoredEvidenceSource.HUMAN_REVIEW,
                sampling_stratum=ScoredSamplingStratum.RANDOM,
                evaluation_role=ScoredEvaluationRole(annotation.evaluation_role),
                provenance_id=annotation.annotation_id,
                rubric_version=annotation.rubric_version,
                confidence=annotation.confidence,
                review_pass_count=annotation.review_pass_count,
            )
            for annotation in accepted
            if annotation.final_label is not None
        ),
    )


def build_scored_view(
    run_a: RunManifest,
    attempts_a: tuple[AttemptEvidence, ...],
    run_b: RunManifest,
    attempts_b: tuple[AttemptEvidence, ...],
    issues: tuple[Issue, ...],
    ground_truth: ScoredGroundTruthSet,
) -> ScoredViewReview:
    populations = available_scored_populations(ground_truth.items)
    if not populations:
        raise MetricIntegrityError("Scored View requires accepted Ground Truth")
    comparisons = tuple(
        build_scored_comparison(
            run_a,
            attempts_a,
            run_b,
            attempts_b,
            issues,
            ground_truth,
            population,
        )
        for population in populations
    )
    default = (
        ScoredPopulation.PRIMARY_HOLDOUT
        if ScoredPopulation.PRIMARY_HOLDOUT in populations
        else populations[0]
    )
    return ScoredViewReview(default_population=default, comparisons=comparisons)


def build_scored_model_summary(
    run: RunManifest,
    attempts: tuple[AttemptEvidence, ...],
    ground_truth: ScoredGroundTruthSet,
    population: ScoredPopulation,
) -> ScoredModelSummary:
    """Build one candidate's stratum-specific quality from persisted evidence."""

    if run.status is not RunStatus.COMPLETE:
        raise MetricIntegrityError("Scored model summary requires a complete run")
    _assert_ground_truth_identity(run, ground_truth)
    selected_truth = _select_ground_truth(ground_truth.items, population)
    if not selected_truth:
        raise MetricIntegrityError(f"No Ground Truth exists for {population.display_name}")
    attempts_by_issue = _benchmark_attempts_for_run(run, attempts)
    selected_numbers = {item.issue_number for item in selected_truth}
    if not selected_numbers <= set(attempts_by_issue):
        raise MetricIntegrityError(
            f"Attempt Evidence is missing {population.display_name} issues"
        )
    return _calculate_model_summary(run, selected_truth, attempts_by_issue)


def available_scored_populations(
    items: tuple[ScoredGroundTruth, ...],
) -> tuple[ScoredPopulation, ...]:
    base = tuple(
        population
        for population in (
            ScoredPopulation.PRIMARY_HOLDOUT,
            ScoredPopulation.PROMPT_DEVELOPMENT,
            ScoredPopulation.DIAGNOSTIC,
            ScoredPopulation.MAPPING_AUDIT,
            ScoredPopulation.CLOSED_ISSUE_MAINTAINER,
        )
        if _select_ground_truth(items, population)
    )
    if len(base) > 1:
        return (*base, ScoredPopulation.COMBINED_DESCRIPTIVE)
    return base


def build_scored_comparison(
    run_a: RunManifest,
    attempts_a: tuple[AttemptEvidence, ...],
    run_b: RunManifest,
    attempts_b: tuple[AttemptEvidence, ...],
    issues: tuple[Issue, ...],
    ground_truth: ScoredGroundTruthSet,
    population: ScoredPopulation,
) -> ScoredComparison:
    assert_comparable_runs(run_a, run_b)
    _assert_ground_truth_identity(run_a, ground_truth)
    selected_truth = _select_ground_truth(ground_truth.items, population)
    if not selected_truth:
        raise MetricIntegrityError(f"No Ground Truth exists for {population.display_name}")

    issues_by_number = _unique_by_issue_number(issues, "Corpus Issue")
    attempts_by_model = (
        _benchmark_attempts_for_run(run_a, attempts_a),
        _benchmark_attempts_for_run(run_b, attempts_b),
    )
    selected_numbers = {item.issue_number for item in selected_truth}
    if not selected_numbers <= set(issues_by_number):
        raise MetricIntegrityError(
            "Scored Ground Truth references an unavailable Corpus Issue"
        )
    for run, attempt_by_issue in zip((run_a, run_b), attempts_by_model, strict=True):
        if not selected_numbers <= set(run.ordered_issue_numbers):
            raise MetricIntegrityError(
                f"Scored population is not contained in run {run.run_id}"
            )
        if not selected_numbers <= set(attempt_by_issue):
            raise MetricIntegrityError(
                f"Attempt Evidence is missing a scored issue for run {run.run_id}"
            )

    rows = tuple(
        _pair_row(
            issues_by_number[item.issue_number],
            item,
            attempts_by_model[0][item.issue_number],
            attempts_by_model[1][item.issue_number],
        )
        for item in selected_truth
    )
    rows = tuple(
        sorted(rows, key=lambda row: (row.predictions_agree, row.issue.issue_number))
    )
    return ScoredComparison(
        population=population,
        population_display_name=population.display_name,
        is_primary_headline=population is ScoredPopulation.PRIMARY_HOLDOUT,
        interpretation=_population_interpretation(population, selected_truth),
        ground_truth_version=ground_truth.ground_truth_version,
        ground_truth_sha256=ground_truth.artifact_sha256,
        run_a=run_a,
        run_b=run_b,
        model_a=_calculate_model_summary(run_a, selected_truth, attempts_by_model[0]),
        model_b=_calculate_model_summary(run_b, selected_truth, attempts_by_model[1]),
        rows=rows,
    )


def filter_scored_rows(
    rows: tuple[ScoredPairRow, ...], filters: ScoredRowFilter
) -> tuple[ScoredPairRow, ...]:
    query = filters.text_query.strip().casefold()
    selected: list[ScoredPairRow] = []
    for row in rows:
        if (
            filters.ground_truth_labels
            and row.ground_truth.label not in filters.ground_truth_labels
        ):
            continue
        if (
            filters.model_a_predictions
            and row.model_a_attempt.parsed_label not in filters.model_a_predictions
        ):
            continue
        if (
            filters.model_b_predictions
            and row.model_b_attempt.parsed_label not in filters.model_b_predictions
        ):
            continue
        if filters.model_a_outcomes and row.model_a_outcome not in filters.model_a_outcomes:
            continue
        if filters.model_b_outcomes and row.model_b_outcome not in filters.model_b_outcomes:
            continue
        if (
            filters.evidence_sources
            and row.ground_truth.ground_truth_source not in filters.evidence_sources
        ):
            continue
        if (
            filters.sampling_strata
            and row.ground_truth.sampling_stratum not in filters.sampling_strata
        ):
            continue
        if filters.agreement == "agreement" and not row.predictions_agree:
            continue
        if filters.agreement == "disagreement" and row.predictions_agree:
            continue
        if query and query not in f"{row.issue.title}\n{row.issue.body or ''}".casefold():
            continue
        selected.append(row)
    return tuple(selected)


def _calculate_model_summary(
    run: RunManifest,
    ground_truth: tuple[ScoredGroundTruth, ...],
    attempts_by_issue: dict[int, AttemptEvidence],
) -> ScoredModelSummary:
    evidence = tuple(
        (
            item,
            attempts_by_issue[item.issue_number],
            _validated_outcome(attempts_by_issue[item.issue_number], item),
        )
        for item in ground_truth
    )
    outcome_counts = {outcome: 0 for outcome in ScoredOutcome}
    request_error_type_counts: dict[ProviderOutcome, int] = {}
    for _, attempt, outcome in evidence:
        outcome_counts[outcome] += 1
        if outcome is ScoredOutcome.REQUEST_ERROR and attempt.provider_outcome is not None:
            request_error_type_counts[attempt.provider_outcome] = (
                request_error_type_counts.get(attempt.provider_outcome, 0) + 1
            )

    per_class = tuple(
        _class_metrics(label, evidence) for label in CUSTOMER_LABEL_ORDER
    )
    supported_f1 = tuple(row.f1 for row in per_class if row.support > 0)
    confusion_cells = tuple(
        ConfusionCell(
            ground_truth=truth_label,
            prediction=prediction,
            issue_numbers=tuple(
                item.issue_number
                for item, attempt, _ in evidence
                if item.label == truth_label and attempt.parsed_label == prediction
            ),
        )
        for truth_label in CUSTOMER_LABEL_ORDER
        for prediction in (*CUSTOMER_LABEL_ORDER, None)
    )
    correct_count = outcome_counts[ScoredOutcome.CORRECT]
    expected_count = len(evidence)
    return ScoredModelSummary(
        model_id=run.model_id,
        run_id=run.run_id,
        expected_count=expected_count,
        correct_count=correct_count,
        accuracy=correct_count / expected_count,
        supported_class_macro_f1=sum(supported_f1) / len(supported_f1),
        supported_class_count=len(supported_f1),
        total_class_count=6,
        invalid_output_count=outcome_counts[ScoredOutcome.INVALID_OUTPUT],
        request_error_count=outcome_counts[ScoredOutcome.REQUEST_ERROR],
        scored_outcome_counts=outcome_counts,
        request_error_type_counts=request_error_type_counts,
        per_class=per_class,
        confusion_cells=confusion_cells,
    )


def _class_metrics(
    label: CustomerLabel,
    evidence: tuple[tuple[ScoredGroundTruth, AttemptEvidence, ScoredOutcome], ...],
) -> ScoredClassMetrics:
    true_positive = sum(
        item.label == label and attempt.parsed_label == label
        for item, attempt, _ in evidence
    )
    false_positive = sum(
        item.label != label and attempt.parsed_label == label
        for item, attempt, _ in evidence
    )
    false_negative = sum(
        item.label == label and attempt.parsed_label != label
        for item, attempt, _ in evidence
    )
    support = true_positive + false_negative
    predicted_count = true_positive + false_positive
    precision = true_positive / predicted_count if predicted_count else None
    recall = true_positive / support if support else None
    f1 = (
        (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        if support
        else None
    )
    return ScoredClassMetrics(
        label=label,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        support=support,
        predicted_count=predicted_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _pair_row(
    issue: Issue,
    ground_truth: ScoredGroundTruth,
    attempt_a: AttemptEvidence,
    attempt_b: AttemptEvidence,
) -> ScoredPairRow:
    outcome_a = _validated_outcome(attempt_a, ground_truth)
    outcome_b = _validated_outcome(attempt_b, ground_truth)
    prediction_a = attempt_a.parsed_label
    prediction_b = attempt_b.parsed_label
    predictions_agree = (
        prediction_a is not None
        and prediction_b is not None
        and prediction_a == prediction_b
    )
    if predictions_agree:
        pair_outcome = "label_agreement"
    elif prediction_a is None and prediction_b is None:
        pair_outcome = "joint_no_valid_prediction"
    elif prediction_a is None:
        pair_outcome = "model_a_no_valid_prediction"
    elif prediction_b is None:
        pair_outcome = "model_b_no_valid_prediction"
    else:
        pair_outcome = "label_disagreement"

    if outcome_a is ScoredOutcome.CORRECT and outcome_b is ScoredOutcome.CORRECT:
        correctness_pattern = "both_correct"
    elif outcome_a is ScoredOutcome.CORRECT:
        correctness_pattern = "model_a_only_correct"
    elif outcome_b is ScoredOutcome.CORRECT:
        correctness_pattern = "model_b_only_correct"
    else:
        correctness_pattern = "neither_correct"
    return ScoredPairRow(
        issue=issue,
        ground_truth=ground_truth,
        model_a_attempt=attempt_a,
        model_b_attempt=attempt_b,
        model_a_outcome=outcome_a,
        model_b_outcome=outcome_b,
        predictions_agree=predictions_agree,
        pair_outcome=pair_outcome,
        correctness_pattern=correctness_pattern,
    )


def _validated_outcome(
    attempt: AttemptEvidence, ground_truth: ScoredGroundTruth
) -> ScoredOutcome:
    if attempt.parsed_label is not None:
        derived = (
            ScoredOutcome.CORRECT
            if attempt.parsed_label == ground_truth.label
            else ScoredOutcome.INCORRECT_LABEL
        )
    elif attempt.provider_outcome is not None:
        derived = (
            ScoredOutcome.INVALID_OUTPUT
            if attempt.provider_outcome is ProviderOutcome.SUCCESS
            else ScoredOutcome.REQUEST_ERROR
        )
    elif attempt.scored_outcome in {
        ScoredOutcome.INVALID_OUTPUT,
        ScoredOutcome.REQUEST_ERROR,
    }:
        derived = attempt.scored_outcome
    else:
        derived = ScoredOutcome.INVALID_OUTPUT
    if attempt.scored_outcome is not None and attempt.scored_outcome is not derived:
        raise MetricIntegrityError(
            "Stored Scored Outcome disagrees with Ground Truth and Attempt Evidence "
            f"for issue {attempt.issue_number}"
        )
    return derived


def _benchmark_attempts_for_run(
    run: RunManifest, attempts: tuple[AttemptEvidence, ...]
) -> dict[int, AttemptEvidence]:
    benchmark = tuple(
        attempt
        for attempt in attempts
        if attempt.attempt_purpose is AttemptPurpose.BENCHMARK
    )
    if any(attempt.run_id != run.run_id for attempt in benchmark):
        raise MetricIntegrityError("Attempt Evidence references the wrong run")
    attempt_by_issue = _unique_by_issue_number(benchmark, "Benchmark Attempt")
    if set(attempt_by_issue) != set(run.ordered_issue_numbers):
        raise MetricIntegrityError(
            f"Attempt Evidence does not match complete run {run.run_id}"
        )
    return attempt_by_issue


def _unique_by_issue_number(items: tuple, name: str) -> dict[int, object]:
    by_issue = {item.issue_number: item for item in items}
    if len(by_issue) != len(items):
        raise MetricIntegrityError(f"Duplicate {name} issue number")
    return by_issue


def _assert_ground_truth_identity(
    run: RunManifest, ground_truth: ScoredGroundTruthSet
) -> None:
    if (
        run.ground_truth_version != ground_truth.ground_truth_version
        or run.ground_truth_sha256 != ground_truth.artifact_sha256
    ):
        raise MetricIntegrityError("Ground Truth identity does not match the Model Evaluation Run")


def _select_ground_truth(
    items: tuple[ScoredGroundTruth, ...], population: ScoredPopulation
) -> tuple[ScoredGroundTruth, ...]:
    if population is ScoredPopulation.COMBINED_DESCRIPTIVE:
        return items
    role = {
        ScoredPopulation.PRIMARY_HOLDOUT: ScoredEvaluationRole.PRIMARY_HOLDOUT,
        ScoredPopulation.PROMPT_DEVELOPMENT: ScoredEvaluationRole.PROMPT_DEVELOPMENT,
        ScoredPopulation.DIAGNOSTIC: ScoredEvaluationRole.DIAGNOSTIC,
        ScoredPopulation.MAPPING_AUDIT: ScoredEvaluationRole.MAPPING_AUDIT,
        ScoredPopulation.CLOSED_ISSUE_MAINTAINER: ScoredEvaluationRole.CLOSED_ISSUE_MAINTAINER,
    }[population]
    return tuple(item for item in items if item.evaluation_role is role)


def _population_interpretation(
    population: ScoredPopulation, items: tuple[ScoredGroundTruth, ...]
) -> str:
    if population is ScoredPopulation.PRIMARY_HOLDOUT:
        return (
            "Headline quality evidence from the random holdout hidden during prompt "
            "development. Every expected item remains in the denominator."
        )
    if population is ScoredPopulation.PROMPT_DEVELOPMENT:
        return "Development evidence used to freeze the prompt; it is not unseen validation."
    if population is ScoredPopulation.DIAGNOSTIC:
        return (
            "Deliberately selected rare and difficult evidence; its class composition "
            "does not estimate ordinary issue prevalence."
        )
    if population is ScoredPopulation.MAPPING_AUDIT:
        return "Human-reviewed mapping-audit evidence; it is outside headline accuracy."
    if population is ScoredPopulation.CLOSED_ISSUE_MAINTAINER:
        return (
            "Audited maintainer-derived sensitivity evidence; it is not equivalent to "
            "Human-Reviewed Ground Truth."
        )
    composition = ", ".join(
        f"{role.value}={sum(item.evaluation_role is role for item in items)}"
        for role in ScoredEvaluationRole
        if any(item.evaluation_role is role for item in items)
    )
    return (
        "Combined descriptive evidence across different sampling strata "
        f"({composition}); it must not replace the Primary Scored Holdout headline."
    )
