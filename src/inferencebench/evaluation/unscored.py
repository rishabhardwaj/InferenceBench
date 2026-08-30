from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from inferencebench.domain import (
    AttemptEvidence,
    AttemptPurpose,
    CustomerLabel,
    Issue,
    ProviderOutcome,
    RunManifest,
    ScoredOutcome,
    StrictModel,
)
from inferencebench.evaluation.metrics import MetricIntegrityError, assert_comparable_runs


NO_VALID_PREDICTION = "No Valid Prediction"
CUSTOMER_LABEL_ORDER = tuple(CustomerLabel)


class UnscoredPairOutcome(StrEnum):
    LABEL_AGREEMENT = "label_agreement"
    LABEL_DISAGREEMENT = "label_disagreement"
    ONE_SIDED_FAILURE = "one_sided_failure"
    JOINT_FAILURE = "joint_failure"


class UnscoredResultState(StrEnum):
    VALID_LABEL = "valid_label"
    INVALID_OUTPUT = "invalid_output"
    REQUEST_ERROR = "request_error"


class SuggestionDistributionRow(StrictModel):
    prediction: CustomerLabel | None
    count: int = Field(ge=0)
    expected_count: int = Field(gt=0)
    rate: float = Field(ge=0, le=1)

    @property
    def display_prediction(self) -> str:
        return self.prediction.value if self.prediction else NO_VALID_PREDICTION

    @model_validator(mode="after")
    def validate_rate(self) -> "SuggestionDistributionRow":
        if self.rate != self.count / self.expected_count:
            raise ValueError("Suggestion rate must match count and expected count")
        return self


class UnscoredModelSummary(StrictModel):
    model_id: str
    run_id: str
    expected_count: int = Field(gt=0)
    suggestion_distribution: tuple[SuggestionDistributionRow, ...]
    result_state_counts: dict[UnscoredResultState, int]
    request_error_type_counts: dict[ProviderOutcome, int]

    @model_validator(mode="after")
    def validate_complete_distribution(self) -> "UnscoredModelSummary":
        expected_predictions = (*CUSTOMER_LABEL_ORDER, None)
        if tuple(row.prediction for row in self.suggestion_distribution) != (
            expected_predictions
        ):
            raise ValueError(
                "Suggestion distribution must use the six-label order followed by "
                "No Valid Prediction"
            )
        if any(
            row.expected_count != self.expected_count
            for row in self.suggestion_distribution
        ):
            raise ValueError("Suggestion distribution uses inconsistent denominators")
        if sum(row.count for row in self.suggestion_distribution) != self.expected_count:
            raise ValueError("Suggestion distribution must cover every expected issue")
        if set(self.result_state_counts) != set(UnscoredResultState):
            raise ValueError("Result-state counts must include all three states")
        if sum(self.result_state_counts.values()) != self.expected_count:
            raise ValueError("Result-state counts must cover every expected issue")
        if self.suggestion_distribution[-1].count != (
            self.result_state_counts[UnscoredResultState.INVALID_OUTPUT]
            + self.result_state_counts[UnscoredResultState.REQUEST_ERROR]
        ):
            raise ValueError("No Valid Prediction must include every failed result")
        if sum(self.request_error_type_counts.values()) != self.result_state_counts[
            UnscoredResultState.REQUEST_ERROR
        ]:
            raise ValueError("Typed request errors must cover every request error")
        return self


class UnscoredPairRow(StrictModel):
    issue: Issue
    model_a_attempt: AttemptEvidence
    model_b_attempt: AttemptEvidence
    model_a_result_state: UnscoredResultState
    model_b_result_state: UnscoredResultState
    pair_outcome: UnscoredPairOutcome


class UnscoredComparison(StrictModel):
    run_a: RunManifest
    run_b: RunManifest
    expected_count: int = Field(gt=0)
    strict_agreement_numerator: int = Field(ge=0)
    strict_agreement_denominator: int = Field(gt=0)
    strict_agreement_rate: float = Field(ge=0, le=1)
    both_valid_agreement_numerator: int = Field(ge=0)
    both_valid_agreement_denominator: int = Field(ge=0)
    both_valid_agreement_rate: float | None = Field(default=None, ge=0, le=1)
    pair_outcome_counts: dict[UnscoredPairOutcome, int]
    model_a: UnscoredModelSummary
    model_b: UnscoredModelSummary
    rows: tuple[UnscoredPairRow, ...]

    @model_validator(mode="after")
    def validate_complete_comparison(self) -> "UnscoredComparison":
        if self.model_a.run_id != self.run_a.run_id:
            raise ValueError("Model A summary references the wrong run")
        if self.model_b.run_id != self.run_b.run_id:
            raise ValueError("Model B summary references the wrong run")
        if self.expected_count != len(self.rows):
            raise ValueError("Unscored comparison must contain every expected row")
        if self.strict_agreement_denominator != self.expected_count:
            raise ValueError("Strict Agreement denominator must include every row")
        if self.model_a.expected_count != self.expected_count:
            raise ValueError("Model A distribution does not cover every row")
        if self.model_b.expected_count != self.expected_count:
            raise ValueError("Model B distribution does not cover every row")
        if set(self.pair_outcome_counts) != set(UnscoredPairOutcome):
            raise ValueError("Pair-outcome counts must include all four outcomes")
        if sum(self.pair_outcome_counts.values()) != self.expected_count:
            raise ValueError("Pair outcomes must exhaust the Unscored Corpus")
        agreement_count = self.pair_outcome_counts[
            UnscoredPairOutcome.LABEL_AGREEMENT
        ]
        both_valid_count = agreement_count + self.pair_outcome_counts[
            UnscoredPairOutcome.LABEL_DISAGREEMENT
        ]
        if self.strict_agreement_numerator != agreement_count:
            raise ValueError("Strict Agreement numerator must count label agreements")
        if self.both_valid_agreement_numerator != agreement_count:
            raise ValueError("Both-Valid numerator must count label agreements")
        if self.both_valid_agreement_denominator != both_valid_count:
            raise ValueError("Both-Valid denominator must count both-valid rows")
        if self.strict_agreement_rate != agreement_count / self.expected_count:
            raise ValueError("Strict Agreement rate disagrees with its fraction")
        if both_valid_count == 0:
            if self.both_valid_agreement_rate is not None:
                raise ValueError("Both-Valid rate is undefined without both-valid rows")
        elif self.both_valid_agreement_rate != agreement_count / both_valid_count:
            raise ValueError("Both-Valid Agreement rate disagrees with its fraction")
        return self


class UnscoredRowFilter(StrictModel):
    model_a_predictions: tuple[CustomerLabel | None, ...] = ()
    model_b_predictions: tuple[CustomerLabel | None, ...] = ()
    pair_outcomes: tuple[UnscoredPairOutcome, ...] = ()
    model_a_result_states: tuple[UnscoredResultState, ...] = ()
    model_b_result_states: tuple[UnscoredResultState, ...] = ()
    text_query: str = ""


def build_unscored_comparison(
    run_a: RunManifest,
    attempts_a: tuple[AttemptEvidence, ...],
    run_b: RunManifest,
    attempts_b: tuple[AttemptEvidence, ...],
    issues: tuple[Issue, ...],
    scored_issue_numbers: tuple[int, ...],
) -> UnscoredComparison:
    """Rebuild the Unscored View from two complete compatible runs."""

    assert_comparable_runs(run_a, run_b)
    issues_by_number = _unique_by_issue_number(issues, "Corpus Issue")
    run_issue_numbers = set(run_a.ordered_issue_numbers)
    if set(issues_by_number) != run_issue_numbers:
        raise MetricIntegrityError("Corpus Issues do not match the complete run population")

    scored_numbers = set(scored_issue_numbers)
    if len(scored_numbers) != len(scored_issue_numbers):
        raise MetricIntegrityError("Scored issue numbers must be unique")
    if not scored_numbers <= run_issue_numbers:
        raise MetricIntegrityError("Scored issues are not contained in the run population")
    unscored_numbers = tuple(
        issue_number
        for issue_number in run_a.ordered_issue_numbers
        if issue_number not in scored_numbers
    )
    if not unscored_numbers:
        raise MetricIntegrityError("Unscored View requires at least one Unscored Corpus issue")

    attempts_by_model = (
        _benchmark_attempts_for_run(run_a, attempts_a),
        _benchmark_attempts_for_run(run_b, attempts_b),
    )
    rows = tuple(
        _pair_row(
            issues_by_number[issue_number],
            attempts_by_model[0][issue_number],
            attempts_by_model[1][issue_number],
        )
        for issue_number in unscored_numbers
    )
    priority = {
        UnscoredPairOutcome.LABEL_DISAGREEMENT: 0,
        UnscoredPairOutcome.ONE_SIDED_FAILURE: 1,
        UnscoredPairOutcome.JOINT_FAILURE: 2,
        UnscoredPairOutcome.LABEL_AGREEMENT: 3,
    }
    rows = tuple(
        sorted(rows, key=lambda row: (priority[row.pair_outcome], row.issue.issue_number))
    )
    outcome_counts = {
        outcome: sum(row.pair_outcome is outcome for row in rows)
        for outcome in UnscoredPairOutcome
    }
    agreement_count = outcome_counts[UnscoredPairOutcome.LABEL_AGREEMENT]
    both_valid_count = agreement_count + outcome_counts[
        UnscoredPairOutcome.LABEL_DISAGREEMENT
    ]
    expected_count = len(rows)
    return UnscoredComparison(
        run_a=run_a,
        run_b=run_b,
        expected_count=expected_count,
        strict_agreement_numerator=agreement_count,
        strict_agreement_denominator=expected_count,
        strict_agreement_rate=agreement_count / expected_count,
        both_valid_agreement_numerator=agreement_count,
        both_valid_agreement_denominator=both_valid_count,
        both_valid_agreement_rate=(
            agreement_count / both_valid_count if both_valid_count else None
        ),
        pair_outcome_counts=outcome_counts,
        model_a=_model_summary(run_a, rows, model="a"),
        model_b=_model_summary(run_b, rows, model="b"),
        rows=rows,
    )


def filter_unscored_rows(
    rows: tuple[UnscoredPairRow, ...], filters: UnscoredRowFilter
) -> tuple[UnscoredPairRow, ...]:
    query = filters.text_query.strip().casefold()
    selected: list[UnscoredPairRow] = []
    for row in rows:
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
        if filters.pair_outcomes and row.pair_outcome not in filters.pair_outcomes:
            continue
        if (
            filters.model_a_result_states
            and row.model_a_result_state not in filters.model_a_result_states
        ):
            continue
        if (
            filters.model_b_result_states
            and row.model_b_result_state not in filters.model_b_result_states
        ):
            continue
        if query and query not in f"{row.issue.title}\n{row.issue.body or ''}".casefold():
            continue
        selected.append(row)
    return tuple(selected)


def result_state(attempt: AttemptEvidence) -> UnscoredResultState:
    if attempt.parsed_label is not None:
        return UnscoredResultState.VALID_LABEL
    if attempt.provider_outcome is None:
        if attempt.scored_outcome is ScoredOutcome.REQUEST_ERROR:
            return UnscoredResultState.REQUEST_ERROR
        return UnscoredResultState.INVALID_OUTPUT
    if attempt.provider_outcome is ProviderOutcome.SUCCESS:
        return UnscoredResultState.INVALID_OUTPUT
    return UnscoredResultState.REQUEST_ERROR


def _pair_row(
    issue: Issue, attempt_a: AttemptEvidence, attempt_b: AttemptEvidence
) -> UnscoredPairRow:
    state_a = result_state(attempt_a)
    state_b = result_state(attempt_b)
    prediction_a = attempt_a.parsed_label
    prediction_b = attempt_b.parsed_label
    if prediction_a is not None and prediction_b is not None:
        outcome = (
            UnscoredPairOutcome.LABEL_AGREEMENT
            if prediction_a == prediction_b
            else UnscoredPairOutcome.LABEL_DISAGREEMENT
        )
    elif (prediction_a is None) != (prediction_b is None):
        outcome = UnscoredPairOutcome.ONE_SIDED_FAILURE
    else:
        outcome = UnscoredPairOutcome.JOINT_FAILURE
    return UnscoredPairRow(
        issue=issue,
        model_a_attempt=attempt_a,
        model_b_attempt=attempt_b,
        model_a_result_state=state_a,
        model_b_result_state=state_b,
        pair_outcome=outcome,
    )


def _model_summary(
    run: RunManifest, rows: tuple[UnscoredPairRow, ...], *, model: str
) -> UnscoredModelSummary:
    attempts = tuple(
        row.model_a_attempt if model == "a" else row.model_b_attempt for row in rows
    )
    expected_count = len(attempts)
    distribution = tuple(
        SuggestionDistributionRow(
            prediction=prediction,
            count=sum(attempt.parsed_label == prediction for attempt in attempts),
            expected_count=expected_count,
            rate=sum(attempt.parsed_label == prediction for attempt in attempts)
            / expected_count,
        )
        for prediction in (*CUSTOMER_LABEL_ORDER, None)
    )
    states = tuple(result_state(attempt) for attempt in attempts)
    state_counts = {state: states.count(state) for state in UnscoredResultState}
    request_error_type_counts: dict[ProviderOutcome, int] = {}
    for attempt, state in zip(attempts, states, strict=True):
        if state is not UnscoredResultState.REQUEST_ERROR:
            continue
        provider_outcome = attempt.provider_outcome or ProviderOutcome.UNKNOWN
        request_error_type_counts[provider_outcome] = (
            request_error_type_counts.get(provider_outcome, 0) + 1
        )
    return UnscoredModelSummary(
        model_id=run.model_id,
        run_id=run.run_id,
        expected_count=expected_count,
        suggestion_distribution=distribution,
        result_state_counts=state_counts,
        request_error_type_counts=request_error_type_counts,
    )


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
