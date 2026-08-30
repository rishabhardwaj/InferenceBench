from __future__ import annotations

from inferencebench.domain import (
    AttemptEvidence,
    AttemptPurpose,
    CustomerLabel,
    GroundTruthAnnotation,
    RunManifest,
    RunStatus,
    ScoredOutcome,
    StrictModel,
)


class MetricIntegrityError(ValueError):
    """Raised when persisted evidence cannot support the requested metric."""


class ModelAccuracy(StrictModel):
    model_id: str
    run_id: str
    correct_count: int
    expected_count: int
    accuracy: float


def assert_comparable_runs(run_a: RunManifest, run_b: RunManifest) -> None:
    if run_a.run_id == run_b.run_id:
        raise MetricIntegrityError("Model Comparison requires two distinct runs")
    if run_a.model_id == run_b.model_id:
        raise MetricIntegrityError("Model Comparison requires two distinct models")
    if run_a.status is not RunStatus.COMPLETE or run_b.status is not RunStatus.COMPLETE:
        raise MetricIntegrityError("Only complete Model Evaluation Runs are comparable")
    if run_a.comparison_identity() != run_b.comparison_identity():
        raise MetricIntegrityError("Run evidence identities are incompatible")


def calculate_accuracy(
    run: RunManifest,
    attempts: tuple[AttemptEvidence, ...],
    annotations: tuple[GroundTruthAnnotation, ...],
) -> ModelAccuracy:
    """Rebuild accuracy from terminal Attempt Evidence and Ground Truth."""

    if run.status is not RunStatus.COMPLETE:
        raise MetricIntegrityError("Accuracy requires a complete Model Evaluation Run")

    truth = {annotation.issue_number: annotation.label for annotation in annotations}
    expected = run.ordered_issue_numbers
    if not truth:
        raise MetricIntegrityError("Accuracy requires accepted Ground Truth")
    if not set(truth) <= set(expected):
        raise MetricIntegrityError("Ground Truth is not contained in the run population")

    benchmark_attempts = tuple(
        attempt
        for attempt in attempts
        if attempt.attempt_purpose is AttemptPurpose.BENCHMARK
    )
    attempt_by_issue = {attempt.issue_number: attempt for attempt in benchmark_attempts}
    if len(attempt_by_issue) != len(benchmark_attempts):
        raise MetricIntegrityError("Duplicate benchmark issue found in Attempt Evidence")
    if set(attempt_by_issue) != set(expected):
        raise MetricIntegrityError("Attempt Evidence does not match expected run population")

    correct = 0
    scored_issue_numbers = tuple(
        issue_number for issue_number in expected if issue_number in truth
    )
    for issue_number in scored_issue_numbers:
        attempt = attempt_by_issue[issue_number]
        derived_outcome = _derive_outcome(attempt.parsed_label, truth[issue_number])
        if attempt.usable:
            if (
                attempt.scored_outcome is not None
                and attempt.scored_outcome is not derived_outcome
            ):
                raise MetricIntegrityError(
                    f"Stored Scored Outcome disagrees with evidence for issue {issue_number}"
                )
        elif attempt.scored_outcome is not None and attempt.scored_outcome not in {
            ScoredOutcome.INVALID_OUTPUT,
            ScoredOutcome.REQUEST_ERROR,
        }:
            raise MetricIntegrityError(
                f"Unusable attempt has a label outcome for issue {issue_number}"
            )
        if derived_outcome is ScoredOutcome.CORRECT:
            correct += 1

    return ModelAccuracy(
        model_id=run.model_id,
        run_id=run.run_id,
        correct_count=correct,
        expected_count=len(scored_issue_numbers),
        accuracy=correct / len(scored_issue_numbers),
    )


def _derive_outcome(
    predicted: CustomerLabel | None, ground_truth: CustomerLabel
) -> ScoredOutcome:
    if predicted is None:
        return ScoredOutcome.INVALID_OUTPUT
    if predicted is ground_truth:
        return ScoredOutcome.CORRECT
    return ScoredOutcome.INCORRECT_LABEL
