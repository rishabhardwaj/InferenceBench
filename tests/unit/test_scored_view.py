from __future__ import annotations

import pytest

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.domain import (
    AttemptEvidence,
    CustomerLabel,
    Issue,
    ParseStatus,
    ProviderOutcome,
    RunManifest,
    ScoredOutcome,
)
from inferencebench.evaluation.scored import (
    ScoredEvidenceSource,
    ScoredEvaluationRole,
    ScoredGroundTruth,
    ScoredGroundTruthSet,
    ScoredPopulation,
    ScoredRowFilter,
    ScoredSamplingStratum,
    build_scored_view,
    filter_scored_rows,
)
from inferencebench.evaluation.metrics import MetricIntegrityError


GROUND_TRUTH_SHA256 = "b" * 64


def test_primary_metrics_keep_wrong_invalid_and_request_errors_visible() -> None:
    view = _scored_view()
    primary = _comparison(view, ScoredPopulation.PRIMARY_HOLDOUT)

    assert primary.is_primary_headline is True
    assert primary.model_a.correct_count == 2
    assert primary.model_a.expected_count == 4
    assert primary.model_a.accuracy == 0.5
    assert primary.model_a.invalid_output_count == 1
    assert primary.model_a.request_error_count == 0
    assert primary.model_a.supported_class_count == 3
    assert primary.model_a.total_class_count == 6
    assert primary.model_a.supported_class_macro_f1 == 4 / 9

    bug = _class_row(primary.model_a, CustomerLabel.BUG)
    assert (
        bug.true_positive,
        bug.false_positive,
        bug.false_negative,
        bug.support,
        bug.predicted_count,
    ) == (1, 0, 1, 2, 1)
    assert bug.precision == 1.0
    assert bug.recall == 0.5
    assert bug.f1 == 2 / 3

    question = _class_row(primary.model_a, CustomerLabel.QUESTION)
    assert question.support == 1
    assert question.predicted_count == 0
    assert question.precision is None
    assert question.recall == 0
    assert question.f1 == 0

    documentation = _class_row(primary.model_a, CustomerLabel.DOCUMENTATION)
    assert documentation.support == 0
    assert documentation.precision is None
    assert documentation.recall is None
    assert documentation.f1 is None


def test_confusion_has_six_by_seven_cells_and_exact_drill_down() -> None:
    primary = _comparison(_scored_view(), ScoredPopulation.PRIMARY_HOLDOUT)

    assert len(primary.model_a.confusion_cells) == 42
    assert _cell(
        primary.model_a, CustomerLabel.BUG, CustomerLabel.ENHANCEMENT
    ).issue_numbers == (2,)
    assert _cell(
        primary.model_a, CustomerLabel.QUESTION, None
    ).issue_numbers == (4,)
    assert primary.rows[0].predictions_agree is False
    assert primary.rows[-1].predictions_agree is True
    issue_four = next(row for row in primary.rows if row.issue.issue_number == 4)
    assert issue_four.model_a_outcome is ScoredOutcome.INVALID_OUTPUT
    assert issue_four.pair_outcome == "model_a_no_valid_prediction"
    assert issue_four.correctness_pattern == "model_b_only_correct"


def test_populations_remain_separate_and_combined_is_explicitly_descriptive() -> None:
    view = _scored_view()

    assert tuple(comparison.population for comparison in view.comparisons) == (
        ScoredPopulation.PRIMARY_HOLDOUT,
        ScoredPopulation.PROMPT_DEVELOPMENT,
        ScoredPopulation.DIAGNOSTIC,
        ScoredPopulation.MAPPING_AUDIT,
        ScoredPopulation.CLOSED_ISSUE_MAINTAINER,
        ScoredPopulation.COMBINED_DESCRIPTIVE,
    )
    assert view.default_population is ScoredPopulation.PRIMARY_HOLDOUT
    combined = _comparison(view, ScoredPopulation.COMBINED_DESCRIPTIVE)
    assert combined.is_primary_headline is False
    assert combined.model_a.expected_count == 8
    assert combined.model_a.correct_count == 3
    assert "must not replace" in combined.interpretation
    prompt = _comparison(view, ScoredPopulation.PROMPT_DEVELOPMENT)
    assert prompt.model_a.request_error_count == 1
    assert prompt.model_a.request_error_type_counts == {ProviderOutcome.TIMEOUT: 1}


def test_filters_cover_truth_predictions_outcomes_provenance_and_text() -> None:
    combined = _comparison(_scored_view(), ScoredPopulation.COMBINED_DESCRIPTIVE)

    no_valid_for_a = filter_scored_rows(
        combined.rows,
        ScoredRowFilter(
            model_a_predictions=(None,),
            model_a_outcomes=(ScoredOutcome.INVALID_OUTPUT,),
        ),
    )
    assert tuple(row.issue.issue_number for row in no_valid_for_a) == (4,)

    maintainer = filter_scored_rows(
        combined.rows,
        ScoredRowFilter(
            evidence_sources=(ScoredEvidenceSource.CLOSED_ISSUE_MAINTAINER,),
            sampling_strata=(ScoredSamplingStratum.CLOSED_ISSUE_MAINTAINER,),
            agreement="agreement",
            text_query="body 8",
        ),
    )
    assert tuple(row.issue.issue_number for row in maintainer) == (8,)


def test_scored_view_rejects_incompatible_runs_and_stored_outcome_drift() -> None:
    combined = _comparison(_scored_view(), ScoredPopulation.COMBINED_DESCRIPTIVE)
    issues = tuple(row.issue for row in combined.rows)
    truth = ScoredGroundTruthSet(
        ground_truth_version=combined.ground_truth_version,
        artifact_sha256=combined.ground_truth_sha256,
        items=tuple(row.ground_truth for row in combined.rows),
    )
    attempts_a = tuple(row.model_a_attempt for row in combined.rows)
    attempts_b = tuple(row.model_b_attempt for row in combined.rows)
    incompatible_b = combined.run_b.model_copy(update={"prompt_version": "different"})

    with pytest.raises(MetricIntegrityError, match="identities are incompatible"):
        build_scored_view(
            combined.run_a,
            attempts_a,
            incompatible_b,
            attempts_b,
            issues,
            truth,
        )

    first = attempts_a[0]
    inconsistent = first.model_copy(
        update={"scored_outcome": ScoredOutcome.INCORRECT_LABEL}
    )
    with pytest.raises(MetricIntegrityError, match="Stored Scored Outcome disagrees"):
        build_scored_view(
            combined.run_a,
            (inconsistent, *attempts_a[1:]),
            combined.run_b,
            attempts_b,
            issues,
            truth,
        )


def _scored_view():
    artifacts = FixtureArtifacts(Settings.from_environment().fixture_root)
    base_manifest, base_issues = artifacts.load_corpus()
    base_issue = base_issues[0]
    bundle = artifacts.load_run_bundle()
    issues = tuple(_issue(base_issue, issue_number) for issue_number in range(1, 9))
    predictions_a = (
        CustomerLabel.BUG,
        CustomerLabel.ENHANCEMENT,
        CustomerLabel.ENHANCEMENT,
        None,
        None,
        CustomerLabel.OTHER,
        CustomerLabel.BUG,
        CustomerLabel.BUG,
    )
    predictions_b = (
        CustomerLabel.ENHANCEMENT,
        CustomerLabel.BUG,
        CustomerLabel.ENHANCEMENT,
        CustomerLabel.QUESTION,
        CustomerLabel.DOCUMENTATION,
        None,
        None,
        CustomerLabel.BUG,
    )
    truth = (
        CustomerLabel.BUG,
        CustomerLabel.BUG,
        CustomerLabel.ENHANCEMENT,
        CustomerLabel.QUESTION,
        CustomerLabel.DOCUMENTATION,
        CustomerLabel.SECURITY,
        CustomerLabel.OTHER,
        CustomerLabel.BUG,
    )
    provider_a = (
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.TIMEOUT,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
    )
    provider_b = (
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.RATE_LIMIT,
        ProviderOutcome.SUCCESS,
    )
    attempts_a = tuple(
        _attempt(
            bundle.attempts[0],
            "synthetic-run-a",
            issue_number,
            prediction,
            truth[issue_number - 1],
            provider_a[issue_number - 1],
        )
        for issue_number, prediction in enumerate(predictions_a, 1)
    )
    attempts_b = tuple(
        _attempt(
            bundle.attempts[1],
            "synthetic-run-b",
            issue_number,
            prediction,
            truth[issue_number - 1],
            provider_b[issue_number - 1],
        )
        for issue_number, prediction in enumerate(predictions_b, 1)
    )
    run_a = _run(
        bundle.runs[0], "synthetic-run-a", "synthetic-model-a", attempts_a
    )
    run_b = _run(
        bundle.runs[1], "synthetic-run-b", "synthetic-model-b", attempts_b
    )
    ground_truth = ScoredGroundTruthSet(
        ground_truth_version="synthetic-ground-truth-v1",
        artifact_sha256=GROUND_TRUTH_SHA256,
        items=(
            _truth(1, truth[0], ScoredEvaluationRole.PRIMARY_HOLDOUT),
            _truth(2, truth[1], ScoredEvaluationRole.PRIMARY_HOLDOUT),
            _truth(3, truth[2], ScoredEvaluationRole.PRIMARY_HOLDOUT),
            _truth(4, truth[3], ScoredEvaluationRole.PRIMARY_HOLDOUT),
            _truth(5, truth[4], ScoredEvaluationRole.PROMPT_DEVELOPMENT),
            _truth(
                6,
                truth[5],
                ScoredEvaluationRole.DIAGNOSTIC,
                stratum=ScoredSamplingStratum.DIAGNOSTIC,
            ),
            _truth(
                7,
                truth[6],
                ScoredEvaluationRole.MAPPING_AUDIT,
                stratum=ScoredSamplingStratum.MAPPING_AUDIT,
            ),
            _truth(
                8,
                truth[7],
                ScoredEvaluationRole.CLOSED_ISSUE_MAINTAINER,
                source=ScoredEvidenceSource.CLOSED_ISSUE_MAINTAINER,
                stratum=ScoredSamplingStratum.CLOSED_ISSUE_MAINTAINER,
            ),
        ),
    )
    return build_scored_view(
        run_a, attempts_a, run_b, attempts_b, issues, ground_truth
    )


def _truth(
    issue_number: int,
    label: CustomerLabel,
    role: ScoredEvaluationRole,
    *,
    source: ScoredEvidenceSource = ScoredEvidenceSource.HUMAN_REVIEW,
    stratum: ScoredSamplingStratum = ScoredSamplingStratum.RANDOM,
) -> ScoredGroundTruth:
    return ScoredGroundTruth(
        issue_number=issue_number,
        label=label,
        ground_truth_source=source,
        sampling_stratum=stratum,
        evaluation_role=role,
        provenance_id=f"truth-{issue_number}",
        rubric_version="test-rubric-v1",
        confidence="high" if source is ScoredEvidenceSource.HUMAN_REVIEW else None,
        review_pass_count=1 if source is ScoredEvidenceSource.HUMAN_REVIEW else None,
    )


def _issue(base: Issue, issue_number: int) -> Issue:
    return Issue.model_validate(
        {
            **base.model_dump(mode="python"),
            "github_issue_id": issue_number,
            "issue_number": issue_number,
            "node_id": f"I_scored_{issue_number}",
            "api_url": f"https://api.github.com/repos/digitalocean/doctl/issues/{issue_number}",
            "html_url": f"https://github.com/digitalocean/doctl/issues/{issue_number}",
            "title": f"Issue {issue_number}",
            "body": f"Body {issue_number}",
            "content_sha256": f"{issue_number:064x}",
        }
    )


def _attempt(
    base: AttemptEvidence,
    run_id: str,
    issue_number: int,
    prediction: CustomerLabel | None,
    truth: CustomerLabel,
    provider_outcome: ProviderOutcome,
) -> AttemptEvidence:
    if provider_outcome is not ProviderOutcome.SUCCESS:
        scored_outcome = ScoredOutcome.REQUEST_ERROR
    elif prediction is None:
        scored_outcome = ScoredOutcome.INVALID_OUTPUT
    elif prediction == truth:
        scored_outcome = ScoredOutcome.CORRECT
    else:
        scored_outcome = ScoredOutcome.INCORRECT_LABEL
    usable = prediction is not None
    return AttemptEvidence.model_validate(
        {
            **base.model_dump(mode="python"),
            "schema_version": "attempt_evidence.v2",
            "attempt_id": f"{run_id}-{issue_number}",
            "run_id": run_id,
            "issue_number": issue_number,
            "dispatch_order": issue_number - 1,
            "provider_outcome": provider_outcome,
            "http_status": 200 if provider_outcome is ProviderOutcome.SUCCESS else None,
            "raw_response": {"response": issue_number}
            if provider_outcome is ProviderOutcome.SUCCESS
            else None,
            "raw_error": None
            if provider_outcome is ProviderOutcome.SUCCESS
            else {"type": provider_outcome.value},
            "raw_model_output": prediction.value if prediction else None,
            "parsed_label": prediction,
            "parse_status": ParseStatus.EXACT if usable else ParseStatus.INVALID,
            "scored_outcome": scored_outcome,
            "usable": usable,
        }
    )


def _run(
    base: RunManifest,
    run_id: str,
    model_id: str,
    attempts: tuple[AttemptEvidence, ...],
) -> RunManifest:
    return RunManifest.model_validate(
        {
            **base.model_dump(mode="python"),
            "run_id": run_id,
            "model_id": model_id,
            "ordered_issue_numbers": tuple(range(1, 9)),
            "issue_order_sha256": "c" * 64,
            "ground_truth_version": "synthetic-ground-truth-v1",
            "ground_truth_sha256": GROUND_TRUTH_SHA256,
            "expected_count": 8,
            "persisted_count": 8,
            "usable_count": sum(attempt.usable for attempt in attempts),
            "normalized_count": 0,
            "invalid_output_count": sum(
                attempt.scored_outcome is ScoredOutcome.INVALID_OUTPUT
                for attempt in attempts
            ),
            "request_error_count": sum(
                attempt.scored_outcome is ScoredOutcome.REQUEST_ERROR
                for attempt in attempts
            ),
        }
    )


def _comparison(view, population: ScoredPopulation):
    return next(
        comparison
        for comparison in view.comparisons
        if comparison.population is population
    )


def _class_row(summary, label: CustomerLabel):
    return next(row for row in summary.per_class if row.label is label)


def _cell(summary, truth: CustomerLabel, prediction: CustomerLabel | None):
    return next(
        cell
        for cell in summary.confusion_cells
        if cell.ground_truth is truth and cell.prediction is prediction
    )
