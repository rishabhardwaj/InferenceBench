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
)
from inferencebench.evaluation.metrics import MetricIntegrityError
from inferencebench.evaluation.unscored import (
    NO_VALID_PREDICTION,
    UnscoredPairOutcome,
    UnscoredResultState,
    UnscoredRowFilter,
    build_unscored_comparison,
    filter_unscored_rows,
)


def test_agreement_denominators_and_all_four_outcomes_are_exhaustive() -> None:
    comparison = _unscored_comparison()

    assert comparison.expected_count == 8
    assert comparison.strict_agreement_numerator == 2
    assert comparison.strict_agreement_denominator == 8
    assert comparison.strict_agreement_rate == 2 / 8
    assert comparison.both_valid_agreement_numerator == 2
    assert comparison.both_valid_agreement_denominator == 3
    assert comparison.both_valid_agreement_rate == 2 / 3
    assert comparison.pair_outcome_counts == {
        UnscoredPairOutcome.LABEL_AGREEMENT: 2,
        UnscoredPairOutcome.LABEL_DISAGREEMENT: 1,
        UnscoredPairOutcome.ONE_SIDED_FAILURE: 2,
        UnscoredPairOutcome.JOINT_FAILURE: 3,
    }
    assert sum(comparison.pair_outcome_counts.values()) == comparison.expected_count
    assert tuple(row.pair_outcome for row in comparison.rows) == (
        UnscoredPairOutcome.LABEL_DISAGREEMENT,
        UnscoredPairOutcome.ONE_SIDED_FAILURE,
        UnscoredPairOutcome.ONE_SIDED_FAILURE,
        UnscoredPairOutcome.JOINT_FAILURE,
        UnscoredPairOutcome.JOINT_FAILURE,
        UnscoredPairOutcome.JOINT_FAILURE,
        UnscoredPairOutcome.LABEL_AGREEMENT,
        UnscoredPairOutcome.LABEL_AGREEMENT,
    )


def test_matching_invalid_outputs_and_errors_are_joint_failures_not_agreement() -> None:
    comparison = _unscored_comparison()
    by_issue = {row.issue.issue_number: row for row in comparison.rows}

    assert by_issue[5].model_a_attempt.raw_model_output == "maybe"
    assert by_issue[5].model_b_attempt.raw_model_output == "maybe"
    assert by_issue[5].pair_outcome is UnscoredPairOutcome.JOINT_FAILURE
    assert by_issue[6].model_a_attempt.provider_outcome is ProviderOutcome.TIMEOUT
    assert by_issue[6].model_b_attempt.provider_outcome is ProviderOutcome.TIMEOUT
    assert by_issue[6].pair_outcome is UnscoredPairOutcome.JOINT_FAILURE
    assert by_issue[3].pair_outcome is UnscoredPairOutcome.ONE_SIDED_FAILURE
    assert by_issue[4].pair_outcome is UnscoredPairOutcome.ONE_SIDED_FAILURE


def test_suggestion_distributions_cover_failures_with_seventh_bucket() -> None:
    comparison = _unscored_comparison()

    for summary in (comparison.model_a, comparison.model_b):
        assert len(summary.suggestion_distribution) == 7
        assert sum(row.count for row in summary.suggestion_distribution) == 8
        assert summary.suggestion_distribution[-1].display_prediction == (
            NO_VALID_PREDICTION
        )
        assert summary.suggestion_distribution[-1].count == 4
    assert comparison.model_a.result_state_counts == {
        UnscoredResultState.VALID_LABEL: 4,
        UnscoredResultState.INVALID_OUTPUT: 1,
        UnscoredResultState.REQUEST_ERROR: 3,
    }
    assert comparison.model_b.result_state_counts == {
        UnscoredResultState.VALID_LABEL: 4,
        UnscoredResultState.INVALID_OUTPUT: 2,
        UnscoredResultState.REQUEST_ERROR: 2,
    }
    assert comparison.model_a.request_error_type_counts == {
        ProviderOutcome.TIMEOUT: 2,
        ProviderOutcome.RATE_LIMIT: 1,
    }
    assert comparison.model_b.request_error_type_counts == {
        ProviderOutcome.TIMEOUT: 1,
        ProviderOutcome.SERVER_ERROR: 1,
    }


def test_filters_cover_predictions_pair_outcomes_error_states_and_text() -> None:
    comparison = _unscored_comparison()

    filtered = filter_unscored_rows(
        comparison.rows,
        UnscoredRowFilter(
            model_a_predictions=(None,),
            model_b_predictions=(CustomerLabel.SECURITY,),
            pair_outcomes=(UnscoredPairOutcome.ONE_SIDED_FAILURE,),
            model_a_result_states=(UnscoredResultState.REQUEST_ERROR,),
            model_b_result_states=(UnscoredResultState.VALID_LABEL,),
            text_query="body 4",
        ),
    )

    assert tuple(row.issue.issue_number for row in filtered) == (4,)


def test_both_valid_rate_is_undefined_when_no_pair_returns_two_valid_labels() -> None:
    comparison = _unscored_comparison(
        predictions_a=(None, None),
        predictions_b=(None, CustomerLabel.BUG),
        providers_a=(ProviderOutcome.SUCCESS, ProviderOutcome.TIMEOUT),
        providers_b=(ProviderOutcome.SUCCESS, ProviderOutcome.SUCCESS),
    )

    assert comparison.strict_agreement_rate == 0
    assert comparison.both_valid_agreement_numerator == 0
    assert comparison.both_valid_agreement_denominator == 0
    assert comparison.both_valid_agreement_rate is None
    assert comparison.pair_outcome_counts == {
        UnscoredPairOutcome.LABEL_AGREEMENT: 0,
        UnscoredPairOutcome.LABEL_DISAGREEMENT: 0,
        UnscoredPairOutcome.ONE_SIDED_FAILURE: 1,
        UnscoredPairOutcome.JOINT_FAILURE: 1,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("corpus_sha256", "f" * 64),
        ("prompt_version", "different-prompt"),
        ("parser_version", "different-parser"),
        ("generation_configuration_sha256", "e" * 64),
        ("timeout_seconds", 31),
        ("concurrency", 2),
    ),
)
def test_unscored_comparison_rejects_incompatible_run_identity(
    field: str, value: object
) -> None:
    comparison = _unscored_comparison()
    incompatible_b = comparison.run_b.model_copy(update={field: value})

    with pytest.raises(MetricIntegrityError, match="identities are incompatible"):
        build_unscored_comparison(
            comparison.run_a,
            tuple(row.model_a_attempt for row in comparison.rows),
            incompatible_b,
            tuple(row.model_b_attempt for row in comparison.rows),
            tuple(row.issue for row in comparison.rows),
            (),
        )


def _unscored_comparison(
    *,
    predictions_a: tuple[CustomerLabel | None, ...] = (
        CustomerLabel.BUG,
        CustomerLabel.BUG,
        CustomerLabel.DOCUMENTATION,
        None,
        None,
        None,
        CustomerLabel.QUESTION,
        None,
    ),
    predictions_b: tuple[CustomerLabel | None, ...] = (
        CustomerLabel.BUG,
        CustomerLabel.ENHANCEMENT,
        None,
        CustomerLabel.SECURITY,
        None,
        None,
        CustomerLabel.QUESTION,
        None,
    ),
    providers_a: tuple[ProviderOutcome, ...] = (
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.TIMEOUT,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.TIMEOUT,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.RATE_LIMIT,
    ),
    providers_b: tuple[ProviderOutcome, ...] = (
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.TIMEOUT,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SERVER_ERROR,
    ),
):
    artifacts = FixtureArtifacts(Settings.from_environment().fixture_root)
    _, base_issues = artifacts.load_corpus()
    base_issue = base_issues[0]
    bundle = artifacts.load_run_bundle()
    count = len(predictions_a)
    assert len(predictions_b) == len(providers_a) == len(providers_b) == count
    issues = tuple(_issue(base_issue, issue_number) for issue_number in range(1, count + 1))
    attempts_a = tuple(
        _attempt(
            bundle.attempts[0],
            "unscored-run-a",
            issue_number,
            prediction,
            providers_a[issue_number - 1],
            raw_invalid_output="maybe" if issue_number == 5 else None,
        )
        for issue_number, prediction in enumerate(predictions_a, 1)
    )
    attempts_b = tuple(
        _attempt(
            bundle.attempts[1],
            "unscored-run-b",
            issue_number,
            prediction,
            providers_b[issue_number - 1],
            raw_invalid_output="maybe" if issue_number == 5 else None,
        )
        for issue_number, prediction in enumerate(predictions_b, 1)
    )
    run_a = _run(bundle.runs[0], "unscored-run-a", "unscored-model-a", attempts_a)
    run_b = _run(bundle.runs[1], "unscored-run-b", "unscored-model-b", attempts_b)
    return build_unscored_comparison(
        run_a, attempts_a, run_b, attempts_b, issues, ()
    )


def _issue(base: Issue, issue_number: int) -> Issue:
    return Issue.model_validate(
        {
            **base.model_dump(mode="python"),
            "github_issue_id": issue_number,
            "issue_number": issue_number,
            "node_id": f"I_unscored_{issue_number}",
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
    provider_outcome: ProviderOutcome,
    *,
    raw_invalid_output: str | None = None,
) -> AttemptEvidence:
    successful = provider_outcome is ProviderOutcome.SUCCESS
    raw_output = prediction.value if prediction else raw_invalid_output
    return AttemptEvidence.model_validate(
        {
            **base.model_dump(mode="python"),
            "schema_version": "attempt_evidence.v2",
            "attempt_id": f"{run_id}-{issue_number}",
            "run_id": run_id,
            "issue_number": issue_number,
            "dispatch_order": issue_number - 1,
            "provider_outcome": provider_outcome,
            "http_status": 200 if successful else None,
            "raw_response": {"response": issue_number} if successful else None,
            "raw_error": None if successful else {"type": provider_outcome.value},
            "raw_model_output": raw_output if successful else None,
            "parsed_label": prediction if successful else None,
            "parse_status": (
                ParseStatus.EXACT
                if successful and prediction is not None
                else ParseStatus.INVALID
            ),
            "normalizations": (),
            "scored_outcome": None,
            "usable": successful and prediction is not None,
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
            "ordered_issue_numbers": tuple(
                attempt.issue_number for attempt in attempts
            ),
            "issue_order_sha256": "c" * 64,
            "expected_count": len(attempts),
            "persisted_count": len(attempts),
            "usable_count": sum(attempt.usable for attempt in attempts),
            "normalized_count": 0,
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
