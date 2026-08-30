from __future__ import annotations

import pytest

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.domain import CustomerLabel, ScoredOutcome
from inferencebench.evaluation.metrics import (
    MetricIntegrityError,
    assert_comparable_runs,
    calculate_accuracy,
)


def test_accuracy_is_rebuilt_from_attempt_evidence_and_ground_truth() -> None:
    artifacts = FixtureArtifacts(Settings.from_environment().fixture_root)
    _, annotations = artifacts.load_ground_truth()
    bundle = artifacts.load_run_bundle()

    attempts_a = tuple(
        attempt for attempt in bundle.attempts if attempt.run_id == bundle.runs[0].run_id
    )
    attempts_b = tuple(
        attempt for attempt in bundle.attempts if attempt.run_id == bundle.runs[1].run_id
    )
    model_a = calculate_accuracy(bundle.runs[0], attempts_a, annotations)
    model_b = calculate_accuracy(bundle.runs[1], attempts_b, annotations)

    assert model_a.correct_count == 1
    assert model_a.expected_count == 1
    assert model_a.accuracy == 1.0
    assert model_b.correct_count == 0
    assert model_b.expected_count == 1
    assert model_b.accuracy == 0.0


def test_metric_rejects_stored_outcome_that_disagrees_with_prediction() -> None:
    artifacts = FixtureArtifacts(Settings.from_environment().fixture_root)
    _, annotations = artifacts.load_ground_truth()
    bundle = artifacts.load_run_bundle()
    attempts_a = tuple(
        attempt for attempt in bundle.attempts if attempt.run_id == bundle.runs[0].run_id
    )
    inconsistent = attempts_a[0].model_copy(
        update={
            "parsed_label": CustomerLabel.ENHANCEMENT,
            "scored_outcome": ScoredOutcome.CORRECT,
        }
    )

    with pytest.raises(MetricIntegrityError, match="Stored Scored Outcome disagrees"):
        calculate_accuracy(bundle.runs[0], (inconsistent, *attempts_a[1:]), annotations)


def test_incompatible_run_identity_is_rejected() -> None:
    bundle = FixtureArtifacts(Settings.from_environment().fixture_root).load_run_bundle()
    incompatible = bundle.runs[1].model_copy(update={"corpus_sha256": "f" * 64})

    with pytest.raises(MetricIntegrityError, match="identities are incompatible"):
        assert_comparable_runs(bundle.runs[0], incompatible)
