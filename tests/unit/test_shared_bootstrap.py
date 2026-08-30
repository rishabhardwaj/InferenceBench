from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from inferencebench.artifacts import ArtifactIntegrityError
from inferencebench.domain import CustomerLabel
from inferencebench.evaluation.bootstrap import (
    BOOTSTRAP_ALGORITHM_VERSION,
    BOOTSTRAP_INTERVAL_VERSION,
    SHARED_BOOTSTRAP_RESAMPLE_COUNT,
    SHARED_BOOTSTRAP_SEED,
    ObservedBootstrapRow,
    build_shared_bootstrap_manifest,
    calculate_candidate_bootstrap_summary,
    calculate_pair_bootstrap_summary,
    canonical_holdout_rows,
    generate_shared_resamples,
    load_shared_bootstrap_plan,
    write_shared_bootstrap_plan,
)


BASELINE_SHA256 = "a" * 64
GROUND_TRUTH_SHA256 = "b" * 64


def _rows():
    return canonical_holdout_rows(
        (
            (101, CustomerLabel.BUG),
            (102, CustomerLabel.BUG),
            (201, CustomerLabel.QUESTION),
            (202, CustomerLabel.QUESTION),
        )
    )


def _observed(predictions, costs):
    rows = _rows()
    return tuple(
        ObservedBootstrapRow(
            row_index=row.row_index,
            issue_number=row.issue_number,
            ground_truth_label=row.ground_truth_label,
            predicted_label=prediction,
            is_invalid_output=False,
            is_request_error=False,
            calculated_request_cost_usd=cost,
        )
        for row, prediction, cost in zip(rows, predictions, costs, strict=True)
    )


def _plan(rows, resamples):
    return build_shared_bootstrap_manifest(
        plan_version="test-shared-bootstrap-v1",
        baseline_plan_sha256=BASELINE_SHA256,
        ground_truth_version="test-ground-truth-v1",
        ground_truth_sha256=GROUND_TRUTH_SHA256,
        rows=rows,
        resamples=resamples,
        seed=SHARED_BOOTSTRAP_SEED,
    )


def test_shared_plan_has_exact_count_support_hash_and_zero_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls = 0

    async def fail_if_called(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("Bootstrap must never call a provider")

    monkeypatch.setattr(httpx.AsyncClient, "request", fail_if_called)
    rows = _rows()
    first = generate_shared_resamples(rows)
    second = generate_shared_resamples(rows)
    manifest = _plan(rows, first)

    assert provider_calls == 0
    assert len(first) == SHARED_BOOTSTRAP_RESAMPLE_COUNT
    assert first == second
    assert all(len(resample.row_indices) == 4 for resample in first)
    assert all(
        sum(index in {0, 1} for index in resample.row_indices) == 2
        and sum(index in {2, 3} for index in resample.row_indices) == 2
        for resample in first
    )
    assert manifest.algorithm_version == BOOTSTRAP_ALGORITHM_VERSION
    assert manifest.interval_version == BOOTSTRAP_INTERVAL_VERSION
    assert manifest.resample_count == 10_000
    assert manifest.resamples_sha256 == (
        "1bda40462adcef1746cfa3e322440e0a6f8f4547d4291a5930200fff762ac43d"
    )


def test_shared_plan_round_trip_reproduces_and_rejects_tampering(tmp_path) -> None:
    rows = _rows()
    resamples = generate_shared_resamples(rows)
    manifest = _plan(rows, resamples)
    directory = tmp_path / "bootstrap"

    write_shared_bootstrap_plan(directory, manifest, resamples)
    loaded_manifest, loaded_resamples = load_shared_bootstrap_plan(directory)

    assert loaded_manifest == manifest
    assert loaded_resamples == resamples

    resamples_path = directory / manifest.resamples_file
    resamples_path.write_bytes(resamples_path.read_bytes() + b"{}\n")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        load_shared_bootstrap_plan(directory)


def test_intervals_and_paired_differences_use_every_shared_resample() -> None:
    rows = _rows()
    resamples = generate_shared_resamples(rows)
    plan = _plan(rows, resamples)
    labels = tuple(row.ground_truth_label for row in rows)
    wrong = (
        CustomerLabel.QUESTION,
        CustomerLabel.QUESTION,
        CustomerLabel.BUG,
        CustomerLabel.BUG,
    )
    candidate_a, vectors_a = calculate_candidate_bootstrap_summary(
        model_id="model-a",
        run_id="run-a",
        source_attempts_sha256="c" * 64,
        plan=plan,
        resamples=resamples,
        observed_rows=_observed(labels, (Decimal("1"),) * 4),
    )
    candidate_b, vectors_b = calculate_candidate_bootstrap_summary(
        model_id="model-b",
        run_id="run-b",
        source_attempts_sha256="d" * 64,
        plan=plan,
        resamples=resamples,
        observed_rows=_observed(wrong, (Decimal("1"),) * 4),
    )
    assert candidate_a.intervals is not None
    assert candidate_b.intervals is not None

    accuracy_a = candidate_a.intervals.accuracy
    assert accuracy_a.point_estimate.value == Decimal("1")
    assert accuracy_a.lower_95.value == Decimal("1")
    assert accuracy_a.upper_95.value == Decimal("1")
    assert accuracy_a.finite_count == 10_000

    zero_correct_cost = candidate_b.intervals.cost_per_correct
    assert zero_correct_cost.point_estimate.state == "positive_infinity"
    assert zero_correct_cost.lower_95.state == "positive_infinity"
    assert zero_correct_cost.upper_95.state == "positive_infinity"
    assert zero_correct_cost.positive_infinity_count == 10_000

    pair = calculate_pair_bootstrap_summary(
        model_a_id="model-a",
        model_b_id="model-b",
        run_a_id="run-a",
        run_b_id="run-b",
        source_a_attempts_sha256="c" * 64,
        source_b_attempts_sha256="d" * 64,
        plan_sha256=plan.content_sha256,
        candidate_a_vectors=vectors_a,
        candidate_b_vectors=vectors_b,
        candidate_a_point=candidate_a.intervals,
        candidate_b_point=candidate_b.intervals,
    )
    assert pair.intervals is not None
    assert pair.paired_difference_direction == "model_a_minus_model_b"
    assert pair.intervals.accuracy.lower_95.value == Decimal("1")
    assert pair.intervals.accuracy.upper_95.value == Decimal("1")
    assert pair.intervals.cost_per_correct.negative_infinity_count == 10_000
    assert pair.intervals.cost_per_correct.lower_95.state == "negative_infinity"
    assert pair.intervals.cost_per_correct.upper_95.state == "negative_infinity"


def test_unknown_cost_resamples_are_counted_instead_of_dropped() -> None:
    rows = _rows()
    resamples = generate_shared_resamples(rows)
    plan = _plan(rows, resamples)
    labels = tuple(row.ground_truth_label for row in rows)

    summary, _ = calculate_candidate_bootstrap_summary(
        model_id="model-a",
        run_id="run-a",
        source_attempts_sha256="c" * 64,
        plan=plan,
        resamples=resamples,
        observed_rows=_observed(
            labels,
            (None, Decimal("1"), Decimal("1"), Decimal("1")),
        ),
    )

    assert summary.intervals is not None
    interval = summary.intervals.cost_per_correct
    assert interval.unknown_count > 0
    assert interval.finite_count > 0
    assert interval.unknown_count + interval.finite_count == 10_000
    assert interval.lower_95.state == "unknown"
    assert interval.upper_95.state == "unknown"


def test_seeded_percentile_fixture_has_exact_frozen_endpoints() -> None:
    rows = _rows()
    resamples = generate_shared_resamples(rows)
    plan = _plan(rows, resamples)

    summary, _ = calculate_candidate_bootstrap_summary(
        model_id="model-a",
        run_id="run-a",
        source_attempts_sha256="c" * 64,
        plan=plan,
        resamples=resamples,
        observed_rows=_observed(
            (
                CustomerLabel.BUG,
                CustomerLabel.QUESTION,
                CustomerLabel.QUESTION,
                CustomerLabel.BUG,
            ),
            (Decimal("1"),) * 4,
        ),
    )

    assert summary.intervals is not None
    assert summary.intervals.accuracy.point_estimate.value == Decimal("0.5")
    assert summary.intervals.accuracy.lower_95.value == Decimal("0")
    assert summary.intervals.accuracy.upper_95.value == Decimal("1")
    assert summary.intervals.supported_class_macro_f1.lower_95.value == Decimal("0")
    assert summary.intervals.supported_class_macro_f1.upper_95.value == Decimal("1")
    cost = summary.intervals.cost_per_correct
    assert cost.point_estimate.value == Decimal("2")
    assert cost.lower_95.value == Decimal("1")
    assert cost.upper_95.state == "positive_infinity"
    assert cost.finite_count == 9_406
    assert cost.positive_infinity_count == 594
