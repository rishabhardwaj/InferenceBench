from __future__ import annotations

from decimal import Decimal
from itertools import combinations
from types import SimpleNamespace

from inferencebench.models.domain import APPROVED_ELIGIBLE_MODEL_IDS
from inferencebench.workflows.candidate_dispositions import (
    derive_candidate_screening_dispositions,
)


def test_same_comparator_with_supported_advantage_screens_out_candidate() -> None:
    evidence, bootstrap = _screening_inputs()
    _set_candidate(evidence.rows[1], accuracy=0.50, f1=0.50, cost="0.20")
    _set_pair_advantage(bootstrap, 0, 1, lower="0.10", upper="0.20")

    result = derive_candidate_screening_dispositions(evidence, bootstrap)
    disposition = result.dispositions[1]

    assert disposition.disposition == "screened_out"
    assert disposition.comparator_model_id == APPROVED_ELIGIBLE_MODEL_IDS[0]
    assert disposition.dominance_evidence.bootstrap_supported_advantages == (
        "accuracy",
        "supported_class_macro_f1",
    )


def test_interval_that_permits_reversal_retains_candidate() -> None:
    evidence, bootstrap = _screening_inputs()
    _set_candidate(evidence.rows[1], accuracy=0.50, f1=0.50, cost="0.20")
    _set_pair_advantage(bootstrap, 0, 1, lower="-0.01", upper="0.20")

    result = derive_candidate_screening_dispositions(evidence, bootstrap)

    assert result.dispositions[1].disposition == "retained"


def test_partial_cost_is_not_comparable_and_cannot_be_screened_out() -> None:
    evidence, bootstrap = _screening_inputs()
    evidence.rows[1].candidate_screening_cost_completeness = "partial"
    evidence.rows[1].candidate_screening_cost_per_correct_usd = None
    _set_pair_advantage(bootstrap, 0, 1, lower="0.10", upper="0.20")

    result = derive_candidate_screening_dispositions(evidence, bootstrap)

    assert result.dispositions[1].disposition == "not_comparable"
    assert "cost completeness" in result.dispositions[1].reason


def test_per_class_advantage_retains_candidate_despite_aggregate_advantage() -> None:
    evidence, bootstrap = _screening_inputs()
    _set_candidate(evidence.rows[1], accuracy=0.50, f1=0.50, cost="0.20")
    evidence.rows[1].candidate_screening_holdout.per_class[0].recall = 1.0
    evidence.rows[0].candidate_screening_holdout.per_class[0].recall = 0.5
    _set_pair_advantage(bootstrap, 0, 1, lower="0.10", upper="0.20")

    result = derive_candidate_screening_dispositions(evidence, bootstrap)

    assert result.dispositions[1].disposition == "retained"


def _screening_inputs() -> tuple[SimpleNamespace, SimpleNamespace]:
    rows = [_candidate_row(model_id) for model_id in APPROVED_ELIGIBLE_MODEL_IDS]
    candidates = [
        SimpleNamespace(model_id=model_id, evidence_status="complete", intervals=object())
        for model_id in APPROVED_ELIGIBLE_MODEL_IDS
    ]
    pairs = [_pair(model_a, model_b) for model_a, model_b in combinations(APPROVED_ELIGIBLE_MODEL_IDS, 2)]
    return (
        SimpleNamespace(
            baseline_version="candidate-screening-test-v1",
            plan_sha256="a" * 64,
            content_sha256="b" * 64,
            rows=rows,
        ),
        SimpleNamespace(
            baseline_version="candidate-screening-test-v1",
            baseline_plan_sha256="a" * 64,
            content_sha256="c" * 64,
            candidates=candidates,
            pairs=pairs,
        ),
    )


def _candidate_row(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=model_id,
        comparable_headlines=True,
        exclusion_reason=None,
        candidate_screening_holdout=SimpleNamespace(
            accuracy=0.80,
            supported_class_macro_f1=0.80,
            invalid_output_count=0,
            request_error_count=0,
            expected_count=40,
            per_class=[SimpleNamespace(precision=0.80, recall=0.80) for _ in range(6)],
        ),
        candidate_screening_cost_per_correct_usd=Decimal("0.10"),
        candidate_screening_cost_completeness="complete",
        output_adherence=SimpleNamespace(exact_rate=1.0),
    )


def _set_candidate(row: SimpleNamespace, *, accuracy: float, f1: float, cost: str) -> None:
    row.candidate_screening_holdout.accuracy = accuracy
    row.candidate_screening_holdout.supported_class_macro_f1 = f1
    row.candidate_screening_cost_per_correct_usd = Decimal(cost)


def _set_pair_advantage(
    bootstrap: SimpleNamespace, candidate_a_index: int, candidate_b_index: int, *, lower: str, upper: str
) -> None:
    model_a = APPROVED_ELIGIBLE_MODEL_IDS[candidate_a_index]
    model_b = APPROVED_ELIGIBLE_MODEL_IDS[candidate_b_index]
    pair = next(
        row
        for row in bootstrap.pairs
        if row.model_a_id == model_a and row.model_b_id == model_b
    )
    pair.intervals.accuracy.lower_95 = _finite(lower)
    pair.intervals.accuracy.upper_95 = _finite(upper)
    pair.intervals.supported_class_macro_f1.lower_95 = _finite(lower)
    pair.intervals.supported_class_macro_f1.upper_95 = _finite(upper)


def _pair(model_a_id: str, model_b_id: str) -> SimpleNamespace:
    neutral = _interval("0", "0")
    return SimpleNamespace(
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        evidence_status="complete",
        intervals=SimpleNamespace(
            accuracy=_interval("0", "0"),
            supported_class_macro_f1=_interval("0", "0"),
            cost_per_correct=neutral,
        ),
    )


def _interval(lower: str, upper: str) -> SimpleNamespace:
    return SimpleNamespace(lower_95=_finite(lower), upper_95=_finite(upper))


def _finite(value: str) -> SimpleNamespace:
    return SimpleNamespace(state="finite", value=Decimal(value))
