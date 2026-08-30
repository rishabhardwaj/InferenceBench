from __future__ import annotations

from decimal import Decimal

from inferencebench.config import Settings
from inferencebench.evaluation.operational import AggregateCostCompleteness
from inferencebench.evaluation.scored import ScoredPopulation
from inferencebench.evaluation.unscored import UnscoredPairOutcome
from inferencebench.persistence.repository import EvidenceRepository
from inferencebench.workflows.saved_comparison import load_saved_comparison


def test_saved_comparison_seeds_then_rebuilds_from_sqlite(settings: Settings) -> None:
    first = load_saved_comparison(settings)
    second = load_saved_comparison(settings)
    repository = EvidenceRepository(settings.database_path)

    assert settings.database_path.exists()
    assert first == second
    assert first.evidence_state == "Saved evidence — no inference request"
    assert first.provider_requests_made == 0
    assert first.model_a.accuracy == 1.0
    assert first.model_b.accuracy == 0.0
    assert first.model_a.run_id == "fixture-v2-run-model-a"
    assert first.model_b.run_id == "fixture-v2-run-model-b"
    assert first.operational_view.pricing_snapshot_id == (
        "do-serverless-pricing-2026-08-30"
    )
    assert first.operational_view.cost_formula_version == (
        "calculated-request-cost-v1"
    )
    assert first.operational_view.percentile_method_version == (
        "linear-percentile-v1"
    )
    operational_a = first.operational_view.model_a
    operational_b = first.operational_view.model_b
    assert operational_a.cost.known_total_cost_usd == Decimal("0.00001235")
    assert operational_a.cost.completeness is AggregateCostCompleteness.PARTIAL
    assert operational_a.cost.is_lower_bound is True
    assert operational_a.latency.p50_usable_request_latency_ms == 85.0
    assert operational_a.latency.p95_usable_request_latency_ms == 98.5
    assert operational_a.latency.usable_count == 4
    assert operational_a.latency.expected_count == 7
    assert operational_a.throughput is not None
    assert operational_a.reliability.timeout_count == 2
    assert operational_a.primary_holdout_cost is not None
    assert operational_a.primary_holdout_cost.cost_per_correct_usd == Decimal(
        "0.00000280"
    )
    assert operational_b.primary_holdout_cost is not None
    assert operational_b.primary_holdout_cost.cost_per_correct_usd is None
    assert operational_b.primary_holdout_cost.cost_per_correct_status == (
        "undefined_zero_correct"
    )
    assert first.scored_view.default_population is ScoredPopulation.PRIMARY_HOLDOUT
    assert len(first.scored_view.comparisons) == 1
    assert len(first.scored_view.comparisons[0].model_a.per_class) == 6
    assert len(first.scored_view.comparisons[0].model_a.confusion_cells) == 42
    assert first.scored_view.comparisons[0].rows[0].pair_outcome == "label_disagreement"
    assert first.unscored_view.strict_agreement_numerator == 1
    assert first.unscored_view.strict_agreement_denominator == 6
    assert first.unscored_view.both_valid_agreement_numerator == 1
    assert first.unscored_view.both_valid_agreement_denominator == 2
    assert first.unscored_view.pair_outcome_counts == {
        UnscoredPairOutcome.LABEL_AGREEMENT: 1,
        UnscoredPairOutcome.LABEL_DISAGREEMENT: 1,
        UnscoredPairOutcome.ONE_SIDED_FAILURE: 2,
        UnscoredPairOutcome.JOINT_FAILURE: 2,
    }
    assert len(repository.get_attempts(first.model_a.run_id)) == 7
    assert len(repository.get_attempts(first.model_b.run_id)) == 7
