from __future__ import annotations

from decimal import Decimal

import streamlit as st

from inferencebench.workflows.recommendation import SelectionDecisionRecord
from inferencebench.workflows.saved_comparison import SavedComparisonReview


def render_recommendation(
    record: SelectionDecisionRecord, review: SavedComparisonReview
) -> None:
    """Present the recorded decision before the supporting drill-down views."""

    primary = next(
        comparison
        for comparison in review.scored_view.comparisons
        if comparison.is_primary_headline
    )
    model_a = review.operational_view.model_a
    model_b = review.operational_view.model_b

    st.subheader("Recommendation")
    st.success(
        f"Run `{record.recommended_model_id}` as the Production Rollout Candidate. "
        f"{record.recommendation_rationale}"
    )
    st.info(
        f"Prefer `{record.decision_challenger_model_id}` when "
        f"{record.challenger_preference_condition}"
    )
    st.caption(
        "Primary Scored Holdout is the headline quality population. Cost per correct "
        "uses that same population; operational metrics use each complete 536-issue run."
    )

    st.dataframe(
        [
            _headline_row("Recommended", primary.model_a, model_a),
            _headline_row("Decision Challenger", primary.model_b, model_b),
        ],
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Evidence limits and production path", expanded=False):
        for limit in record.evidence_limits:
            st.markdown(f"- {limit}")
        st.markdown(
            "This recommendation is for the evaluated `doctl` workload. Validate it on "
            "representative repositories, shadow real traffic, then use customer-approved "
            "quality, reliability, latency, and cost gates before broader rollout."
        )


def _headline_row(position: str, scored: object, operational: object) -> dict[str, str]:
    # The two view models deliberately remain separate: this small presentation adapter
    # combines only fields already derived from immutable persisted evidence.
    return {
        "Role": position,
        "Model": scored.model_id,
        "Primary accuracy": f"{scored.accuracy:.1%} ({scored.correct_count}/{scored.expected_count})",
        "Primary macro-F1": f"{scored.supported_class_macro_f1:.3f}",
        "Cost / correct": _money(operational.primary_holdout_cost.cost_per_correct_usd),
        "p95 request latency": _milliseconds(
            operational.latency.p95_usable_request_latency_ms
        ),
        "Throughput": f"{operational.throughput.sustained_requests_per_second:.2f} req/s",
        "Wall clock": f"{operational.throughput.run_wall_clock_seconds:.1f} s",
        "Failures": str(operational.reliability.unusable_count),
        "Concurrency": str(operational.concurrency),
    }


def _money(value: Decimal | None) -> str:
    return "Unknown" if value is None else f"${value:.8f}"


def _milliseconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value / 1000:.2f} s"
