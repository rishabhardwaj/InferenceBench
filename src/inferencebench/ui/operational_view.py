from __future__ import annotations

from decimal import Decimal

import streamlit as st

from inferencebench.evaluation.operational import (
    AggregateCostCompleteness,
    OperationalComparison,
    RunOperationalSummary,
)


def render_operational_view(comparison: OperationalComparison) -> None:
    st.subheader("Cost and Operational Evidence")
    st.caption(
        "Calculated benchmark costs are traceable estimates, not invoice values. "
        "Latency uses usable classifications; reliability and throughput retain every "
        "expected request, including the first request."
    )
    st.markdown(
        f"**Pricing Snapshot:** `{comparison.pricing_snapshot_id}` · "
        f"**Cost Formula:** `{comparison.cost_formula_version}` · "
        f"**Percentiles:** `{comparison.percentile_method_version}`"
    )
    st.caption(
        f"Pricing evidence SHA-256 `{comparison.pricing_snapshot_sha256}` · "
        f"Source: {comparison.pricing_source_url}"
    )

    left, right = st.columns(2)
    for column, heading, summary in (
        (left, "Model A", comparison.model_a),
        (right, "Model B", comparison.model_b),
    ):
        with column:
            _render_model_summary(heading, summary)


def _render_model_summary(heading: str, summary: RunOperationalSummary) -> None:
    st.markdown(f"### {heading}: `{summary.model_id}`")
    st.caption(
        f"Full Corpus · Run `{summary.run_id}` · concurrency `{summary.concurrency}` · "
        f"status `{summary.status.value}`"
    )
    if not summary.latency.is_comparable:
        st.warning(
            "This run is incomplete. Raw evidence remains visible, but comparable "
            "latency and throughput headlines are intentionally withheld."
        )

    _render_cost(summary)
    _render_latency_and_throughput(summary)
    _render_reliability(summary)
    _render_cost_trace(summary)


def _render_cost(summary: RunOperationalSummary) -> None:
    cost = summary.cost
    lower_bound = (
        " (partial lower bound)"
        if cost.completeness is AggregateCostCompleteness.PARTIAL
        else ""
    )
    st.markdown("#### Economics")
    st.metric(
        "Known Total Calculated Cost",
        _money(cost.known_total_cost_usd),
        help=(
            f"{cost.known_cost_count}/{cost.expected_count} expected request costs "
            f"are known{lower_bound}."
        ),
    )
    st.markdown(
        f"**Evidence completeness:** `{cost.completeness.value}`{lower_bound} · "
        f"known `{cost.known_cost_count}/{cost.expected_count}` · "
        f"unobserved `{cost.unobserved_count}`"
    )
    st.markdown(
        f"**Mean known request cost:** `{_money_or_unknown(cost.mean_known_request_cost_usd)}` "
        f"over `{cost.known_cost_count}` known requests"
    )
    st.markdown(
        "**Known cost per expected classification:** "
        f"`{_money(cost.cost_per_expected_classification_usd)}` "
        f"(denominator `{cost.expected_count}`){lower_bound}"
    )

    holdout = summary.primary_holdout_cost
    if holdout is not None:
        if holdout.cost_per_correct_usd is None:
            per_correct = "undefined (0 correct classifications)"
        else:
            per_correct = _money(holdout.cost_per_correct_usd)
        partial = (
            "; partial lower bound"
            if holdout.completeness is AggregateCostCompleteness.PARTIAL
            else ""
        )
        st.markdown(
            f"**Primary Scored Holdout cost per correct:** `{per_correct}` · "
            f"correct `{holdout.correct_count}/{holdout.expected_count}` · "
            f"cost evidence `{holdout.completeness.value}`{partial}"
        )


def _render_latency_and_throughput(summary: RunOperationalSummary) -> None:
    latency = summary.latency
    st.markdown("#### Latency and throughput")
    st.markdown(
        "**Usable Classification Request Latency:** "
        f"p50 `{_milliseconds(latency.p50_usable_request_latency_ms)}` · "
        f"p95 `{_milliseconds(latency.p95_usable_request_latency_ms)}` · "
        f"usable `{latency.usable_count}/{latency.expected_count}` · "
        f"concurrency `{latency.concurrency}`"
    )
    st.markdown(
        "**Queue Wait (separate from Request Latency):** "
        f"p50 `{_milliseconds(latency.p50_queue_wait_ms)}` · "
        f"p95 `{_milliseconds(latency.p95_queue_wait_ms)}`"
    )
    if summary.throughput is not None:
        throughput = summary.throughput
        st.markdown(
            f"**Run Wall-Clock Time:** `{_milliseconds(throughput.run_wall_clock_ms)}`"
        )
        st.markdown(
            "**Sustained Request Throughput:** "
            f"`{_rate(throughput.sustained_requests_per_second)} requests/s` · "
            "**Usable Classification Throughput:** "
            f"`{_rate(throughput.usable_classifications_per_second)} usable/s`"
        )


def _render_reliability(summary: RunOperationalSummary) -> None:
    reliability = summary.reliability
    st.markdown("#### Reliability")
    st.markdown(
        f"**Usable:** `{reliability.usable_count}/{reliability.expected_count}` · "
        f"**Invalid output:** `{reliability.invalid_output_count}` "
        f"(`{_percent(reliability.invalid_output_rate)}`) · "
        f"**Request error:** `{reliability.request_error_count}` "
        f"(`{_percent(reliability.request_error_rate)}`) · "
        f"**Total unusable:** `{reliability.unusable_count}` "
        f"(`{_percent(reliability.unusable_rate)}`)"
    )
    st.markdown(
        f"**Rate limit:** `{reliability.rate_limit_count}` "
        f"(`{_percent(reliability.rate_limit_rate)}`) · "
        f"**Timeout:** `{reliability.timeout_count}` "
        f"(`{_percent(reliability.timeout_rate)}`) · "
        f"**Other request errors:** `{reliability.other_request_error_count}` "
        f"(`{_percent(reliability.other_request_error_rate)}`)"
    )
    taxonomy = ", ".join(
        f"{outcome.value}={count}"
        for outcome, count in sorted(
            reliability.detailed_request_error_counts.items(),
            key=lambda item: item[0].value,
        )
    )
    st.caption(f"Detailed typed request-error taxonomy: {taxonomy or 'none'}")


def _render_cost_trace(summary: RunOperationalSummary) -> None:
    with st.expander(f"Exact per-request cost evidence — {summary.model_id}"):
        st.caption(
            "Each term is tokens × the matching immutable USD rate ÷ rate unit. "
            "Decimals below are persisted without display rounding."
        )
        st.json(
            [
                {
                    "attempt_id": item.attempt_id,
                    "issue_number": item.issue_number,
                    "dispatch_order": item.dispatch_order,
                    **item.calculation.model_dump(mode="json"),
                }
                for item in summary.request_costs
            ]
        )


def _money(value: Decimal) -> str:
    return f"${value:.8f}"


def _money_or_unknown(value: Decimal | None) -> str:
    return "unknown" if value is None else _money(value)


def _rate(value: Decimal) -> str:
    return f"{value:.3f}"


def _milliseconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _percent(value: float) -> str:
    return f"{value:.1%}"
