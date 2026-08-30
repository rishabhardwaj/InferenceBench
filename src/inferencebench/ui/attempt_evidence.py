from __future__ import annotations

from decimal import Decimal

import streamlit as st

from inferencebench.domain import AttemptEvidence


NO_VALID_PREDICTION = "No Valid Prediction"


def render_attempt_evidence(
    heading: str,
    attempt: AttemptEvidence,
    *,
    result_label: str | None = None,
    result_value: str | None = None,
) -> None:
    prediction = (
        attempt.parsed_label.value
        if attempt.parsed_label is not None
        else NO_VALID_PREDICTION
    )
    st.markdown(f"#### {heading}")
    st.markdown(f"**Prediction:** `{prediction}`")
    if result_label is not None and result_value is not None:
        st.markdown(f"**{result_label}:** `{result_value}`")
    st.markdown(
        f"**Output Adherence:** `{attempt.parse_status.value}` · "
        f"Normalizations: `{', '.join(attempt.normalizations) or 'none'}`"
    )
    st.markdown("**Raw model output**")
    st.code(attempt.raw_model_output or "", language=None)
    if attempt.raw_error is not None:
        st.markdown("**Typed terminal error**")
        st.json(attempt.raw_error)
    st.markdown(
        f"**Request Latency:** `{attempt.request_latency_ms:.2f} ms` · "
        f"**Queue Wait:** `{attempt.queue_wait_ms:.2f} ms`"
    )
    st.markdown(
        "**Provider outcome:** "
        f"`{attempt.provider_outcome.value if attempt.provider_outcome else 'fixture-not-recorded'}` · "
        f"**HTTP status:** `{attempt.http_status if attempt.http_status is not None else 'N/A'}`"
    )
    st.markdown("**Provider usage**")
    st.json(attempt.usage)
    st.markdown(
        f"**Calculated Request Cost:** `{_cost_display(attempt.calculated_request_cost_usd)}` · "
        f"**Cost Completeness:** `{attempt.cost_completeness.value}` · "
        f"**Pricing Snapshot:** `{attempt.pricing_snapshot_id}`"
    )
    if attempt.cost_unknown_reasons:
        st.caption(
            "Cost is unknown because: " + "; ".join(attempt.cost_unknown_reasons)
        )
    if attempt.cost_calculation_terms:
        st.markdown("**Exact cost terms**")
        st.json(
            [
                term.model_dump(mode="json")
                for term in attempt.cost_calculation_terms
            ]
        )
    st.caption(
        f"Attempt `{attempt.attempt_id}` · Provider request "
        f"`{attempt.provider_request_id or 'not available'}`"
    )


def _cost_display(value: Decimal | None) -> str:
    return "unknown" if value is None else f"${value}"
