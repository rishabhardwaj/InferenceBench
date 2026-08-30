from __future__ import annotations

from decimal import Decimal

import streamlit as st

from inferencebench.workflows.candidates import CandidateCatalogReview


def render_candidate_catalog(review: CandidateCatalogReview) -> None:
    st.subheader("Eligible Candidate Pool and Pricing Snapshot")
    st.caption(review.evidence_state)
    st.info(
        "The active pool contains 25 runnable candidates. The frozen discovery "
        "snapshot is retained separately. Model "
        "selection and paid execution are isolated in the Run Comparison View."
    )
    st.warning(
        "Excluded after provider validation: arcee-trinity-large-thinking — "
        "this model is not available for your subscription tier."
    )

    with st.expander("Model and pricing evidence", expanded=True):
        st.markdown(f"**Eligible candidates:** `{len(review.candidates)}`")
        st.markdown(
            f"**Model Catalog:** `{review.catalog_version}` · `{review.catalog_sha256}`"
        )
        st.markdown(f"**Source snapshot:** `{review.source_snapshot_id}`")
        st.markdown(
            f"**Pricing Snapshot:** `{review.pricing_snapshot_id}` · "
            f"`{review.pricing_sha256}`"
        )
        st.markdown(
            f"**Pricing source:** [DigitalOcean Inference Pricing]"
            f"({review.pricing_source_url}) · updated "
            f"`{review.pricing_source_last_updated.isoformat()}`"
        )

    with st.expander("All 25 active candidates", expanded=False):
        st.dataframe(
            [
                {
                    "Model ID": candidate.model_id,
                    "Context": _known_integer(candidate.context_length),
                    "Max output": _known_integer(candidate.max_output_tokens),
                    "Compatibility": candidate.compatibility.replace("_", " "),
                    "Family": candidate.family or "Unknown",
                    "Reasoning": candidate.reasoning_characteristic or "Unknown",
                    "Parameters": candidate.parameter_summary or "Unknown",
                    "Input / 1M": _rate(candidate.standard_input_rate_usd),
                    "Output / 1M": _rate(candidate.output_rate_usd),
                    "Cache read / 1M": _rate(candidate.cache_read_input_rate_usd),
                    "Cache write / 1M": _rate(candidate.cache_write_input_rate_usd),
                }
                for candidate in review.candidates
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Unknown and Not published are evidence states, not zero values. Family, "
            "reasoning behavior, parameters, and request compatibility remain unknown "
            "until supported by reviewed sources or empirical preflight evidence."
        )


def _known_integer(value: int | None) -> str:
    return f"{value:,}" if value is not None else "Unknown"


def _rate(value: Decimal | None) -> str:
    return f"${value}" if value is not None else "Not published"
