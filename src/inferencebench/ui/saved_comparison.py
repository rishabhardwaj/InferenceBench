from __future__ import annotations

import streamlit as st

from inferencebench.ui.scored_view import render_scored_view
from inferencebench.ui.operational_view import render_operational_view
from inferencebench.ui.unscored_view import render_unscored_view
from inferencebench.workflows.saved_comparison import SavedComparisonReview


def render_saved_comparison(review: SavedComparisonReview) -> None:
    st.caption(review.evidence_state)
    st.info(
        "This review is rebuilt from persisted Run Manifests, Attempt Evidence, "
        "and Ground Truth. Opening it made 0 provider requests."
    )

    st.subheader("Saved Comparison Review")
    with st.expander("Evidence identity", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Repository:** `{review.repository}`")
            st.markdown(f"**Run Corpus:** `{review.corpus_version}`")
            st.markdown(f"**Corpus SHA-256:** `{review.corpus_sha256}`")
            st.markdown(f"**Run Corpus issues:** `{review.corpus_issue_count}`")
        with right:
            st.markdown(f"**Ground Truth:** `{review.ground_truth_version}`")
            st.markdown(
                "**Shared Inference Contract:** "
                f"`{review.shared_inference_contract_version}`"
            )
            st.markdown(f"**Parser:** `{review.parser_version}`")
            st.markdown(f"**Concurrency:** `{review.concurrency}`")
    render_operational_view(review.operational_view)
    st.divider()
    render_scored_view(review.scored_view)
    st.divider()
    render_unscored_view(review.unscored_view)
