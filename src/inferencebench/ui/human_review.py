from __future__ import annotations

import streamlit as st

from inferencebench.workflows.human_review import CompletedHumanReviewSummary


def render_human_review_populations(summary: CompletedHumanReviewSummary | None) -> None:
    st.subheader("Human-Reviewed Ground Truth")
    if summary is None:
        st.info(
            "No completed Random Human-Reviewed Sample is bundled yet. "
            "Prompt Development and Primary Scored Holdout results are therefore unavailable."
        )
        return
    st.caption(summary.evidence_state)
    st.markdown(
        f"**Ground Truth:** `{summary.ground_truth_version}` · "
        f"**Corpus:** `{summary.corpus_version}`"
    )
    st.markdown(
        f"**Reviewed:** `{summary.reviewed_count}` · **Accepted:** "
        f"`{summary.accepted_count}` · **Excluded:** `{summary.excluded_count}`"
    )
    st.markdown(
        "**Named populations:** "
        f"`{summary.prompt_development_count}` Prompt Development Sample; "
        f"`{summary.primary_holdout_count}` Primary Scored Holdout. "
        "Prompt Development is not unseen evidence."
    )
    st.markdown(
        "**Accepted-label support:** "
        + ", ".join(f"`{label}`: {count}" for label, count in summary.support_counts.items())
    )
