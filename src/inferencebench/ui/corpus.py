from __future__ import annotations

import streamlit as st

from inferencebench.workflows.corpus import CorpusSnapshotSummary


def render_corpus_summary(summary: CorpusSnapshotSummary) -> None:
    st.subheader("Complete doctl Corpus")
    st.caption(summary.evidence_state)
    st.info(
        "Loaded from the bundled immutable snapshot. Opening this application made "
        "0 GitHub requests. The demonstration comparison below remains separate and "
        "identifies its own one-issue fixture Corpus."
    )
    count_column, pr_column, page_column = st.columns(3)
    count_column.metric("Issues retained", summary.issue_count)
    pr_column.metric("Pull requests excluded", summary.excluded_pull_request_count)
    page_column.metric("API pages followed", summary.page_count)
    with st.expander("Corpus Snapshot Manifest", expanded=True):
        st.markdown(f"**Repository:** `{summary.repository}`")
        st.markdown(f"**Corpus version:** `{summary.corpus_version}`")
        st.markdown(f"**Corpus SHA-256:** `{summary.corpus_sha256}`")
        st.markdown(
            "**Retrieval interval:** "
            f"`{summary.retrieval_started_at.isoformat()}` to "
            f"`{summary.retrieval_completed_at.isoformat()}`"
        )
        st.markdown(f"**API objects fetched:** `{summary.api_object_count}`")

