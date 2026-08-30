from __future__ import annotations

import sqlite3

import streamlit as st
from pydantic import ValidationError

from inferencebench.artifacts import ArtifactIntegrityError
from inferencebench.config import Settings
from inferencebench.evaluation.metrics import MetricIntegrityError
from inferencebench.persistence.repository import (
    EvidenceConflictError,
    EvidenceNotFoundError,
)
from inferencebench.ui.corpus import render_corpus_summary
from inferencebench.ui.candidates import render_candidate_catalog
from inferencebench.ui.saved_comparison import render_saved_comparison
from inferencebench.ui.live_comparison import render_run_comparison
from inferencebench.workflows.corpus import load_active_corpus_summary
from inferencebench.workflows.candidates import load_candidate_catalog
from inferencebench.workflows.saved_comparison import (
    FixtureEvidenceError,
    load_saved_comparison,
)
from inferencebench.workflows.live_comparison import prepare_live_comparison
from inferencebench.workflows.recommendation import (
    load_recommended_comparison,
    load_selection_decision,
)
from inferencebench.ui.recommendation import render_recommendation


def main() -> None:
    st.set_page_config(page_title="InferenceBench", page_icon="🧪", layout="wide")
    try:
        settings = Settings.from_environment()
        corpus_summary = load_active_corpus_summary(settings)
        candidate_catalog = load_candidate_catalog(settings)
        selection_decision = load_selection_decision(settings)
        comparison = (
            load_recommended_comparison(settings, selection_decision)
            if selection_decision is not None
            else load_saved_comparison(settings)
        )
        live_preparation = prepare_live_comparison(settings)
    except (
        ArtifactIntegrityError,
        EvidenceConflictError,
        EvidenceNotFoundError,
        FixtureEvidenceError,
        MetricIntegrityError,
        ValidationError,
        sqlite3.Error,
        OSError,
    ) as error:
        st.error(f"Saved evidence could not be loaded: {error}")
        st.stop()
    st.title("InferenceBench")
    active_view = st.radio(
        "Application view",
        ("Saved Comparison Review", "Run Comparison View"),
        horizontal=True,
    )
    render_corpus_summary(corpus_summary)
    st.divider()
    render_candidate_catalog(candidate_catalog)
    st.divider()
    if active_view == "Saved Comparison Review":
        if selection_decision is not None:
            render_recommendation(selection_decision, comparison)
            st.divider()
        render_saved_comparison(comparison)
    else:
        render_run_comparison(settings, live_preparation)


main()
