from __future__ import annotations

import asyncio
import os
from decimal import Decimal

import streamlit as st
from inferencebench.config import Settings
from inferencebench.inference.digitalocean import redact_text
from inferencebench.ui.saved_comparison import render_saved_comparison
from inferencebench.workflows.live_comparison import (
    LiveComparisonConfiguration,
    LiveComparisonExecution,
    LiveComparisonPreparation,
    LiveModelProgress,
    execute_live_comparison,
)
from inferencebench.workflows.saved_comparison import load_persisted_comparison


_SESSION_KEY = "live-session-api-key"
_LIVE_RUN_IDS = "live-comparison-run-ids"


def render_run_comparison(
    settings: Settings,
    preparation: LiveComparisonPreparation,
) -> None:
    st.subheader("Run Comparison View")
    st.warning(
        "Paid action: starting creates two new full-Corpus runs and makes "
        f"{preparation.expected_provider_requests:,} DigitalOcean inference requests. "
        "Changing a selector or refreshing this page never starts inference."
    )
    st.info(
        "Execution is sequential: Model A completes its independent run, then Model B "
        "runs with the same displayed concurrency. The models never share a hidden "
        "combined request pool. Cost is unknown until usage is returned."
    )

    environment_key = os.environ.get("DO_INFERENCE_API_KEY", "")
    session_key = ""
    if environment_key:
        st.success(
            "A DigitalOcean credential is configured in the server environment. "
            "Its value is never displayed or persisted."
        )
    else:
        st.markdown("#### Live credential")
        st.caption(
            "The masked key is transmitted to and temporarily held in this Streamlit "
            "server's active session memory. It is not browser-only. Use the local "
            "Docker workflow if you do not trust the hosted server, and use HTTPS for "
            "a hosted demonstration."
        )
        session_key = st.text_input(
            "DigitalOcean API key (active session only)",
            type="password",
            key=_SESSION_KEY,
            autocomplete="off",
        )
        if session_key and st.button("Clear session API key"):
            del st.session_state[_SESSION_KEY]
            st.rerun()

    api_key = environment_key or session_key
    for blocker in preparation.blockers:
        st.error(blocker)

    with st.form("live-comparison-form", clear_on_submit=False):
        model_columns = st.columns(2)
        with model_columns[0]:
            model_a = st.selectbox(
                "Model A",
                preparation.eligible_model_ids,
                index=0,
                key="live-model-a",
            )
        with model_columns[1]:
            model_b = st.selectbox(
                "Model B",
                preparation.eligible_model_ids,
                index=1,
                key="live-model-b",
            )
        concurrency = int(
            st.number_input(
                "Shared concurrency per independent model run",
                min_value=1,
                max_value=64,
                value=preparation.default_concurrency,
                step=1,
            )
        )
        _render_effective_configuration(preparation, model_a, model_b, concurrency)
        if model_a == model_b:
            st.error("Model A and Model B must be distinct.")
        if not api_key:
            st.caption(
                "Live start is disabled until an environment or active-session key exists. "
                "Saved Comparison Review remains available without a key."
            )
        submitted = st.form_submit_button(
            "Start new comparison — paid action",
            type="primary",
            disabled=(
                not preparation.can_start or not api_key or model_a == model_b
            ),
        )

    if submitted:
        _start_and_render(
            settings,
            preparation,
            model_a,
            model_b,
            concurrency,
            api_key,
        )
    elif _LIVE_RUN_IDS in st.session_state:
        _render_completed_live_review(
            settings,
            tuple(st.session_state[_LIVE_RUN_IDS]),
        )


def _render_effective_configuration(
    preparation: LiveComparisonPreparation,
    model_a: str,
    model_b: str,
    concurrency: int,
) -> None:
    st.markdown("#### Effective frozen configuration")
    st.markdown(
        f"**Corpus:** `{preparation.repository}` / `{preparation.corpus_version}` · "
        f"SHA-256 `{preparation.corpus_sha256}` · "
        f"`{preparation.corpus_issue_count}` issues"
    )
    st.markdown(
        f"**Evaluation Corpus:** `{preparation.ground_truth_version}` · "
        f"`{preparation.scored_issue_count}` scored / "
        f"`{preparation.unscored_issue_count}` unscored"
    )
    st.markdown(
        f"**Shared Inference Contract:** `{preparation.contract_version}` "
        f"(`{preparation.contract_status}`) · prompt `{preparation.prompt_version}` · "
        f"parser `{preparation.parser_version}`"
    )
    st.markdown(
        "**Shared Generation Configuration:** "
        f"`{preparation.generation_configuration_version}` · "
        f"max completion tokens `{preparation.max_completion_tokens}` · "
        f"SHA-256 `{preparation.generation_configuration_sha256}`"
    )
    st.markdown(
        f"**Models:** `{model_a}` then `{model_b}` · "
        f"**Concurrency per run:** `{concurrency}` · "
        f"**Shared Benchmark Timeout:** `{preparation.shared_timeout_seconds:g}s`"
    )
    st.markdown(
        f"**Expected provider requests:** `2 × {preparation.corpus_issue_count} = "
        f"{preparation.expected_provider_requests}` · **Retries:** `0`"
    )


def _start_and_render(
    settings: Settings,
    preparation: LiveComparisonPreparation,
    model_a: str,
    model_b: str,
    concurrency: int,
    api_key: str,
) -> None:
    progress_slots = {
        "Model A": st.empty(),
        "Model B": st.empty(),
    }
    latest_run_ids: dict[str, str] = {}

    def report(progress: LiveModelProgress) -> None:
        latest_run_ids[progress.position] = progress.run_id
        progress_slots[progress.position].markdown(_progress_text(progress))

    try:
        configuration = LiveComparisonConfiguration(
            model_a_id=model_a,
            model_b_id=model_b,
            concurrency=concurrency,
        )
        execution = asyncio.run(
            execute_live_comparison(
                settings,
                configuration,
                api_key=api_key,
                progress_callback=report,
            )
        )
    except Exception as error:
        # This is the presentation boundary for provider and persistence errors.
        # Render only a credential-redacted message; the durable run evidence
        # already contains the typed, sanitized failure details.
        safe_message = redact_text(str(error), api_key)
        st.error(f"Live Comparison stopped: {safe_message}")
        if latest_run_ids:
            st.warning(
                "Any completed Attempt Evidence remains durable. Starting again creates "
                "new run IDs; the interrupted evidence is not repaired or overwritten."
            )
            for position, run_id in latest_run_ids.items():
                st.markdown(f"**{position} interrupted run:** `{run_id}`")
        return

    _render_execution_outcome(settings, execution)


def _render_execution_outcome(
    settings: Settings,
    execution: LiveComparisonExecution,
) -> None:
    run_ids = execution.run_ids
    if not execution.comparison_complete:
        st.warning(
            "The pair is incomplete and is not presented as a finished comparison. "
            "Completed attempts remain persisted; starting again creates fresh run IDs."
        )
        for position, result in (
            ("Model A", execution.model_a),
            ("Model B", execution.model_b),
        ):
            manifest = result.manifest
            st.markdown(
                f"**{position}:** `{manifest.model_id}` · run `{manifest.run_id}` · "
                f"status `{manifest.status.value}` · persisted "
                f"`{manifest.persisted_count}/{manifest.expected_count}`"
            )
        return
    st.session_state[_LIVE_RUN_IDS] = run_ids
    st.success(
        "Both fresh independent runs completed. The comparison below is rebuilt from "
        "their persisted evidence; no saved operational measurement was substituted."
    )
    _render_completed_live_review(settings, run_ids)


def _render_completed_live_review(
    settings: Settings,
    run_ids: tuple[str, str],
) -> None:
    try:
        review = load_persisted_comparison(settings, run_ids)
    except Exception as error:
        st.error(
            "Persisted live comparison could not be opened: "
            f"{redact_text(str(error))}"
        )
        return
    st.divider()
    render_saved_comparison(review)


def _progress_text(progress: LiveModelProgress) -> str:
    cost_note = (
        "complete so far"
        if progress.unknown_cost_count == 0
        else f"{progress.unknown_cost_count} observed costs unknown"
    )
    return (
        f"**{progress.position}: `{progress.model_id}`** · run `{progress.run_id}` · "
        f"status `{progress.status.value}`  \n"
        f"Persisted `{progress.persisted_count}/{progress.expected_count}` · "
        f"usable `{progress.usable_count}` · failures `{progress.failure_count}` "
        f"(invalid `{progress.invalid_output_count}`, request errors "
        f"`{progress.request_error_count}`) · elapsed "
        f"`{progress.elapsed_wall_clock_ms / 1000:.2f}s` · known cost "
        f"`{_money(progress.known_cost_usd)}` ({cost_note})"
    )


def _money(value: Decimal) -> str:
    return f"${value:.8f}"
