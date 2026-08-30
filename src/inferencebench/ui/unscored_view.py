from __future__ import annotations

import streamlit as st

from inferencebench.domain import AttemptEvidence, CustomerLabel
from inferencebench.evaluation.unscored import (
    CUSTOMER_LABEL_ORDER,
    NO_VALID_PREDICTION,
    UnscoredComparison,
    UnscoredModelSummary,
    UnscoredPairOutcome,
    UnscoredPairRow,
    UnscoredResultState,
    UnscoredRowFilter,
    filter_unscored_rows,
)
from inferencebench.ui.attempt_evidence import render_attempt_evidence


def render_unscored_view(comparison: UnscoredComparison) -> None:
    st.subheader("Unscored View")
    st.caption(
        "These issues have no accepted label. This view reports model agreement, "
        "suggestion distributions, and failures."
    )
    _render_headlines(comparison)
    _render_pair_outcomes(comparison)
    _render_distributions(comparison)
    filtered_rows = _render_filters(comparison)
    _render_issue_table_and_detail(comparison, filtered_rows)


def _render_headlines(comparison: UnscoredComparison) -> None:
    strict_column, both_valid_column = st.columns(2)
    with strict_column:
        st.metric("Strict Agreement Rate", f"{comparison.strict_agreement_rate:.1%}")
        st.caption(
            f"{comparison.strict_agreement_numerator}/"
            f"{comparison.strict_agreement_denominator} expected unscored issues"
        )
    with both_valid_column:
        rate = comparison.both_valid_agreement_rate
        st.metric(
            "Both-Valid Agreement Rate",
            "N/A" if rate is None else f"{rate:.1%}",
        )
        st.caption(
            f"{comparison.both_valid_agreement_numerator}/"
            f"{comparison.both_valid_agreement_denominator} issues with two valid labels"
        )


def _render_pair_outcomes(comparison: UnscoredComparison) -> None:
    st.markdown("### Paired outcomes")
    columns = st.columns(4)
    labels = {
        UnscoredPairOutcome.LABEL_AGREEMENT: "Label agreement",
        UnscoredPairOutcome.LABEL_DISAGREEMENT: "Label disagreement",
        UnscoredPairOutcome.ONE_SIDED_FAILURE: "One-sided failure",
        UnscoredPairOutcome.JOINT_FAILURE: "Joint failure",
    }
    for column, outcome in zip(columns, UnscoredPairOutcome, strict=True):
        with column:
            st.metric(labels[outcome], comparison.pair_outcome_counts[outcome])
    st.caption(
        "Matching invalid outputs or request errors are joint failures, not label agreement."
    )


def _render_distributions(comparison: UnscoredComparison) -> None:
    st.markdown("### Suggestion distributions")
    st.caption(
        "Each distribution uses every expected unscored issue. Failed results remain "
        "visible as No Valid Prediction."
    )
    model_a_column, model_b_column = st.columns(2)
    with model_a_column:
        _render_model_distribution("Model A", comparison.model_a)
    with model_b_column:
        _render_model_distribution("Model B", comparison.model_b)


def _render_model_distribution(label: str, summary: UnscoredModelSummary) -> None:
    st.markdown(f"**{label} — `{summary.model_id}`**")
    st.dataframe(
        tuple(
            {
                "Suggestion": row.display_prediction,
                "Count": row.count,
                "Expected": row.expected_count,
                "Rate": f"{row.rate:.1%}",
            }
            for row in summary.suggestion_distribution
        ),
        hide_index=True,
        width="stretch",
    )
    states = summary.result_state_counts
    st.caption(
        f"Valid label: {states[UnscoredResultState.VALID_LABEL]}/"
        f"{summary.expected_count} · Invalid output: "
        f"{states[UnscoredResultState.INVALID_OUTPUT]}/{summary.expected_count} · "
        f"Request error: {states[UnscoredResultState.REQUEST_ERROR]}/"
        f"{summary.expected_count}"
    )
    if summary.request_error_type_counts:
        st.caption(
            "Typed request errors: "
            + ", ".join(
                f"{outcome.value}={count}"
                for outcome, count in sorted(
                    summary.request_error_type_counts.items(),
                    key=lambda item: item[0].value,
                )
            )
        )


def _render_filters(comparison: UnscoredComparison) -> tuple[UnscoredPairRow, ...]:
    st.markdown("### Issue-level comparison")
    st.caption(
        "The default order shows label disagreements and failures before agreements. "
        "Filters never change the headline denominators above."
    )
    with st.expander("Filter unscored evidence", expanded=False):
        prediction_options = (
            *(label.value for label in CUSTOMER_LABEL_ORDER),
            NO_VALID_PREDICTION,
        )
        model_a_predictions = st.multiselect(
            "Unscored Model A prediction",
            options=prediction_options,
            key="unscored_model_a_prediction_filter",
        )
        model_b_predictions = st.multiselect(
            "Unscored Model B prediction",
            options=prediction_options,
            key="unscored_model_b_prediction_filter",
        )
        pair_outcomes = st.multiselect(
            "Unscored Pair Outcome",
            options=tuple(outcome.value for outcome in UnscoredPairOutcome),
            key="unscored_pair_outcome_filter",
        )
        state_options = tuple(state.value for state in UnscoredResultState)
        model_a_states = st.multiselect(
            "Unscored Model A result state",
            options=state_options,
            key="unscored_model_a_state_filter",
        )
        model_b_states = st.multiselect(
            "Unscored Model B result state",
            options=state_options,
            key="unscored_model_b_state_filter",
        )
        text_query = st.text_input(
            "Unscored issue title/body contains",
            key="unscored_text_filter",
        )
    return filter_unscored_rows(
        comparison.rows,
        UnscoredRowFilter(
            model_a_predictions=tuple(
                _prediction_value(value) for value in model_a_predictions
            ),
            model_b_predictions=tuple(
                _prediction_value(value) for value in model_b_predictions
            ),
            pair_outcomes=tuple(
                UnscoredPairOutcome(value) for value in pair_outcomes
            ),
            model_a_result_states=tuple(
                UnscoredResultState(value) for value in model_a_states
            ),
            model_b_result_states=tuple(
                UnscoredResultState(value) for value in model_b_states
            ),
            text_query=text_query,
        ),
    )


def _render_issue_table_and_detail(
    comparison: UnscoredComparison, rows: tuple[UnscoredPairRow, ...]
) -> None:
    st.dataframe(
        tuple(_issue_table_row(row) for row in rows),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Showing {len(rows)}/{len(comparison.rows)} unscored issues")
    if not rows:
        st.info("No unscored issues match the current filters.")
        return
    rows_by_number = {row.issue.issue_number: row for row in rows}
    selected_issue_number = st.selectbox(
        "Unscored issue detail",
        options=tuple(rows_by_number),
        format_func=lambda issue_number: (
            f"#{issue_number} — {rows_by_number[issue_number].issue.title}"
        ),
        key="unscored_issue_detail",
    )
    _render_issue_detail(rows_by_number[selected_issue_number])


def _render_issue_detail(row: UnscoredPairRow) -> None:
    with st.expander(
        f"Unscored issue #{row.issue.issue_number} evidence",
        expanded=True,
    ):
        st.markdown(f"**[{row.issue.title}]({row.issue.html_url})**")
        st.markdown(f"**Unscored Pair Outcome:** `{row.pair_outcome.value}`")
        st.markdown("**Model-Visible Issue Input**")
        st.code(row.model_a_attempt.request_messages[-1].content, language="json")
        st.markdown(f"**Title:** {row.issue.title}")
        st.markdown("**Body:**")
        st.write(
            row.issue.body if row.issue.body is not None else "_Absent body (`null`)._"
        )
        model_a_column, model_b_column = st.columns(2)
        with model_a_column:
            render_attempt_evidence(
                "Model A",
                row.model_a_attempt,
                result_label="Result state",
                result_value=row.model_a_result_state.value,
            )
        with model_b_column:
            render_attempt_evidence(
                "Model B",
                row.model_b_attempt,
                result_label="Result state",
                result_value=row.model_b_result_state.value,
            )


def _issue_table_row(row: UnscoredPairRow) -> dict[str, object]:
    return {
        "Issue": row.issue.issue_number,
        "Title": row.issue.title,
        "Model A": _prediction_display(row.model_a_attempt),
        "A result state": row.model_a_result_state.value,
        "Model B": _prediction_display(row.model_b_attempt),
        "B result state": row.model_b_result_state.value,
        "Pair outcome": row.pair_outcome.value,
    }


def _prediction_display(attempt: AttemptEvidence) -> str:
    return attempt.parsed_label.value if attempt.parsed_label else NO_VALID_PREDICTION


def _prediction_value(value: str) -> CustomerLabel | None:
    return None if value == NO_VALID_PREDICTION else CustomerLabel(value)
