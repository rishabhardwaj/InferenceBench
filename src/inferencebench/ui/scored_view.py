from __future__ import annotations

import streamlit as st

from inferencebench.domain import AttemptEvidence, CustomerLabel, ScoredOutcome
from inferencebench.evaluation.scored import (
    CUSTOMER_LABEL_ORDER,
    NO_VALID_PREDICTION,
    ScoredComparison,
    ScoredEvidenceSource,
    ScoredModelSummary,
    ScoredPairRow,
    ScoredRowFilter,
    ScoredSamplingStratum,
    ScoredViewReview,
    filter_scored_rows,
)
from inferencebench.ui.attempt_evidence import render_attempt_evidence


def render_scored_view(review: ScoredViewReview) -> None:
    st.subheader("Scored View")
    by_display_name = {
        comparison.population_display_name: comparison
        for comparison in review.comparisons
    }
    default = next(
        comparison
        for comparison in review.comparisons
        if comparison.population is review.default_population
    )
    selected_name = st.selectbox(
        "Evidence stratum",
        options=tuple(by_display_name),
        index=tuple(by_display_name).index(default.population_display_name),
        help=(
            "Primary Scored Holdout is the headline population. Development, "
            "diagnostic, mapping-audit, maintainer-derived, and combined evidence "
            "remain separately identified."
        ),
        key="scored_population",
    )
    comparison = by_display_name[selected_name]
    if comparison.is_primary_headline:
        st.success(comparison.interpretation)
    else:
        st.warning(comparison.interpretation)

    _render_headlines(comparison)
    _render_per_class_and_confusion(comparison)
    filtered_rows = _render_filters(comparison)
    _render_issue_table_and_detail(comparison, filtered_rows)


def _render_headlines(comparison: ScoredComparison) -> None:
    model_a_column, model_b_column = st.columns(2)
    with model_a_column:
        _render_model_headline("Model A", comparison.model_a)
    with model_b_column:
        _render_model_headline("Model B", comparison.model_b)


def _render_model_headline(label: str, summary: ScoredModelSummary) -> None:
    st.markdown(f"### {label}: `{summary.model_id}`")
    accuracy_column, macro_column = st.columns(2)
    with accuracy_column:
        st.metric(
            "Accuracy",
            f"{summary.accuracy:.1%}",
            help="Correct outcomes divided by every expected item in this scored population.",
        )
        st.caption(
            f"{summary.correct_count}/{summary.expected_count} correct · "
            f"Run `{summary.run_id}`"
        )
    with macro_column:
        st.metric(
            "Supported-Class Macro-F1",
            f"{summary.supported_class_macro_f1:.3f}",
            help="Unweighted mean F1 over labels with positive Ground Truth support.",
        )
        st.caption(
            f"Coverage {summary.supported_class_count}/{summary.total_class_count} labels"
        )
    st.caption(
        f"Invalid output: {summary.invalid_output_count}/{summary.expected_count} · "
        f"Request error: {summary.request_error_count}/{summary.expected_count}"
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


def _render_per_class_and_confusion(comparison: ScoredComparison) -> None:
    st.markdown("### Per-class behavior")
    model_a_column, model_b_column = st.columns(2)
    with model_a_column:
        st.markdown(f"**Model A — `{comparison.model_a.model_id}`**")
        st.dataframe(
            _per_class_rows(comparison.model_a),
            hide_index=True,
            width="stretch",
        )
    with model_b_column:
        st.markdown(f"**Model B — `{comparison.model_b.model_id}`**")
        st.dataframe(
            _per_class_rows(comparison.model_b),
            hide_index=True,
            width="stretch",
        )

    st.markdown("### Confusion matrices")
    st.caption(
        "Rows are Ground Truth. Invalid outputs and request errors share the "
        "No Valid Prediction column; their typed causes remain in the evidence below."
    )
    matrix_a_column, matrix_b_column = st.columns(2)
    with matrix_a_column:
        st.markdown(f"**Model A — `{comparison.model_a.model_id}`**")
        st.dataframe(
            _confusion_rows(comparison.model_a),
            hide_index=True,
            width="stretch",
        )
    with matrix_b_column:
        st.markdown(f"**Model B — `{comparison.model_b.model_id}`**")
        st.dataframe(
            _confusion_rows(comparison.model_b),
            hide_index=True,
            width="stretch",
        )


def _render_filters(comparison: ScoredComparison) -> tuple[ScoredPairRow, ...]:
    st.markdown("### Issue-level disagreements")
    st.caption(
        "The table starts with model disagreements and no-valid-prediction cases. "
        "Filters never change the headline denominator above."
    )
    with st.expander("Filter scored evidence", expanded=False):
        truth_values = st.multiselect(
            "Ground Truth Label",
            options=tuple(label.value for label in CUSTOMER_LABEL_ORDER),
            key="scored_truth_filter",
        )
        prediction_options = (
            *(label.value for label in CUSTOMER_LABEL_ORDER),
            NO_VALID_PREDICTION,
        )
        model_a_predictions = st.multiselect(
            "Model A prediction",
            options=prediction_options,
            key="scored_model_a_prediction_filter",
        )
        model_b_predictions = st.multiselect(
            "Model B prediction",
            options=prediction_options,
            key="scored_model_b_prediction_filter",
        )
        outcome_options = tuple(outcome.value for outcome in ScoredOutcome)
        model_a_outcomes = st.multiselect(
            "Model A Scored Outcome",
            options=outcome_options,
            key="scored_model_a_outcome_filter",
        )
        model_b_outcomes = st.multiselect(
            "Model B Scored Outcome",
            options=outcome_options,
            key="scored_model_b_outcome_filter",
        )
        sources = st.multiselect(
            "Ground Truth provenance",
            options=tuple(
                sorted(
                    {
                        row.ground_truth.ground_truth_source.value
                        for row in comparison.rows
                    }
                )
            ),
            key="scored_source_filter",
        )
        strata = st.multiselect(
            "Sampling stratum",
            options=tuple(
                sorted(
                    {row.ground_truth.sampling_stratum.value for row in comparison.rows}
                )
            ),
            key="scored_stratum_filter",
        )
        agreement = st.selectbox(
            "Model prediction relationship",
            options=("all", "agreement", "disagreement"),
            key="scored_agreement_filter",
        )
        text_query = st.text_input(
            "Issue title/body contains",
            key="scored_text_filter",
        )
    return filter_scored_rows(
        comparison.rows,
        ScoredRowFilter(
            ground_truth_labels=tuple(CustomerLabel(value) for value in truth_values),
            model_a_predictions=tuple(
                _prediction_value(value) for value in model_a_predictions
            ),
            model_b_predictions=tuple(
                _prediction_value(value) for value in model_b_predictions
            ),
            model_a_outcomes=tuple(
                ScoredOutcome(value) for value in model_a_outcomes
            ),
            model_b_outcomes=tuple(
                ScoredOutcome(value) for value in model_b_outcomes
            ),
            evidence_sources=tuple(ScoredEvidenceSource(value) for value in sources),
            sampling_strata=tuple(ScoredSamplingStratum(value) for value in strata),
            agreement=agreement,
            text_query=text_query,
        ),
    )


def _render_issue_table_and_detail(
    comparison: ScoredComparison, rows: tuple[ScoredPairRow, ...]
) -> None:
    st.dataframe(
        tuple(_issue_table_row(row) for row in rows),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Showing {len(rows)}/{len(comparison.rows)} scored issues")
    if not rows:
        st.info("No scored issues match the current filters.")
        return
    rows_by_number = {row.issue.issue_number: row for row in rows}
    selected_issue_number = st.selectbox(
        "Issue detail",
        options=tuple(rows_by_number),
        format_func=lambda issue_number: (
            f"#{issue_number} — {rows_by_number[issue_number].issue.title}"
        ),
        key="scored_issue_detail",
    )
    _render_issue_detail(rows_by_number[selected_issue_number])


def _render_issue_detail(row: ScoredPairRow) -> None:
    with st.expander(
        f"Issue #{row.issue.issue_number} evidence",
        expanded=True,
    ):
        st.markdown(f"**[{row.issue.title}]({row.issue.html_url})**")
        st.markdown("**Model-Visible Issue Input**")
        st.code(row.model_a_attempt.request_messages[-1].content, language="json")
        st.markdown(f"**Title:** {row.issue.title}")
        st.markdown("**Body:**")
        st.write(row.issue.body if row.issue.body is not None else "_Absent body (`null`)._")
        truth = row.ground_truth
        st.markdown(
            f"**Ground Truth Label:** `{truth.label.value}` · "
            f"**Source:** `{truth.ground_truth_source.value}` · "
            f"**Sampling stratum:** `{truth.sampling_stratum.value}` · "
            f"**Evaluation role:** `{truth.evaluation_role.value}`"
        )
        st.caption(
            f"Provenance `{truth.provenance_id}` · Rubric `{truth.rubric_version}` · "
            f"Confidence `{truth.confidence or 'not applicable'}`"
        )
        model_a_column, model_b_column = st.columns(2)
        with model_a_column:
            render_attempt_evidence(
                "Model A",
                row.model_a_attempt,
                result_label="Scored Outcome",
                result_value=row.model_a_outcome.value,
            )
        with model_b_column:
            render_attempt_evidence(
                "Model B",
                row.model_b_attempt,
                result_label="Scored Outcome",
                result_value=row.model_b_outcome.value,
            )


def _per_class_rows(summary: ScoredModelSummary) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "Label": row.label.value,
            "TP": row.true_positive,
            "FP": row.false_positive,
            "FN": row.false_negative,
            "Support": row.support,
            "Predicted": row.predicted_count,
            "Precision": _metric_display(row.precision),
            "Recall": _metric_display(row.recall),
            "F1": _metric_display(row.f1),
        }
        for row in summary.per_class
    )


def _confusion_rows(summary: ScoredModelSummary) -> tuple[dict[str, object], ...]:
    counts = {
        (cell.ground_truth, cell.prediction): len(cell.issue_numbers)
        for cell in summary.confusion_cells
    }
    return tuple(
        {
            "Ground Truth": truth.value,
            **{
                prediction.value: counts[(truth, prediction)]
                for prediction in CUSTOMER_LABEL_ORDER
            },
            NO_VALID_PREDICTION: counts[(truth, None)],
        }
        for truth in CUSTOMER_LABEL_ORDER
    )


def _issue_table_row(row: ScoredPairRow) -> dict[str, object]:
    return {
        "Issue": row.issue.issue_number,
        "Title": row.issue.title,
        "Ground Truth": row.ground_truth.label.value,
        "Model A": _prediction_display(row.model_a_attempt),
        "A outcome": row.model_a_outcome.value,
        "Model B": _prediction_display(row.model_b_attempt),
        "B outcome": row.model_b_outcome.value,
        "Pair outcome": row.pair_outcome,
        "Source": row.ground_truth.ground_truth_source.value,
        "Stratum": row.ground_truth.sampling_stratum.value,
        "Role": row.ground_truth.evaluation_role.value,
    }


def _prediction_display(attempt: AttemptEvidence) -> str:
    return attempt.parsed_label.value if attempt.parsed_label else NO_VALID_PREDICTION


def _prediction_value(value: str) -> CustomerLabel | None:
    return None if value == NO_VALID_PREDICTION else CustomerLabel(value)


def _metric_display(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"

