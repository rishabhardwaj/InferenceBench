from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.ground_truth.annotations import HumanReviewArtifacts
from inferencebench.ground_truth.strata import (
    DiagnosticSelection,
    DiagnosticSelectionManifest,
    EvidenceStrataError,
    assess_closed_issue_candidate,
    build_mapping_audit_report,
    load_mapping_manifest,
    validate_diagnostic_selection,
    validate_mapping_manifest,
)


def _corpus_and_mapping():
    manifest, issues = CorpusArtifacts(Path("artifacts/corpus")).load_active()
    mapping = load_mapping_manifest(
        Path("artifacts/ground_truth/mappings/doctl-closed-labels-v1.json")
    )
    validate_mapping_manifest(mapping, manifest, issues)
    return manifest, issues, mapping


def test_mapping_accounts_for_exact_frozen_label_vocabulary() -> None:
    _, _, mapping = _corpus_and_mapping()

    administrative = {
        rule.source_label for rule in mapping.rules if rule.disposition == "administrative"
    }
    direct_labels = {
        rule.customer_label for rule in mapping.rules if rule.disposition == "direct"
    }

    assert administrative == {"duplicate", "waiting-response"}
    assert direct_labels == {"bug", "enhancement", "question", "documentation"}
    assert next(rule for rule in mapping.rules if rule.source_label == "security vulnerability").disposition == "ignored"


def test_closed_candidate_rejects_administrative_signals_and_not_planned() -> None:
    _, issues, mapping = _corpus_and_mapping()
    duplicate_issue = next(issue for issue in issues if "duplicate" in {label.name for label in issue.labels})
    not_planned_issue = next(issue for issue in issues if issue.state_reason == "not_planned")

    duplicate = assess_closed_issue_candidate(duplicate_issue, mapping)
    not_planned = assess_closed_issue_candidate(not_planned_issue, mapping)

    assert not duplicate.eligible
    assert duplicate.administrative_signals == ("duplicate",)
    assert not not_planned.eligible
    assert "state_reason=not_planned" in not_planned.exclusion_reasons


def test_diagnostic_selection_is_bounded_and_disjoint_from_random_reviews() -> None:
    manifest, issues, _ = _corpus_and_mapping()
    reviewed = {181}
    selection = DiagnosticSelectionManifest(
        schema_version="diagnostic_selection_manifest.v1",
        corpus_version=manifest.corpus_version,
        corpus_sha256=manifest.artifact_sha256,
        selection_version="doctl-diagnostic-selection-v1",
        selections=(
            DiagnosticSelection(issue_number=569, signals=("ambiguous",), rationale="Boundary case"),
        ),
    )

    validate_diagnostic_selection(selection, manifest, issues, reviewed)
    with pytest.raises(EvidenceStrataError, match="disjoint"):
        validate_diagnostic_selection(
            selection.model_copy(
                update={"selections": (selection.selections[0].model_copy(update={"issue_number": 181}),)}
            ),
            manifest,
            issues,
            reviewed,
        )


def test_mapping_audit_rejects_rules_without_five_matching_human_reviews() -> None:
    manifest, issues, mapping = _corpus_and_mapping()
    _, reviews = HumanReviewArtifacts(Path("artifacts/ground_truth")).load_version(
        "doctl-human-review-v1", manifest, issues
    )

    report = build_mapping_audit_report(manifest, issues, mapping, reviews)

    assert report.candidate_count == sum(issue.state == "closed" for issue in issues)
    assert all(not result.accepted for result in report.audit_results)
    assert all(result.rejection_reason for result in report.audit_results)
