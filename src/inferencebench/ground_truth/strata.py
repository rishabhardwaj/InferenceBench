"""Auditable targeted-review and closed-label evidence rules.

This module deliberately separates Human-Reviewed Ground Truth from the
weaker Closed-Issue Maintainer Evidence.  It contains no inference code: all
selection and audit decisions are derived from the frozen Corpus and recorded
human-review outcomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from inferencebench.artifacts import ArtifactIntegrityError
from inferencebench.domain import CorpusManifest, CustomerLabel, Issue, StrictModel
from inferencebench.ground_truth.annotations import HumanReviewAnnotation


class EvidenceStrataError(ArtifactIntegrityError):
    """Raised when targeted or closed-label evidence is not reproducible."""


class LabelMappingRule(StrictModel):
    source_label: str
    normalized_meaning: str
    disposition: Literal["direct", "administrative", "ignored"]
    customer_label: CustomerLabel | None = None

    def is_direct(self) -> bool:
        return self.disposition == "direct"


class ClosedLabelMappingManifest(StrictModel):
    schema_version: Literal["closed_label_mapping_manifest.v1"]
    mapping_version: str
    corpus_version: str
    corpus_sha256: str
    rules: tuple[LabelMappingRule, ...]
    excluded_state_reasons: tuple[Literal["not_planned"], ...] = ("not_planned",)


class DiagnosticSelection(StrictModel):
    """A pre-inference, human-curated Diagnostic Scored Supplement entry."""

    issue_number: int
    signals: tuple[
        Literal[
            "rare",
            "conflicting",
            "multi_label",
            "suspected_security",
            "other",
            "ambiguous",
            "difficult",
        ],
        ...,
    ]
    rationale: str


class DiagnosticSelectionManifest(StrictModel):
    schema_version: Literal["diagnostic_selection_manifest.v1"]
    corpus_version: str
    corpus_sha256: str
    selection_version: str
    selections: tuple[DiagnosticSelection, ...]


class ClosedIssueCandidate(StrictModel):
    issue_number: int
    eligible: bool
    mapped_label: CustomerLabel | None
    matched_direct_labels: tuple[str, ...]
    administrative_signals: tuple[str, ...]
    exclusion_reasons: tuple[str, ...]


class MappingAuditResult(StrictModel):
    source_label: str
    mapped_label: CustomerLabel
    audit_issue_numbers: tuple[int, ...]
    matching_issue_numbers: tuple[int, ...]
    disagreement_issue_numbers: tuple[int, ...]
    accepted: bool
    rejection_reason: str | None


class MappingAuditReport(StrictModel):
    mapping_version: str
    corpus_version: str
    corpus_sha256: str
    observed_labels: tuple[str, ...]
    administrative_labels: tuple[str, ...]
    candidate_count: int
    eligible_candidate_count: int
    audit_results: tuple[MappingAuditResult, ...]


def load_mapping_manifest(path: Path) -> ClosedLabelMappingManifest:
    manifest = ClosedLabelMappingManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if Path(manifest.mapping_version).name != manifest.mapping_version:
        raise EvidenceStrataError("Mapping version must be a directory-safe name")
    return manifest


def observed_label_names(issues: tuple[Issue, ...]) -> tuple[str, ...]:
    return tuple(sorted({label.name for issue in issues for label in issue.labels}))


def validate_mapping_manifest(
    manifest: ClosedLabelMappingManifest,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
) -> None:
    if manifest.corpus_version != corpus_manifest.corpus_version:
        raise EvidenceStrataError("Mapping references a different Corpus version")
    if manifest.corpus_sha256 != corpus_manifest.artifact_sha256:
        raise EvidenceStrataError("Mapping references a different Corpus hash")
    rules = {rule.source_label: rule for rule in manifest.rules}
    if len(rules) != len(manifest.rules):
        raise EvidenceStrataError("Each observed maintainer label needs one mapping rule")
    observed = observed_label_names(issues)
    if tuple(sorted(rules)) != observed:
        raise EvidenceStrataError("Mapping rules must account for exactly the frozen label vocabulary")
    for rule in manifest.rules:
        if rule.disposition == "direct":
            if rule.customer_label not in {
                CustomerLabel.BUG,
                CustomerLabel.ENHANCEMENT,
                CustomerLabel.QUESTION,
                CustomerLabel.DOCUMENTATION,
            }:
                raise EvidenceStrataError("Only four direct labels may be automatically mapped")
        elif rule.customer_label is not None:
            raise EvidenceStrataError("Only direct mapping rules may carry a customer label")


def validate_diagnostic_selection(
    selection: DiagnosticSelectionManifest,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    random_reviewed_issue_numbers: set[int],
) -> None:
    if selection.corpus_version != corpus_manifest.corpus_version:
        raise EvidenceStrataError("Diagnostic selection references a different Corpus version")
    if selection.corpus_sha256 != corpus_manifest.artifact_sha256:
        raise EvidenceStrataError("Diagnostic selection references a different Corpus hash")
    if not selection.selections or len(selection.selections) > 20:
        raise EvidenceStrataError("Diagnostic Scored Supplement must contain one to 20 issues")
    selected = [item.issue_number for item in selection.selections]
    corpus_numbers = {issue.issue_number for issue in issues}
    if len(selected) != len(set(selected)) or not set(selected) <= corpus_numbers:
        raise EvidenceStrataError("Diagnostic selections must be unique Corpus issues")
    if set(selected) & random_reviewed_issue_numbers:
        raise EvidenceStrataError("Diagnostic selections must be disjoint from the random sample")
    if any(not item.signals or not item.rationale.strip() for item in selection.selections):
        raise EvidenceStrataError("Each diagnostic selection needs pre-inference signals and a rationale")


def assess_closed_issue_candidate(
    issue: Issue, manifest: ClosedLabelMappingManifest
) -> ClosedIssueCandidate:
    rules = {rule.source_label: rule for rule in manifest.rules}
    administrative = tuple(
        sorted(
            rule.source_label
            for label in issue.labels
            if (rule := rules.get(label.name)) and rule.disposition == "administrative"
        )
    )
    direct = tuple(
        sorted(
            rule.source_label
            for label in issue.labels
            if (rule := rules.get(label.name)) and rule.is_direct()
        )
    )
    mapped = {rules[name].customer_label for name in direct}
    reasons: list[str] = []
    if issue.state != "closed":
        reasons.append("issue is not closed")
    if issue.state_reason in manifest.excluded_state_reasons:
        reasons.append(f"state_reason={issue.state_reason}")
    if administrative:
        reasons.append("Administrative Closure Signal: " + ", ".join(administrative))
    if not direct:
        reasons.append("no direct closed-label mapping")
    elif len(mapped) != 1:
        reasons.append("conflicting direct closed-label mappings")
    return ClosedIssueCandidate(
        issue_number=issue.issue_number,
        eligible=not reasons,
        mapped_label=next(iter(mapped)) if not reasons else None,
        matched_direct_labels=direct,
        administrative_signals=administrative,
        exclusion_reasons=tuple(reasons),
    )


def build_mapping_audit_report(
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    mapping: ClosedLabelMappingManifest,
    human_reviews: tuple[HumanReviewAnnotation, ...],
) -> MappingAuditReport:
    validate_mapping_manifest(mapping, corpus_manifest, issues)
    candidates = tuple(assess_closed_issue_candidate(issue, mapping) for issue in issues)
    candidate_by_number = {candidate.issue_number: candidate for candidate in candidates}
    accepted_reviews = {
        review.issue_number: review.final_label
        for review in human_reviews
        if review.review_status == "accepted" and review.final_label is not None
    }
    direct_rules = tuple(rule for rule in mapping.rules if rule.is_direct())
    reports: list[MappingAuditResult] = []
    for rule in direct_rules:
        audited = tuple(
            sorted(
                issue_number
                for issue_number, truth in accepted_reviews.items()
                if (candidate := candidate_by_number.get(issue_number))
                and candidate.eligible
                and rule.source_label in candidate.matched_direct_labels
                and candidate.mapped_label is rule.customer_label
            )
        )
        matching = tuple(number for number in audited if accepted_reviews[number] is rule.customer_label)
        disagreements = tuple(number for number in audited if accepted_reviews[number] is not rule.customer_label)
        accepted = len(audited) >= 5 and not disagreements
        reports.append(
            MappingAuditResult(
                source_label=rule.source_label,
                mapped_label=rule.customer_label,
                audit_issue_numbers=audited,
                matching_issue_numbers=matching,
                disagreement_issue_numbers=disagreements,
                accepted=accepted,
                rejection_reason=(
                    None
                    if accepted
                    else "human-review disagreement"
                    if disagreements
                    else f"insufficient audit support ({len(audited)}/5)"
                ),
            )
        )
    return MappingAuditReport(
        mapping_version=mapping.mapping_version,
        corpus_version=corpus_manifest.corpus_version,
        corpus_sha256=corpus_manifest.artifact_sha256,
        observed_labels=observed_label_names(issues),
        administrative_labels=tuple(
            sorted(rule.source_label for rule in mapping.rules if rule.disposition == "administrative")
        ),
        candidate_count=sum(issue.state == "closed" for issue in issues),
        eligible_candidate_count=sum(candidate.eligible for candidate in candidates),
        audit_results=tuple(reports),
    )
