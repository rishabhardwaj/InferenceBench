"""Freeze the Evaluation Corpus from accepted human-review evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from inferencebench.artifacts import ArtifactIntegrityError
from inferencebench.domain import (
    CorpusManifest,
    CustomerLabel,
    GroundTruthAnnotation,
    GroundTruthManifest,
    Issue,
    StrictModel,
)
from inferencebench.ground_truth.annotations import HumanReviewAnnotation


class EvaluationCorpusError(ArtifactIntegrityError):
    """Raised when accepted evidence cannot form a traceable Evaluation Corpus."""


class DiagnosticDecision(StrictModel):
    issue_number: int
    label: CustomerLabel
    confidence: str = "medium"
    review_notes: str


class DiagnosticDecisionManifest(StrictModel):
    schema_version: str
    decision_version: str
    corpus_version: str
    corpus_sha256: str
    source_packet: str
    decisions: tuple[DiagnosticDecision, ...]


def load_diagnostic_decisions(path: Path) -> DiagnosticDecisionManifest:
    return DiagnosticDecisionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_evaluation_corpus(
    *,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    random_reviews: tuple[HumanReviewAnnotation, ...],
    diagnostics: DiagnosticDecisionManifest,
    ground_truth_version: str,
) -> tuple[GroundTruthManifest, tuple[GroundTruthAnnotation, ...]]:
    if diagnostics.corpus_version != corpus_manifest.corpus_version:
        raise EvaluationCorpusError("Diagnostic decisions reference a different Corpus version")
    if diagnostics.corpus_sha256 != corpus_manifest.artifact_sha256:
        raise EvaluationCorpusError("Diagnostic decisions reference a different Corpus hash")
    if len(diagnostics.decisions) > 20:
        raise EvaluationCorpusError("Diagnostic Scored Supplement is capped at 20 decisions")
    diagnostic_numbers = tuple(decision.issue_number for decision in diagnostics.decisions)
    if len(set(diagnostic_numbers)) != len(diagnostic_numbers):
        raise EvaluationCorpusError("Diagnostic decisions must be unique")
    corpus_numbers = {issue.issue_number for issue in issues}
    if not set(diagnostic_numbers) <= corpus_numbers:
        raise EvaluationCorpusError("Diagnostic decision references an issue outside the Corpus")
    accepted_random = tuple(review for review in random_reviews if review.review_status == "accepted")
    random_numbers = {review.issue_number for review in accepted_random}
    if set(diagnostic_numbers) & random_numbers:
        raise EvaluationCorpusError("Diagnostic decisions overlap the Random Human-Reviewed Sample")
    if len(accepted_random) != 100:
        raise EvaluationCorpusError("Evaluation Corpus requires 100 accepted random reviews")

    rows: list[GroundTruthAnnotation] = []
    for review in accepted_random:
        assert review.final_label is not None and review.confidence is not None
        annotation = GroundTruthAnnotation(
            schema_version="ground_truth_annotation.v1",
            annotation_id=f"{ground_truth_version}-{review.issue_number}",
            corpus_version=corpus_manifest.corpus_version,
            issue_number=review.issue_number,
            label=review.final_label,
            ground_truth_source="human_review",
            sampling_stratum="random",
            evaluation_role=review.evaluation_role,
            rubric_version=review.rubric_version,
            confidence=review.confidence,
            review_status="accepted",
            review_pass_count=review.review_pass_count,
            input_sufficiency="sufficient",
            reviewed_at=review.reviewed_at,
            annotation_sha256="0" * 64,
        )
        rows.append(annotation.model_copy(update={"annotation_sha256": _annotation_hash(annotation)}))

    reviewed_at = datetime.now(UTC)
    rubric_version = accepted_random[0].rubric_version
    for decision in diagnostics.decisions:
        annotation = GroundTruthAnnotation(
            schema_version="ground_truth_annotation.v1",
            annotation_id=f"{ground_truth_version}-{decision.issue_number}",
            corpus_version=corpus_manifest.corpus_version,
            issue_number=decision.issue_number,
            label=decision.label,
            ground_truth_source="human_review",
            sampling_stratum="diagnostic",
            evaluation_role="diagnostic",
            rubric_version=rubric_version,
            confidence=decision.confidence,
            review_status="accepted",
            review_pass_count=1,
            input_sufficiency="sufficient",
            reviewed_at=reviewed_at,
            annotation_sha256="0" * 64,
        )
        rows.append(annotation.model_copy(update={"annotation_sha256": _annotation_hash(annotation)}))

    content = _jsonl(rows)
    annotations = tuple(rows)
    manifest = GroundTruthManifest(
        schema_version="ground_truth_manifest.v1",
        ground_truth_version=ground_truth_version,
        corpus_version=corpus_manifest.corpus_version,
        artifact_file="annotations.jsonl",
        artifact_sha256=hashlib.sha256(content).hexdigest(),
        annotation_count=len(annotations),
        ordered_issue_numbers=tuple(annotation.issue_number for annotation in annotations),
    )
    return manifest, annotations


def write_evaluation_corpus(
    root: Path,
    manifest: GroundTruthManifest,
    annotations: tuple[GroundTruthAnnotation, ...],
) -> Path:
    directory = root / manifest.ground_truth_version
    if directory.exists():
        raise EvaluationCorpusError(f"Evaluation Corpus version is immutable: {directory}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=root))
    try:
        (temporary / manifest.artifact_file).write_bytes(_jsonl(annotations))
        (temporary / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, directory)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return directory


def _annotation_hash(annotation: GroundTruthAnnotation) -> str:
    value = annotation.model_dump(mode="json", exclude={"annotation_sha256"})
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _jsonl(rows: list[GroundTruthAnnotation] | tuple[GroundTruthAnnotation, ...]) -> bytes:
    return b"".join(
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )
