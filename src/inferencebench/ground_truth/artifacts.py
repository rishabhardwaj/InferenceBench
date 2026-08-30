from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inferencebench.artifacts import ArtifactIntegrityError, sha256_file
from inferencebench.domain import (
    CorpusManifest,
    GroundTruthAnnotation,
    GroundTruthManifest,
    Issue,
)


class EvaluationCorpusArtifactError(ArtifactIntegrityError):
    """Raised when a frozen Evaluation Corpus cannot support a live run."""


class EvaluationCorpusArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load_version(
        self,
        version: str,
        corpus_manifest: CorpusManifest,
        issues: tuple[Issue, ...],
    ) -> tuple[GroundTruthManifest, tuple[GroundTruthAnnotation, ...]]:
        if Path(version).name != version:
            raise EvaluationCorpusArtifactError(
                "Evaluation Corpus version must be a directory name"
            )
        directory = self.root / version
        manifest = GroundTruthManifest.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.ground_truth_version != version:
            raise EvaluationCorpusArtifactError(
                "Evaluation Corpus directory and manifest versions differ"
            )
        artifact_path = directory / manifest.artifact_file
        if sha256_file(artifact_path) != manifest.artifact_sha256:
            raise EvaluationCorpusArtifactError(
                "Evaluation Corpus artifact hash does not match its manifest"
            )
        annotations = _read_annotations(artifact_path)
        _validate_evaluation_corpus(
            corpus_manifest,
            issues,
            manifest,
            annotations,
        )
        return manifest, annotations


def _read_annotations(path: Path) -> tuple[GroundTruthAnnotation, ...]:
    rows: list[GroundTruthAnnotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(GroundTruthAnnotation.model_validate_json(line))
        except ValueError as error:
            raise EvaluationCorpusArtifactError(
                f"{path}:{line_number}: {error}"
            ) from error
    return tuple(rows)


def _validate_evaluation_corpus(
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    manifest: GroundTruthManifest,
    annotations: tuple[GroundTruthAnnotation, ...],
) -> None:
    if manifest.corpus_version != corpus_manifest.corpus_version:
        raise EvaluationCorpusArtifactError(
            "Evaluation Corpus references a different Corpus version"
        )
    issue_numbers = {issue.issue_number for issue in issues}
    if issue_numbers != set(corpus_manifest.ordered_issue_numbers):
        raise EvaluationCorpusArtifactError(
            "Loaded issues do not match the complete frozen Corpus"
        )
    annotation_numbers = tuple(annotation.issue_number for annotation in annotations)
    if len(annotations) != manifest.annotation_count:
        raise EvaluationCorpusArtifactError(
            "Evaluation Corpus annotation count does not match its manifest"
        )
    if annotation_numbers != manifest.ordered_issue_numbers:
        raise EvaluationCorpusArtifactError(
            "Evaluation Corpus annotation order does not match its manifest"
        )
    if not set(annotation_numbers) <= issue_numbers:
        raise EvaluationCorpusArtifactError(
            "Evaluation Corpus contains an issue outside the frozen Corpus"
        )
    for annotation in annotations:
        if annotation.corpus_version != corpus_manifest.corpus_version:
            raise EvaluationCorpusArtifactError(
                "Evaluation annotation references a different Corpus version"
            )
        expected_hash = hashlib.sha256(
            json.dumps(
                annotation.model_dump(
                    mode="json", exclude={"annotation_sha256"}
                ),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if annotation.annotation_sha256 != expected_hash:
            raise EvaluationCorpusArtifactError(
                f"Evaluation annotation hash mismatch: {annotation.annotation_id}"
            )
