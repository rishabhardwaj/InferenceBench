from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from inferencebench.domain import (
    ActiveCorpusPointer,
    CorpusManifest,
    FixtureRunBundle,
    GroundTruthAnnotation,
    GroundTruthManifest,
    Issue,
)


class ArtifactIntegrityError(ValueError):
    """Raised when a frozen artifact does not match its manifest."""


ModelT = TypeVar("ModelT", bound=BaseModel)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except ValueError as error:
            raise ArtifactIntegrityError(f"{path}:{line_number}: {error}") from error
    return tuple(records)


class FixtureArtifacts:
    FIXTURE_VERSION = "v1"
    RUN_FIXTURE_VERSION = "v2"

    def __init__(self, fixture_root: Path) -> None:
        self.fixture_root = fixture_root

    def load_corpus(self) -> tuple[CorpusManifest, tuple[Issue, ...]]:
        directory = self.fixture_root / "corpus" / self.FIXTURE_VERSION
        return _load_corpus_directory(directory, enforce_directory_version=False)

    def load_ground_truth(
        self,
    ) -> tuple[GroundTruthManifest, tuple[GroundTruthAnnotation, ...]]:
        directory = self.fixture_root / "ground_truth" / self.FIXTURE_VERSION
        manifest = _read_json(directory / "manifest.json", GroundTruthManifest)
        artifact_path = directory / manifest.artifact_file
        self._verify_hash(artifact_path, manifest.artifact_sha256)
        annotations = _read_jsonl(artifact_path, GroundTruthAnnotation)
        issue_numbers = tuple(annotation.issue_number for annotation in annotations)
        if len(annotations) != manifest.annotation_count:
            raise ArtifactIntegrityError("Ground Truth count does not match manifest")
        if issue_numbers != manifest.ordered_issue_numbers:
            raise ArtifactIntegrityError("Ground Truth order does not match manifest")
        if any(
            annotation.corpus_version != manifest.corpus_version
            for annotation in annotations
        ):
            raise ArtifactIntegrityError("Ground Truth references a different Corpus version")
        return manifest, annotations

    def load_run_bundle(self) -> FixtureRunBundle:
        path = (
            self.fixture_root
            / "runs"
            / self.RUN_FIXTURE_VERSION
            / "evidence.json"
        )
        return _read_json(path, FixtureRunBundle)

    @staticmethod
    def _verify_hash(path: Path, expected: str) -> None:
        actual = sha256_file(path)
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Artifact hash mismatch for {path}: expected {expected}, got {actual}"
            )


class CorpusArtifacts:
    """Read frozen Corpus versions without contacting GitHub."""

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    def load_active(self) -> tuple[CorpusManifest, tuple[Issue, ...]]:
        pointer = _read_json(
            self.corpus_root / "default.json", ActiveCorpusPointer
        )
        manifest, issues = self.load_version(pointer.corpus_version)
        if manifest.artifact_sha256 != pointer.artifact_sha256:
            raise ArtifactIntegrityError(
                "Active Corpus pointer hash does not match its manifest"
            )
        return manifest, issues

    def load_version(
        self, corpus_version: str
    ) -> tuple[CorpusManifest, tuple[Issue, ...]]:
        if Path(corpus_version).name != corpus_version:
            raise ArtifactIntegrityError("Corpus version must be a directory name")
        directory = self.corpus_root / corpus_version
        return _load_corpus_directory(directory)


def _load_corpus_directory(
    directory: Path,
    *,
    enforce_directory_version: bool = True,
) -> tuple[CorpusManifest, tuple[Issue, ...]]:
    manifest = _read_json(directory / "manifest.json", CorpusManifest)
    if enforce_directory_version and manifest.corpus_version != directory.name:
        raise ArtifactIntegrityError(
            "Corpus directory name does not match manifest version"
        )
    artifact_path = directory / manifest.artifact_file
    FixtureArtifacts._verify_hash(artifact_path, manifest.artifact_sha256)
    if artifact_path.stat().st_size != manifest.artifact_byte_count:
        raise ArtifactIntegrityError("Corpus byte count does not match manifest")
    issues = _read_jsonl(artifact_path, Issue)
    issue_numbers = tuple(issue.issue_number for issue in issues)
    if len(issues) != manifest.issue_count:
        raise ArtifactIntegrityError("Corpus record count does not match manifest")
    if issue_numbers != manifest.ordered_issue_numbers:
        raise ArtifactIntegrityError("Corpus order does not match manifest")
    if any(issue.corpus_version != manifest.corpus_version for issue in issues):
        raise ArtifactIntegrityError("Issue references a different Corpus version")
    if any(issue.repository != manifest.repository for issue in issues):
        raise ArtifactIntegrityError("Issue references a different repository")
    return manifest, issues


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
