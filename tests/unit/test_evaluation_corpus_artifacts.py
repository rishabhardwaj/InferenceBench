from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.ground_truth.artifacts import (
    EvaluationCorpusArtifactError,
    EvaluationCorpusArtifacts,
)


def test_evaluation_corpus_loader_validates_the_frozen_artifact(
    tmp_path: Path,
) -> None:
    corpus_root, ground_truth_root, version = _copy_fixture_artifacts(tmp_path)
    corpus_manifest, issues = CorpusArtifacts(corpus_root).load_active()

    manifest, annotations = EvaluationCorpusArtifacts(
        ground_truth_root
    ).load_version(version, corpus_manifest, issues)

    assert manifest.ground_truth_version == version
    assert manifest.annotation_count == 1
    assert annotations[0].issue_number in corpus_manifest.ordered_issue_numbers


def test_evaluation_corpus_loader_rejects_changed_annotation_evidence(
    tmp_path: Path,
) -> None:
    corpus_root, ground_truth_root, version = _copy_fixture_artifacts(tmp_path)
    corpus_manifest, issues = CorpusArtifacts(corpus_root).load_active()
    artifact_path = ground_truth_root / version / "annotations.jsonl"
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8").replace('"bug"', '"other"'),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationCorpusArtifactError, match="artifact hash"):
        EvaluationCorpusArtifacts(ground_truth_root).load_version(
            version,
            corpus_manifest,
            issues,
        )


def _copy_fixture_artifacts(tmp_path: Path) -> tuple[Path, Path, str]:
    corpus_root = tmp_path / "corpus"
    corpus_version = "fixture-corpus-v1"
    corpus_directory = corpus_root / corpus_version
    shutil.copytree(Path("artifacts/fixtures/corpus/v1"), corpus_directory)
    corpus_manifest = json.loads(
        (corpus_directory / "manifest.json").read_text(encoding="utf-8")
    )
    (corpus_root / "default.json").write_text(
        json.dumps(
            {
                "schema_version": "active_corpus.v1",
                "corpus_version": corpus_version,
                "manifest_file": f"{corpus_version}/manifest.json",
                "artifact_sha256": corpus_manifest["artifact_sha256"],
            }
        ),
        encoding="utf-8",
    )

    ground_truth_root = tmp_path / "ground-truth"
    evaluation_version = "fixture-ground-truth-v1"
    shutil.copytree(
        Path("artifacts/fixtures/ground_truth/v1"),
        ground_truth_root / evaluation_version,
    )
    return corpus_root, ground_truth_root, evaluation_version
