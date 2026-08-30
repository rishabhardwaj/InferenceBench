from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from inferencebench.artifacts import ArtifactIntegrityError, FixtureArtifacts


def test_fixture_artifacts_load_full_population_with_one_scored_issue() -> None:
    artifacts = FixtureArtifacts(Path("artifacts/fixtures"))

    corpus_manifest, issues = artifacts.load_corpus()
    ground_truth_manifest, annotations = artifacts.load_ground_truth()

    assert corpus_manifest.issue_count == 7
    assert ground_truth_manifest.annotation_count == 1
    assert issues[0].issue_number == annotations[0].issue_number == 1


def test_corpus_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(Path("artifacts/fixtures"), fixture_root)
    issue_path = (
        fixture_root
        / "corpus"
        / FixtureArtifacts.FIXTURE_VERSION
        / "issues.jsonl"
    )
    issue_path.write_text(issue_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Artifact hash mismatch"):
        FixtureArtifacts(fixture_root).load_corpus()
