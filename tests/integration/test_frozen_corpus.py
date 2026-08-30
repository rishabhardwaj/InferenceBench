from __future__ import annotations

from pathlib import Path

from inferencebench.artifacts import CorpusArtifacts


def test_bundled_doctl_corpus_matches_its_frozen_manifest() -> None:
    manifest, issues = CorpusArtifacts(Path("artifacts/corpus")).load_active()

    assert manifest.corpus_version == "doctl-2026-08-30"
    assert manifest.artifact_sha256 == (
        "2f1db01f91a3ccc21c2ac0c3b10dc9720dde450d4643e0a4de7ece80a7e3b711"
    )
    assert manifest.page_count == 20
    assert manifest.api_object_count == 1_922
    assert manifest.excluded_pull_request_count == 1_386
    assert manifest.issue_count == len(issues) == 536
    assert {issue.state for issue in issues} == {"open", "closed"}
    assert tuple(issue.issue_number for issue in issues) == tuple(
        sorted(issue.issue_number for issue in issues)
    )

