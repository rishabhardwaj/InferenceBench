from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.config import Settings
from inferencebench.corpus.github import PageTransport, fetch_doctl_issue_objects
from inferencebench.corpus.snapshot import (
    CorpusVersionExistsError,
    CreatedCorpusSnapshot,
    freeze_corpus,
    validate_corpus_version,
)
from inferencebench.domain import StrictModel


class CorpusSnapshotSummary(StrictModel):
    evidence_state: str
    github_requests_made: int
    corpus_version: str
    corpus_sha256: str
    repository: str
    issue_count: int
    excluded_pull_request_count: int
    api_object_count: int
    page_count: int
    retrieval_started_at: datetime
    retrieval_completed_at: datetime


def refresh_doctl_corpus(
    settings: Settings,
    *,
    corpus_version: str,
    token: str | None = None,
    transport: PageTransport | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CreatedCorpusSnapshot:
    validate_corpus_version(corpus_version)
    target = settings.corpus_root / corpus_version
    if target.exists():
        raise CorpusVersionExistsError(
            f"Corpus version already exists and is immutable: {target}"
        )
    fetched = fetch_doctl_issue_objects(
        transport=transport,
        token=token,
        clock=clock,
    )
    return freeze_corpus(
        fetched,
        corpus_root=settings.corpus_root,
        corpus_version=corpus_version,
        activate=True,
    )


def load_active_corpus_summary(settings: Settings) -> CorpusSnapshotSummary:
    manifest, _ = CorpusArtifacts(settings.corpus_root).load_active()
    return CorpusSnapshotSummary(
        evidence_state="Frozen Corpus — no GitHub request",
        github_requests_made=0,
        corpus_version=manifest.corpus_version,
        corpus_sha256=manifest.artifact_sha256,
        repository=manifest.repository,
        issue_count=manifest.issue_count,
        excluded_pull_request_count=manifest.excluded_pull_request_count,
        api_object_count=manifest.api_object_count,
        page_count=manifest.page_count,
        retrieval_started_at=manifest.retrieval_started_at,
        retrieval_completed_at=manifest.retrieval_completed_at,
    )
