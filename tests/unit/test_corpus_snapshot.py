from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pytest

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.config import Settings
from inferencebench.corpus.github import (
    ENDPOINT_URL,
    GITHUB_API_VERSION,
    HttpResponse,
    fetch_doctl_issue_objects,
)
from inferencebench.corpus.snapshot import (
    CorpusSnapshotError,
    CorpusVersionExistsError,
    freeze_corpus,
)
from inferencebench.workflows.corpus import load_active_corpus_summary


STARTED_AT = datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc)
COMPLETED_AT = datetime(2026, 8, 30, 10, 0, 1, tzinfo=timezone.utc)
FIRST_URL = f"{ENDPOINT_URL}?state=all&per_page=100"
SECOND_URL = f"{ENDPOINT_URL}?state=all&per_page=100&page=2"


class FakeTransport:
    def __init__(self, responses: Mapping[str, HttpResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.calls.append((url, dict(headers)))
        return self.responses[url]


def test_pagination_exclusion_raw_text_and_manifest_provenance(tmp_path: Path) -> None:
    transport = _transport_for_standard_pages()
    fetched = fetch_doctl_issue_objects(
        transport=transport,
        token="secret-token-not-evidence",
        clock=_clock(),
    )
    created = freeze_corpus(
        fetched,
        corpus_root=tmp_path / "corpus",
        corpus_version="doctl-test-v1",
    )

    assert [url for url, _ in transport.calls] == [FIRST_URL, SECOND_URL]
    assert transport.calls[0][1]["X-GitHub-Api-Version"] == GITHUB_API_VERSION
    assert transport.calls[0][1]["Authorization"] == "Bearer secret-token-not-evidence"
    assert [issue.issue_number for issue in created.issues] == [1, 2, 3]
    assert created.issues[0].body is None
    assert created.issues[1].body == ""
    assert created.issues[1].title == "  Preserve title whitespace  "
    assert {issue.state for issue in created.issues} == {"open", "closed"}

    manifest = created.manifest
    assert manifest.query_parameters == {"state": "all", "per_page": "100"}
    assert manifest.page_count == 2
    assert manifest.api_object_count == 4
    assert manifest.excluded_pull_request_count == 1
    assert manifest.issue_count == 3
    assert manifest.artifact_byte_count == (
        created.directory / "issues.jsonl"
    ).stat().st_size
    assert "secret-token-not-evidence" not in (
        created.directory / "manifest.json"
    ).read_text()
    assert "secret-token-not-evidence" not in (
        created.directory / "issues.jsonl"
    ).read_text()

    loaded_manifest, loaded_issues = CorpusArtifacts(tmp_path / "corpus").load_active()
    assert loaded_manifest == manifest
    assert loaded_issues == created.issues
    summary = load_active_corpus_summary(
        Settings.for_test(tmp_path / "evidence.sqlite3", tmp_path / "corpus")
    )
    assert summary.github_requests_made == 0
    assert summary.corpus_sha256 == manifest.artifact_sha256


def test_identical_pages_create_identical_jsonl_hash_and_manifest(tmp_path: Path) -> None:
    first = _freeze_standard_snapshot(tmp_path / "first")
    second = _freeze_standard_snapshot(tmp_path / "second")

    assert (first.directory / "issues.jsonl").read_bytes() == (
        second.directory / "issues.jsonl"
    ).read_bytes()
    assert first.manifest.artifact_sha256 == second.manifest.artifact_sha256
    assert (first.directory / "manifest.json").read_bytes() == (
        second.directory / "manifest.json"
    ).read_bytes()


def test_duplicate_issue_number_is_rejected_before_writing(tmp_path: Path) -> None:
    duplicate = _issue_object(1, body="duplicate")
    transport = FakeTransport(
        {
            FIRST_URL: _response(
                [_issue_object(1, body=None), duplicate],
            )
        }
    )
    fetched = fetch_doctl_issue_objects(transport=transport, clock=_clock())

    with pytest.raises(CorpusSnapshotError, match="Duplicate GitHub issue numbers"):
        freeze_corpus(
            fetched,
            corpus_root=tmp_path / "corpus",
            corpus_version="doctl-duplicate",
        )
    assert not (tmp_path / "corpus" / "doctl-duplicate").exists()


def test_refresh_creates_new_version_without_altering_history(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    fetched_v1 = fetch_doctl_issue_objects(
        transport=_transport_for_standard_pages(), clock=_clock()
    )
    first = freeze_corpus(
        fetched_v1, corpus_root=corpus_root, corpus_version="doctl-v1"
    )
    historical_bytes = (first.directory / "issues.jsonl").read_bytes()
    historical_manifest = (first.directory / "manifest.json").read_bytes()

    fetched_v2 = fetch_doctl_issue_objects(
        transport=_transport_for_standard_pages(), clock=_clock()
    )
    second = freeze_corpus(
        fetched_v2, corpus_root=corpus_root, corpus_version="doctl-v2"
    )

    assert first.directory != second.directory
    assert (first.directory / "issues.jsonl").read_bytes() == historical_bytes
    assert (first.directory / "manifest.json").read_bytes() == historical_manifest
    assert CorpusArtifacts(corpus_root).load_active()[0].corpus_version == "doctl-v2"
    with pytest.raises(CorpusVersionExistsError, match="immutable"):
        freeze_corpus(
            fetched_v1, corpus_root=corpus_root, corpus_version="doctl-v1"
        )


def _freeze_standard_snapshot(root: Path):
    fetched = fetch_doctl_issue_objects(
        transport=_transport_for_standard_pages(), clock=_clock()
    )
    return freeze_corpus(
        fetched,
        corpus_root=root / "corpus",
        corpus_version="doctl-stable",
    )


def _transport_for_standard_pages() -> FakeTransport:
    first_page = [
        _issue_object(3, body="third", state="closed"),
        _issue_object(99, body="pull request", pull_request=True),
    ]
    second_page = [
        _issue_object(2, body="", title="  Preserve title whitespace  "),
        _issue_object(1, body=None),
    ]
    return FakeTransport(
        {
            FIRST_URL: _response(
                first_page,
                link=f'<{SECOND_URL}>; rel="next", <{SECOND_URL}>; rel="last"',
            ),
            SECOND_URL: _response(second_page),
        }
    )


def _issue_object(
    number: int,
    *,
    body: str | None,
    title: str | None = None,
    state: str = "open",
    pull_request: bool = False,
) -> dict[str, object]:
    item: dict[str, object] = {
        "id": 10_000 + number,
        "number": number,
        "node_id": f"I_{number}",
        "url": f"https://api.github.com/repos/digitalocean/doctl/issues/{number}",
        "html_url": f"https://github.com/digitalocean/doctl/issues/{number}",
        "title": title if title is not None else f"Issue {number}",
        "body": body,
        "state": state,
        "state_reason": "completed" if state == "closed" else None,
        "labels": [
            {
                "id": 50,
                "name": "kind/bug",
                "description": None,
                "color": "d73a4a",
            }
        ],
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-02T00:00:00Z",
        "closed_at": "2026-08-02T00:00:00Z" if state == "closed" else None,
    }
    if pull_request:
        item["pull_request"] = {
            "url": f"https://api.github.com/repos/digitalocean/doctl/pulls/{number}"
        }
    return item


def _response(
    page: list[dict[str, object]], link: str | None = None
) -> HttpResponse:
    headers = {"LiNk": link} if link else {}
    return HttpResponse(status_code=200, headers=headers, body=json.dumps(page).encode())


def _clock():
    timestamps = iter((STARTED_AT, COMPLETED_AT))
    return lambda: next(timestamps)

