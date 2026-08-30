from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from inferencebench.corpus.github import GitHubFetchResult
from inferencebench.domain import ActiveCorpusPointer, CorpusManifest, GitHubLabel, Issue


ARTIFACT_FILE = "issues.jsonl"
MANIFEST_FILE = "manifest.json"
ACTIVE_POINTER_FILE = "default.json"
SERIALIZATION = "utf-8-json-lines-sorted-keys-compact-lf"
PAGINATION = "github-link-header-until-exhaustion"
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CorpusSnapshotError(ValueError):
    """Raised when source records cannot form a valid immutable Corpus."""


class CorpusVersionExistsError(CorpusSnapshotError):
    """Raised rather than overwriting a previously frozen Corpus version."""


@dataclass(frozen=True, slots=True)
class CreatedCorpusSnapshot:
    directory: Path
    manifest: CorpusManifest
    issues: tuple[Issue, ...]


def freeze_corpus(
    fetch_result: GitHubFetchResult,
    *,
    corpus_root: Path,
    corpus_version: str,
    activate: bool = True,
) -> CreatedCorpusSnapshot:
    validate_corpus_version(corpus_version)
    issues, excluded_pull_requests = _build_issues(fetch_result, corpus_version)
    artifact_bytes = serialize_issues(issues)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    manifest = CorpusManifest(
        schema_version="corpus_manifest.v1",
        corpus_version=corpus_version,
        repository=fetch_result.repository,
        endpoint_url=fetch_result.endpoint_url,
        query_parameters=dict(fetch_result.query_parameters),
        github_api_version=fetch_result.github_api_version,
        retrieval_started_at=fetch_result.retrieval_started_at,
        retrieval_completed_at=fetch_result.retrieval_completed_at,
        page_count=fetch_result.page_count,
        api_object_count=len(fetch_result.api_objects),
        excluded_pull_request_count=excluded_pull_requests,
        artifact_file=ARTIFACT_FILE,
        artifact_sha256=artifact_sha256,
        artifact_byte_count=len(artifact_bytes),
        issue_count=len(issues),
        ordered_issue_numbers=tuple(issue.issue_number for issue in issues),
        ordering="issue_number_ascending",
        serialization=SERIALIZATION,
        pagination=PAGINATION,
    )

    corpus_root.mkdir(parents=True, exist_ok=True)
    target = corpus_root / corpus_version
    if target.exists():
        raise CorpusVersionExistsError(
            f"Corpus version already exists and is immutable: {target}"
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{corpus_version}-", dir=corpus_root))
    try:
        (temporary / ARTIFACT_FILE).write_bytes(artifact_bytes)
        (temporary / MANIFEST_FILE).write_bytes(_model_json_bytes(manifest))
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    created = CreatedCorpusSnapshot(directory=target, manifest=manifest, issues=issues)
    if activate:
        activate_corpus(corpus_root, manifest)
    return created


def activate_corpus(corpus_root: Path, manifest: CorpusManifest) -> None:
    pointer = ActiveCorpusPointer(
        schema_version="active_corpus.v1",
        corpus_version=manifest.corpus_version,
        manifest_file=f"{manifest.corpus_version}/{MANIFEST_FILE}",
        artifact_sha256=manifest.artifact_sha256,
    )
    temporary = corpus_root / f".{ACTIVE_POINTER_FILE}.tmp"
    temporary.write_bytes(_model_json_bytes(pointer))
    os.replace(temporary, corpus_root / ACTIVE_POINTER_FILE)


def serialize_issues(issues: tuple[Issue, ...]) -> bytes:
    return b"".join(
        _canonical_json_bytes(issue.model_dump(mode="json")) + b"\n"
        for issue in issues
    )


def issue_content_sha256(values: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(values)).hexdigest()


def _build_issues(
    fetch_result: GitHubFetchResult, corpus_version: str
) -> tuple[tuple[Issue, ...], int]:
    issues: list[Issue] = []
    excluded_pull_requests = 0
    for api_object in fetch_result.api_objects:
        if "pull_request" in api_object:
            excluded_pull_requests += 1
            continue
        values = _issue_values(api_object, fetch_result.repository, corpus_version)
        values["content_sha256"] = issue_content_sha256(values)
        try:
            issues.append(Issue.model_validate(values))
        except ValidationError as error:
            number = api_object.get("number", "unknown")
            raise CorpusSnapshotError(
                f"GitHub issue {number} does not match issue.v1: {error}"
            ) from error

    if not issues:
        raise CorpusSnapshotError("Corpus contains no issues after PR exclusion")
    issue_numbers = [issue.issue_number for issue in issues]
    duplicate_numbers = _duplicates(issue_numbers)
    if duplicate_numbers:
        raise CorpusSnapshotError(
            f"Duplicate GitHub issue numbers: {sorted(duplicate_numbers)}"
        )
    github_ids = [issue.github_issue_id for issue in issues]
    duplicate_ids = _duplicates(github_ids)
    if duplicate_ids:
        raise CorpusSnapshotError(f"Duplicate GitHub issue IDs: {sorted(duplicate_ids)}")
    return tuple(sorted(issues, key=lambda issue: issue.issue_number)), excluded_pull_requests


def _issue_values(
    api_object: dict[str, object], repository: str, corpus_version: str
) -> dict[str, object]:
    raw_labels = api_object.get("labels")
    if not isinstance(raw_labels, list):
        raise CorpusSnapshotError("GitHub issue labels must be an array")
    labels: list[dict[str, object]] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, dict):
            raise CorpusSnapshotError("GitHub issue labels must be objects")
        try:
            label = GitHubLabel(
                label_id=raw_label["id"],
                name=raw_label["name"],
                description=raw_label.get("description"),
                color=raw_label["color"],
            )
        except (KeyError, ValidationError) as error:
            raise CorpusSnapshotError(f"Invalid GitHub label: {raw_label}") from error
        labels.append(label.model_dump(mode="json"))

    try:
        return {
            "schema_version": "issue.v1",
            "corpus_version": corpus_version,
            "repository": repository,
            "github_issue_id": api_object["id"],
            "issue_number": api_object["number"],
            "node_id": api_object["node_id"],
            "api_url": api_object["url"],
            "html_url": api_object["html_url"],
            "title": api_object["title"],
            "body": api_object.get("body"),
            "state": api_object["state"],
            "state_reason": api_object.get("state_reason"),
            "labels": labels,
            "created_at": api_object["created_at"],
            "updated_at": api_object["updated_at"],
            "closed_at": api_object.get("closed_at"),
        }
    except KeyError as error:
        raise CorpusSnapshotError(f"GitHub issue is missing field: {error.args[0]}") from error


def _duplicates(values: list[int]) -> set[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return duplicates


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _model_json_bytes(model: CorpusManifest | ActiveCorpusPointer) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def validate_corpus_version(corpus_version: str) -> None:
    if not _VERSION_PATTERN.fullmatch(corpus_version):
        raise CorpusSnapshotError(
            "Corpus version must contain only letters, numbers, '.', '_' or '-'"
        )
