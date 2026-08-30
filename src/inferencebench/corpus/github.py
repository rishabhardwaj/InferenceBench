from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlencode, urlparse


GITHUB_API_VERSION = "2026-03-10"
REPOSITORY = "digitalocean/doctl"
ENDPOINT_URL = f"https://api.github.com/repos/{REPOSITORY}/issues"
QUERY_PARAMETERS = {"state": "all", "per_page": "100"}
_LINK_PATTERN = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')


class GitHubFetchError(RuntimeError):
    """Raised when the GitHub issue population cannot be fetched safely."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


class PageTransport(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse: ...


class UrllibPageTransport:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                return HttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise GitHubFetchError(
                f"GitHub returned HTTP {error.code}: {detail[:500]}"
            ) from error
        except urllib.error.URLError as error:
            raise GitHubFetchError(f"GitHub request failed: {error.reason}") from error


@dataclass(frozen=True, slots=True)
class GitHubFetchResult:
    repository: str
    endpoint_url: str
    query_parameters: Mapping[str, str]
    github_api_version: str
    retrieval_started_at: datetime
    retrieval_completed_at: datetime
    page_count: int
    api_objects: tuple[dict[str, object], ...]


def fetch_doctl_issue_objects(
    *,
    transport: PageTransport | None = None,
    token: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> GitHubFetchResult:
    """Follow GitHub Link headers until every issue/PR object has been fetched."""

    resolved_transport = transport or UrllibPageTransport()
    resolved_clock = clock or (lambda: datetime.now(timezone.utc))
    started_at = _require_aware_utc(resolved_clock())
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "InferenceBench/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    next_url: str | None = f"{ENDPOINT_URL}?{urlencode(QUERY_PARAMETERS)}"
    visited_urls: set[str] = set()
    api_objects: list[dict[str, object]] = []
    page_count = 0

    while next_url is not None:
        _validate_github_url(next_url)
        if next_url in visited_urls:
            raise GitHubFetchError(f"GitHub pagination repeated URL: {next_url}")
        visited_urls.add(next_url)

        response = resolved_transport.get(next_url, headers)
        if response.status_code != 200:
            raise GitHubFetchError(f"GitHub returned HTTP {response.status_code}")
        try:
            page = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitHubFetchError("GitHub returned invalid JSON") from error
        if not isinstance(page, list):
            raise GitHubFetchError("GitHub issues response must be a JSON array")
        for item in page:
            if not isinstance(item, dict):
                raise GitHubFetchError("GitHub issue entries must be JSON objects")
            api_objects.append(item)
        page_count += 1
        next_url = _next_link(_header(response.headers, "link"))

    completed_at = _require_aware_utc(resolved_clock())
    if completed_at < started_at:
        raise GitHubFetchError("Retrieval clock moved backwards")
    if not api_objects:
        raise GitHubFetchError("GitHub returned an empty issue population")

    return GitHubFetchResult(
        repository=REPOSITORY,
        endpoint_url=ENDPOINT_URL,
        query_parameters=QUERY_PARAMETERS,
        github_api_version=GITHUB_API_VERSION,
        retrieval_started_at=started_at,
        retrieval_completed_at=completed_at,
        page_count=page_count,
        api_objects=tuple(api_objects),
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered_name = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == lowered_name), None
    )


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for url, relations in _LINK_PATTERN.findall(link_header):
        if "next" in relations.split():
            return url
    return None


def _validate_github_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com":
        raise GitHubFetchError(f"Refusing non-GitHub pagination URL: {url}")


def _require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise GitHubFetchError("Retrieval timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)

