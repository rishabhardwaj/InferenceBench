from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.domain import CustomerLabel, ParseStatus
from inferencebench.inference.domain import ProviderOutcome
from inferencebench.models.domain import APPROVED_ELIGIBLE_MODEL_IDS
from inferencebench.workflows.classification import classify_one_frozen_issue


API_KEY = "doo_v1_super-secret-classification-key"


def test_one_frozen_issue_makes_one_exact_request_and_preserves_evidence() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            headers={
                "x-request-id": "request-safe-123",
                "ratelimit-remaining": "42",
                "authorization": f"Bearer {API_KEY}",
            },
            json={
                "id": "response-id-fallback",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "'Bug'."},
                    }
                ],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": 2,
                    "total_tokens": 323,
                    "prompt_tokens_details": {"cached_tokens": 120},
                },
                "debug": {"authorization": f"Bearer {API_KEY}"},
            },
        )

    result = _classify(handler, timeout_seconds=2.5)

    assert len(seen_requests) == 1
    sent = seen_requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://inference.do-ai.run/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    payload = json.loads(sent.content)
    assert payload == {
        "model": APPROVED_ELIGIBLE_MODEL_IDS[0],
        "messages": [
            {
                "role": "system",
                "content": result.request_messages[0].content,
            },
            {
                "role": "user",
                "content": result.request_messages[1].content,
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "max_completion_tokens": 256,
    }
    assert sent.extensions["timeout"] == {
        "connect": 2.5,
        "read": 2.5,
        "write": 2.5,
        "pool": 2.5,
    }
    assert "tools" not in payload
    assert "tool_choice" not in payload

    assert result.provider_outcome is ProviderOutcome.SUCCESS
    assert result.http_status == 200
    assert result.provider_request_id == "request-safe-123"
    assert result.response_headers == {
        "content-type": "application/json",
        "ratelimit-remaining": "42",
        "x-request-id": "request-safe-123",
    }
    assert result.raw_model_output == "'Bug'."
    assert result.parse_result.parse_status is ParseStatus.NORMALIZED
    assert result.parse_result.parsed_label is CustomerLabel.BUG
    assert result.finish_reason == "stop"
    assert result.usage == {
        "prompt_tokens": 321,
        "completion_tokens": 2,
        "total_tokens": 323,
        "prompt_tokens_details": {"cached_tokens": 120},
    }
    assert result.effective_settings.stream is False
    assert result.effective_settings.tools_enabled is False
    assert API_KEY not in result.model_dump_json()
    assert API_KEY not in repr(result)
    assert "authorization" not in result.response_headers
    assert result.raw_response["debug"]["authorization"] == "[REDACTED]"


def test_invalid_model_output_is_preserved_without_repair_call() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _completion_response("The label is bug")

    result = _classify(handler)

    assert call_count == 1
    assert result.provider_outcome is ProviderOutcome.SUCCESS
    assert result.raw_model_output == "The label is bug"
    assert result.parse_result.parse_status is ParseStatus.INVALID
    assert result.parse_result.parsed_label is None


def test_shared_timeout_is_a_total_deadline_and_is_not_retried() -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return _completion_response("bug")

    result = _classify(handler, timeout_seconds=0.001)

    assert call_count == 1
    assert result.provider_outcome is ProviderOutcome.TIMEOUT
    assert result.http_status is None
    assert result.parse_result.parse_status is ParseStatus.INVALID


@pytest.mark.parametrize(
    ("response_body", "expected_message"),
    [
        ({"choices": [], "usage": {}}, "exactly one choice"),
        (
            {"choices": [{"message": {"content": "bug"}}]},
            "usage must be an object",
        ),
        (
            {"choices": [{"message": {"content": 7}}], "usage": {}},
            "string message content",
        ),
    ],
)
def test_malformed_success_envelope_is_typed_and_raw_json_is_preserved(
    response_body: dict[str, Any], expected_message: str
) -> None:
    result = _classify(lambda request: httpx.Response(200, json=response_body))

    assert result.provider_outcome is ProviderOutcome.PROTOCOL_ERROR
    assert result.raw_response == response_body
    assert expected_message in result.raw_error["message"]
    assert result.parse_result.parse_status is ParseStatus.INVALID


def test_non_json_success_body_is_a_protocol_error() -> None:
    result = _classify(
        lambda request: httpx.Response(
            200,
            text=f"upstream accidentally echoed Bearer {API_KEY}",
        )
    )

    assert result.provider_outcome is ProviderOutcome.PROTOCOL_ERROR
    assert result.raw_response is None
    assert API_KEY not in result.model_dump_json()
    assert result.raw_error["body"] == "upstream accidentally echoed Bearer [REDACTED]"


@pytest.mark.parametrize(
    ("status_code", "expected_outcome"),
    [
        (400, ProviderOutcome.INVALID_REQUEST),
        (401, ProviderOutcome.AUTHENTICATION),
        (403, ProviderOutcome.AUTHENTICATION),
        (408, ProviderOutcome.TIMEOUT),
        (429, ProviderOutcome.RATE_LIMIT),
        (500, ProviderOutcome.SERVER_ERROR),
        (503, ProviderOutcome.SERVER_ERROR),
        (302, ProviderOutcome.UNKNOWN),
    ],
)
def test_http_failures_are_typed_and_sanitized(
    status_code: int, expected_outcome: ProviderOutcome
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={
                "retry-after": "3",
                "x-request-id": API_KEY,
                "authorization": f"Bearer {API_KEY}",
            },
            json={
                "error": f"credential Bearer {API_KEY} was rejected",
                "token": API_KEY,
            },
        )

    result = _classify(handler)

    assert result.provider_outcome is expected_outcome
    assert result.provider_request_id is None
    assert result.response_headers["retry-after"] == "3"
    assert API_KEY not in result.model_dump_json()
    assert result.parse_result.parse_status is ParseStatus.INVALID


@pytest.mark.parametrize(
    ("error_factory", "expected_outcome"),
    [
        (
            lambda request: httpx.ReadTimeout(
                f"timeout exposed {API_KEY}", request=request
            ),
            ProviderOutcome.TIMEOUT,
        ),
        (
            lambda request: httpx.ConnectError(
                f"network exposed {API_KEY}", request=request
            ),
            ProviderOutcome.NETWORK_ERROR,
        ),
        (
            lambda request: RuntimeError(f"unexpected exposed {API_KEY}"),
            ProviderOutcome.UNKNOWN,
        ),
    ],
)
def test_transport_failures_are_terminal_typed_and_secret_safe(
    error_factory: Callable[[httpx.Request], Exception],
    expected_outcome: ProviderOutcome,
) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise error_factory(request)

    result = _classify(handler)

    assert call_count == 1
    assert result.provider_outcome is expected_outcome
    assert result.http_status is None
    assert result.raw_response is None
    assert API_KEY not in result.model_dump_json()
    assert result.parse_result.parse_status is ParseStatus.INVALID


def _classify(
    handler: Callable[
        [httpx.Request], httpx.Response | Awaitable[httpx.Response]
    ],
    *,
    timeout_seconds: float = 30.0,
):
    async def execute():
        _, issues = CorpusArtifacts(Path("artifacts/corpus")).load_active()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await classify_one_frozen_issue(
                issues[0],
                model_id=APPROVED_ELIGIBLE_MODEL_IDS[0],
                api_key=API_KEY,
                timeout_seconds=timeout_seconds,
                client=client,
            )

    return asyncio.run(execute())


def _completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 3,
                "total_tokens": 103,
            },
        },
    )
