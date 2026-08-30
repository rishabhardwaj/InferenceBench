from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import JsonValue

from inferencebench.inference.domain import (
    OutputParseResult,
    PreparedClassificationRequest,
    ProviderOutcome,
    SingleIssueClassificationResult,
)
from inferencebench.inference.parser import invalid_parse_result, parse_customer_label


DIGITALOCEAN_CHAT_COMPLETIONS_URL = (
    "https://inference.do-ai.run/v1/chat/completions"
)
REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api_key",
        "access_key",
        "access_token",
        "token",
    }
)
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "date",
        "ratelimit-limit",
        "ratelimit-remaining",
        "ratelimit-reset",
        "retry-after",
        "request-id",
        "x-request-id",
        "x-do-request-id",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;\"']+")
_DIGITALOCEAN_TOKEN_PATTERN = re.compile(r"\b(?:dop|doo|dor)_v1_[A-Za-z0-9_-]+\b")
_SAFE_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


class DigitalOceanChatCompletionsAdapter:
    """One-attempt async boundary for DigitalOcean Chat Completions."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str = DIGITALOCEAN_CHAT_COMPLETIONS_URL,
    ) -> None:
        self._client = client
        self._endpoint_url = endpoint_url

    async def classify(
        self,
        request: PreparedClassificationRequest,
        *,
        api_key: str,
        timeout_seconds: float,
    ) -> SingleIssueClassificationResult:
        if not api_key:
            raise ValueError("DigitalOcean API key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive")

        started_at = datetime.now(UTC)
        started_monotonic = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await self._client.post(
                    self._endpoint_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request.api_payload(),
                    timeout=httpx.Timeout(timeout_seconds),
                )
        except (TimeoutError, httpx.TimeoutException) as error:
            return _transport_failure_result(
                request,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                started_monotonic=started_monotonic,
                outcome=ProviderOutcome.TIMEOUT,
                error=error,
            )
        except httpx.NetworkError as error:
            return _transport_failure_result(
                request,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                started_monotonic=started_monotonic,
                outcome=ProviderOutcome.NETWORK_ERROR,
                error=error,
            )
        except httpx.RequestError as error:
            return _transport_failure_result(
                request,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                started_monotonic=started_monotonic,
                outcome=ProviderOutcome.NETWORK_ERROR,
                error=error,
            )
        except Exception as error:  # Defensive provider boundary; cancellation is not caught.
            return _transport_failure_result(
                request,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                started_monotonic=started_monotonic,
                outcome=ProviderOutcome.UNKNOWN,
                error=error,
            )

        decoded_body, body_is_json = _decode_response_body(response, api_key)
        ended_at = datetime.now(UTC)
        latency_ms = (time.perf_counter() - started_monotonic) * 1000
        response_headers = _safe_headers(response.headers, api_key)
        provider_request_id = _provider_request_id(
            response.headers, decoded_body if body_is_json else None, api_key
        )

        if not 200 <= response.status_code < 300:
            outcome = _http_outcome(response.status_code)
            return _result(
                request,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                provider_outcome=outcome,
                http_status=response.status_code,
                provider_request_id=provider_request_id,
                response_headers=response_headers,
                raw_response=None,
                raw_error={
                    "type": outcome.value,
                    "message": f"DigitalOcean returned HTTP {response.status_code}",
                    "body": decoded_body,
                },
                usage=_usage_from_body(decoded_body),
            )

        if not body_is_json:
            return _protocol_error_result(
                request,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                http_status=response.status_code,
                provider_request_id=provider_request_id,
                response_headers=response_headers,
                raw_response=None,
                raw_error_body=decoded_body,
                message="Response body was not valid JSON",
            )

        envelope = _read_success_envelope(decoded_body)
        if isinstance(envelope, str):
            return _protocol_error_result(
                request,
                timeout_seconds=timeout_seconds,
                started_at=started_at,
                ended_at=ended_at,
                latency_ms=latency_ms,
                http_status=response.status_code,
                provider_request_id=provider_request_id,
                response_headers=response_headers,
                raw_response=decoded_body,
                raw_error_body=None,
                message=envelope,
                usage=_usage_from_body(decoded_body),
            )

        raw_output, finish_reason, usage = envelope
        parse_result = parse_customer_label(raw_output)
        return _result(
            request,
            timeout_seconds=timeout_seconds,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            provider_outcome=ProviderOutcome.SUCCESS,
            http_status=response.status_code,
            provider_request_id=provider_request_id,
            response_headers=response_headers,
            raw_response=decoded_body,
            raw_error=None,
            raw_model_output=raw_output,
            finish_reason=finish_reason,
            usage=usage,
            parse_result=parse_result,
        )


def redact_text(value: str, *secrets: str) -> str:
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, REDACTED)
    redacted = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", redacted)
    return _DIGITALOCEAN_TOKEN_PATTERN.sub(REDACTED, redacted)


def sanitize_json(value: Any, *secrets: str) -> JsonValue:
    if isinstance(value, dict):
        sanitized: dict[str, JsonValue] = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key.lower() in _SENSITIVE_KEYS:
                sanitized[text_key] = REDACTED
            else:
                sanitized[text_key] = sanitize_json(item, *secrets)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item, *secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, *secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value), *secrets)


def _transport_failure_result(
    request: PreparedClassificationRequest,
    *,
    api_key: str,
    timeout_seconds: float,
    started_at: datetime,
    started_monotonic: float,
    outcome: ProviderOutcome,
    error: Exception,
) -> SingleIssueClassificationResult:
    return _result(
        request,
        timeout_seconds=timeout_seconds,
        started_at=started_at,
        ended_at=datetime.now(UTC),
        latency_ms=(time.perf_counter() - started_monotonic) * 1000,
        provider_outcome=outcome,
        http_status=None,
        provider_request_id=None,
        response_headers={},
        raw_response=None,
        raw_error={
            "type": outcome.value,
            "exception_class": type(error).__name__,
            "message": redact_text(str(error), api_key),
        },
        usage={},
    )


def _protocol_error_result(
    request: PreparedClassificationRequest,
    *,
    timeout_seconds: float,
    started_at: datetime,
    ended_at: datetime,
    latency_ms: float,
    http_status: int,
    provider_request_id: str | None,
    response_headers: dict[str, str],
    raw_response: JsonValue | None,
    raw_error_body: JsonValue | None,
    message: str,
    usage: dict[str, JsonValue] | None = None,
) -> SingleIssueClassificationResult:
    return _result(
        request,
        timeout_seconds=timeout_seconds,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        provider_outcome=ProviderOutcome.PROTOCOL_ERROR,
        http_status=http_status,
        provider_request_id=provider_request_id,
        response_headers=response_headers,
        raw_response=raw_response,
        raw_error={
            "type": ProviderOutcome.PROTOCOL_ERROR.value,
            "message": message,
            "body": raw_error_body,
        },
        usage=usage or {},
    )


def _result(
    request: PreparedClassificationRequest,
    *,
    timeout_seconds: float,
    started_at: datetime,
    ended_at: datetime,
    latency_ms: float,
    provider_outcome: ProviderOutcome,
    http_status: int | None,
    provider_request_id: str | None,
    response_headers: dict[str, str],
    raw_response: JsonValue | None,
    raw_error: dict[str, JsonValue] | None,
    usage: dict[str, JsonValue],
    raw_model_output: str | None = None,
    finish_reason: str | None = None,
    parse_result: OutputParseResult | None = None,
) -> SingleIssueClassificationResult:
    return SingleIssueClassificationResult(
        schema_version="single_issue_classification_result.v1",
        model_id=request.model_id,
        issue_number=request.issue_number,
        contract_version=request.contract_version,
        prompt_version=request.prompt_version,
        parser_version=request.parser_version,
        rubric_version=request.rubric_version,
        system_message_sha256=request.system_message_sha256,
        generation_configuration_sha256=request.generation_configuration_sha256,
        request_messages=request.request_messages,
        request_messages_sha256=request.request_messages_sha256,
        effective_settings=request.effective_settings,
        configured_timeout_seconds=timeout_seconds,
        request_started_at=started_at,
        request_ended_at=ended_at,
        request_latency_ms=latency_ms,
        provider_outcome=provider_outcome,
        http_status=http_status,
        provider_request_id=provider_request_id,
        response_headers=response_headers,
        raw_response=raw_response,
        raw_error=raw_error,
        raw_model_output=raw_model_output,
        finish_reason=finish_reason,
        usage=usage,
        parse_result=parse_result or invalid_parse_result(),
    )


def _decode_response_body(
    response: httpx.Response, api_key: str
) -> tuple[JsonValue, bool]:
    try:
        return sanitize_json(response.json(), api_key), True
    except ValueError:
        return redact_text(response.text, api_key), False


def _read_success_envelope(
    body: JsonValue,
) -> tuple[str, str | None, dict[str, JsonValue]] | str:
    if not isinstance(body, dict):
        return "Response JSON must be an object"
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return "Response must contain exactly one choice"
    choice = choices[0]
    if not isinstance(choice, dict):
        return "Response choice must be an object"
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return "Response choice must contain string message content"
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        return "Response finish_reason must be a string or null"
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return "Response usage must be an object"
    return message["content"], finish_reason, usage


def _usage_from_body(body: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(body, dict) or not isinstance(body.get("usage"), dict):
        return {}
    return body["usage"]


def _safe_headers(headers: httpx.Headers, api_key: str) -> dict[str, str]:
    return {
        name.lower(): redact_text(value, api_key)
        for name, value in headers.items()
        if name.lower() in _SAFE_RESPONSE_HEADERS
    }


def _provider_request_id(
    headers: httpx.Headers, body: JsonValue | None, api_key: str
) -> str | None:
    candidates = (
        headers.get("x-request-id"),
        headers.get("x-do-request-id"),
        headers.get("request-id"),
        body.get("id") if isinstance(body, dict) else None,
    )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        sanitized = redact_text(candidate, api_key)
        if REDACTED not in sanitized and _SAFE_REQUEST_ID_PATTERN.fullmatch(sanitized):
            return sanitized
    return None


def _http_outcome(status_code: int) -> ProviderOutcome:
    if status_code in {401, 403}:
        return ProviderOutcome.AUTHENTICATION
    if status_code == 408:
        return ProviderOutcome.TIMEOUT
    if status_code == 429:
        return ProviderOutcome.RATE_LIMIT
    if 500 <= status_code < 600:
        return ProviderOutcome.SERVER_ERROR
    if 400 <= status_code < 500:
        return ProviderOutcome.INVALID_REQUEST
    return ProviderOutcome.UNKNOWN
