from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import httpx

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.domain import (
    CorpusManifest,
    CostCompleteness,
    CustomerLabel,
    Issue,
    ParseStatus,
    ProviderOutcome,
    RunStatus,
    ScoredOutcome,
)
from inferencebench.inference.domain import (
    ModelEvaluationConfiguration,
    ModelEvaluationProgress,
)
from inferencebench.inference.runner import execute_model_evaluation
from inferencebench.models.domain import APPROVED_ELIGIBLE_MODEL_IDS
from inferencebench.persistence.repository import EvidenceRepository


API_KEY = "doo_v1_test-runner-secret"
GROUND_TRUTH_SHA256 = "a" * 64


@dataclass
class ProviderState:
    database_path: Any
    active: int = 0
    max_active: int = 0
    calls: list[tuple[str, int]] = field(default_factory=list)
    manifest_seen_before_every_call: bool = True


class RecordingRepository(EvidenceRepository):
    def __init__(self, database_path, provider_state: ProviderState) -> None:
        super().__init__(database_path)
        self.provider_state = provider_state
        self.writer_tasks: set[int] = set()
        self.provider_activity_during_writes: list[int] = []
        self.persisted_issue_numbers: list[int] = []

    def insert_attempt(self, attempt) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.writer_tasks.add(id(task))
        self.provider_activity_during_writes.append(self.provider_state.active)
        super().insert_attempt(attempt)
        self.persisted_issue_numbers.append(attempt.issue_number)


def test_run_persists_manifest_bounds_calls_and_records_every_terminal_outcome(
    settings: Settings, fixture_artifacts: FixtureArtifacts
) -> None:
    corpus_manifest, issues = _population(fixture_artifacts, 6)
    state = ProviderState(settings.database_path)
    repository = RecordingRepository(settings.database_path, state)
    progress: list[ModelEvaluationProgress] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        issue_input = json.loads(payload["messages"][1]["content"])
        issue_number = int(issue_input["title"].removeprefix("Issue "))
        assert set(issue_input) == {"title", "body"}
        assert payload["model"] == APPROVED_ELIGIBLE_MODEL_IDS[0]
        state.calls.append((payload["model"], issue_number))
        with sqlite3.connect(settings.database_path) as connection:
            rows = connection.execute(
                "SELECT status, manifest_json FROM run_manifests"
            ).fetchall()
        state.manifest_seen_before_every_call &= (
            len(rows) == 1
            and rows[0][0] == RunStatus.RUNNING.value
            and json.loads(rows[0][1])["expected_count"] == len(issues)
        )

        state.active += 1
        state.max_active = max(state.max_active, state.active)
        try:
            if issue_number == 5:
                await asyncio.sleep(0.08)
            else:
                await asyncio.sleep(0.01)
            if issue_number == 3:
                return _completion("The label is bug", issue_number)
            if issue_number == 4:
                return httpx.Response(
                    429,
                    headers={"retry-after": "2", "x-request-id": "rate-limited-4"},
                    json={"error": "rate limited"},
                )
            outputs = {1: "bug", 2: '"BUG".', 6: "question"}
            return _completion(outputs[issue_number], issue_number)
        finally:
            state.active -= 1

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await execute_model_evaluation(
                settings,
                corpus_manifest,
                issues,
                _configuration(
                    model_id=APPROVED_ELIGIBLE_MODEL_IDS[0],
                    timeout_seconds=0.03,
                    concurrency=2,
                    labels={
                        1: CustomerLabel.BUG,
                        2: CustomerLabel.BUG,
                        3: CustomerLabel.BUG,
                        4: CustomerLabel.BUG,
                        5: CustomerLabel.BUG,
                        6: CustomerLabel.QUESTION,
                    },
                ),
                api_key=API_KEY,
                client=client,
                repository=repository,
                progress_callback=progress.append,
            )

    execution = asyncio.run(execute())

    assert state.manifest_seen_before_every_call is True
    assert state.max_active == 2
    assert state.calls == [
        (APPROVED_ELIGIBLE_MODEL_IDS[0], issue_number)
        for issue_number in range(1, 7)
    ]
    assert len(repository.writer_tasks) == 1
    assert any(active > 0 for active in repository.provider_activity_during_writes)

    assert execution.manifest.status is RunStatus.COMPLETE
    assert execution.manifest.expected_count == 6
    assert execution.manifest.persisted_count == 6
    assert execution.manifest.usable_count == 3
    assert execution.manifest.normalized_count == 1
    assert execution.manifest.invalid_output_count == 1
    assert execution.manifest.request_error_count == 2
    assert execution.manifest.concurrency == 2
    assert execution.manifest.timeout_seconds == 0.03
    assert execution.manifest.retries == 0
    assert execution.manifest.wall_clock_ms is not None

    assert tuple(attempt.issue_number for attempt in execution.attempts) == tuple(
        range(1, 7)
    )
    assert len({attempt.attempt_id for attempt in execution.attempts}) == 6
    assert all(attempt.attempt_purpose.value == "benchmark" for attempt in execution.attempts)
    assert all(len(attempt.request_messages) == 2 for attempt in execution.attempts)
    assert max(attempt.queue_wait_ms for attempt in execution.attempts[2:]) > 1
    assert tuple(attempt.provider_outcome for attempt in execution.attempts) == (
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.RATE_LIMIT,
        ProviderOutcome.TIMEOUT,
        ProviderOutcome.SUCCESS,
    )
    assert tuple(attempt.scored_outcome for attempt in execution.attempts) == (
        ScoredOutcome.CORRECT,
        ScoredOutcome.CORRECT,
        ScoredOutcome.INVALID_OUTPUT,
        ScoredOutcome.REQUEST_ERROR,
        ScoredOutcome.REQUEST_ERROR,
        ScoredOutcome.CORRECT,
    )
    assert execution.attempts[1].parse_status is ParseStatus.NORMALIZED
    assert execution.attempts[0].usage["prompt_tokens"] == 101
    assert execution.attempts[4].raw_error["type"] == "timeout"
    assert all(
        attempt.schema_version == "attempt_evidence.v3"
        for attempt in execution.attempts
    )
    assert tuple(
        attempt.cost_completeness for attempt in execution.attempts
    ) == (
        CostCompleteness.COMPLETE,
        CostCompleteness.COMPLETE,
        CostCompleteness.COMPLETE,
        CostCompleteness.UNKNOWN,
        CostCompleteness.UNKNOWN,
        CostCompleteness.COMPLETE,
    )
    assert execution.attempts[0].calculated_request_cost_usd is not None
    assert len(execution.attempts[0].cost_calculation_terms) == 2
    assert execution.attempts[3].cost_unknown_reasons == (
        "missing prompt_tokens",
        "missing completion_tokens",
    )
    assert API_KEY not in execution.model_dump_json()

    assert progress[0].status is RunStatus.RUNNING
    assert progress[0].persisted_count == 0
    assert [item.persisted_count for item in progress] == list(range(0, 6)) + [6]
    assert progress[-1].status is RunStatus.COMPLETE
    assert repository.get_run(execution.manifest.run_id) == execution.manifest
    assert repository.get_attempts(execution.manifest.run_id) == execution.attempts


def test_interruption_is_durable_and_restart_creates_a_fresh_complete_run(
    settings: Settings, fixture_artifacts: FixtureArtifacts
) -> None:
    corpus_manifest, issues = _population(fixture_artifacts, 4)
    state = ProviderState(settings.database_path)
    repository = RecordingRepository(settings.database_path, state)
    configuration = _configuration(
        model_id=APPROVED_ELIGIBLE_MODEL_IDS[1],
        timeout_seconds=1,
        concurrency=1,
        labels={},
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        issue_input = json.loads(payload["messages"][1]["content"])
        issue_number = int(issue_input["title"].removeprefix("Issue "))
        state.active += 1
        try:
            await asyncio.sleep(0.005)
            return _completion("other", issue_number)
        finally:
            state.active -= 1

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interrupted = await execute_model_evaluation(
                settings,
                corpus_manifest,
                issues,
                configuration,
                api_key=API_KEY,
                client=client,
                repository=repository,
                cancel_requested=lambda: len(repository.persisted_issue_numbers) >= 2,
            )
            restarted = await execute_model_evaluation(
                settings,
                corpus_manifest,
                issues,
                configuration,
                api_key=API_KEY,
                client=client,
                repository=repository,
            )
        return interrupted, restarted

    interrupted, restarted = asyncio.run(execute())

    assert interrupted.manifest.status is RunStatus.INCOMPLETE
    assert interrupted.manifest.persisted_count == 2
    assert tuple(attempt.issue_number for attempt in interrupted.attempts) == (1, 2)
    assert repository.get_run(interrupted.manifest.run_id).status is RunStatus.INCOMPLETE

    assert restarted.manifest.status is RunStatus.COMPLETE
    assert restarted.manifest.persisted_count == 4
    assert restarted.manifest.run_id != interrupted.manifest.run_id
    assert tuple(attempt.issue_number for attempt in restarted.attempts) == (1, 2, 3, 4)
    assert repository.get_attempts(interrupted.manifest.run_id) == interrupted.attempts


def test_two_models_are_separate_sequential_runs(
    settings: Settings, fixture_artifacts: FixtureArtifacts
) -> None:
    corpus_manifest, issues = _population(fixture_artifacts, 2)
    active_models: set[str] = set()
    model_overlap = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_overlap
        payload = json.loads(request.content)
        model_id = payload["model"]
        active_models.add(model_id)
        model_overlap |= len(active_models) > 1
        try:
            await asyncio.sleep(0.005)
            return _completion("other", 1)
        finally:
            active_models.remove(model_id)

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = await execute_model_evaluation(
                settings,
                corpus_manifest,
                issues,
                _configuration(
                    model_id=APPROVED_ELIGIBLE_MODEL_IDS[0],
                    timeout_seconds=1,
                    concurrency=2,
                    labels={},
                ),
                api_key=API_KEY,
                client=client,
            )
            second = await execute_model_evaluation(
                settings,
                corpus_manifest,
                issues,
                _configuration(
                    model_id=APPROVED_ELIGIBLE_MODEL_IDS[1],
                    timeout_seconds=1,
                    concurrency=2,
                    labels={},
                ),
                api_key=API_KEY,
                client=client,
            )
        return first, second

    first, second = asyncio.run(execute())

    assert model_overlap is False
    assert first.manifest.run_id != second.manifest.run_id
    assert first.manifest.model_id != second.manifest.model_id
    assert first.manifest.expected_count == second.manifest.expected_count == 2
    assert first.manifest.status is second.manifest.status is RunStatus.COMPLETE


def _configuration(
    *,
    model_id: str,
    timeout_seconds: float,
    concurrency: int,
    labels: dict[int, CustomerLabel],
) -> ModelEvaluationConfiguration:
    return ModelEvaluationConfiguration(
        schema_version="model_evaluation_configuration.v1",
        model_id=model_id,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        ground_truth_version="synthetic-ground-truth-v1",
        ground_truth_sha256=GROUND_TRUTH_SHA256,
        ground_truth_labels=labels,
    )


def _population(
    fixture_artifacts: FixtureArtifacts, count: int
) -> tuple[CorpusManifest, tuple[Issue, ...]]:
    base_manifest, base_issues = fixture_artifacts.load_corpus()
    base_issue = base_issues[0]
    issues = tuple(
        Issue.model_validate(
            {
                **base_issue.model_dump(mode="python"),
                "github_issue_id": issue_number,
                "issue_number": issue_number,
                "node_id": f"I_synthetic_{issue_number}",
                "api_url": (
                    "https://api.github.com/repos/digitalocean/doctl/issues/"
                    f"{issue_number}"
                ),
                "html_url": (
                    "https://github.com/digitalocean/doctl/issues/"
                    f"{issue_number}"
                ),
                "title": f"Issue {issue_number}",
                "body": f"Synthetic body {issue_number}",
                "content_sha256": f"{issue_number:064x}",
            }
        )
        for issue_number in range(1, count + 1)
    )
    manifest = CorpusManifest.model_validate(
        {
            **base_manifest.model_dump(mode="python"),
            "api_object_count": count,
            "issue_count": count,
            "ordered_issue_numbers": tuple(range(1, count + 1)),
        }
    )
    return manifest, issues


def _completion(content: str, issue_number: int) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"x-request-id": f"request-{issue_number}"},
        json={
            "id": f"response-{issue_number}",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 100 + issue_number,
                "completion_tokens": 1,
                "total_tokens": 101 + issue_number,
            },
        },
    )
