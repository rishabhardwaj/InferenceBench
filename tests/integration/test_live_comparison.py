from __future__ import annotations

import asyncio
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import httpx

from inferencebench.config import Settings
from inferencebench.domain import ProviderOutcome, RunStatus
from inferencebench.workflows.live_comparison import (
    LiveComparisonConfiguration,
    LiveModelProgress,
    execute_live_comparison,
    prepare_live_comparison,
)
from inferencebench.workflows.saved_comparison import load_persisted_comparison


API_KEY = "doo_v1_live-comparison-test-secret"


def test_fake_provider_runs_models_independently_persists_and_reopens_review(
    settings: Settings,
    tmp_path: Path,
) -> None:
    live_settings = _fixture_live_settings(settings, tmp_path)
    preparation = prepare_live_comparison(live_settings)
    calls: list[str] = []
    active_models: set[str] = set()
    max_active_by_model: Counter[str] = Counter()
    current_active_by_model: Counter[str] = Counter()
    model_overlap = False
    progress: list[LiveModelProgress] = []

    model_a, model_b = preparation.eligible_model_ids[:2]

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_overlap
        payload = json.loads(request.content)
        model_id = payload["model"]
        calls.append(model_id)
        model_call_number = calls.count(model_id)
        active_models.add(model_id)
        current_active_by_model[model_id] += 1
        max_active_by_model[model_id] = max(
            max_active_by_model[model_id], current_active_by_model[model_id]
        )
        model_overlap |= len(active_models) > 1
        try:
            await asyncio.sleep(0.001)
            if model_id == model_a and model_call_number == 1:
                return httpx.Response(
                    500,
                    headers={"x-request-id": "sanitized-error"},
                    json={"error": f"Bearer {API_KEY}"},
                )
            return httpx.Response(
                200,
                headers={"x-request-id": f"request-{len(calls)}"},
                json={
                    "id": f"response-{len(calls)}",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "bug"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "total_tokens": 101,
                    },
                },
            )
        finally:
            current_active_by_model[model_id] -= 1
            if current_active_by_model[model_id] == 0:
                active_models.remove(model_id)

    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    configuration = LiveComparisonConfiguration(
        model_a_id=model_a,
        model_b_id=model_b,
        concurrency=2,
    )

    first = asyncio.run(
        execute_live_comparison(
            live_settings,
            configuration,
            api_key=API_KEY,
            progress_callback=progress.append,
            client_factory=client_factory,
        )
    )
    second = asyncio.run(
        execute_live_comparison(
            live_settings,
            configuration,
            api_key=API_KEY,
            client_factory=client_factory,
        )
    )

    expected_per_run = preparation.corpus_issue_count
    assert preparation.can_start is True
    assert preparation.expected_provider_requests == 2 * expected_per_run
    assert first.comparison_complete is True
    assert first.model_a.manifest.status is RunStatus.COMPLETE
    assert first.model_b.manifest.status is RunStatus.COMPLETE
    assert first.model_a.manifest.concurrency == 2
    assert first.model_b.manifest.concurrency == 2
    assert first.run_ids[0] != first.run_ids[1]
    assert set(first.run_ids).isdisjoint(second.run_ids)
    assert calls[:expected_per_run] == [model_a] * expected_per_run
    assert calls[expected_per_run : 2 * expected_per_run] == [model_b] * expected_per_run
    assert model_overlap is False
    assert max_active_by_model[model_a] <= 2
    assert max_active_by_model[model_b] <= 2

    terminal_a_index = next(
        index
        for index, item in enumerate(progress)
        if item.position == "Model A" and item.status is RunStatus.COMPLETE
    )
    first_b_index = next(
        index for index, item in enumerate(progress) if item.position == "Model B"
    )
    assert terminal_a_index < first_b_index
    assert progress[-1].status is RunStatus.COMPLETE
    assert progress[-1].known_cost_count == expected_per_run

    failed_attempt = next(
        attempt
        for attempt in first.model_a.attempts
        if attempt.provider_outcome is ProviderOutcome.SERVER_ERROR
    )
    assert failed_attempt.provider_outcome is ProviderOutcome.SERVER_ERROR
    assert API_KEY not in failed_attempt.model_dump_json()
    assert "[REDACTED]" in failed_attempt.model_dump_json()
    assert API_KEY.encode() not in live_settings.database_path.read_bytes()

    review = load_persisted_comparison(live_settings, first.run_ids)
    assert review.evidence_state == "Live result — fresh persisted comparison"
    assert review.corpus_issue_count == expected_per_run
    assert review.model_a.run_id == first.run_ids[0]
    assert review.model_b.run_id == first.run_ids[1]
    assert review.unscored_view.expected_count == expected_per_run - 1


def test_live_preparation_blocks_development_contract_without_provider_calls(
    settings: Settings,
) -> None:
    preparation = prepare_live_comparison(settings)

    assert preparation.corpus_issue_count == 536
    assert preparation.scored_issue_count == 120
    assert preparation.unscored_issue_count == 416
    assert preparation.expected_provider_requests == 1072
    assert preparation.contract_status == "development"
    assert preparation.can_start is False
    assert "Issue 12" in preparation.blockers[0]


def _fixture_live_settings(settings: Settings, tmp_path: Path) -> Settings:
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

    contract_directory = tmp_path / "frozen-contract"
    shutil.copytree(Path("artifacts/prompts/development-v1"), contract_directory)
    contract_manifest_path = contract_directory / "manifest.json"
    contract_manifest = json.loads(
        contract_manifest_path.read_text(encoding="utf-8")
    )
    contract_manifest.update(
        {
            "contract_version": "shared-inference-contract-test-frozen-v1",
            "contract_status": "frozen",
            "prompt_version": "zero-shot-test-frozen-v1",
        }
    )
    contract_manifest_path.write_text(
        json.dumps(contract_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return replace(
        settings,
        corpus_root=corpus_root,
        ground_truth_root=ground_truth_root,
        evaluation_ground_truth_version=evaluation_version,
        shared_contract_directory=contract_directory,
        shared_timeout_seconds=2,
        live_default_concurrency=2,
    )
