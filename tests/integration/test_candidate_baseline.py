from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.config import Settings
from inferencebench.evaluation.bootstrap import (
    build_shared_bootstrap_manifest,
    generate_shared_resamples,
)
from inferencebench.models.domain import APPROVED_ELIGIBLE_MODEL_IDS
from inferencebench.workflows.candidate_baseline import (
    CandidateBaselineExecution,
    CandidateBaselineRunRecord,
    baseline_bootstrap_rows,
    build_candidate_bootstrap_analysis,
    build_candidate_evidence_table,
    create_candidate_baseline_authorization,
    execute_candidate_baseline,
    prepare_candidate_baseline,
)


API_KEY = "doo_v1_candidate-baseline-test-secret"


@dataclass
class ProviderState:
    active: int = 0
    max_active: int = 0
    calls: list[tuple[str, int]] = field(default_factory=list)


def _frozen_settings(settings: Settings, tmp_path: Path) -> Settings:
    contract_directory = tmp_path / "frozen-contract"
    shutil.copytree(Path("artifacts/prompts/development-v1"), contract_directory)
    manifest_path = contract_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "contract_version": "shared-inference-contract-candidate-test-v1",
            "contract_status": "frozen",
            "prompt_version": "candidate-test-frozen-v1",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return replace(settings, shared_contract_directory=contract_directory)


def _issue_lookup(settings: Settings, issue_numbers: tuple[int, ...]):
    _, issues = CorpusArtifacts(settings.corpus_root).load_active()
    return {
        (issue.title, issue.body): issue.issue_number
        for issue in issues
        if issue.issue_number in set(issue_numbers)
    }


def test_preparation_is_local_exact_and_blocks_development_contract(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_calls = 0

    async def fail_if_called(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("Candidate Baseline preparation must be local")

    monkeypatch.setattr(httpx.AsyncClient, "request", fail_if_called)
    plan = prepare_candidate_baseline(settings)

    assert provider_calls == 0
    assert plan.eligible_model_ids == APPROVED_ELIGIBLE_MODEL_IDS
    assert len(plan.random_sample_issue_numbers) == 100
    assert len(plan.prompt_development_issue_numbers) == 20
    assert len(plan.primary_holdout_issue_numbers) == 80
    assert len(plan.diagnostic_issue_numbers) == 20
    assert len(plan.candidate_screening_issue_numbers) == 40
    assert plan.ordered_baseline_issue_numbers == plan.candidate_screening_issue_numbers
    assert set(plan.candidate_screening_issue_numbers) <= set(
        plan.primary_holdout_issue_numbers
    )
    assert plan.expected_attempts_per_model == 40
    assert plan.expected_provider_requests == 25 * 40
    assert plan.concurrency == 4
    assert plan.retries == 0
    assert plan.can_authorize is False
    assert "not frozen" in plan.blockers[0]
    with pytest.raises(ValueError, match="blocked"):
        create_candidate_baseline_authorization(
            plan,
            confirmed_plan_sha256=plan.content_sha256,
            authorized_by="test reviewer",
            authorized_at=datetime.now(UTC),
        )


def test_explicit_authorized_baseline_runs_all_25_sequentially_and_reports_strata(
    settings: Settings, tmp_path: Path
) -> None:
    frozen = _frozen_settings(settings, tmp_path)
    plan = prepare_candidate_baseline(frozen)
    with pytest.raises(ValueError, match="hash does not match"):
        create_candidate_baseline_authorization(
            plan,
            confirmed_plan_sha256="0" * 64,
            authorized_by="test reviewer",
            authorized_at=datetime.now(UTC),
        )
    authorization = create_candidate_baseline_authorization(
        plan,
        confirmed_plan_sha256=plan.content_sha256,
        authorized_by="test reviewer",
        authorized_at=datetime.now(UTC),
    )
    lookup = _issue_lookup(frozen, plan.ordered_baseline_issue_numbers)
    state = ProviderState()
    request_number = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_number
        payload = json.loads(request.content)
        issue_input = json.loads(payload["messages"][1]["content"])
        issue_number = lookup[(issue_input["title"], issue_input["body"])]
        state.active += 1
        state.max_active = max(state.max_active, state.active)
        state.calls.append((payload["model"], issue_number))
        request_number += 1
        try:
            await asyncio.sleep(0.001)
            return httpx.Response(
                200,
                headers={"x-request-id": f"baseline-request-{request_number}"},
                json={
                    "id": f"baseline-response-{request_number}",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "other",
                            },
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
            state.active -= 1

    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    execution = asyncio.run(
        execute_candidate_baseline(
            frozen,
            plan,
            authorization,
            api_key=API_KEY,
            client_factory=client_factory,
        )
    )

    assert state.max_active == 4
    assert len(state.calls) == 25 * 40
    assert state.calls == [
        (model_id, issue_number)
        for model_id in APPROVED_ELIGIBLE_MODEL_IDS
        for issue_number in plan.ordered_baseline_issue_numbers
    ]
    assert tuple(row.model_id for row in execution.runs) == APPROVED_ELIGIBLE_MODEL_IDS
    assert all(row.status == "complete" for row in execution.runs)
    assert all(row.persisted_count == 40 for row in execution.runs)
    assert len({row.run_id for row in execution.runs}) == 25

    evidence = build_candidate_evidence_table(frozen, plan, execution)
    assert evidence.row_count == 25
    assert all(row.comparable_headlines for row in evidence.rows)
    assert all(row.candidate_screening_holdout is not None for row in evidence.rows)
    assert all(
        row.candidate_screening_holdout.expected_count == 40 for row in evidence.rows
    )
    assert all(row.candidate_screening_latency.issue_count == 40 for row in evidence.rows)
    assert all(row.output_adherence.expected_count == 40 for row in evidence.rows)
    assert all(row.operational.concurrency == 4 for row in evidence.rows)
    assert all(row.source_attempts_sha256 is not None for row in evidence.rows)
    assert all(len(row.source_attempts_sha256) == 64 for row in evidence.rows)
    assert all(
        row.candidate_screening_cost_completeness == "complete"
        for row in evidence.rows
    )
    assert API_KEY not in execution.model_dump_json()
    assert API_KEY.encode() not in frozen.database_path.read_bytes()


def test_bootstrap_analysis_retains_every_incomplete_candidate_and_pair(
    settings: Settings, tmp_path: Path
) -> None:
    frozen = _frozen_settings(settings, tmp_path)
    plan = prepare_candidate_baseline(frozen)
    authorization = create_candidate_baseline_authorization(
        plan,
        confirmed_plan_sha256=plan.content_sha256,
        authorized_by="test reviewer",
        authorized_at=datetime.now(UTC),
    )
    execution = CandidateBaselineExecution(
        schema_version="candidate_baseline_execution.v1",
        baseline_version=plan.baseline_version,
        plan_sha256=plan.content_sha256,
        authorization=authorization,
        runs=tuple(
            CandidateBaselineRunRecord(
                candidate_position=position,
                model_id=model_id,
                run_id=None,
                status="runner_error",
                    expected_count=40,
                persisted_count=0,
                usable_count=0,
                invalid_output_count=0,
                request_error_count=0,
                error_type="UnavailableModel",
                sanitized_error="model unavailable",
            )
            for position, model_id in enumerate(APPROVED_ELIGIBLE_MODEL_IDS, 1)
        ),
    )
    rows = baseline_bootstrap_rows(frozen)
    resamples = generate_shared_resamples(rows)
    bootstrap_plan = build_shared_bootstrap_manifest(
        plan_version="candidate-test-shared-bootstrap-v1",
        baseline_plan_sha256=plan.content_sha256,
        ground_truth_version=plan.ground_truth_version,
        ground_truth_sha256=plan.ground_truth_sha256,
        rows=rows,
        resamples=resamples,
        seed=plan.shared_bootstrap_seed,
    )

    analysis = build_candidate_bootstrap_analysis(
        frozen, plan, execution, bootstrap_plan, resamples
    )

    assert analysis.candidate_count == 25
    assert analysis.pair_count == 300
    assert len(analysis.candidates) == 25
    assert len(analysis.pairs) == 300
    assert all(
        row.evidence_status == "unavailable_incomplete"
        for row in analysis.candidates
    )
    assert all(row.intervals is None for row in analysis.pairs)
