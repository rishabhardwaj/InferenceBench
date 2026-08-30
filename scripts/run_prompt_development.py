"""Run the persisted 26-by-20 Prompt Development matrix.

The key is read from an external, mode-600 file and is never written to the
database, summary, or console output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import httpx

from inferencebench.artifacts import CorpusArtifacts, sha256_file
from inferencebench.config import Settings
from inferencebench.domain import GroundTruthAnnotation, GroundTruthManifest
from inferencebench.inference.domain import ModelEvaluationConfiguration
from inferencebench.inference.contract import load_shared_inference_contract
from inferencebench.inference.runner import execute_model_evaluation
from inferencebench.models.domain import APPROVED_ELIGIBLE_MODEL_IDS
from inferencebench.persistence.repository import EvidenceRepository


def _print_progress(progress) -> None:
    if progress.status.value == "running" and progress.persisted_count > 0:
        print(
            json.dumps(
                {
                    "model_id": progress.model_id,
                    "status": "running",
                    "persisted_count": progress.persisted_count,
                    "expected_count": progress.expected_count,
                    "latest_issue_number": progress.latest_issue_number,
                    "usable_count": progress.usable_count,
                    "invalid_output_count": progress.invalid_output_count,
                    "request_error_count": progress.request_error_count,
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--issue-limit", type=int, default=20)
    parser.add_argument(
        "--model-ids",
        help="comma-separated Eligible Candidate Pool IDs; defaults to all candidates",
    )
    parser.add_argument(
        "--database-name",
        help="results database filename; defaults by issue-limit",
    )
    parser.add_argument(
        "--contract-directory",
        type=Path,
        default=Path("artifacts/prompts/development-v2"),
    )
    arguments = parser.parse_args()
    if not arguments.key_file.is_file():
        raise SystemExit("--key-file must be an existing regular file")
    model_ids = (
        tuple(arguments.model_ids.split(","))
        if arguments.model_ids
        else APPROVED_ELIGIBLE_MODEL_IDS
    )
    if not model_ids or any(model_id not in APPROVED_ELIGIBLE_MODEL_IDS for model_id in model_ids):
        raise SystemExit("--model-ids must contain only eligible model IDs")
    if not 0 <= arguments.start_index < len(model_ids):
        raise SystemExit("--start-index is outside the Eligible Candidate Pool")
    if not 1 <= arguments.issue_limit <= 20:
        raise SystemExit("--issue-limit must be between 1 and 20")
    asyncio.run(
        _run(
            arguments.key_file,
            arguments.start_index,
            arguments.issue_limit,
            arguments.contract_directory,
            model_ids,
            arguments.database_name,
        )
    )


async def _run(
    key_file: Path,
    start_index: int,
    issue_limit: int,
    contract_directory: Path,
    model_ids: tuple[str, ...],
    database_name_override: str | None,
) -> None:
    root = Path(__file__).resolve().parents[1]
    database_name = database_name_override or (
        "prompt-development-v2-preflight.sqlite3"
        if issue_limit < 20
        else "prompt-development-v2.sqlite3"
    )
    settings = replace(
        Settings.from_environment(),
        database_path=root / "artifacts" / "results" / database_name,
    )
    corpus_manifest, corpus_issues = CorpusArtifacts(settings.corpus_root).load_active()
    ground_truth_dir = root / "artifacts" / "ground_truth" / "doctl-evaluation-v1"
    ground_truth_manifest = GroundTruthManifest.model_validate_json(
        (ground_truth_dir / "manifest.json").read_text(encoding="utf-8")
    )
    annotation_path = ground_truth_dir / ground_truth_manifest.artifact_file
    if sha256_file(annotation_path) != ground_truth_manifest.artifact_sha256:
        raise RuntimeError("Ground Truth artifact hash mismatch")
    annotations = tuple(
        GroundTruthAnnotation.model_validate_json(line)
        for line in annotation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    development = tuple(
        item for item in annotations if item.evaluation_role == "prompt_development"
    )
    if len(development) != 20:
        raise RuntimeError("Prompt Development must contain exactly 20 annotations")
    issues_by_number = {issue.issue_number: issue for issue in corpus_issues}
    development = development[:issue_limit]
    issues = tuple(issues_by_number[item.issue_number] for item in development)
    labels = {item.issue_number: item.label for item in development}
    contract = load_shared_inference_contract(contract_directory)
    api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("API key file is empty")

    summary_path = settings.database_path.with_suffix(".summary.json")
    results: list[dict[str, object]] = []
    EvidenceRepository(settings.database_path).initialize()
    with sqlite3.connect(settings.database_path) as connection:
        completed_models = {
            row[0]
            for row in connection.execute(
                """
                SELECT model_id FROM run_manifests
                WHERE status = 'complete' AND expected_count = ?
                """
                , (issue_limit,)
            )
        }
    async with httpx.AsyncClient() as client:
        for position, model_id in enumerate(
            model_ids[start_index:], start_index + 1
        ):
            if model_id in completed_models:
                print(
                    json.dumps(
                        {
                            "position": position,
                            "model_id": model_id,
                            "status": "skipped_existing_complete_run",
                        }
                    ),
                    flush=True,
                )
                continue
            configuration = ModelEvaluationConfiguration(
                schema_version="model_evaluation_configuration.v1",
                model_id=model_id,
                timeout_seconds=60,
                concurrency=1,
                ground_truth_version=ground_truth_manifest.ground_truth_version,
                ground_truth_sha256=ground_truth_manifest.artifact_sha256,
                ground_truth_labels=labels,
            )
            try:
                execution = await execute_model_evaluation(
                    settings,
                    corpus_manifest,
                    issues,
                    configuration,
                    api_key=api_key,
                    client=client,
                    evaluation_stage="prompt_development",
                    contract=contract,
                    progress_callback=lambda progress: _print_progress(progress),
                )
                result: dict[str, object] = {
                    "position": position,
                    "model_id": model_id,
                    "run_id": execution.manifest.run_id,
                    "status": execution.manifest.status.value,
                    "persisted_count": execution.manifest.persisted_count,
                    "usable_count": execution.manifest.usable_count,
                    "invalid_output_count": execution.manifest.invalid_output_count,
                    "request_error_count": execution.manifest.request_error_count,
                }
            except Exception as error:
                result = {
                    "position": position,
                    "model_id": model_id,
                    "status": "runner_error",
                    "error_type": type(error).__name__,
                    "message": str(error).replace(api_key, "[REDACTED]"),
                }
            results.append(result)
            summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
