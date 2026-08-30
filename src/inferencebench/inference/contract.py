from __future__ import annotations

import json
from pathlib import Path

from inferencebench.artifacts import ArtifactIntegrityError, canonical_sha256, sha256_file
from inferencebench.domain import Issue, RequestMessage
from inferencebench.config import PROJECT_ROOT
from inferencebench.inference.domain import (
    PreparedClassificationRequest,
    SharedGenerationConfiguration,
    SharedInferenceContract,
    SharedInferenceContractManifest,
)


DEFAULT_CONTRACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "prompts" / "development-v1"


def load_shared_inference_contract(
    directory: Path = DEFAULT_CONTRACT_DIRECTORY,
) -> SharedInferenceContract:
    manifest = SharedInferenceContractManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    system_path = directory / manifest.system_message_file
    generation_path = directory / manifest.generation_configuration_file
    rubric_path = directory.parents[2] / manifest.rubric_file
    _verify_hash(system_path, manifest.system_message_sha256)
    _verify_hash(generation_path, manifest.generation_configuration_sha256)
    _verify_hash(rubric_path, manifest.rubric_sha256)
    system_message = system_path.read_text(encoding="utf-8")
    generation_configuration = SharedGenerationConfiguration.model_validate_json(
        generation_path.read_text(encoding="utf-8")
    )
    return SharedInferenceContract(
        manifest=manifest,
        system_message=system_message,
        generation_configuration=generation_configuration,
    )


def prepare_classification_request(
    issue: Issue,
    model_id: str,
    contract: SharedInferenceContract,
) -> PreparedClassificationRequest:
    user_message = json.dumps(
        {"title": issue.title, "body": issue.body},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages = (
        RequestMessage(role="system", content=contract.system_message),
        RequestMessage(role="user", content=user_message),
    )
    messages_hash = canonical_sha256(
        [message.model_dump(mode="json") for message in messages]
    )
    manifest = contract.manifest
    return PreparedClassificationRequest(
        schema_version="prepared_classification_request.v1",
        model_id=model_id,
        issue_number=issue.issue_number,
        contract_version=manifest.contract_version,
        prompt_version=manifest.prompt_version,
        parser_version=manifest.parser_version,
        rubric_version=manifest.rubric_version,
        system_message_sha256=manifest.system_message_sha256,
        generation_configuration_sha256=manifest.generation_configuration_sha256,
        request_messages=messages,
        request_messages_sha256=messages_hash,
        effective_settings=contract.generation_configuration,
    )


def _verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"Shared Inference Contract hash mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
