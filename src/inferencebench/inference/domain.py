from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from inferencebench.domain import (
    AttemptEvidence,
    CustomerLabel,
    ParseStatus,
    PositiveMilliseconds,
    ProviderOutcome,
    RequestMessage,
    RunManifest,
    RunStatus,
    Sha256,
    StrictModel,
)
from inferencebench.models.domain import APPROVED_ELIGIBLE_MODEL_IDS


class OutputNormalization(StrEnum):
    SURROUNDING_WHITESPACE = "trim_surrounding_whitespace"
    ASCII_CASE = "normalize_ascii_case"
    SINGLE_QUOTE_WRAPPER = "remove_matching_single_quotes"
    DOUBLE_QUOTE_WRAPPER = "remove_matching_double_quotes"
    BACKTICK_WRAPPER = "remove_matching_backticks"
    TERMINAL_PERIOD = "remove_single_terminal_period"


class SharedGenerationConfiguration(StrictModel):
    schema_version: Literal["shared_generation_configuration.v1"]
    configuration_version: str
    temperature: Literal[0, 1]
    top_p: Literal[1] | None = None
    n: Literal[1]
    stream: Literal[False]
    tools_enabled: Literal[False]
    max_completion_tokens: Annotated[int, Field(gt=0)]

    def api_parameters(self) -> dict[str, JsonValue]:
        parameters: dict[str, JsonValue] = {
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            parameters["top_p"] = self.top_p
        parameters.update(
            {
                "n": self.n,
                "stream": self.stream,
                "max_completion_tokens": self.max_completion_tokens,
            }
        )
        return parameters


class SharedInferenceContractManifest(StrictModel):
    schema_version: Literal["shared_inference_contract_manifest.v1"]
    contract_version: str
    contract_status: Literal["development", "frozen"]
    prompt_version: str
    parser_version: Literal["bare-label-parser-v1"]
    rubric_version: Literal["v1"]
    rubric_file: Literal["docs/label-rubric-v1.md"]
    rubric_sha256: Sha256
    system_message_file: Literal["system.txt"]
    system_message_sha256: Sha256
    generation_configuration_file: Literal["generation.json"]
    generation_configuration_sha256: Sha256
    input_schema: Literal["fixed-order-compact-json-title-body-v1"]
    output_schema: Literal["one-bare-customer-taxonomy-label-v1"]
    normalization_policy: Literal["bounded-formatting-only-v1"]
    serialization: Literal["utf-8"]

    @model_validator(mode="after")
    def validate_local_artifacts(self) -> "SharedInferenceContractManifest":
        for filename in (
            self.system_message_file,
            self.generation_configuration_file,
        ):
            if PurePosixPath(filename).name != filename:
                raise ValueError("Contract artifact references must be local filenames")
        return self


class SharedInferenceContract(StrictModel):
    manifest: SharedInferenceContractManifest
    system_message: str
    generation_configuration: SharedGenerationConfiguration


class PreparedClassificationRequest(StrictModel):
    schema_version: Literal["prepared_classification_request.v1"]
    model_id: str
    issue_number: Annotated[int, Field(gt=0)]
    contract_version: str
    prompt_version: str
    parser_version: str
    rubric_version: str
    system_message_sha256: Sha256
    generation_configuration_sha256: Sha256
    request_messages: tuple[RequestMessage, RequestMessage]
    request_messages_sha256: Sha256
    effective_settings: SharedGenerationConfiguration

    @model_validator(mode="after")
    def validate_request(self) -> "PreparedClassificationRequest":
        if self.model_id not in APPROVED_ELIGIBLE_MODEL_IDS:
            raise ValueError(
                f"Model is not in the approved Eligible Candidate Pool: {self.model_id}"
            )
        if tuple(message.role for message in self.request_messages) != (
            "system",
            "user",
        ):
            raise ValueError("Classification request requires one system and one user message")
        return self

    def api_payload(self) -> dict[str, JsonValue]:
        return {
            "model": self.model_id,
            "messages": [
                message.model_dump(mode="json") for message in self.request_messages
            ],
            **self.effective_settings.api_parameters(),
        }


class OutputParseResult(StrictModel):
    parser_version: Literal["bare-label-parser-v1"]
    parse_status: ParseStatus
    parsed_label: CustomerLabel | None
    normalizations: tuple[OutputNormalization, ...]

    @model_validator(mode="after")
    def validate_parse_result(self) -> "OutputParseResult":
        if self.parse_status is ParseStatus.INVALID:
            if self.parsed_label is not None:
                raise ValueError("Invalid output cannot have a parsed label")
        elif self.parsed_label is None:
            raise ValueError("Exact or normalized output requires a parsed label")
        if self.parse_status is ParseStatus.EXACT and self.normalizations:
            raise ValueError("Exact output cannot record normalizations")
        if self.parse_status is ParseStatus.NORMALIZED and not self.normalizations:
            raise ValueError("Normalized output must record at least one normalization")
        return self


class SingleIssueClassificationResult(StrictModel):
    schema_version: Literal["single_issue_classification_result.v1"]
    model_id: str
    issue_number: Annotated[int, Field(gt=0)]
    contract_version: str
    prompt_version: str
    parser_version: str
    rubric_version: str
    system_message_sha256: Sha256
    generation_configuration_sha256: Sha256
    request_messages: tuple[RequestMessage, RequestMessage]
    request_messages_sha256: Sha256
    effective_settings: SharedGenerationConfiguration
    configured_timeout_seconds: Annotated[float, Field(gt=0)]
    request_started_at: datetime
    request_ended_at: datetime
    request_latency_ms: PositiveMilliseconds
    provider_outcome: ProviderOutcome
    http_status: int | None
    provider_request_id: str | None
    response_headers: dict[str, str]
    raw_response: JsonValue | None
    raw_error: dict[str, JsonValue] | None
    raw_model_output: str | None
    finish_reason: str | None
    usage: dict[str, JsonValue]
    parse_result: OutputParseResult

    @field_validator("response_headers")
    @classmethod
    def reject_sensitive_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        sensitive = {"authorization", "proxy-authorization", "x-api-key"}
        if any(name.lower() in sensitive for name in headers):
            raise ValueError("Sensitive headers are not inference evidence")
        return headers

    @model_validator(mode="after")
    def validate_terminal_result(self) -> "SingleIssueClassificationResult":
        if self.request_ended_at < self.request_started_at:
            raise ValueError("Request end cannot precede request start")
        if self.provider_outcome is ProviderOutcome.SUCCESS:
            if self.http_status is None or not 200 <= self.http_status < 300:
                raise ValueError("Successful provider outcome requires a 2xx status")
            if self.raw_response is None or self.raw_error is not None:
                raise ValueError("Successful provider outcome requires only raw response")
            if self.raw_model_output is None:
                raise ValueError("Successful provider outcome requires model output")
        else:
            if self.raw_error is None:
                raise ValueError("Provider failure requires sanitized raw error evidence")
            if self.parse_result.parsed_label is not None:
                raise ValueError("Provider failure cannot produce a parsed label")
        return self


class ModelEvaluationConfiguration(StrictModel):
    schema_version: Literal["model_evaluation_configuration.v1"]
    model_id: str
    timeout_seconds: Annotated[float, Field(gt=0)]
    concurrency: Annotated[int, Field(gt=0)]
    ground_truth_version: str
    ground_truth_sha256: Sha256
    ground_truth_labels: dict[int, CustomerLabel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model(self) -> "ModelEvaluationConfiguration":
        if self.model_id not in APPROVED_ELIGIBLE_MODEL_IDS:
            raise ValueError(
                f"Model is not in the approved Eligible Candidate Pool: {self.model_id}"
            )
        if any(issue_number <= 0 for issue_number in self.ground_truth_labels):
            raise ValueError("Ground Truth issue numbers must be positive")
        return self


class ModelEvaluationProgress(StrictModel):
    schema_version: Literal["model_evaluation_progress.v1"]
    run_id: str
    model_id: str
    status: RunStatus
    expected_count: Annotated[int, Field(gt=0)]
    persisted_count: Annotated[int, Field(ge=0)]
    usable_count: Annotated[int, Field(ge=0)]
    normalized_count: Annotated[int, Field(ge=0)]
    invalid_output_count: Annotated[int, Field(ge=0)]
    request_error_count: Annotated[int, Field(ge=0)]
    elapsed_wall_clock_ms: PositiveMilliseconds
    latest_issue_number: Annotated[int, Field(gt=0)] | None


class ModelEvaluationExecution(StrictModel):
    schema_version: Literal["model_evaluation_execution.v1"]
    manifest: RunManifest
    attempts: tuple[AttemptEvidence, ...]
