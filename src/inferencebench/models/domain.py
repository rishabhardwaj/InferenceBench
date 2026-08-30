from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from inferencebench.domain import Sha256, StrictModel


CATALOG_SNAPSHOT_MODEL_IDS = (
    "arcee-trinity-large-thinking",
    "deepseek-3.2",
    "deepseek-4-flash",
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro",
    "deepseek-v4-pro-0813",
    "gemma-4-31B-it",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "glm-5.3-flash",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k3",
    "llama-4-maverick",
    "mimo-v2.5-pro",
    "minimax-m2.5",
    "mistral-3-14B",
    "nemotron-3-nano-omni",
    "nemotron-3-ultra-550b",
    "nemotron-nano-12b-v2-vl",
    "nvidia-nemotron-3-super-120b",
    "openai-gpt-oss-20b",
    "openai-gpt-oss-120b",
    "qwen3.5-397b-a17b",
    "qwen3.8-max",
)
APPROVED_ELIGIBLE_MODEL_IDS = tuple(
    model_id
    for model_id in CATALOG_SNAPSHOT_MODEL_IDS
    if model_id != "arcee-trinity-large-thinking"
)
UNAVAILABLE_CANDIDATE_MODELS = {
    "arcee-trinity-large-thinking": (
        "DigitalOcean returned: this model is not available for your subscription tier"
    )
}
_APPROVED_MODEL_ID_SET = frozenset(APPROVED_ELIGIBLE_MODEL_IDS)
_CATALOG_MODEL_ID_SET = frozenset(CATALOG_SNAPSHOT_MODEL_IDS)

PositiveTokenCount = Annotated[int, Field(gt=0)]


class DigitalOceanModelSourceRecord(StrictModel):
    created: Annotated[int, Field(gt=0)]
    id: str
    object: Literal["model"]
    owned_by: Literal["digitalocean"]
    context_length: PositiveTokenCount | None = None
    max_output_tokens: PositiveTokenCount | None = None

    @model_validator(mode="after")
    def validate_approved_id(self) -> "DigitalOceanModelSourceRecord":
        _require_catalog_model_id(self.id)
        return self


class ModelMetadata(StrictModel):
    schema_version: Literal["model_metadata.v1"]
    catalog_version: str
    model_id: str
    owned_by: Literal["digitalocean"]
    listed_in_source_snapshot: Literal[True]
    eligibility_status: Literal["eligible"]
    open_weight_status: Literal["approved_by_adr_0002"]
    supported_task: Literal["text_generation"]
    context_length: PositiveTokenCount | None
    max_output_tokens: PositiveTokenCount | None
    observed_chat_completions_compatibility: Literal["not_tested"]
    family: None = None
    reasoning_characteristic: None = None
    parameter_summary: None = None

    @model_validator(mode="after")
    def validate_approved_id(self) -> "ModelMetadata":
        _require_catalog_model_id(self.model_id)
        return self


class ModelCatalogManifest(StrictModel):
    schema_version: Literal["model_catalog_manifest.v1"]
    catalog_version: str
    source_snapshot_id: str
    source_snapshot_date: date
    source_endpoint_url: str
    source_artifact_file: str
    source_artifact_sha256: Sha256
    artifact_file: str
    artifact_sha256: Sha256
    content_sha256: Sha256
    model_count: Literal[26]
    ordered_model_ids: tuple[str, ...]
    selection_policy_reference: Literal["docs/adr/0002-eligible-model-pool.md"]
    serialization: Literal["utf-8-json-lines-sorted-keys-compact-lf"]

    @model_validator(mode="after")
    def validate_pool(self) -> "ModelCatalogManifest":
        if self.ordered_model_ids != CATALOG_SNAPSHOT_MODEL_IDS:
            raise ValueError("Model Catalog must preserve the discovered 26 IDs in ADR order")
        _require_local_filename(self.source_artifact_file)
        _require_local_filename(self.artifact_file)
        return self


class RateAvailability(StrEnum):
    PUBLISHED = "published"
    NOT_PUBLISHED = "not_published"


class TokenRate(StrictModel):
    category: Literal[
        "standard_input", "output", "cache_read_input", "cache_write_input"
    ]
    availability: RateAvailability
    rate_usd: Decimal | None
    currency: Literal["USD"]
    unit_tokens: Literal[1_000_000]

    @model_validator(mode="after")
    def validate_rate(self) -> "TokenRate":
        if self.availability is RateAvailability.PUBLISHED:
            if self.rate_usd is None or self.rate_usd <= 0:
                raise ValueError("Published token rate must be greater than zero")
        elif self.rate_usd is not None:
            raise ValueError("An unpublished token rate must remain null, not zero")
        return self


class ModelPricing(StrictModel):
    schema_version: Literal["model_pricing.v1"]
    pricing_snapshot_id: str
    model_id: str
    published_catalog_name: str
    published_model_reference_url: str
    rates: tuple[TokenRate, ...]

    @model_validator(mode="after")
    def validate_model_and_categories(self) -> "ModelPricing":
        _require_catalog_model_id(self.model_id)
        expected_categories = (
            "standard_input",
            "output",
            "cache_read_input",
            "cache_write_input",
        )
        if tuple(rate.category for rate in self.rates) != expected_categories:
            raise ValueError("Pricing must record all four token categories in order")
        return self


class PricingSnapshotManifest(StrictModel):
    schema_version: Literal["pricing_snapshot_manifest.v1"]
    pricing_snapshot_id: str
    model_catalog_version: str
    model_catalog_sha256: Sha256
    source_url: str
    source_last_updated: date
    source_retrieved_at: datetime
    source_content_sha256: Sha256
    artifact_file: str
    artifact_sha256: Sha256
    content_sha256: Sha256
    price_entry_count: Literal[26]
    ordered_model_ids: tuple[str, ...]
    currency: Literal["USD"]
    unit_tokens: Literal[1_000_000]
    serialization: Literal["utf-8-json-lines-sorted-keys-compact-lf"]

    @model_validator(mode="after")
    def validate_snapshot(self) -> "PricingSnapshotManifest":
        if self.ordered_model_ids != CATALOG_SNAPSHOT_MODEL_IDS:
            raise ValueError("Pricing Snapshot must preserve the discovered 26 IDs in ADR order")
        if self.source_retrieved_at.tzinfo is None:
            raise ValueError("Pricing retrieval timestamp must be timezone-aware")
        _require_local_filename(self.artifact_file)
        return self


def _require_approved_model_id(model_id: str) -> None:
    if model_id not in _APPROVED_MODEL_ID_SET:
        raise ValueError(f"Model is not in the approved Eligible Candidate Pool: {model_id}")


def _require_catalog_model_id(model_id: str) -> None:
    if model_id not in _CATALOG_MODEL_ID_SET:
        raise ValueError(f"Model is not in the frozen Model Catalog: {model_id}")


def _require_local_filename(filename: str) -> None:
    if PurePosixPath(filename).name != filename:
        raise ValueError("Artifact reference must be a local filename")
