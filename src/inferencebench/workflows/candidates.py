from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from inferencebench.config import Settings
from inferencebench.domain import StrictModel
from inferencebench.models.artifacts import ModelCatalogArtifacts, PricingArtifacts
from inferencebench.models.domain import ModelPricing, TokenRate
from inferencebench.models.domain import (
    APPROVED_ELIGIBLE_MODEL_IDS,
    UNAVAILABLE_CANDIDATE_MODELS,
)


class CandidateModelView(StrictModel):
    model_id: str
    context_length: int | None
    max_output_tokens: int | None
    compatibility: str
    family: str | None
    reasoning_characteristic: str | None
    parameter_summary: str | None
    published_catalog_name: str
    standard_input_rate_usd: Decimal
    output_rate_usd: Decimal
    cache_read_input_rate_usd: Decimal | None
    cache_write_input_rate_usd: Decimal | None


class CandidateCatalogReview(StrictModel):
    evidence_state: str
    live_catalog_requests_made: int
    catalog_version: str
    catalog_sha256: str
    source_snapshot_id: str
    source_snapshot_date: date
    pricing_snapshot_id: str
    pricing_sha256: str
    pricing_source_url: str
    pricing_source_last_updated: date
    pricing_source_retrieved_at: datetime
    candidates: tuple[CandidateModelView, ...]
    excluded_models: dict[str, str]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(candidate.model_id for candidate in self.candidates)


def load_candidate_catalog(settings: Settings) -> CandidateCatalogReview:
    catalog_manifest, models, _ = ModelCatalogArtifacts(
        settings.model_catalog_directory
    ).load()
    pricing_manifest, pricing_entries = PricingArtifacts(
        settings.pricing_directory
    ).load_for_catalog(catalog_manifest)
    prices_by_model = {entry.model_id: entry for entry in pricing_entries}
    models_by_id = {model.model_id: model for model in models}
    candidates = tuple(
        _candidate_view(models_by_id[model_id], prices_by_model[model_id])
        for model_id in APPROVED_ELIGIBLE_MODEL_IDS
    )
    return CandidateCatalogReview(
        evidence_state="Frozen model and pricing evidence — no live catalog request",
        live_catalog_requests_made=0,
        catalog_version=catalog_manifest.catalog_version,
        catalog_sha256=catalog_manifest.content_sha256,
        source_snapshot_id=catalog_manifest.source_snapshot_id,
        source_snapshot_date=catalog_manifest.source_snapshot_date,
        pricing_snapshot_id=pricing_manifest.pricing_snapshot_id,
        pricing_sha256=pricing_manifest.content_sha256,
        pricing_source_url=pricing_manifest.source_url,
        pricing_source_last_updated=pricing_manifest.source_last_updated,
        pricing_source_retrieved_at=pricing_manifest.source_retrieved_at,
        candidates=candidates,
        excluded_models=UNAVAILABLE_CANDIDATE_MODELS,
    )


def _candidate_view(model, pricing: ModelPricing) -> CandidateModelView:
    rates = {rate.category: rate for rate in pricing.rates}
    return CandidateModelView(
        model_id=model.model_id,
        context_length=model.context_length,
        max_output_tokens=model.max_output_tokens,
        compatibility=model.observed_chat_completions_compatibility,
        family=model.family,
        reasoning_characteristic=model.reasoning_characteristic,
        parameter_summary=model.parameter_summary,
        published_catalog_name=pricing.published_catalog_name,
        standard_input_rate_usd=_published_rate(rates["standard_input"]),
        output_rate_usd=_published_rate(rates["output"]),
        cache_read_input_rate_usd=rates["cache_read_input"].rate_usd,
        cache_write_input_rate_usd=rates["cache_write_input"].rate_usd,
    )


def _published_rate(rate: TokenRate) -> Decimal:
    if rate.rate_usd is None:
        raise ValueError(f"Required published rate is missing: {rate.category}")
    return rate.rate_usd
