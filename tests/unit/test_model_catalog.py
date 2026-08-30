from __future__ import annotations

import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferencebench.artifacts import ArtifactIntegrityError, FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.models.artifacts import (
    ModelCatalogArtifacts,
    PricingArtifacts,
    assert_run_uses_model_sources,
)
from inferencebench.models.domain import (
    APPROVED_ELIGIBLE_MODEL_IDS,
    CATALOG_SNAPSHOT_MODEL_IDS,
    ModelMetadata,
    RateAvailability,
    TokenRate,
)
from inferencebench.workflows.candidates import load_candidate_catalog


CATALOG_DIRECTORY = Path("artifacts/model_catalog/v1")
PRICING_DIRECTORY = Path("artifacts/pricing/v1")


def test_frozen_catalog_and_pricing_cover_exact_approved_pool() -> None:
    catalog, models, source_records = ModelCatalogArtifacts(CATALOG_DIRECTORY).load()
    pricing, price_entries = PricingArtifacts(PRICING_DIRECTORY).load_for_catalog(
        catalog
    )

    assert tuple(model.model_id for model in models) == CATALOG_SNAPSHOT_MODEL_IDS
    assert tuple(record.id for record in source_records) == CATALOG_SNAPSHOT_MODEL_IDS
    assert tuple(entry.model_id for entry in price_entries) == CATALOG_SNAPSHOT_MODEL_IDS
    assert catalog.model_count == pricing.price_entry_count == 26
    assert catalog.source_snapshot_id == "user-supplied-do-models-2026-08-29"
    assert catalog.content_sha256 == pricing.model_catalog_sha256

    models_by_id = {model.model_id: model for model in models}
    assert models_by_id["kimi-k3"].context_length is None
    assert models_by_id["kimi-k3"].max_output_tokens is None
    assert models_by_id["mistral-3-14B"].max_output_tokens is None
    assert all(model.family is None for model in models)
    assert all(model.reasoning_characteristic is None for model in models)
    assert all(model.parameter_summary is None for model in models)

    prices_by_id = {entry.model_id: entry for entry in price_entries}
    llama_rates = {rate.category: rate for rate in prices_by_id["llama-4-maverick"].rates}
    assert llama_rates["standard_input"].rate_usd == Decimal("0.20")
    assert llama_rates["cache_read_input"].rate_usd is None
    assert (
        llama_rates["cache_read_input"].availability
        is RateAvailability.NOT_PUBLISHED
    )
    assert all(
        rate.unit_tokens == 1_000_000
        for entry in price_entries
        for rate in entry.rates
    )


@pytest.mark.parametrize(
    "ineligible_model_id",
    [
        "openai-gpt-4o",
        "all-mini-lm-l6-v2",
        "router:general",
        "stable-diffusion-3.5-large",
    ],
)
def test_model_metadata_rejects_entries_outside_approved_pool(
    ineligible_model_id: str,
) -> None:
    with pytest.raises(ValidationError, match="not in the frozen Model Catalog"):
        ModelMetadata.model_validate(_model_payload(ineligible_model_id))


def test_model_metadata_rejects_invented_family_fact() -> None:
    payload = _model_payload(APPROVED_ELIGIBLE_MODEL_IDS[0])
    payload["family"] = "Invented from model name"

    with pytest.raises(ValidationError):
        ModelMetadata.model_validate(payload)


def test_token_rate_rejects_ambiguous_units_and_unknown_as_zero() -> None:
    with pytest.raises(ValidationError):
        TokenRate(
            category="standard_input",
            availability="published",
            rate_usd=Decimal("0.25"),
            currency="USD",
            unit_tokens=1_000,
        )
    with pytest.raises(ValidationError, match="must remain null"):
        TokenRate(
            category="cache_read_input",
            availability="not_published",
            rate_usd=Decimal("0"),
            currency="USD",
            unit_tokens=1_000_000,
        )


def test_rehashed_ineligible_replacement_is_still_rejected(tmp_path: Path) -> None:
    copied_catalog = tmp_path / "catalog"
    shutil.copytree(CATALOG_DIRECTORY, copied_catalog)
    model_path = copied_catalog / "models.jsonl"
    lines = model_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["model_id"] = "openai-gpt-4o"
    lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
    replacement_bytes = ("\n".join(lines) + "\n").encode()
    model_path.write_bytes(replacement_bytes)
    manifest_path = copied_catalog / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = hashlib.sha256(replacement_bytes).hexdigest()
    manifest_without_content_hash = {
        key: value for key, value in manifest.items() if key != "content_sha256"
    }
    manifest["content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest_without_content_hash, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="not in the frozen Model Catalog"):
        ModelCatalogArtifacts(copied_catalog).load()


def test_run_references_make_catalog_and_pricing_mismatch_visible() -> None:
    catalog, _, _ = ModelCatalogArtifacts(CATALOG_DIRECTORY).load()
    pricing, _ = PricingArtifacts(PRICING_DIRECTORY).load_for_catalog(catalog)
    fixture_run = FixtureArtifacts(Path("artifacts/fixtures")).load_run_bundle().runs[0]
    run = fixture_run.model_copy(
        update={
            "model_id": APPROVED_ELIGIBLE_MODEL_IDS[0],
            "model_catalog_version": catalog.catalog_version,
            "model_catalog_sha256": catalog.content_sha256,
            "pricing_snapshot_id": pricing.pricing_snapshot_id,
            "pricing_snapshot_sha256": pricing.content_sha256,
        }
    )
    assert_run_uses_model_sources(run, catalog, pricing)

    mismatched = run.model_copy(update={"pricing_snapshot_sha256": "f" * 64})
    with pytest.raises(ArtifactIntegrityError, match="different Pricing Snapshot"):
        assert_run_uses_model_sources(mismatched, catalog, pricing)


def test_pricing_provenance_change_requires_a_new_content_hash(
    tmp_path: Path,
) -> None:
    copied_pricing = tmp_path / "pricing"
    shutil.copytree(PRICING_DIRECTORY, copied_pricing)
    manifest_path = copied_pricing / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_url"] = "https://example.invalid/replaced-pricing-source"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    catalog, _, _ = ModelCatalogArtifacts(CATALOG_DIRECTORY).load()
    with pytest.raises(ArtifactIntegrityError, match="Manifest content hash mismatch"):
        PricingArtifacts(copied_pricing).load_for_catalog(catalog)


def test_candidate_workflow_uses_same_frozen_pool_as_selectors(tmp_path: Path) -> None:
    review = load_candidate_catalog(Settings.for_test(tmp_path / "evidence.sqlite3"))

    assert review.live_catalog_requests_made == 0
    assert review.model_ids == APPROVED_ELIGIBLE_MODEL_IDS
    assert len(review.candidates) == 25
    assert review.excluded_models == {
        "arcee-trinity-large-thinking": (
            "DigitalOcean returned: this model is not available for your subscription tier"
        )
    }


def _model_payload(model_id: str) -> dict[str, object]:
    return {
        "schema_version": "model_metadata.v1",
        "catalog_version": "eligible-candidates-2026-08-29",
        "model_id": model_id,
        "owned_by": "digitalocean",
        "listed_in_source_snapshot": True,
        "eligibility_status": "eligible",
        "open_weight_status": "approved_by_adr_0002",
        "supported_task": "text_generation",
        "context_length": 128_000,
        "max_output_tokens": 4_096,
        "observed_chat_completions_compatibility": "not_tested",
        "family": None,
        "reasoning_characteristic": None,
        "parameter_summary": None,
    }
