from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from inferencebench.artifacts import ArtifactIntegrityError, sha256_file
from inferencebench.domain import RunManifest
from inferencebench.models.domain import (
    CATALOG_SNAPSHOT_MODEL_IDS,
    DigitalOceanModelSourceRecord,
    ModelCatalogManifest,
    ModelMetadata,
    ModelPricing,
    PricingSnapshotManifest,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


class ModelCatalogArtifacts:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(
        self,
    ) -> tuple[
        ModelCatalogManifest,
        tuple[ModelMetadata, ...],
        tuple[DigitalOceanModelSourceRecord, ...],
    ]:
        manifest = _read_json(
            self.directory / "manifest.json", ModelCatalogManifest
        )
        _verify_manifest_content_hash(manifest)
        source_path = self.directory / manifest.source_artifact_file
        model_path = self.directory / manifest.artifact_file
        _verify_hash(source_path, manifest.source_artifact_sha256)
        _verify_hash(model_path, manifest.artifact_sha256)
        source_records = _read_jsonl(source_path, DigitalOceanModelSourceRecord)
        models = _read_jsonl(model_path, ModelMetadata)
        _validate_order(
            tuple(record.id for record in source_records), "source model records"
        )
        _validate_order(tuple(model.model_id for model in models), "Model Catalog")
        if len(models) != manifest.model_count:
            raise ArtifactIntegrityError("Model count does not match manifest")
        if tuple(model.model_id for model in models) != manifest.ordered_model_ids:
            raise ArtifactIntegrityError("Model order does not match manifest")
        sources_by_id = {record.id: record for record in source_records}
        for model in models:
            source = sources_by_id[model.model_id]
            if (
                model.owned_by != source.owned_by
                or model.context_length != source.context_length
                or model.max_output_tokens != source.max_output_tokens
            ):
                raise ArtifactIntegrityError(
                    f"Model metadata is not supported by source record: {model.model_id}"
                )
        return manifest, models, source_records


class PricingArtifacts:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load_for_catalog(
        self, catalog_manifest: ModelCatalogManifest
    ) -> tuple[PricingSnapshotManifest, tuple[ModelPricing, ...]]:
        manifest = _read_json(
            self.directory / "manifest.json", PricingSnapshotManifest
        )
        _verify_manifest_content_hash(manifest)
        artifact_path = self.directory / manifest.artifact_file
        _verify_hash(artifact_path, manifest.artifact_sha256)
        entries = _read_jsonl(artifact_path, ModelPricing)
        entry_ids = tuple(entry.model_id for entry in entries)
        _validate_order(entry_ids, "Pricing Snapshot")
        if entry_ids != manifest.ordered_model_ids:
            raise ArtifactIntegrityError("Pricing entry order does not match manifest")
        if len(entries) != manifest.price_entry_count:
            raise ArtifactIntegrityError("Pricing entry count does not match manifest")
        if (
            manifest.model_catalog_version != catalog_manifest.catalog_version
            or manifest.model_catalog_sha256 != catalog_manifest.content_sha256
        ):
            raise ArtifactIntegrityError(
                "Pricing Snapshot references a different Model Catalog"
            )
        return manifest, entries


def assert_run_uses_model_sources(
    run: RunManifest,
    catalog_manifest: ModelCatalogManifest,
    pricing_manifest: PricingSnapshotManifest,
) -> None:
    if run.model_id not in catalog_manifest.ordered_model_ids:
        raise ArtifactIntegrityError(
            f"Run model is not in its referenced Eligible Candidate Pool: {run.model_id}"
        )
    if (
        run.model_catalog_version != catalog_manifest.catalog_version
        or run.model_catalog_sha256 != catalog_manifest.content_sha256
    ):
        raise ArtifactIntegrityError("Run references a different Model Catalog")
    if (
        run.pricing_snapshot_id != pricing_manifest.pricing_snapshot_id
        or run.pricing_snapshot_sha256 != pricing_manifest.content_sha256
    ):
        raise ArtifactIntegrityError("Run references a different Pricing Snapshot")


def _read_json(path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ArtifactIntegrityError(f"Invalid artifact {path}: {error}") from error


def _read_jsonl(path: Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            records.append(model_type.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ArtifactIntegrityError(f"{path}:{line_number}: {error}") from error
    return tuple(records)


def _verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"Artifact hash mismatch for {path}: expected {expected}, got {actual}"
        )


def _verify_manifest_content_hash(
    manifest: ModelCatalogManifest | PricingSnapshotManifest,
) -> None:
    payload = manifest.model_dump(mode="json", exclude={"content_sha256"})
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != manifest.content_sha256:
        raise ArtifactIntegrityError(
            f"Manifest content hash mismatch: expected {manifest.content_sha256}, got {actual}"
        )


def _validate_order(model_ids: tuple[str, ...], artifact_name: str) -> None:
    if model_ids != CATALOG_SNAPSHOT_MODEL_IDS:
        raise ArtifactIntegrityError(
            f"{artifact_name} must preserve exactly the discovered 26 model IDs"
        )
