from __future__ import annotations

import json
from pathlib import Path

from inferencebench.config import PROJECT_ROOT
from inferencebench.evaluation.cost import calculate_request_cost
from inferencebench.models.artifacts import ModelCatalogArtifacts, PricingArtifacts


MODEL_IDS = ("openai-gpt-oss-20b", "openai-gpt-oss-120b")


def main() -> None:
    """Upgrade the small saved-comparison fixture with traceable cost evidence."""

    source_path = PROJECT_ROOT / "artifacts" / "fixtures" / "runs" / "v1" / "evidence.json"
    destination_path = (
        PROJECT_ROOT / "artifacts" / "fixtures" / "runs" / "v2" / "evidence.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    catalog, _, _ = ModelCatalogArtifacts(
        PROJECT_ROOT / "artifacts" / "model_catalog" / "v1"
    ).load()
    pricing_manifest, pricing_entries = PricingArtifacts(
        PROJECT_ROOT / "artifacts" / "pricing" / "v1"
    ).load_for_catalog(catalog)
    pricing_by_model = {entry.model_id: entry for entry in pricing_entries}

    old_to_new_run: dict[str, str] = {}
    for index, run in enumerate(source["runs"]):
        model_id = MODEL_IDS[index]
        old_run_id = run["run_id"]
        new_run_id = f"fixture-v2-run-model-{'a' if index == 0 else 'b'}"
        old_to_new_run[old_run_id] = new_run_id
        run.update(
            {
                "run_id": new_run_id,
                "model_id": model_id,
                "model_catalog_version": catalog.catalog_version,
                "model_catalog_sha256": catalog.content_sha256,
                "pricing_snapshot_id": pricing_manifest.pricing_snapshot_id,
                "pricing_snapshot_sha256": pricing_manifest.content_sha256,
                "schema_revision": 3,
                "metric_version": "classification-and-operational-metrics-v1",
            }
        )

    source["default_run_ids"] = [old_to_new_run[item] for item in source["default_run_ids"]]
    model_by_run = {
        source["runs"][index]["run_id"]: MODEL_IDS[index] for index in range(2)
    }
    for attempt in source["attempts"]:
        old_run_id = attempt["run_id"]
        new_run_id = old_to_new_run[old_run_id]
        model_id = model_by_run[new_run_id]
        suffix = "a" if model_id == MODEL_IDS[0] else "b"
        calculation = calculate_request_cost(
            attempt["usage"], pricing_by_model[model_id]
        )
        attempt.update(
            {
                "schema_version": "attempt_evidence.v3",
                "attempt_id": f"fixture-v2-attempt-model-{suffix}-{attempt['issue_number']}",
                "run_id": new_run_id,
                "provider_outcome": attempt.get("provider_outcome") or "success",
                "pricing_snapshot_id": pricing_manifest.pricing_snapshot_id,
                "cost_formula_version": calculation.formula_version,
                "calculated_request_cost_usd": (
                    str(calculation.calculated_cost_usd)
                    if calculation.calculated_cost_usd is not None
                    else None
                ),
                "cost_completeness": calculation.completeness.value,
                "cost_calculation_terms": [
                    term.model_dump(mode="json") for term in calculation.terms
                ],
                "cost_unknown_reasons": list(calculation.unknown_reasons),
            }
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(source, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(destination_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
