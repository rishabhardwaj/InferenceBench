from __future__ import annotations

from pathlib import Path

from pydantic import model_validator

from inferencebench.config import Settings
from inferencebench.domain import StrictModel
from inferencebench.persistence.repository import EvidenceNotFoundError, EvidenceRepository
from inferencebench.workflows.saved_comparison import (
    SavedComparisonReview,
    load_persisted_comparison,
)


class SelectionDecisionRecord(StrictModel):
    """The human engineering decision over persisted candidate and final-run evidence."""

    schema_version: str
    recommended_model_id: str
    decision_challenger_model_id: str
    recommended_run_id: str
    decision_challenger_run_id: str
    concurrency: int
    candidate_evidence_sha256: str
    candidate_dispositions_sha256: str
    recommendation_rationale: str
    challenger_preference_condition: str
    evidence_limits: tuple[str, ...]

    @model_validator(mode="after")
    def validate_distinct_models_and_runs(self) -> "SelectionDecisionRecord":
        if self.recommended_model_id == self.decision_challenger_model_id:
            raise ValueError("Selection Decision Record models must be distinct")
        if self.recommended_run_id == self.decision_challenger_run_id:
            raise ValueError("Selection Decision Record runs must be distinct")
        return self


def selection_decision_path(settings: Settings) -> Path:
    return settings.project_root / "artifacts" / "results" / "selection-decision.json"


def load_selection_decision(settings: Settings) -> SelectionDecisionRecord | None:
    """Return the recorded decision only when its two runs exist in this evidence DB."""

    path = selection_decision_path(settings)
    if not path.exists():
        return None
    record = SelectionDecisionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    try:
        recommended = repository.get_run(record.recommended_run_id)
        challenger = repository.get_run(record.decision_challenger_run_id)
    except EvidenceNotFoundError:
        return None
    if recommended.model_id != record.recommended_model_id:
        raise ValueError("Selection Decision Record recommended model disagrees with Run Manifest")
    if challenger.model_id != record.decision_challenger_model_id:
        raise ValueError("Selection Decision Record challenger disagrees with Run Manifest")
    if recommended.concurrency != record.concurrency or challenger.concurrency != record.concurrency:
        raise ValueError("Selection Decision Record concurrency disagrees with Run Manifest")
    return record


def load_recommended_comparison(
    settings: Settings, record: SelectionDecisionRecord
) -> SavedComparisonReview:
    return load_persisted_comparison(
        settings,
        (record.recommended_run_id, record.decision_challenger_run_id),
    )
