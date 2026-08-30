from __future__ import annotations

import shutil
from pathlib import Path

from inferencebench.config import Settings
from inferencebench.persistence.repository import EvidenceRepository
from inferencebench.workflows.recommendation import load_selection_decision


def test_selection_record_is_ignored_when_its_runs_are_not_in_this_database(
    tmp_path: Path,
) -> None:
    settings = Settings.for_test(tmp_path / "empty.sqlite3")

    # The project-level record identifies submitted full-Corpus evidence, while a
    # fresh reviewer/test DB should continue to use the credential-free fixture.
    assert load_selection_decision(settings) is None


def test_selection_record_is_loaded_when_both_runs_exist(tmp_path: Path) -> None:
    source = Path("artifacts/results/fixture.sqlite3")
    database_path = tmp_path / "evidence.sqlite3"
    shutil.copy2(source, database_path)
    settings = Settings.for_test(database_path)

    record = load_selection_decision(settings)

    assert record is not None
    assert record.recommended_model_id == "mistral-3-14B"
    assert record.decision_challenger_model_id == "deepseek-v4-flash-0731"
    assert EvidenceRepository(database_path).get_run(record.recommended_run_id).status.value == "complete"
