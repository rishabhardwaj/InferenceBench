from __future__ import annotations

import pytest

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.domain import RunStatus
from inferencebench.persistence.repository import (
    EvidenceConflictError,
    EvidenceRepository,
)


def test_database_initializes_migrates_and_round_trips_fixture(settings: Settings) -> None:
    bundle = FixtureArtifacts(settings.fixture_root).load_run_bundle()
    repository = EvidenceRepository(settings.database_path)

    repository.initialize()
    inserted = repository.seed_fixture_bundle(bundle)

    assert inserted is True
    assert repository.migration_versions() == (
        "0001_initial.sql",
        "0002_generated_attempt_outcomes.sql",
    )
    assert repository.get_run(bundle.runs[0].run_id) == bundle.runs[0]
    assert repository.get_attempts(bundle.runs[0].run_id) == tuple(
        attempt
        for attempt in bundle.attempts
        if attempt.run_id == bundle.runs[0].run_id
    )
    assert repository.seed_fixture_bundle(bundle) is False


def test_duplicate_canonical_benchmark_attempt_is_rejected(settings: Settings) -> None:
    bundle = FixtureArtifacts(settings.fixture_root).load_run_bundle()
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    running = bundle.runs[0].model_copy(
        update={
            "run_id": "running-duplicate-test",
            "status": RunStatus.RUNNING,
            "ended_at": None,
            "wall_clock_ms": None,
            "persisted_count": 0,
            "usable_count": 0,
        }
    )
    repository.insert_run(running)
    attempt = bundle.attempts[0].model_copy(update={"run_id": running.run_id})
    repository.insert_attempt(attempt)
    duplicate = attempt.model_copy(update={"attempt_id": "another-id"})

    with pytest.raises(EvidenceConflictError, match="UNIQUE constraint failed"):
        repository.insert_attempt(duplicate)


def test_terminal_run_and_attempt_evidence_are_immutable(settings: Settings) -> None:
    bundle = FixtureArtifacts(settings.fixture_root).load_run_bundle()
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    repository.seed_fixture_bundle(bundle)

    with pytest.raises(EvidenceConflictError, match="terminal Run Manifest"):
        repository.insert_attempt(bundle.attempts[0])

    with pytest.raises(EvidenceConflictError, match="immutable"):
        repository.finalize_run(bundle.runs[0])


def test_attempt_without_run_is_rejected(settings: Settings) -> None:
    bundle = FixtureArtifacts(settings.fixture_root).load_run_bundle()
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()

    with pytest.raises(EvidenceConflictError, match="FOREIGN KEY constraint failed"):
        repository.insert_attempt(bundle.attempts[0])
