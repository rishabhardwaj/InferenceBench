from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from inferencebench.domain import (
    AttemptEvidence,
    AttemptPurpose,
    FixtureRunBundle,
    ParseStatus,
    ProviderOutcome,
    RunManifest,
    RunStatus,
)


_RUN_LIFECYCLE_FIELDS = {
    "status",
    "ended_at",
    "wall_clock_ms",
    "persisted_count",
    "usable_count",
    "normalized_count",
    "invalid_output_count",
    "request_error_count",
}


class EvidenceConflictError(RuntimeError):
    """Raised when immutable or uniquely constrained evidence conflicts."""


class EvidenceNotFoundError(LookupError):
    """Raised when requested evidence does not exist."""


class EvidenceRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.migrations_path = Path(__file__).parent / "migrations"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for migration in sorted(self.migrations_path.glob("*.sql")):
                if migration.name in applied:
                    continue
                connection.executescript(migration.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)",
                    (migration.name,),
                )
            connection.commit()

    def migration_versions(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        return tuple(row["version"] for row in rows)

    def insert_run(self, run: RunManifest) -> None:
        with self._connect() as connection:
            try:
                self._insert_run(connection, run)
                connection.commit()
            except sqlite3.IntegrityError as error:
                raise EvidenceConflictError(str(error)) from error

    def insert_attempt(self, attempt: AttemptEvidence) -> None:
        with self._connect() as connection:
            parent = connection.execute(
                "SELECT status FROM run_manifests WHERE run_id = ?",
                (attempt.run_id,),
            ).fetchone()
            if parent is not None and parent["status"] != RunStatus.RUNNING.value:
                raise EvidenceConflictError(
                    "Attempt Evidence cannot be added to a terminal Run Manifest"
                )
            try:
                self._insert_attempt(connection, attempt)
                connection.commit()
            except sqlite3.IntegrityError as error:
                raise EvidenceConflictError(str(error)) from error

    def finalize_run(self, terminal_run: RunManifest) -> None:
        if terminal_run.status is RunStatus.RUNNING:
            raise EvidenceConflictError("finalize_run requires a terminal Run Manifest")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM run_manifests WHERE run_id = ?",
                (terminal_run.run_id,),
            ).fetchone()
            if row is None:
                raise EvidenceNotFoundError(
                    f"Run Manifest not found: {terminal_run.run_id}"
                )
            current = RunManifest.model_validate_json(row["manifest_json"])
            if current.status is not RunStatus.RUNNING:
                raise EvidenceConflictError("Terminal Run Manifest is immutable")
            if _without_lifecycle(current) != _without_lifecycle(terminal_run):
                raise EvidenceConflictError(
                    "Run configuration cannot change during finalization"
                )
            attempts = tuple(
                AttemptEvidence.model_validate_json(attempt_row["evidence_json"])
                for attempt_row in connection.execute(
                    """
                    SELECT evidence_json
                    FROM attempt_evidence
                    WHERE run_id = ? AND attempt_purpose = ?
                    ORDER BY dispatch_order, attempt_id
                    """,
                    (terminal_run.run_id, AttemptPurpose.BENCHMARK.value),
                )
            )
            _validate_terminal_counts_and_population(terminal_run, attempts)
            cursor = connection.execute(
                """
                UPDATE run_manifests
                SET status = ?, manifest_json = ?
                WHERE run_id = ? AND status = ?
                """,
                (
                    terminal_run.status.value,
                    terminal_run.model_dump_json(),
                    terminal_run.run_id,
                    RunStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise EvidenceConflictError("Run Manifest terminal transition conflicted")
            connection.commit()

    def seed_fixture_bundle(self, bundle: FixtureRunBundle) -> bool:
        """Insert the complete fixture atomically, or verify the existing immutable copy."""

        with self._connect() as connection:
            existing_runs = {
                row["run_id"]: row["manifest_json"]
                for row in connection.execute(
                    "SELECT run_id, manifest_json FROM run_manifests WHERE run_id IN (?, ?)",
                    bundle.default_run_ids,
                )
            }
            if existing_runs:
                if set(existing_runs) != set(bundle.default_run_ids):
                    raise EvidenceConflictError("fixture evidence is only partially present")
                existing_run_models = {
                    run_id: RunManifest.model_validate_json(payload)
                    for run_id, payload in existing_runs.items()
                }
                expected_runs = {run.run_id: run for run in bundle.runs}
                if existing_run_models != expected_runs:
                    raise EvidenceConflictError("persisted fixture Run Manifest has changed")

                rows = connection.execute(
                    """
                    SELECT attempt_id, evidence_json
                    FROM attempt_evidence
                    WHERE run_id IN (?, ?)
                    """,
                    bundle.default_run_ids,
                ).fetchall()
                existing_attempts = {
                    row["attempt_id"]: AttemptEvidence.model_validate_json(
                        row["evidence_json"]
                    )
                    for row in rows
                }
                expected_attempts = {
                    attempt.attempt_id: attempt
                    for attempt in bundle.attempts
                }
                if existing_attempts != expected_attempts:
                    raise EvidenceConflictError("persisted fixture Attempt Evidence has changed")
                return False

            try:
                for run in bundle.runs:
                    self._insert_run(connection, run)
                for attempt in bundle.attempts:
                    self._insert_attempt(connection, attempt)
                connection.commit()
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise EvidenceConflictError(str(error)) from error
        return True

    def get_run(self, run_id: str) -> RunManifest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM run_manifests WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise EvidenceNotFoundError(f"Run Manifest not found: {run_id}")
        return RunManifest.model_validate_json(row["manifest_json"])

    def get_attempts(self, run_id: str) -> tuple[AttemptEvidence, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT evidence_json
                FROM attempt_evidence
                WHERE run_id = ?
                ORDER BY dispatch_order, attempt_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(
            AttemptEvidence.model_validate_json(row["evidence_json"]) for row in rows
        )

    @staticmethod
    def _insert_run(connection: sqlite3.Connection, run: RunManifest) -> None:
        connection.execute(
            """
            INSERT INTO run_manifests (
                run_id,
                model_id,
                status,
                corpus_version,
                corpus_sha256,
                prompt_version,
                parser_version,
                generation_configuration_sha256,
                expected_count,
                manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.model_id,
                run.status.value,
                run.corpus_version,
                run.corpus_sha256,
                run.prompt_version,
                run.parser_version,
                run.generation_configuration_sha256,
                run.expected_count,
                run.model_dump_json(),
            ),
        )

    @staticmethod
    def _insert_attempt(
        connection: sqlite3.Connection, attempt: AttemptEvidence
    ) -> None:
        connection.execute(
            """
            INSERT INTO attempt_evidence (
                attempt_id,
                run_id,
                issue_number,
                attempt_purpose,
                dispatch_order,
                provider_outcome,
                scored_outcome,
                parsed_label,
                usable,
                evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt.attempt_id,
                attempt.run_id,
                attempt.issue_number,
                attempt.attempt_purpose.value,
                attempt.dispatch_order,
                attempt.provider_outcome.value if attempt.provider_outcome else None,
                attempt.scored_outcome.value if attempt.scored_outcome else None,
                attempt.parsed_label.value if attempt.parsed_label else None,
                int(attempt.usable),
                attempt.model_dump_json(),
            ),
        )


def _without_lifecycle(run: RunManifest) -> dict[str, object]:
    return run.model_dump(mode="json", exclude=_RUN_LIFECYCLE_FIELDS)


def _validate_terminal_counts_and_population(
    run: RunManifest, attempts: tuple[AttemptEvidence, ...]
) -> None:
    if any(attempt.run_id != run.run_id for attempt in attempts):
        raise EvidenceConflictError("Attempt Evidence references a different run")
    dispatch_orders = tuple(attempt.dispatch_order for attempt in attempts)
    if len(set(dispatch_orders)) != len(dispatch_orders):
        raise EvidenceConflictError("Benchmark dispatch order must be unique")
    for attempt in attempts:
        if attempt.dispatch_order >= run.expected_count:
            raise EvidenceConflictError("Benchmark dispatch order is outside the run")
        if run.ordered_issue_numbers[attempt.dispatch_order] != attempt.issue_number:
            raise EvidenceConflictError(
                "Attempt issue does not match the frozen run dispatch order"
            )
        if run.run_type == "model_evaluation" and attempt.provider_outcome is None:
            raise EvidenceConflictError(
                "Generated Attempt Evidence is missing Provider Outcome"
            )
    if run.status is RunStatus.COMPLETE:
        issue_numbers = tuple(attempt.issue_number for attempt in attempts)
        if issue_numbers != run.ordered_issue_numbers:
            raise EvidenceConflictError(
                "Complete run does not contain its exact ordered issue population"
            )

    usable_count = sum(attempt.usable for attempt in attempts)
    normalized_count = sum(
        attempt.parse_status is ParseStatus.NORMALIZED for attempt in attempts
    )
    invalid_output_count = sum(
        attempt.provider_outcome is ProviderOutcome.SUCCESS and not attempt.usable
        for attempt in attempts
    )
    request_error_count = sum(
        attempt.provider_outcome is not None
        and attempt.provider_outcome is not ProviderOutcome.SUCCESS
        for attempt in attempts
    )
    actual_counts = (
        len(attempts),
        usable_count,
        normalized_count,
        invalid_output_count,
        request_error_count,
    )
    manifest_counts = (
        run.persisted_count,
        run.usable_count,
        run.normalized_count,
        run.invalid_output_count,
        run.request_error_count,
    )
    if actual_counts != manifest_counts:
        raise EvidenceConflictError(
            "Run terminal counts do not match persisted Attempt Evidence"
        )
