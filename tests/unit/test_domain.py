from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.domain import AttemptEvidence, RunManifest


def test_run_and_attempt_round_trip_through_json() -> None:
    bundle = FixtureArtifacts(Settings.from_environment().fixture_root).load_run_bundle()

    run = bundle.runs[0]
    attempt = bundle.attempts[0]

    assert RunManifest.model_validate_json(run.model_dump_json()) == run
    assert AttemptEvidence.model_validate_json(attempt.model_dump_json()) == attempt


def test_run_manifest_forbids_authorization_field() -> None:
    bundle = FixtureArtifacts(Settings.from_environment().fixture_root).load_run_bundle()
    payload = bundle.runs[0].model_dump(mode="json")
    payload["authorization"] = "Bearer secret"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunManifest.model_validate(payload)


def test_attempt_evidence_rejects_sensitive_response_headers() -> None:
    bundle = FixtureArtifacts(Settings.from_environment().fixture_root).load_run_bundle()
    payload = bundle.attempts[0].model_dump(mode="json")
    payload["response_headers"]["Authorization"] = "Bearer secret"

    with pytest.raises(ValidationError, match="sensitive headers are not evidence"):
        AttemptEvidence.model_validate(payload)

