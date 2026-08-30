from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.for_test(tmp_path / "evidence.sqlite3")


@pytest.fixture
def fixture_artifacts() -> FixtureArtifacts:
    return FixtureArtifacts(Settings.from_environment().fixture_root)

