from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("INFERENCEBENCH_PROJECT_ROOT", Path.cwd())).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime paths only; benchmark semantics live in versioned evidence."""

    project_root: Path
    database_path: Path
    fixture_root: Path
    corpus_root: Path
    model_catalog_directory: Path
    pricing_directory: Path
    ground_truth_root: Path
    evaluation_ground_truth_version: str
    shared_contract_directory: Path
    live_default_concurrency: int
    shared_timeout_seconds: float
    human_review_version: str | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        database_path = Path(
            os.environ.get(
                "INFERENCEBENCH_DB_PATH",
                PROJECT_ROOT / "artifacts" / "results" / "fixture.sqlite3",
            )
        ).expanduser()
        return cls(
            project_root=PROJECT_ROOT,
            database_path=database_path,
            fixture_root=PROJECT_ROOT / "artifacts" / "fixtures",
            corpus_root=Path(
                os.environ.get(
                    "INFERENCEBENCH_CORPUS_ROOT", PROJECT_ROOT / "artifacts" / "corpus"
                )
            ).expanduser(),
            model_catalog_directory=Path(
                os.environ.get(
                    "INFERENCEBENCH_MODEL_CATALOG_PATH",
                    PROJECT_ROOT / "artifacts" / "model_catalog" / "v1",
                )
            ).expanduser(),
            pricing_directory=Path(
                os.environ.get(
                    "INFERENCEBENCH_PRICING_PATH",
                    PROJECT_ROOT / "artifacts" / "pricing" / "v1",
                )
            ).expanduser(),
            ground_truth_root=Path(
                os.environ.get(
                    "INFERENCEBENCH_GROUND_TRUTH_ROOT",
                    PROJECT_ROOT / "artifacts" / "ground_truth",
                )
            ).expanduser(),
            evaluation_ground_truth_version=os.environ.get(
                "INFERENCEBENCH_EVALUATION_VERSION", "doctl-evaluation-v1"
            ),
            shared_contract_directory=Path(
                os.environ.get(
                    "INFERENCEBENCH_CONTRACT_PATH",
                    PROJECT_ROOT / "artifacts" / "prompts" / "frozen-v1-timeboxed",
                )
            ).expanduser(),
            live_default_concurrency=_positive_int_environment(
                "INFERENCEBENCH_DEFAULT_CONCURRENCY", 1
            ),
            shared_timeout_seconds=_positive_float_environment(
                "INFERENCEBENCH_SHARED_TIMEOUT_SECONDS", 30.0
            ),
            human_review_version=os.environ.get("INFERENCEBENCH_HUMAN_REVIEW_VERSION"),
        )

    @classmethod
    def for_test(
        cls, database_path: Path, corpus_root: Path | None = None
    ) -> "Settings":
        return cls(
            project_root=PROJECT_ROOT,
            database_path=database_path,
            fixture_root=PROJECT_ROOT / "artifacts" / "fixtures",
            corpus_root=corpus_root or PROJECT_ROOT / "artifacts" / "corpus",
            model_catalog_directory=PROJECT_ROOT
            / "artifacts"
            / "model_catalog"
            / "v1",
            pricing_directory=PROJECT_ROOT / "artifacts" / "pricing" / "v1",
            ground_truth_root=PROJECT_ROOT / "artifacts" / "ground_truth",
            evaluation_ground_truth_version="doctl-evaluation-v1",
            shared_contract_directory=PROJECT_ROOT
            / "artifacts"
            / "prompts"
            / "development-v1",
            live_default_concurrency=1,
            shared_timeout_seconds=30.0,
            human_review_version=None,
        )


def _positive_int_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
