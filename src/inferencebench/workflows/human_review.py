from __future__ import annotations

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.config import Settings
from inferencebench.domain import StrictModel
from inferencebench.ground_truth.annotations import (
    HumanReviewArtifacts,
    HumanReviewPopulationSummary,
    prepare_review_queue,
)


class CompletedHumanReviewSummary(StrictModel):
    evidence_state: str
    ground_truth_version: str
    corpus_version: str
    reviewed_count: int
    accepted_count: int
    excluded_count: int
    prompt_development_count: int
    primary_holdout_count: int
    support_counts: dict[str, int]


def prepare_active_review_queue(settings: Settings, random_order_seed: int):
    """Prepare title/body-only review rows from the active frozen Corpus."""
    _, issues = CorpusArtifacts(settings.corpus_root).load_active()
    return prepare_review_queue(issues, random_order_seed)


def load_completed_human_review(
    settings: Settings, version: str
) -> CompletedHumanReviewSummary:
    corpus_manifest, issues = CorpusArtifacts(settings.corpus_root).load_active()
    manifest, annotations = HumanReviewArtifacts(
        settings.project_root / "artifacts" / "ground_truth"
    ).load_version(version, corpus_manifest, issues)
    population = HumanReviewPopulationSummary(annotations)
    return CompletedHumanReviewSummary(
        evidence_state="Completed Human-Reviewed Ground Truth — no provider request",
        ground_truth_version=manifest.ground_truth_version,
        corpus_version=manifest.corpus_version,
        reviewed_count=population.reviewed_count,
        accepted_count=population.accepted_count,
        excluded_count=population.excluded_count,
        prompt_development_count=population.prompt_development_count,
        primary_holdout_count=population.primary_holdout_count,
        support_counts=population.support_counts,
    )
