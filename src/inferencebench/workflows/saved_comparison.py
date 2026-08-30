from __future__ import annotations

from inferencebench.artifacts import (
    ArtifactIntegrityError,
    CorpusArtifacts,
    FixtureArtifacts,
)
from inferencebench.config import Settings
from inferencebench.domain import (
    AttemptEvidence,
    CorpusManifest,
    CustomerLabel,
    GroundTruthAnnotation,
    GroundTruthManifest,
    Issue,
    RunManifest,
    StrictModel,
)
from inferencebench.evaluation.metrics import (
    ModelAccuracy,
    assert_comparable_runs,
)
from inferencebench.evaluation.operational import (
    OperationalComparison,
    build_operational_comparison,
)
from inferencebench.evaluation.scored import (
    ScoredGroundTruthSet,
    ScoredModelSummary,
    ScoredPairRow,
    ScoredViewReview,
    build_scored_view,
    scored_ground_truth_from_evaluation_corpus,
    scored_ground_truth_from_fixture,
)
from inferencebench.evaluation.unscored import (
    UnscoredComparison,
    build_unscored_comparison,
)
from inferencebench.persistence.repository import EvidenceRepository
from inferencebench.ground_truth.artifacts import EvaluationCorpusArtifacts
from inferencebench.models.artifacts import (
    ModelCatalogArtifacts,
    PricingArtifacts,
    assert_run_uses_model_sources,
)
from inferencebench.models.domain import ModelPricing, PricingSnapshotManifest


class FixtureEvidenceError(ArtifactIntegrityError):
    """Raised when seeded run evidence disagrees with frozen fixture artifacts."""


class IssueModelResult(StrictModel):
    model_id: str
    run_id: str
    raw_output: str | None
    prediction: CustomerLabel | None
    scored_outcome: str
    attempt_id: str


class IssueComparison(StrictModel):
    issue: Issue
    ground_truth: GroundTruthAnnotation
    model_a: IssueModelResult
    model_b: IssueModelResult


class SavedComparisonReview(StrictModel):
    evidence_state: str
    provider_requests_made: int
    corpus_version: str
    corpus_sha256: str
    corpus_issue_count: int
    repository: str
    ground_truth_version: str
    shared_inference_contract_version: str
    parser_version: str
    concurrency: int
    model_a: ModelAccuracy
    model_b: ModelAccuracy
    issue_comparison: IssueComparison
    operational_view: OperationalComparison
    scored_view: ScoredViewReview
    unscored_view: UnscoredComparison


def load_saved_comparison(settings: Settings) -> SavedComparisonReview:
    artifacts = FixtureArtifacts(settings.fixture_root)
    corpus_manifest, issues = artifacts.load_corpus()
    ground_truth_manifest, annotations = artifacts.load_ground_truth()
    bundle = artifacts.load_run_bundle()
    catalog_manifest, _, _ = ModelCatalogArtifacts(
        settings.model_catalog_directory
    ).load()
    pricing_manifest, pricing_entries = PricingArtifacts(
        settings.pricing_directory
    ).load_for_catalog(catalog_manifest)
    _validate_bundle_against_artifacts(
        bundle.runs,
        bundle.attempts,
        corpus_manifest.artifact_sha256,
        corpus_manifest.corpus_version,
        corpus_manifest.ordered_issue_numbers,
        ground_truth_manifest.artifact_sha256,
        ground_truth_manifest.ground_truth_version,
    )

    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    repository.seed_fixture_bundle(bundle)

    run_a = repository.get_run(bundle.default_run_ids[0])
    run_b = repository.get_run(bundle.default_run_ids[1])
    assert_run_uses_model_sources(run_a, catalog_manifest, pricing_manifest)
    assert_run_uses_model_sources(run_b, catalog_manifest, pricing_manifest)
    assert_comparable_runs(run_a, run_b)
    attempts_a = repository.get_attempts(run_a.run_id)
    attempts_b = repository.get_attempts(run_b.run_id)
    scored_ground_truth = scored_ground_truth_from_fixture(
        ground_truth_manifest, annotations
    )
    return _build_comparison_review(
        evidence_state="Saved evidence — no inference request",
        corpus_manifest=corpus_manifest,
        issues=issues,
        ground_truth_manifest=ground_truth_manifest,
        annotations=annotations,
        run_a=run_a,
        attempts_a=attempts_a,
        run_b=run_b,
        attempts_b=attempts_b,
        scored_ground_truth=scored_ground_truth,
        pricing_manifest=pricing_manifest,
        pricing_entries=pricing_entries,
    )


def load_persisted_comparison(
    settings: Settings,
    run_ids: tuple[str, str],
) -> SavedComparisonReview:
    """Open two newly persisted full-Corpus runs without provider requests."""

    corpus_manifest, issues = CorpusArtifacts(settings.corpus_root).load_active()
    ground_truth_manifest, annotations = EvaluationCorpusArtifacts(
        settings.ground_truth_root
    ).load_version(
        settings.evaluation_ground_truth_version,
        corpus_manifest,
        issues,
    )
    catalog_manifest, _, _ = ModelCatalogArtifacts(
        settings.model_catalog_directory
    ).load()
    pricing_manifest, pricing_entries = PricingArtifacts(
        settings.pricing_directory
    ).load_for_catalog(catalog_manifest)
    repository = EvidenceRepository(settings.database_path)
    repository.initialize()
    run_a = repository.get_run(run_ids[0])
    run_b = repository.get_run(run_ids[1])
    _validate_bundle_against_artifacts(
        (run_a, run_b),
        (*repository.get_attempts(run_a.run_id), *repository.get_attempts(run_b.run_id)),
        corpus_manifest.artifact_sha256,
        corpus_manifest.corpus_version,
        corpus_manifest.ordered_issue_numbers,
        ground_truth_manifest.artifact_sha256,
        ground_truth_manifest.ground_truth_version,
    )
    assert_run_uses_model_sources(run_a, catalog_manifest, pricing_manifest)
    assert_run_uses_model_sources(run_b, catalog_manifest, pricing_manifest)
    assert_comparable_runs(run_a, run_b)
    attempts_a = repository.get_attempts(run_a.run_id)
    attempts_b = repository.get_attempts(run_b.run_id)
    scored_ground_truth = scored_ground_truth_from_evaluation_corpus(
        ground_truth_manifest,
        annotations,
    )
    return _build_comparison_review(
        evidence_state="Live result — fresh persisted comparison",
        corpus_manifest=corpus_manifest,
        issues=issues,
        ground_truth_manifest=ground_truth_manifest,
        annotations=annotations,
        run_a=run_a,
        attempts_a=attempts_a,
        run_b=run_b,
        attempts_b=attempts_b,
        scored_ground_truth=scored_ground_truth,
        pricing_manifest=pricing_manifest,
        pricing_entries=pricing_entries,
    )


def _build_comparison_review(
    *,
    evidence_state: str,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    ground_truth_manifest: GroundTruthManifest,
    annotations: tuple[GroundTruthAnnotation, ...],
    run_a: RunManifest,
    attempts_a: tuple[AttemptEvidence, ...],
    run_b: RunManifest,
    attempts_b: tuple[AttemptEvidence, ...],
    scored_ground_truth: ScoredGroundTruthSet,
    pricing_manifest: PricingSnapshotManifest,
    pricing_entries: tuple[ModelPricing, ...],
) -> SavedComparisonReview:
    scored_view = build_scored_view(
        run_a,
        attempts_a,
        run_b,
        attempts_b,
        issues,
        scored_ground_truth,
    )
    unscored_view = build_unscored_comparison(
        run_a,
        attempts_a,
        run_b,
        attempts_b,
        issues,
        tuple(item.issue_number for item in scored_ground_truth.items),
    )
    pricing_by_model = {entry.model_id: entry for entry in pricing_entries}
    primary_holdout_issue_numbers = tuple(
        item.issue_number
        for item in scored_ground_truth.items
        if item.evaluation_role.value == "primary_holdout"
    )
    operational_view = build_operational_comparison(
        run_a,
        attempts_a,
        pricing_by_model[run_a.model_id],
        run_b,
        attempts_b,
        pricing_by_model[run_b.model_id],
        pricing_manifest,
        primary_holdout_issue_numbers,
    )
    primary = next(
        comparison
        for comparison in scored_view.comparisons
        if comparison.population is scored_view.default_population
    )
    score_a = _accuracy_from_summary(primary.model_a)
    score_b = _accuracy_from_summary(primary.model_b)
    first_row = primary.rows[0]

    return SavedComparisonReview(
        evidence_state=evidence_state,
        provider_requests_made=0,
        corpus_version=corpus_manifest.corpus_version,
        corpus_sha256=corpus_manifest.artifact_sha256,
        corpus_issue_count=corpus_manifest.issue_count,
        repository=corpus_manifest.repository,
        ground_truth_version=ground_truth_manifest.ground_truth_version,
        shared_inference_contract_version=run_a.prompt_version,
        parser_version=run_a.parser_version,
        concurrency=run_a.concurrency,
        model_a=score_a,
        model_b=score_b,
        issue_comparison=_legacy_issue_comparison(
            first_row, annotations, run_a.model_id, run_b.model_id
        ),
        operational_view=operational_view,
        scored_view=scored_view,
        unscored_view=unscored_view,
    )


def _issue_model_result(
    model_id: str, attempt: AttemptEvidence, scored_outcome: str
) -> IssueModelResult:
    return IssueModelResult(
        model_id=model_id,
        run_id=attempt.run_id,
        raw_output=attempt.raw_model_output,
        prediction=attempt.parsed_label,
        scored_outcome=scored_outcome,
        attempt_id=attempt.attempt_id,
    )


def _accuracy_from_summary(summary: ScoredModelSummary) -> ModelAccuracy:
    return ModelAccuracy(
        model_id=summary.model_id,
        run_id=summary.run_id,
        correct_count=summary.correct_count,
        expected_count=summary.expected_count,
        accuracy=summary.accuracy,
    )


def _legacy_issue_comparison(
    row: ScoredPairRow,
    annotations: tuple[GroundTruthAnnotation, ...],
    model_a_id: str,
    model_b_id: str,
) -> IssueComparison:
    annotation = next(
        annotation
        for annotation in annotations
        if annotation.issue_number == row.issue.issue_number
    )
    return IssueComparison(
        issue=row.issue,
        ground_truth=annotation,
        model_a=_issue_model_result(
            model_a_id,
            row.model_a_attempt,
            row.model_a_outcome.value,
        ),
        model_b=_issue_model_result(
            model_b_id,
            row.model_b_attempt,
            row.model_b_outcome.value,
        ),
    )


def _validate_bundle_against_artifacts(
    runs: tuple[RunManifest, RunManifest],
    attempts: tuple[AttemptEvidence, ...],
    corpus_sha256: str,
    corpus_version: str,
    issue_numbers: tuple[int, ...],
    ground_truth_sha256: str,
    ground_truth_version: str,
) -> None:
    for run in runs:
        if (
            run.corpus_sha256 != corpus_sha256
            or run.corpus_version != corpus_version
            or run.ordered_issue_numbers != issue_numbers
        ):
            raise FixtureEvidenceError(
                f"Run {run.run_id} does not reference the supplied frozen Corpus"
            )
        if (
            run.ground_truth_sha256 != ground_truth_sha256
            or run.ground_truth_version != ground_truth_version
        ):
            raise FixtureEvidenceError(
                f"Run {run.run_id} does not reference the supplied Ground Truth"
            )
        run_attempts = tuple(attempt for attempt in attempts if attempt.run_id == run.run_id)
        if len(run_attempts) != run.expected_count:
            raise FixtureEvidenceError(
                f"Run {run.run_id} does not have its expected Attempt Evidence"
            )
