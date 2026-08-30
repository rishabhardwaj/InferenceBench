from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

from inferencebench.artifacts import CorpusArtifacts
from inferencebench.domain import CustomerLabel, HumanReviewAnnotation, HumanReviewManifest
from inferencebench.ground_truth.annotations import (
    annotation_sha256,
    model_visible_input,
    prepare_review_queue,
    validate_human_review_artifact,
)
from inferencebench.ground_truth.drafts import (
    append_entry,
    create_draft,
    new_entry,
    next_issue,
    record_second_pass,
)


def test_review_queue_is_seeded_and_exposes_only_model_visible_input() -> None:
    _, issues = CorpusArtifacts(Path("artifacts/corpus")).load_active()

    first = prepare_review_queue(issues, 41)
    second = prepare_review_queue(issues, 41)
    different = prepare_review_queue(issues, 42)

    assert first == second
    assert tuple(item.issue_number for item in first) != tuple(
        item.issue_number for item in different
    )
    assert model_visible_input(issues[0]) == {
        "title": issues[0].title,
        "body": issues[0].body,
    }


def test_completed_sample_requires_seeded_split_and_quality_control() -> None:
    corpus_manifest, issues = CorpusArtifacts(Path("artifacts/corpus")).load_active()
    random_order_seed = 41
    partition_seed = 84
    quality_control_seed = 99
    queue = prepare_review_queue(issues, random_order_seed)
    accepted_numbers = tuple(item.issue_number for item in queue[:100])
    partitioned = list(accepted_numbers)
    random.Random(partition_seed).shuffle(partitioned)
    development = set(partitioned[:20])
    quality_control = set(random.Random(quality_control_seed).sample(list(accepted_numbers), 20))

    annotations = tuple(
        _annotation(
            corpus_manifest.corpus_version,
            item.issue_number,
            item.random_order_position,
            "prompt_development" if item.issue_number in development else "primary_holdout",
            item.issue_number in quality_control,
        )
        for item in queue[:100]
    )
    manifest = HumanReviewManifest(
        schema_version="human_review_manifest.v1",
        ground_truth_version="human-review-test-v1",
        corpus_version=corpus_manifest.corpus_version,
        corpus_sha256=corpus_manifest.artifact_sha256,
        rubric_version="label-rubric-v1",
        rubric_sha256="a" * 64,
        random_order_seed=random_order_seed,
        partition_seed=partition_seed,
        quality_control_seed=quality_control_seed,
        artifact_file="annotations.jsonl",
        artifact_sha256="b" * 64,
        reviewed_count=100,
        accepted_count=100,
        excluded_count=0,
        prompt_development_count=20,
        primary_holdout_count=80,
        quality_control_count=20,
        random_ordered_issue_numbers=tuple(item.issue_number for item in queue),
        reviewed_issue_numbers=accepted_numbers,
    )

    validate_human_review_artifact(corpus_manifest, issues, manifest, annotations)


def test_draft_advances_only_after_a_recorded_outcome() -> None:
    corpus_manifest, issues = CorpusArtifacts(Path("artifacts/corpus")).load_active()
    draft = create_draft(
        corpus_manifest=corpus_manifest,
        rubric_path=Path("docs/label-rubric-v1.md"),
        random_order_seed=41,
        partition_seed=84,
        quality_control_seed=99,
    )
    _, first = next_issue(draft, issues)
    draft = append_entry(
        draft,
        issues,
        new_entry(
            issue_number=first.issue_number,
            initial_label="bug",
            final_label="bug",
            confidence="high",
            review_status="accepted",
            review_pass_count=1,
            requires_second_pass=False,
            input_sufficiency="sufficient",
            exclusion_reason=None,
            review_notes=None,
        ),
    )

    position, second = next_issue(draft, issues)

    assert position == 2
    assert second.issue_number != first.issue_number

    revised = record_second_pass(
        draft,
        issue_number=first.issue_number,
        final_label="enhancement",
        confidence="medium",
        review_notes="Solo quality-control pass",
    )
    assert revised.entries[0].initial_label is CustomerLabel.BUG
    assert revised.entries[0].final_label is CustomerLabel.ENHANCEMENT
    assert revised.entries[0].review_pass_count == 2
    assert revised.entries[0].second_pass_reviewed_at is not None


def _annotation(
    corpus_version: str,
    issue_number: int,
    position: int,
    role: str,
    quality_control_reviewed: bool,
) -> HumanReviewAnnotation:
    draft = HumanReviewAnnotation(
        schema_version="human_review_annotation.v1",
        annotation_id=f"test-{issue_number}",
        corpus_version=corpus_version,
        issue_number=issue_number,
        random_order_position=position,
        sampling_stratum="random",
        evaluation_role=role,
        initial_label=CustomerLabel.BUG,
        final_label=CustomerLabel.BUG,
        confidence="high",
        review_status="accepted",
        review_pass_count=2 if quality_control_reviewed else 1,
        quality_control_reviewed=quality_control_reviewed,
        input_sufficiency="sufficient",
        rubric_version="label-rubric-v1",
        rubric_sha256="a" * 64,
        reviewed_at=datetime(2026, 8, 30, tzinfo=UTC),
        second_pass_reviewed_at=(
            datetime(2026, 8, 31, tzinfo=UTC) if quality_control_reviewed else None
        ),
        annotation_sha256="0" * 64,
    )
    return draft.model_copy(update={"annotation_sha256": annotation_sha256(draft)})
