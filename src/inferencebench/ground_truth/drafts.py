from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from inferencebench.artifacts import sha256_file
from inferencebench.domain import (
    CorpusManifest,
    HumanReviewAnnotation,
    HumanReviewDraft,
    HumanReviewDraftEntry,
    HumanReviewManifest,
    Issue,
)
from inferencebench.ground_truth.annotations import (
    HumanReviewArtifactError,
    annotation_sha256,
    prepare_review_queue,
    validate_human_review_artifact,
)


class HumanReviewDraftError(ValueError):
    """Raised when a draft would violate prediction-blind review order."""


def create_draft(
    *,
    corpus_manifest: CorpusManifest,
    rubric_path: Path,
    random_order_seed: int,
    partition_seed: int,
    quality_control_seed: int,
) -> HumanReviewDraft:
    return HumanReviewDraft(
        schema_version="human_review_draft.v1",
        corpus_version=corpus_manifest.corpus_version,
        corpus_sha256=corpus_manifest.artifact_sha256,
        rubric_version="label-rubric-v1",
        rubric_sha256=sha256_file(rubric_path),
        random_order_seed=random_order_seed,
        partition_seed=partition_seed,
        quality_control_seed=quality_control_seed,
    )


def next_issue(
    draft: HumanReviewDraft, issues: tuple[Issue, ...]
) -> tuple[int, Issue]:
    queue = prepare_review_queue(issues, draft.random_order_seed)
    position = len(draft.entries)
    if position >= len(queue):
        raise HumanReviewDraftError("The frozen random review order is exhausted")
    item = queue[position]
    issue = next(issue for issue in issues if issue.issue_number == item.issue_number)
    return item.random_order_position, issue


def append_entry(
    draft: HumanReviewDraft,
    issues: tuple[Issue, ...],
    entry: HumanReviewDraftEntry,
) -> HumanReviewDraft:
    _, expected = next_issue(draft, issues)
    if entry.issue_number != expected.issue_number:
        raise HumanReviewDraftError("Review decisions must follow the frozen random order")
    if entry.review_status == "accepted":
        if entry.final_label is None or entry.confidence is None:
            raise HumanReviewDraftError("Accepted review requires final label and confidence")
        if entry.input_sufficiency != "sufficient" or entry.exclusion_reason is not None:
            raise HumanReviewDraftError("Accepted review must be input-sufficient and unexcluded")
    elif entry.final_label is not None or entry.exclusion_reason is None:
        raise HumanReviewDraftError("Excluded or unresolved review requires a reason and no final label")
    if entry.requires_second_pass and entry.review_pass_count < 2:
        raise HumanReviewDraftError("Second-pass review must record at least two passes")
    return draft.model_copy(update={"entries": (*draft.entries, entry)})


def record_second_pass(
    draft: HumanReviewDraft,
    *,
    issue_number: int,
    final_label: str,
    confidence: str,
    review_notes: str | None = None,
) -> HumanReviewDraft:
    matches = [index for index, entry in enumerate(draft.entries) if entry.issue_number == issue_number]
    if len(matches) != 1:
        raise HumanReviewDraftError("Second-pass issue must exist exactly once in the draft")
    index = matches[0]
    entry = draft.entries[index]
    if entry.review_status != "accepted":
        raise HumanReviewDraftError("Only accepted annotations receive a label second pass")
    updated = entry.model_dump()
    updated.update(
        {
            "final_label": final_label,
            "confidence": confidence,
            "review_pass_count": max(2, entry.review_pass_count + 1),
            "requires_second_pass": True,
            "review_notes": review_notes if review_notes is not None else entry.review_notes,
            "second_pass_reviewed_at": datetime.now(UTC),
        }
    )
    entries = list(draft.entries)
    entries[index] = HumanReviewDraftEntry.model_validate(updated)
    return draft.model_copy(update={"entries": tuple(entries)})


def finalize_draft(
    draft: HumanReviewDraft,
    *,
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    ground_truth_version: str,
) -> tuple[HumanReviewManifest, tuple[HumanReviewAnnotation, ...]]:
    if draft.corpus_version != corpus_manifest.corpus_version or draft.corpus_sha256 != corpus_manifest.artifact_sha256:
        raise HumanReviewDraftError("Draft does not reference the active frozen Corpus")
    accepted = tuple(entry for entry in draft.entries if entry.review_status == "accepted")
    if len(accepted) != 100:
        raise HumanReviewDraftError("Finalize only after exactly 100 accepted annotations")
    accepted_numbers = [entry.issue_number for entry in accepted]
    partitioned = list(accepted_numbers)
    random.Random(draft.partition_seed).shuffle(partitioned)
    development = set(partitioned[:20])
    quality_control = set(
        random.Random(draft.quality_control_seed).sample(accepted_numbers, 20)
    )
    incomplete_quality_control = quality_control - {
        entry.issue_number for entry in accepted if entry.review_pass_count >= 2
    }
    if incomplete_quality_control:
        raise HumanReviewDraftError(
            "Quality-control second pass is incomplete for issues: "
            f"{sorted(incomplete_quality_control)}"
        )
    annotations: list[HumanReviewAnnotation] = []
    for position, entry in enumerate(draft.entries, 1):
        accepted_entry = entry.review_status == "accepted"
        annotation = HumanReviewAnnotation(
            schema_version="human_review_annotation.v1",
            annotation_id=f"{ground_truth_version}-{entry.issue_number}",
            corpus_version=draft.corpus_version,
            issue_number=entry.issue_number,
            random_order_position=position,
            sampling_stratum="random",
            evaluation_role=(
                "prompt_development"
                if entry.issue_number in development
                else "primary_holdout"
                if accepted_entry
                else "excluded"
            ),
            initial_label=entry.initial_label,
            final_label=entry.final_label,
            confidence=entry.confidence,
            review_status=entry.review_status,
            review_pass_count=entry.review_pass_count,
            requires_second_pass=entry.requires_second_pass,
            quality_control_reviewed=entry.issue_number in quality_control,
            input_sufficiency=entry.input_sufficiency,
            exclusion_reason=entry.exclusion_reason,
            review_notes=entry.review_notes,
            rubric_version=draft.rubric_version,
            rubric_sha256=draft.rubric_sha256,
            reviewed_at=entry.reviewed_at,
            second_pass_reviewed_at=entry.second_pass_reviewed_at,
            annotation_sha256="0" * 64,
        )
        annotations.append(annotation.model_copy(update={"annotation_sha256": annotation_sha256(annotation)}))
    artifact_bytes = b"".join(_json_bytes(row.model_dump(mode="json")) + b"\n" for row in annotations)
    manifest = HumanReviewManifest(
        schema_version="human_review_manifest.v1",
        ground_truth_version=ground_truth_version,
        corpus_version=draft.corpus_version,
        corpus_sha256=draft.corpus_sha256,
        rubric_version=draft.rubric_version,
        rubric_sha256=draft.rubric_sha256,
        random_order_seed=draft.random_order_seed,
        partition_seed=draft.partition_seed,
        quality_control_seed=draft.quality_control_seed,
        artifact_file="annotations.jsonl",
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        reviewed_count=len(annotations),
        accepted_count=100,
        excluded_count=len(annotations) - 100,
        prompt_development_count=20,
        primary_holdout_count=80,
        quality_control_count=20,
        random_ordered_issue_numbers=tuple(
            item.issue_number for item in prepare_review_queue(issues, draft.random_order_seed)
        ),
        reviewed_issue_numbers=tuple(entry.issue_number for entry in draft.entries),
    )
    validate_human_review_artifact(corpus_manifest, issues, manifest, tuple(annotations))
    return manifest, tuple(annotations)


def load_draft(path: Path) -> HumanReviewDraft:
    return HumanReviewDraft.model_validate_json(path.read_text(encoding="utf-8"))


def save_draft(path: Path, draft: HumanReviewDraft) -> None:
    _write_atomically(path, _pretty_json(draft.model_dump(mode="json")))


def write_completed_artifact(
    root: Path,
    manifest: HumanReviewManifest,
    annotations: tuple[HumanReviewAnnotation, ...],
) -> Path:
    if Path(manifest.ground_truth_version).name != manifest.ground_truth_version:
        raise HumanReviewDraftError("Ground Truth version must be a directory name")
    directory = root / manifest.ground_truth_version
    if directory.exists():
        raise HumanReviewDraftError(f"Ground Truth version is immutable: {directory}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=root))
    try:
        (temporary / manifest.artifact_file).write_bytes(
            b"".join(_json_bytes(row.model_dump(mode="json")) + b"\n" for row in annotations)
        )
        (temporary / "manifest.json").write_bytes(_pretty_json(manifest.model_dump(mode="json")))
        os.rename(temporary, directory)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return directory


def new_entry(
    *,
    issue_number: int,
    initial_label: str | None,
    final_label: str | None,
    confidence: str | None,
    review_status: str,
    review_pass_count: int,
    requires_second_pass: bool,
    input_sufficiency: str,
    exclusion_reason: str | None,
    review_notes: str | None,
) -> HumanReviewDraftEntry:
    return HumanReviewDraftEntry(
        issue_number=issue_number,
        initial_label=initial_label,
        final_label=final_label,
        confidence=confidence,
        review_status=review_status,
        review_pass_count=review_pass_count,
        requires_second_pass=requires_second_pass,
        input_sufficiency=input_sufficiency,
        exclusion_reason=exclusion_reason,
        review_notes=review_notes,
        reviewed_at=datetime.now(UTC),
    )


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
