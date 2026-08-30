from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from inferencebench.artifacts import ArtifactIntegrityError, sha256_file
from inferencebench.domain import (
    CorpusManifest,
    HumanReviewAnnotation,
    HumanReviewManifest,
    Issue,
)


class HumanReviewArtifactError(ArtifactIntegrityError):
    """Raised when a human-review artifact is not a defensible random sample."""


@dataclass(frozen=True, slots=True)
class ReviewQueueItem:
    random_order_position: int
    issue_number: int
    title: str
    body: str | None


class HumanReviewPopulationSummary:
    def __init__(self, annotations: tuple[HumanReviewAnnotation, ...]) -> None:
        accepted = tuple(a for a in annotations if a.review_status == "accepted")
        self.reviewed_count = len(annotations)
        self.accepted_count = len(accepted)
        self.excluded_count = self.reviewed_count - self.accepted_count
        self.prompt_development_count = sum(
            a.evaluation_role == "prompt_development" for a in accepted
        )
        self.primary_holdout_count = sum(
            a.evaluation_role == "primary_holdout" for a in accepted
        )
        self.support_counts = dict(
            sorted(Counter(a.final_label.value for a in accepted if a.final_label).items())
        )


class HumanReviewArtifacts:
    """Read a versioned completed Human-Reviewed Ground Truth artifact locally."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load_version(
        self, version: str, corpus_manifest: CorpusManifest, issues: tuple[Issue, ...]
    ) -> tuple[HumanReviewManifest, tuple[HumanReviewAnnotation, ...]]:
        if Path(version).name != version:
            raise HumanReviewArtifactError("Ground Truth version must be a directory name")
        directory = self.root / version
        manifest = HumanReviewManifest.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        artifact_path = directory / manifest.artifact_file
        if sha256_file(artifact_path) != manifest.artifact_sha256:
            raise HumanReviewArtifactError("Human review artifact hash does not match manifest")
        annotations = _read_annotations(artifact_path)
        validate_human_review_artifact(corpus_manifest, issues, manifest, annotations)
        return manifest, annotations

    def available_versions(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        return tuple(sorted(path.name for path in self.root.iterdir() if path.is_dir()))


def model_visible_input(issue: Issue) -> dict[str, str | None]:
    """The sole data exposed during first-pass annotation and model inference."""
    return {"title": issue.title, "body": issue.body}


def prepare_review_queue(issues: tuple[Issue, ...], random_order_seed: int) -> tuple[ReviewQueueItem, ...]:
    """Return the persisted-seed review order without inspecting labels or state."""
    issue_numbers = [issue.issue_number for issue in issues]
    if len(issue_numbers) != len(set(issue_numbers)):
        raise HumanReviewArtifactError("Corpus issue numbers must be unique")
    random.Random(random_order_seed).shuffle(issue_numbers)
    issues_by_number = {issue.issue_number: issue for issue in issues}
    return tuple(
        ReviewQueueItem(
            random_order_position=position,
            issue_number=issue_number,
            title=issues_by_number[issue_number].title,
            body=issues_by_number[issue_number].body,
        )
        for position, issue_number in enumerate(issue_numbers, 1)
    )


def validate_human_review_artifact(
    corpus_manifest: CorpusManifest,
    issues: tuple[Issue, ...],
    manifest: HumanReviewManifest,
    annotations: tuple[HumanReviewAnnotation, ...],
) -> None:
    if manifest.corpus_version != corpus_manifest.corpus_version:
        raise HumanReviewArtifactError("Human review references a different Corpus version")
    if manifest.corpus_sha256 != corpus_manifest.artifact_sha256:
        raise HumanReviewArtifactError("Human review references a different Corpus hash")
    expected_order = tuple(
        item.issue_number for item in prepare_review_queue(issues, manifest.random_order_seed)
    )
    if manifest.random_ordered_issue_numbers != expected_order:
        raise HumanReviewArtifactError("Persisted random order does not match its seed")
    if len(annotations) != manifest.reviewed_count:
        raise HumanReviewArtifactError("Review record count does not match manifest")
    if tuple(a.issue_number for a in annotations) != manifest.reviewed_issue_numbers:
        raise HumanReviewArtifactError("Review record order does not match manifest")
    corpus_numbers = set(corpus_manifest.ordered_issue_numbers)
    reviewed_numbers = [a.issue_number for a in annotations]
    if not set(reviewed_numbers) <= corpus_numbers:
        raise HumanReviewArtifactError("Review contains an issue outside the frozen Corpus")
    if len(set(reviewed_numbers)) != len(reviewed_numbers):
        raise HumanReviewArtifactError("An issue may be reviewed only once in this artifact")
    expected_prefix = expected_order[: len(annotations)]
    if tuple(reviewed_numbers) != expected_prefix:
        raise HumanReviewArtifactError("Reviewed issues must be a contiguous random-order prefix")
    for position, annotation in enumerate(annotations, 1):
        if annotation.corpus_version != corpus_manifest.corpus_version:
            raise HumanReviewArtifactError("Annotation references a different Corpus version")
        if annotation.random_order_position != position:
            raise HumanReviewArtifactError("Annotation position does not match review order")
        if annotation.rubric_version != manifest.rubric_version:
            raise HumanReviewArtifactError("Annotation references a different rubric version")
        if annotation.rubric_sha256 != manifest.rubric_sha256:
            raise HumanReviewArtifactError("Annotation references a different rubric hash")
        _validate_annotation_hash(annotation)
    accepted = tuple(a for a in annotations if a.review_status == "accepted")
    if len(accepted) != 100:
        raise HumanReviewArtifactError("Artifact must contain exactly 100 accepted random annotations")
    if any(a.sampling_stratum != "random" for a in accepted):
        raise HumanReviewArtifactError("Accepted sample annotations must have random stratum")
    split = _partition(tuple(a.issue_number for a in accepted), manifest.partition_seed)
    for annotation in accepted:
        expected_role = (
            "prompt_development" if annotation.issue_number in split[:20] else "primary_holdout"
        )
        if annotation.evaluation_role != expected_role:
            raise HumanReviewArtifactError("Development/holdout membership does not match seed")
    quality_control = set(_quality_control(tuple(a.issue_number for a in accepted), manifest.quality_control_seed))
    actual_quality_control = {a.issue_number for a in accepted if a.quality_control_reviewed}
    if actual_quality_control != quality_control:
        raise HumanReviewArtifactError("Quality-control membership does not match seed")
    if sum(a.requires_second_pass for a in accepted) and any(
        a.requires_second_pass and a.review_pass_count < 2 for a in accepted
    ):
        raise HumanReviewArtifactError("Second-pass annotations must record two passes")


def annotation_sha256(annotation: HumanReviewAnnotation) -> str:
    values = annotation.model_dump(mode="json", exclude={"annotation_sha256"})
    return hashlib.sha256(_canonical_json(values)).hexdigest()


def _validate_annotation_hash(annotation: HumanReviewAnnotation) -> None:
    if annotation.annotation_sha256 != annotation_sha256(annotation):
        raise HumanReviewArtifactError(
            f"Annotation {annotation.annotation_id} hash does not match its content"
        )


def _partition(issue_numbers: tuple[int, ...], seed: int) -> tuple[int, ...]:
    selected = list(issue_numbers)
    random.Random(seed).shuffle(selected)
    return tuple(selected)


def _quality_control(issue_numbers: tuple[int, ...], seed: int) -> tuple[int, ...]:
    return tuple(sorted(random.Random(seed).sample(list(issue_numbers), 20)))


def _read_annotations(path: Path) -> tuple[HumanReviewAnnotation, ...]:
    rows: list[HumanReviewAnnotation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(HumanReviewAnnotation.model_validate_json(line))
        except ValueError as error:
            raise HumanReviewArtifactError(f"{path}:{line_number}: {error}") from error
    return tuple(rows)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
