from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from inferencebench.artifacts import ArtifactIntegrityError, canonical_sha256, sha256_file
from inferencebench.domain import CustomerLabel, Sha256, StrictModel


SHARED_BOOTSTRAP_SEED = 20_260_833
SHARED_BOOTSTRAP_RESAMPLE_COUNT = 10_000
BOOTSTRAP_ALGORITHM_VERSION = "sha256-counter-stratified-bootstrap-v1"
BOOTSTRAP_INTERVAL_VERSION = "linear-percentile-2.5-97.5-v1"
BOOTSTRAP_METRIC_VERSION = "candidate-bootstrap-metrics-v1"
CUSTOMER_LABEL_ORDER = tuple(CustomerLabel)


class BootstrapHoldoutRow(StrictModel):
    row_index: int = Field(ge=0)
    issue_number: int = Field(gt=0)
    ground_truth_label: CustomerLabel


class BootstrapResample(StrictModel):
    resample_index: int = Field(ge=0)
    row_indices: tuple[int, ...]


class SharedBootstrapPlanManifest(StrictModel):
    schema_version: Literal["shared_bootstrap_plan_manifest.v1"]
    plan_version: str
    baseline_plan_sha256: Sha256
    ground_truth_version: str
    ground_truth_sha256: Sha256
    algorithm_version: Literal["sha256-counter-stratified-bootstrap-v1"]
    interval_version: Literal["linear-percentile-2.5-97.5-v1"]
    metric_version: Literal["candidate-bootstrap-metrics-v1"]
    seed: int
    resample_count: int = Field(gt=0)
    holdout_count: int = Field(gt=0)
    canonical_holdout_rows: tuple[BootstrapHoldoutRow, ...]
    canonical_holdout_sha256: Sha256
    represented_class_support: dict[CustomerLabel, int]
    resamples_file: str
    resamples_sha256: Sha256
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan_identity(self) -> "SharedBootstrapPlanManifest":
        if PurePosixPath(self.resamples_file).name != self.resamples_file:
            raise ValueError("Bootstrap resamples_file must be a local filename")
        if len(self.canonical_holdout_rows) != self.holdout_count:
            raise ValueError("Bootstrap holdout count does not match canonical rows")
        if tuple(row.row_index for row in self.canonical_holdout_rows) != tuple(
            range(self.holdout_count)
        ):
            raise ValueError("Bootstrap holdout row indices must be contiguous and zero-based")
        if len({row.issue_number for row in self.canonical_holdout_rows}) != (
            self.holdout_count
        ):
            raise ValueError("Bootstrap holdout issues must be unique")
        expected_support = {
            label: sum(
                row.ground_truth_label is label
                for row in self.canonical_holdout_rows
            )
            for label in CUSTOMER_LABEL_ORDER
            if any(
                row.ground_truth_label is label
                for row in self.canonical_holdout_rows
            )
        }
        if self.represented_class_support != expected_support:
            raise ValueError("Bootstrap class support disagrees with canonical holdout")
        holdout_payload = [
            row.model_dump(mode="json") for row in self.canonical_holdout_rows
        ]
        if canonical_sha256(holdout_payload) != self.canonical_holdout_sha256:
            raise ValueError("Bootstrap canonical holdout hash is invalid")
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(payload) != self.content_sha256:
            raise ValueError("Bootstrap manifest content hash is invalid")
        return self


class ExtendedMetricValue(StrictModel):
    state: Literal[
        "finite",
        "positive_infinity",
        "negative_infinity",
        "unknown",
        "indeterminate",
    ]
    value: Decimal | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "ExtendedMetricValue":
        if (self.state == "finite") != (self.value is not None):
            raise ValueError("Only a finite metric value may contain a number")
        return self


class BootstrapInterval(StrictModel):
    point_estimate: ExtendedMetricValue
    lower_95: ExtendedMetricValue
    upper_95: ExtendedMetricValue
    resample_count: int = Field(gt=0)
    finite_count: int = Field(ge=0)
    positive_infinity_count: int = Field(ge=0)
    negative_infinity_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    indeterminate_count: int = Field(ge=0)
    interval_version: Literal["linear-percentile-2.5-97.5-v1"]

    @model_validator(mode="after")
    def validate_partition(self) -> "BootstrapInterval":
        if (
            self.finite_count
            + self.positive_infinity_count
            + self.negative_infinity_count
            + self.unknown_count
            + self.indeterminate_count
            != self.resample_count
        ):
            raise ValueError("Bootstrap value states must cover every resample")
        return self


class BootstrapMetricIntervals(StrictModel):
    accuracy: BootstrapInterval
    supported_class_macro_f1: BootstrapInterval
    cost_per_correct: BootstrapInterval
    invalid_output_rate: BootstrapInterval
    request_error_rate: BootstrapInterval


class CandidateBootstrapSummary(StrictModel):
    schema_version: Literal["candidate_bootstrap_summary.v1"]
    model_id: str
    run_id: str | None
    source_attempts_sha256: Sha256 | None
    evidence_status: Literal["complete", "unavailable_incomplete"]
    unavailable_reason: str | None = None
    bootstrap_plan_sha256: Sha256
    intervals: BootstrapMetricIntervals | None

    @model_validator(mode="after")
    def validate_availability(self) -> "CandidateBootstrapSummary":
        complete = self.evidence_status == "complete"
        if complete != (self.intervals is not None):
            raise ValueError("Complete bootstrap evidence requires metric intervals")
        if complete and self.source_attempts_sha256 is None:
            raise ValueError("Complete bootstrap evidence requires its Attempt Evidence hash")
        if complete == (self.unavailable_reason is not None):
            raise ValueError("Only unavailable bootstrap evidence requires a reason")
        return self


class PairBootstrapSummary(StrictModel):
    schema_version: Literal["pair_bootstrap_summary.v1"]
    model_a_id: str
    model_b_id: str
    run_a_id: str | None
    run_b_id: str | None
    source_a_attempts_sha256: Sha256 | None
    source_b_attempts_sha256: Sha256 | None
    evidence_status: Literal["complete", "unavailable_incomplete"]
    unavailable_reason: str | None = None
    bootstrap_plan_sha256: Sha256
    paired_difference_direction: Literal["model_a_minus_model_b"]
    intervals: BootstrapMetricIntervals | None

    @model_validator(mode="after")
    def validate_availability(self) -> "PairBootstrapSummary":
        if self.model_a_id == self.model_b_id:
            raise ValueError("Paired bootstrap requires two distinct models")
        complete = self.evidence_status == "complete"
        if complete != (self.intervals is not None):
            raise ValueError("Complete paired bootstrap requires metric intervals")
        if complete and (
            self.source_a_attempts_sha256 is None
            or self.source_b_attempts_sha256 is None
        ):
            raise ValueError("Complete paired bootstrap requires both Attempt Evidence hashes")
        if complete == (self.unavailable_reason is not None):
            raise ValueError("Only unavailable paired bootstrap requires a reason")
        return self


class ObservedBootstrapRow(StrictModel):
    row_index: int = Field(ge=0)
    issue_number: int = Field(gt=0)
    ground_truth_label: CustomerLabel
    predicted_label: CustomerLabel | None
    is_invalid_output: bool
    is_request_error: bool
    calculated_request_cost_usd: Decimal | None

    @model_validator(mode="after")
    def validate_outcome(self) -> "ObservedBootstrapRow":
        states = (
            self.predicted_label is not None,
            self.is_invalid_output,
            self.is_request_error,
        )
        if sum(states) != 1:
            raise ValueError("Each bootstrap row requires one terminal result state")
        return self


@dataclass(frozen=True, slots=True)
class _Extended:
    state: str
    value: Decimal | None = None


@dataclass(frozen=True, slots=True)
class _MetricVector:
    accuracy: _Extended
    supported_class_macro_f1: _Extended
    cost_per_correct: _Extended
    invalid_output_rate: _Extended
    request_error_rate: _Extended


def canonical_holdout_rows(
    issue_labels: Iterable[tuple[int, CustomerLabel]],
) -> tuple[BootstrapHoldoutRow, ...]:
    return tuple(
        BootstrapHoldoutRow(
            row_index=row_index,
            issue_number=issue_number,
            ground_truth_label=label,
        )
        for row_index, (issue_number, label) in enumerate(issue_labels)
    )


def generate_shared_resamples(
    rows: tuple[BootstrapHoldoutRow, ...],
    *,
    seed: int = SHARED_BOOTSTRAP_SEED,
    resample_count: int = SHARED_BOOTSTRAP_RESAMPLE_COUNT,
) -> tuple[BootstrapResample, ...]:
    if not rows:
        raise ValueError("Shared Bootstrap Plan requires holdout rows")
    if resample_count <= 0:
        raise ValueError("Shared Bootstrap Plan resample count must be positive")
    if tuple(row.row_index for row in rows) != tuple(range(len(rows))):
        raise ValueError("Bootstrap row indices must be contiguous and zero-based")
    strata = {
        label: tuple(
            row.row_index for row in rows if row.ground_truth_label is label
        )
        for label in CUSTOMER_LABEL_ORDER
    }
    counter = 0
    generated: list[BootstrapResample] = []
    for resample_index in range(resample_count):
        indices: list[int] = []
        for label in CUSTOMER_LABEL_ORDER:
            source = strata[label]
            for _ in range(len(source)):
                selected, counter = _uniform_index(seed, counter, len(source))
                indices.append(source[selected])
        generated.append(
            BootstrapResample(
                resample_index=resample_index,
                row_indices=tuple(indices),
            )
        )
    return tuple(generated)


def build_shared_bootstrap_manifest(
    *,
    plan_version: str,
    baseline_plan_sha256: str,
    ground_truth_version: str,
    ground_truth_sha256: str,
    rows: tuple[BootstrapHoldoutRow, ...],
    resamples: tuple[BootstrapResample, ...],
    seed: int,
    resamples_file: str = "resamples.jsonl",
) -> SharedBootstrapPlanManifest:
    _validate_resamples(rows, resamples)
    resample_bytes = serialize_resamples(resamples)
    payload = {
        "schema_version": "shared_bootstrap_plan_manifest.v1",
        "plan_version": plan_version,
        "baseline_plan_sha256": baseline_plan_sha256,
        "ground_truth_version": ground_truth_version,
        "ground_truth_sha256": ground_truth_sha256,
        "algorithm_version": BOOTSTRAP_ALGORITHM_VERSION,
        "interval_version": BOOTSTRAP_INTERVAL_VERSION,
        "metric_version": BOOTSTRAP_METRIC_VERSION,
        "seed": seed,
        "resample_count": len(resamples),
        "holdout_count": len(rows),
        "canonical_holdout_rows": [row.model_dump(mode="json") for row in rows],
        "canonical_holdout_sha256": canonical_sha256(
            [row.model_dump(mode="json") for row in rows]
        ),
        "represented_class_support": {
            label.value: sum(row.ground_truth_label is label for row in rows)
            for label in CUSTOMER_LABEL_ORDER
            if any(row.ground_truth_label is label for row in rows)
        },
        "resamples_file": resamples_file,
        "resamples_sha256": hashlib.sha256(resample_bytes).hexdigest(),
    }
    return SharedBootstrapPlanManifest.model_validate(
        {**payload, "content_sha256": canonical_sha256(payload)}
    )


def write_shared_bootstrap_plan(
    directory: Path,
    manifest: SharedBootstrapPlanManifest,
    resamples: tuple[BootstrapResample, ...],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "bootstrap-manifest.json"
    resamples_path = directory / manifest.resamples_file
    if manifest_path.exists() or resamples_path.exists():
        raise FileExistsError("Shared Bootstrap Plan artifacts are immutable")
    payload = serialize_resamples(resamples)
    if hashlib.sha256(payload).hexdigest() != manifest.resamples_sha256:
        raise ArtifactIntegrityError("Bootstrap resamples disagree with their manifest")
    resamples_path.write_bytes(payload)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_shared_bootstrap_plan(
    directory: Path,
) -> tuple[SharedBootstrapPlanManifest, tuple[BootstrapResample, ...]]:
    manifest = SharedBootstrapPlanManifest.model_validate_json(
        (directory / "bootstrap-manifest.json").read_text(encoding="utf-8")
    )
    resamples_path = directory / manifest.resamples_file
    if sha256_file(resamples_path) != manifest.resamples_sha256:
        raise ArtifactIntegrityError("Shared Bootstrap Plan resample hash mismatch")
    resamples = tuple(
        BootstrapResample.model_validate_json(line)
        for line in resamples_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(resamples) != manifest.resample_count:
        raise ArtifactIntegrityError("Shared Bootstrap Plan resample count mismatch")
    _validate_resamples(manifest.canonical_holdout_rows, resamples)
    regenerated = generate_shared_resamples(
        manifest.canonical_holdout_rows,
        seed=manifest.seed,
        resample_count=manifest.resample_count,
    )
    if regenerated != resamples:
        raise ArtifactIntegrityError("Shared Bootstrap Plan is not reproducible from its seed")
    return manifest, resamples


def assert_shared_bootstrap_resamples(
    manifest: SharedBootstrapPlanManifest,
    resamples: tuple[BootstrapResample, ...],
) -> None:
    """Verify in-memory resamples against the persisted, reproducible plan identity."""

    if len(resamples) != manifest.resample_count:
        raise ArtifactIntegrityError("Shared Bootstrap Plan resample count mismatch")
    if hashlib.sha256(serialize_resamples(resamples)).hexdigest() != (
        manifest.resamples_sha256
    ):
        raise ArtifactIntegrityError("Shared Bootstrap Plan resample hash mismatch")
    _validate_resamples(manifest.canonical_holdout_rows, resamples)
    if generate_shared_resamples(
        manifest.canonical_holdout_rows,
        seed=manifest.seed,
        resample_count=manifest.resample_count,
    ) != resamples:
        raise ArtifactIntegrityError(
            "Shared Bootstrap Plan is not reproducible from its seed"
        )


def serialize_resamples(resamples: tuple[BootstrapResample, ...]) -> bytes:
    return (
        "".join(
            json.dumps(
                resample.model_dump(mode="json"),
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for resample in resamples
        )
    ).encode("utf-8")


def calculate_candidate_bootstrap_summary(
    *,
    model_id: str,
    run_id: str,
    source_attempts_sha256: str,
    plan: SharedBootstrapPlanManifest,
    resamples: tuple[BootstrapResample, ...],
    observed_rows: tuple[ObservedBootstrapRow, ...],
) -> tuple[CandidateBootstrapSummary, tuple[_MetricVector, ...]]:
    _validate_observed_rows(plan.canonical_holdout_rows, observed_rows)
    point = _calculate_metrics(observed_rows, tuple(range(len(observed_rows))))
    vectors = tuple(
        _calculate_metrics(observed_rows, resample.row_indices)
        for resample in resamples
    )
    return (
        CandidateBootstrapSummary(
            schema_version="candidate_bootstrap_summary.v1",
            model_id=model_id,
            run_id=run_id,
            source_attempts_sha256=source_attempts_sha256,
            evidence_status="complete",
            unavailable_reason=None,
            bootstrap_plan_sha256=plan.content_sha256,
            intervals=_summarize_vectors(point, vectors),
        ),
        vectors,
    )


def calculate_pair_bootstrap_summary(
    *,
    model_a_id: str,
    model_b_id: str,
    run_a_id: str,
    run_b_id: str,
    source_a_attempts_sha256: str,
    source_b_attempts_sha256: str,
    plan_sha256: str,
    candidate_a_vectors: tuple[_MetricVector, ...],
    candidate_b_vectors: tuple[_MetricVector, ...],
    candidate_a_point: BootstrapMetricIntervals,
    candidate_b_point: BootstrapMetricIntervals,
) -> PairBootstrapSummary:
    if len(candidate_a_vectors) != len(candidate_b_vectors):
        raise ValueError("Paired bootstrap candidates require the same resamples")
    vectors = tuple(
        _subtract_vectors(a, b)
        for a, b in zip(candidate_a_vectors, candidate_b_vectors, strict=True)
    )
    point = _MetricVector(
        accuracy=_subtract_extended(
            _from_public(candidate_a_point.accuracy.point_estimate),
            _from_public(candidate_b_point.accuracy.point_estimate),
        ),
        supported_class_macro_f1=_subtract_extended(
            _from_public(candidate_a_point.supported_class_macro_f1.point_estimate),
            _from_public(candidate_b_point.supported_class_macro_f1.point_estimate),
        ),
        cost_per_correct=_subtract_extended(
            _from_public(candidate_a_point.cost_per_correct.point_estimate),
            _from_public(candidate_b_point.cost_per_correct.point_estimate),
        ),
        invalid_output_rate=_subtract_extended(
            _from_public(candidate_a_point.invalid_output_rate.point_estimate),
            _from_public(candidate_b_point.invalid_output_rate.point_estimate),
        ),
        request_error_rate=_subtract_extended(
            _from_public(candidate_a_point.request_error_rate.point_estimate),
            _from_public(candidate_b_point.request_error_rate.point_estimate),
        ),
    )
    return PairBootstrapSummary(
        schema_version="pair_bootstrap_summary.v1",
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        source_a_attempts_sha256=source_a_attempts_sha256,
        source_b_attempts_sha256=source_b_attempts_sha256,
        evidence_status="complete",
        unavailable_reason=None,
        bootstrap_plan_sha256=plan_sha256,
        paired_difference_direction="model_a_minus_model_b",
        intervals=_summarize_vectors(point, vectors),
    )


def unavailable_candidate_bootstrap_summary(
    model_id: str,
    run_id: str | None,
    source_attempts_sha256: str | None,
    plan_sha256: str,
    reason: str,
) -> CandidateBootstrapSummary:
    return CandidateBootstrapSummary(
        schema_version="candidate_bootstrap_summary.v1",
        model_id=model_id,
        run_id=run_id,
        source_attempts_sha256=source_attempts_sha256,
        evidence_status="unavailable_incomplete",
        unavailable_reason=reason,
        bootstrap_plan_sha256=plan_sha256,
        intervals=None,
    )


def unavailable_pair_bootstrap_summary(
    model_a: CandidateBootstrapSummary,
    model_b: CandidateBootstrapSummary,
) -> PairBootstrapSummary:
    return PairBootstrapSummary(
        schema_version="pair_bootstrap_summary.v1",
        model_a_id=model_a.model_id,
        model_b_id=model_b.model_id,
        run_a_id=model_a.run_id,
        run_b_id=model_b.run_id,
        source_a_attempts_sha256=model_a.source_attempts_sha256,
        source_b_attempts_sha256=model_b.source_attempts_sha256,
        evidence_status="unavailable_incomplete",
        unavailable_reason="At least one candidate lacks complete baseline evidence",
        bootstrap_plan_sha256=model_a.bootstrap_plan_sha256,
        paired_difference_direction="model_a_minus_model_b",
        intervals=None,
    )


def _uniform_index(seed: int, counter: int, bound: int) -> tuple[int, int]:
    if bound <= 0:
        raise ValueError("Cannot sample an unrepresented class")
    maximum = 1 << 256
    limit = maximum - (maximum % bound)
    while True:
        digest = hashlib.sha256(f"{seed}:{counter}".encode("ascii")).digest()
        counter += 1
        candidate = int.from_bytes(digest, "big")
        if candidate < limit:
            return candidate % bound, counter


def _validate_resamples(
    rows: tuple[BootstrapHoldoutRow, ...],
    resamples: tuple[BootstrapResample, ...],
) -> None:
    if tuple(resample.resample_index for resample in resamples) != tuple(
        range(len(resamples))
    ):
        raise ValueError("Bootstrap resample indices must be contiguous and zero-based")
    support = {
        label: sum(row.ground_truth_label is label for row in rows)
        for label in CUSTOMER_LABEL_ORDER
    }
    labels_by_index = {row.row_index: row.ground_truth_label for row in rows}
    for resample in resamples:
        if len(resample.row_indices) != len(rows):
            raise ValueError("Every bootstrap resample must preserve holdout size")
        if any(index not in labels_by_index for index in resample.row_indices):
            raise ValueError("Bootstrap resample references an unknown row index")
        observed = {
            label: sum(labels_by_index[index] is label for index in resample.row_indices)
            for label in CUSTOMER_LABEL_ORDER
        }
        if observed != support:
            raise ValueError("Bootstrap resample does not preserve class support")


def _validate_observed_rows(
    canonical: tuple[BootstrapHoldoutRow, ...],
    observed: tuple[ObservedBootstrapRow, ...],
) -> None:
    identity = tuple(
        (row.row_index, row.issue_number, row.ground_truth_label) for row in observed
    )
    expected = tuple(
        (row.row_index, row.issue_number, row.ground_truth_label) for row in canonical
    )
    if identity != expected:
        raise ValueError("Candidate bootstrap rows do not match canonical holdout order")


def _calculate_metrics(
    rows: tuple[ObservedBootstrapRow, ...], indices: tuple[int, ...]
) -> _MetricVector:
    selected = tuple(rows[index] for index in indices)
    expected = Decimal(len(selected))
    correct_count = sum(
        row.predicted_label is row.ground_truth_label for row in selected
    )
    accuracy = Decimal(correct_count) / expected
    supported_f1: list[Decimal] = []
    for label in CUSTOMER_LABEL_ORDER:
        support = sum(row.ground_truth_label is label for row in selected)
        if support == 0:
            continue
        true_positive = sum(
            row.ground_truth_label is label and row.predicted_label is label
            for row in selected
        )
        false_positive = sum(
            row.ground_truth_label is not label and row.predicted_label is label
            for row in selected
        )
        false_negative = support - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        supported_f1.append(
            Decimal(2 * true_positive) / Decimal(denominator)
            if denominator
            else Decimal("0")
        )
    macro_f1 = sum(supported_f1, Decimal("0")) / Decimal(len(supported_f1))
    costs = tuple(row.calculated_request_cost_usd for row in selected)
    if any(cost is None for cost in costs):
        cost_per_correct = _Extended("unknown")
    elif correct_count == 0:
        cost_per_correct = _Extended("positive_infinity")
    else:
        known_costs = tuple(cost for cost in costs if cost is not None)
        cost_per_correct = _Extended(
            "finite",
            sum(known_costs, Decimal("0")) / Decimal(correct_count),
        )
    return _MetricVector(
        accuracy=_Extended("finite", accuracy),
        supported_class_macro_f1=_Extended("finite", macro_f1),
        cost_per_correct=cost_per_correct,
        invalid_output_rate=_Extended(
            "finite",
            Decimal(sum(row.is_invalid_output for row in selected)) / expected,
        ),
        request_error_rate=_Extended(
            "finite",
            Decimal(sum(row.is_request_error for row in selected)) / expected,
        ),
    )


def _summarize_vectors(
    point: _MetricVector, vectors: tuple[_MetricVector, ...]
) -> BootstrapMetricIntervals:
    return BootstrapMetricIntervals(
        accuracy=_interval(point.accuracy, tuple(row.accuracy for row in vectors)),
        supported_class_macro_f1=_interval(
            point.supported_class_macro_f1,
            tuple(row.supported_class_macro_f1 for row in vectors),
        ),
        cost_per_correct=_interval(
            point.cost_per_correct,
            tuple(row.cost_per_correct for row in vectors),
        ),
        invalid_output_rate=_interval(
            point.invalid_output_rate,
            tuple(row.invalid_output_rate for row in vectors),
        ),
        request_error_rate=_interval(
            point.request_error_rate,
            tuple(row.request_error_rate for row in vectors),
        ),
    )


def _interval(point: _Extended, values: tuple[_Extended, ...]) -> BootstrapInterval:
    counts = {
        state: sum(value.state == state for value in values)
        for state in (
            "finite",
            "positive_infinity",
            "negative_infinity",
            "unknown",
            "indeterminate",
        )
    }
    if counts["unknown"]:
        lower = upper = _Extended("unknown")
    elif counts["indeterminate"]:
        lower = upper = _Extended("indeterminate")
    else:
        lower = _extended_percentile(values, Decimal("0.025"))
        upper = _extended_percentile(values, Decimal("0.975"))
    return BootstrapInterval(
        point_estimate=_to_public(point),
        lower_95=_to_public(lower),
        upper_95=_to_public(upper),
        resample_count=len(values),
        finite_count=counts["finite"],
        positive_infinity_count=counts["positive_infinity"],
        negative_infinity_count=counts["negative_infinity"],
        unknown_count=counts["unknown"],
        indeterminate_count=counts["indeterminate"],
        interval_version=BOOTSTRAP_INTERVAL_VERSION,
    )


def _extended_percentile(
    values: tuple[_Extended, ...], percentile: Decimal
) -> _Extended:
    ordered = tuple(sorted(values, key=_extended_sort_key))
    position = Decimal(len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    if lower_index == upper_index or lower == upper:
        return lower
    weight = position - Decimal(lower_index)
    if lower.state == "finite" and upper.state == "finite":
        assert lower.value is not None and upper.value is not None
        return _Extended(
            "finite", lower.value + (upper.value - lower.value) * weight
        )
    if lower.state == "negative_infinity" and upper.state == "positive_infinity":
        return _Extended("indeterminate")
    if lower.state == "negative_infinity":
        return _Extended("negative_infinity")
    if upper.state == "positive_infinity":
        return _Extended("positive_infinity")
    raise ValueError("Unsupported extended percentile state")


def _extended_sort_key(value: _Extended) -> tuple[int, Decimal]:
    if value.state == "negative_infinity":
        return (0, Decimal("0"))
    if value.state == "finite":
        assert value.value is not None
        return (1, value.value)
    if value.state == "positive_infinity":
        return (2, Decimal("0"))
    raise ValueError("Unknown or indeterminate values cannot be ordered")


def _subtract_vectors(a: _MetricVector, b: _MetricVector) -> _MetricVector:
    return _MetricVector(
        accuracy=_subtract_extended(a.accuracy, b.accuracy),
        supported_class_macro_f1=_subtract_extended(
            a.supported_class_macro_f1, b.supported_class_macro_f1
        ),
        cost_per_correct=_subtract_extended(a.cost_per_correct, b.cost_per_correct),
        invalid_output_rate=_subtract_extended(
            a.invalid_output_rate, b.invalid_output_rate
        ),
        request_error_rate=_subtract_extended(
            a.request_error_rate, b.request_error_rate
        ),
    )


def _subtract_extended(a: _Extended, b: _Extended) -> _Extended:
    if "unknown" in {a.state, b.state}:
        return _Extended("unknown")
    if "indeterminate" in {a.state, b.state}:
        return _Extended("indeterminate")
    if a.state == b.state and a.state in {
        "positive_infinity",
        "negative_infinity",
    }:
        return _Extended("indeterminate")
    if a.state == "positive_infinity" or b.state == "negative_infinity":
        return _Extended("positive_infinity")
    if a.state == "negative_infinity" or b.state == "positive_infinity":
        return _Extended("negative_infinity")
    assert a.value is not None and b.value is not None
    return _Extended("finite", a.value - b.value)


def _to_public(value: _Extended) -> ExtendedMetricValue:
    return ExtendedMetricValue(state=value.state, value=value.value)


def _from_public(value: ExtendedMetricValue) -> _Extended:
    return _Extended(state=value.state, value=value.value)
