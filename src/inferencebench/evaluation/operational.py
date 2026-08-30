from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from inferencebench.domain import (
    AttemptEvidence,
    AttemptPurpose,
    CostCompleteness,
    ProviderOutcome,
    RunManifest,
    RunStatus,
    ScoredOutcome,
    StrictModel,
)
from inferencebench.evaluation.cost import (
    RequestCostCalculation,
    calculate_request_cost,
)
from inferencebench.evaluation.metrics import MetricIntegrityError, assert_comparable_runs
from inferencebench.models.domain import ModelPricing, PricingSnapshotManifest


PERCENTILE_METHOD_VERSION = "linear-percentile-v1"


class AggregateCostCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class RunCostSummary(StrictModel):
    expected_count: int = Field(gt=0)
    observed_count: int = Field(ge=0)
    known_cost_count: int = Field(ge=0)
    unknown_cost_count: int = Field(ge=0)
    unobserved_count: int = Field(ge=0)
    known_total_cost_usd: Decimal
    mean_known_request_cost_usd: Decimal | None
    cost_per_expected_classification_usd: Decimal
    completeness: AggregateCostCompleteness
    is_lower_bound: bool

    @model_validator(mode="after")
    def validate_counts(self) -> "RunCostSummary":
        if self.observed_count + self.unobserved_count != self.expected_count:
            raise ValueError("Observed and unobserved counts must cover expected attempts")
        if self.known_cost_count + self.unknown_cost_count != self.expected_count:
            raise ValueError("Known and unknown cost counts must cover expected attempts")
        if self.known_cost_count > self.observed_count:
            raise ValueError("Known cost cannot exceed observed attempts")
        expected_mean = (
            self.known_total_cost_usd / self.known_cost_count
            if self.known_cost_count
            else None
        )
        if self.mean_known_request_cost_usd != expected_mean:
            raise ValueError("Mean known request cost disagrees with its inputs")
        if self.cost_per_expected_classification_usd != (
            self.known_total_cost_usd / self.expected_count
        ):
            raise ValueError("Cost per expected classification disagrees with its inputs")
        complete = self.unknown_cost_count == 0
        if complete != (self.completeness is AggregateCostCompleteness.COMPLETE):
            raise ValueError("Aggregate Cost Completeness disagrees with unknown costs")
        if self.is_lower_bound == complete:
            raise ValueError("Only partial cost is a disclosed lower bound")
        return self


class PrimaryHoldoutCostSummary(StrictModel):
    population_name: Literal["Primary Scored Holdout"]
    expected_count: int = Field(gt=0)
    correct_count: int = Field(ge=0)
    known_total_cost_usd: Decimal
    unknown_cost_count: int = Field(ge=0)
    completeness: AggregateCostCompleteness
    cost_per_correct_usd: Decimal | None
    cost_per_correct_status: Literal["defined", "undefined_zero_correct"]

    @model_validator(mode="after")
    def validate_cost_per_correct(self) -> "PrimaryHoldoutCostSummary":
        if self.correct_count > self.expected_count:
            raise ValueError("Correct count cannot exceed Primary Holdout size")
        if self.unknown_cost_count > self.expected_count:
            raise ValueError("Unknown cost cannot exceed Primary Holdout size")
        complete = self.unknown_cost_count == 0
        if complete != (self.completeness is AggregateCostCompleteness.COMPLETE):
            raise ValueError("Primary Holdout cost completeness is inconsistent")
        if self.correct_count == 0:
            if (
                self.cost_per_correct_usd is not None
                or self.cost_per_correct_status != "undefined_zero_correct"
            ):
                raise ValueError("Zero correct classifications make cost per correct undefined")
        elif (
            self.cost_per_correct_usd
            != self.known_total_cost_usd / self.correct_count
            or self.cost_per_correct_status != "defined"
        ):
            raise ValueError("Primary Holdout cost per correct disagrees with its inputs")
        return self


class AttemptCostSummary(StrictModel):
    attempt_id: str
    issue_number: int = Field(gt=0)
    dispatch_order: int = Field(ge=0)
    calculation: RequestCostCalculation


class LatencySummary(StrictModel):
    percentile_method_version: Literal["linear-percentile-v1"]
    is_comparable: bool
    concurrency: int = Field(gt=0)
    usable_count: int = Field(ge=0)
    expected_count: int = Field(gt=0)
    p50_usable_request_latency_ms: float | None = Field(default=None, ge=0)
    p95_usable_request_latency_ms: float | None = Field(default=None, ge=0)
    p50_queue_wait_ms: float | None = Field(default=None, ge=0)
    p95_queue_wait_ms: float | None = Field(default=None, ge=0)


class ThroughputSummary(StrictModel):
    run_wall_clock_ms: float = Field(gt=0)
    run_wall_clock_seconds: Decimal
    sustained_requests_per_second: Decimal
    usable_classifications_per_second: Decimal


class ReliabilitySummary(StrictModel):
    is_comparable: bool
    expected_count: int = Field(gt=0)
    observed_count: int = Field(ge=0)
    unobserved_count: int = Field(ge=0)
    usable_count: int = Field(ge=0)
    invalid_output_count: int = Field(ge=0)
    request_error_count: int = Field(ge=0)
    unusable_count: int = Field(ge=0)
    invalid_output_rate: float = Field(ge=0, le=1)
    request_error_rate: float = Field(ge=0, le=1)
    unusable_rate: float = Field(ge=0, le=1)
    rate_limit_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    other_request_error_count: int = Field(ge=0)
    rate_limit_rate: float = Field(ge=0, le=1)
    timeout_rate: float = Field(ge=0, le=1)
    other_request_error_rate: float = Field(ge=0, le=1)
    detailed_request_error_counts: dict[ProviderOutcome, int]

    @model_validator(mode="after")
    def validate_partition(self) -> "ReliabilitySummary":
        if self.observed_count + self.unobserved_count != self.expected_count:
            raise ValueError("Reliability counts must expose unobserved attempts")
        if (
            self.usable_count + self.invalid_output_count + self.request_error_count
            != self.observed_count
        ):
            raise ValueError("Observed attempts must have one terminal result state")
        if self.unusable_count != self.invalid_output_count + self.request_error_count:
            raise ValueError("Unusable count must include invalid outputs and request errors")
        if (
            self.rate_limit_count
            + self.timeout_count
            + self.other_request_error_count
            != self.request_error_count
        ):
            raise ValueError("Top-level request-error groups must be exhaustive")
        if sum(self.detailed_request_error_counts.values()) != self.request_error_count:
            raise ValueError("Detailed request errors must cover every request error")
        expected_rates = (
            self.invalid_output_count / self.expected_count,
            self.request_error_count / self.expected_count,
            self.unusable_count / self.expected_count,
            self.rate_limit_count / self.expected_count,
            self.timeout_count / self.expected_count,
            self.other_request_error_count / self.expected_count,
        )
        actual_rates = (
            self.invalid_output_rate,
            self.request_error_rate,
            self.unusable_rate,
            self.rate_limit_rate,
            self.timeout_rate,
            self.other_request_error_rate,
        )
        if actual_rates != expected_rates:
            raise ValueError("Reliability rates must divide by every expected attempt")
        return self


class RunOperationalSummary(StrictModel):
    model_id: str
    run_id: str
    status: RunStatus
    population_name: Literal["Full Corpus"]
    concurrency: int = Field(gt=0)
    source_expected_count: int = Field(gt=0)
    request_costs: tuple[AttemptCostSummary, ...]
    cost: RunCostSummary
    primary_holdout_cost: PrimaryHoldoutCostSummary | None
    latency: LatencySummary
    throughput: ThroughputSummary | None
    reliability: ReliabilitySummary

    @model_validator(mode="after")
    def validate_headline_status(self) -> "RunOperationalSummary":
        complete = self.status is RunStatus.COMPLETE
        if self.latency.is_comparable != complete:
            raise ValueError("Only complete runs publish comparable latency")
        if self.reliability.is_comparable != complete:
            raise ValueError("Only complete runs publish comparable reliability")
        if (self.throughput is not None) != complete:
            raise ValueError("Only complete runs publish comparable throughput")
        if self.source_expected_count != self.cost.expected_count:
            raise ValueError("Operational summaries must share one expected population")
        return self


class OperationalComparison(StrictModel):
    pricing_snapshot_id: str
    pricing_snapshot_sha256: str
    pricing_source_url: str
    cost_formula_version: str
    percentile_method_version: Literal["linear-percentile-v1"]
    model_a: RunOperationalSummary
    model_b: RunOperationalSummary


def build_operational_comparison(
    run_a: RunManifest,
    attempts_a: tuple[AttemptEvidence, ...],
    pricing_a: ModelPricing,
    run_b: RunManifest,
    attempts_b: tuple[AttemptEvidence, ...],
    pricing_b: ModelPricing,
    pricing_manifest: PricingSnapshotManifest,
    primary_holdout_issue_numbers: tuple[int, ...],
) -> OperationalComparison:
    assert_comparable_runs(run_a, run_b)
    for run, pricing in ((run_a, pricing_a), (run_b, pricing_b)):
        if run.model_id != pricing.model_id:
            raise MetricIntegrityError("Model Pricing references the wrong run model")
        if (
            run.pricing_snapshot_id != pricing_manifest.pricing_snapshot_id
            or run.pricing_snapshot_sha256 != pricing_manifest.content_sha256
            or pricing.pricing_snapshot_id != pricing_manifest.pricing_snapshot_id
        ):
            raise MetricIntegrityError("Run references a different Pricing Snapshot")
    summary_a = build_run_operational_summary(
        run_a,
        attempts_a,
        pricing_a,
        primary_holdout_issue_numbers,
    )
    summary_b = build_run_operational_summary(
        run_b,
        attempts_b,
        pricing_b,
        primary_holdout_issue_numbers,
    )
    return OperationalComparison(
        pricing_snapshot_id=pricing_manifest.pricing_snapshot_id,
        pricing_snapshot_sha256=pricing_manifest.content_sha256,
        pricing_source_url=pricing_manifest.source_url,
        cost_formula_version=summary_a.request_costs[0].calculation.formula_version,
        percentile_method_version=PERCENTILE_METHOD_VERSION,
        model_a=summary_a,
        model_b=summary_b,
    )


def build_run_operational_summary(
    run: RunManifest,
    attempts: tuple[AttemptEvidence, ...],
    pricing: ModelPricing,
    primary_holdout_issue_numbers: tuple[int, ...] = (),
) -> RunOperationalSummary:
    if run.model_id != pricing.model_id:
        raise MetricIntegrityError("Model Pricing references the wrong run model")
    if run.pricing_snapshot_id != pricing.pricing_snapshot_id:
        raise MetricIntegrityError("Model Pricing references a different snapshot")
    benchmark = _validated_benchmark_attempts(run, attempts)
    calculations = tuple(
        calculate_request_cost(attempt.usage, pricing) for attempt in benchmark
    )
    for attempt, calculation in zip(benchmark, calculations, strict=True):
        _verify_persisted_cost(attempt, calculation)

    complete = run.status is RunStatus.COMPLETE
    if complete and (run.wall_clock_ms is None or run.wall_clock_ms <= 0):
        raise MetricIntegrityError("Completed run requires positive Run Wall-Clock Time")
    cost = _cost_summary(run.expected_count, calculations)
    reliability = _reliability_summary(run, benchmark)
    if complete:
        usable_latency = tuple(
            attempt.request_latency_ms for attempt in benchmark if attempt.usable
        )
        queue_wait = tuple(attempt.queue_wait_ms for attempt in benchmark)
        latency = LatencySummary(
            percentile_method_version=PERCENTILE_METHOD_VERSION,
            is_comparable=True,
            concurrency=run.concurrency,
            usable_count=len(usable_latency),
            expected_count=run.expected_count,
            p50_usable_request_latency_ms=linear_percentile(usable_latency, 50),
            p95_usable_request_latency_ms=linear_percentile(usable_latency, 95),
            p50_queue_wait_ms=linear_percentile(queue_wait, 50),
            p95_queue_wait_ms=linear_percentile(queue_wait, 95),
        )
        wall_clock_seconds = Decimal(str(run.wall_clock_ms)) / Decimal("1000")
        throughput = ThroughputSummary(
            run_wall_clock_ms=run.wall_clock_ms,
            run_wall_clock_seconds=wall_clock_seconds,
            sustained_requests_per_second=(
                Decimal(run.expected_count) / wall_clock_seconds
            ),
            usable_classifications_per_second=(
                Decimal(run.usable_count) / wall_clock_seconds
            ),
        )
    else:
        latency = LatencySummary(
            percentile_method_version=PERCENTILE_METHOD_VERSION,
            is_comparable=False,
            concurrency=run.concurrency,
            usable_count=sum(attempt.usable for attempt in benchmark),
            expected_count=run.expected_count,
        )
        throughput = None

    primary = (
        _primary_holdout_cost(
            run,
            benchmark,
            calculations,
            primary_holdout_issue_numbers,
        )
        if primary_holdout_issue_numbers
        else None
    )
    return RunOperationalSummary(
        model_id=run.model_id,
        run_id=run.run_id,
        status=run.status,
        population_name="Full Corpus",
        concurrency=run.concurrency,
        source_expected_count=run.expected_count,
        request_costs=tuple(
            AttemptCostSummary(
                attempt_id=attempt.attempt_id,
                issue_number=attempt.issue_number,
                dispatch_order=attempt.dispatch_order,
                calculation=calculation,
            )
            for attempt, calculation in zip(benchmark, calculations, strict=True)
        ),
        cost=cost,
        primary_holdout_cost=primary,
        latency=latency,
        throughput=throughput,
        reliability=reliability,
    )


def linear_percentile(values: tuple[float, ...], percentile: float) -> float | None:
    """NumPy-compatible linear interpolation without a runtime dependency."""

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if not values:
        return None
    ordered = tuple(sorted(values))
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def _validated_benchmark_attempts(
    run: RunManifest, attempts: tuple[AttemptEvidence, ...]
) -> tuple[AttemptEvidence, ...]:
    benchmark = tuple(
        attempt
        for attempt in attempts
        if attempt.attempt_purpose is AttemptPurpose.BENCHMARK
    )
    if any(attempt.run_id != run.run_id for attempt in benchmark):
        raise MetricIntegrityError("Attempt Evidence references the wrong run")
    if len({attempt.issue_number for attempt in benchmark}) != len(benchmark):
        raise MetricIntegrityError("Duplicate benchmark issue found in Attempt Evidence")
    if len({attempt.dispatch_order for attempt in benchmark}) != len(benchmark):
        raise MetricIntegrityError("Duplicate benchmark dispatch order")
    if len(benchmark) != run.persisted_count:
        raise MetricIntegrityError("Attempt Evidence does not match persisted run count")
    ordered = tuple(sorted(benchmark, key=lambda attempt: attempt.dispatch_order))
    for attempt in ordered:
        if attempt.dispatch_order >= run.expected_count:
            raise MetricIntegrityError("Benchmark dispatch order is outside the run")
        if run.ordered_issue_numbers[attempt.dispatch_order] != attempt.issue_number:
            raise MetricIntegrityError("Attempt issue disagrees with frozen dispatch order")
        if attempt.pricing_snapshot_id != run.pricing_snapshot_id:
            raise MetricIntegrityError("Attempt references a different Pricing Snapshot")
    if run.status is RunStatus.COMPLETE and tuple(
        attempt.issue_number for attempt in ordered
    ) != run.ordered_issue_numbers:
        raise MetricIntegrityError("Complete run lacks its exact Benchmark Attempt population")
    if run.status is RunStatus.RUNNING:
        raise MetricIntegrityError("Operational summary requires a terminal run")
    return ordered


def _verify_persisted_cost(
    attempt: AttemptEvidence, calculation: RequestCostCalculation
) -> None:
    if attempt.schema_version != "attempt_evidence.v3":
        return
    if attempt.cost_formula_version != calculation.formula_version:
        raise MetricIntegrityError("Stored cost formula version disagrees with calculator")
    if attempt.cost_completeness is not calculation.completeness:
        raise MetricIntegrityError("Stored Cost Completeness disagrees with raw usage")
    if attempt.calculated_request_cost_usd != calculation.calculated_cost_usd:
        raise MetricIntegrityError("Stored Calculated Request Cost disagrees with raw usage")
    if attempt.cost_calculation_terms != calculation.terms:
        raise MetricIntegrityError("Stored cost terms disagree with raw usage and prices")
    if attempt.cost_unknown_reasons != calculation.unknown_reasons:
        raise MetricIntegrityError("Stored unknown-cost reasons disagree with raw usage")


def _cost_summary(
    expected_count: int, calculations: tuple[RequestCostCalculation, ...]
) -> RunCostSummary:
    known = tuple(
        calculation.calculated_cost_usd
        for calculation in calculations
        if calculation.calculated_cost_usd is not None
    )
    known_total = sum(known, Decimal("0"))
    known_count = len(known)
    unknown_count = expected_count - known_count
    complete = unknown_count == 0
    return RunCostSummary(
        expected_count=expected_count,
        observed_count=len(calculations),
        known_cost_count=known_count,
        unknown_cost_count=unknown_count,
        unobserved_count=expected_count - len(calculations),
        known_total_cost_usd=known_total,
        mean_known_request_cost_usd=(
            known_total / known_count if known_count else None
        ),
        cost_per_expected_classification_usd=known_total / expected_count,
        completeness=(
            AggregateCostCompleteness.COMPLETE
            if complete
            else AggregateCostCompleteness.PARTIAL
        ),
        is_lower_bound=not complete,
    )


def _primary_holdout_cost(
    run: RunManifest,
    attempts: tuple[AttemptEvidence, ...],
    calculations: tuple[RequestCostCalculation, ...],
    issue_numbers: tuple[int, ...],
) -> PrimaryHoldoutCostSummary:
    if len(set(issue_numbers)) != len(issue_numbers):
        raise MetricIntegrityError("Primary Holdout issue numbers must be unique")
    if not set(issue_numbers) <= set(run.ordered_issue_numbers):
        raise MetricIntegrityError("Primary Holdout is not contained in the run")
    by_issue = {
        attempt.issue_number: (attempt, calculation)
        for attempt, calculation in zip(attempts, calculations, strict=True)
    }
    if not set(issue_numbers) <= set(by_issue):
        raise MetricIntegrityError("Primary Holdout Attempt Evidence is incomplete")
    selected = tuple(by_issue[issue_number] for issue_number in issue_numbers)
    if any(attempt.scored_outcome is None for attempt, _ in selected):
        raise MetricIntegrityError(
            "Primary Holdout cost per correct requires a Scored Outcome for every item"
        )
    known = tuple(
        calculation.calculated_cost_usd
        for _, calculation in selected
        if calculation.calculated_cost_usd is not None
    )
    known_total = sum(known, Decimal("0"))
    correct_count = sum(
        attempt.scored_outcome is ScoredOutcome.CORRECT for attempt, _ in selected
    )
    unknown_count = len(selected) - len(known)
    return PrimaryHoldoutCostSummary(
        population_name="Primary Scored Holdout",
        expected_count=len(selected),
        correct_count=correct_count,
        known_total_cost_usd=known_total,
        unknown_cost_count=unknown_count,
        completeness=(
            AggregateCostCompleteness.COMPLETE
            if unknown_count == 0
            else AggregateCostCompleteness.PARTIAL
        ),
        cost_per_correct_usd=(
            known_total / correct_count if correct_count else None
        ),
        cost_per_correct_status=(
            "defined" if correct_count else "undefined_zero_correct"
        ),
    )


def _reliability_summary(
    run: RunManifest, attempts: tuple[AttemptEvidence, ...]
) -> ReliabilitySummary:
    invalid_count = 0
    detailed: dict[ProviderOutcome, int] = {}
    for attempt in attempts:
        if attempt.usable:
            continue
        if attempt.provider_outcome is ProviderOutcome.SUCCESS or (
            attempt.provider_outcome is None
            and attempt.scored_outcome is not ScoredOutcome.REQUEST_ERROR
        ):
            invalid_count += 1
            continue
        outcome = attempt.provider_outcome or ProviderOutcome.UNKNOWN
        detailed[outcome] = detailed.get(outcome, 0) + 1
    request_error_count = sum(detailed.values())
    usable_count = sum(attempt.usable for attempt in attempts)
    if (
        usable_count,
        invalid_count,
        request_error_count,
    ) != (
        run.usable_count,
        run.invalid_output_count,
        run.request_error_count,
    ):
        raise MetricIntegrityError("Run terminal counts disagree with Attempt Evidence")
    rate_limit_count = detailed.get(ProviderOutcome.RATE_LIMIT, 0)
    timeout_count = detailed.get(ProviderOutcome.TIMEOUT, 0)
    other_count = request_error_count - rate_limit_count - timeout_count
    unusable_count = invalid_count + request_error_count
    expected = run.expected_count
    return ReliabilitySummary(
        is_comparable=run.status is RunStatus.COMPLETE,
        expected_count=expected,
        observed_count=len(attempts),
        unobserved_count=expected - len(attempts),
        usable_count=usable_count,
        invalid_output_count=invalid_count,
        request_error_count=request_error_count,
        unusable_count=unusable_count,
        invalid_output_rate=invalid_count / expected,
        request_error_rate=request_error_count / expected,
        unusable_rate=unusable_count / expected,
        rate_limit_count=rate_limit_count,
        timeout_count=timeout_count,
        other_request_error_count=other_count,
        rate_limit_rate=rate_limit_count / expected,
        timeout_rate=timeout_count / expected,
        other_request_error_rate=other_count / expected,
        detailed_request_error_counts=detailed,
    )
