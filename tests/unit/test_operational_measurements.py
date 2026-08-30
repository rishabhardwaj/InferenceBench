from __future__ import annotations

from decimal import Decimal

from inferencebench.artifacts import FixtureArtifacts
from inferencebench.config import Settings
from inferencebench.domain import (
    AttemptEvidence,
    CostCompleteness,
    CustomerLabel,
    ParseStatus,
    ProviderOutcome,
    RunManifest,
    RunStatus,
    ScoredOutcome,
)
from inferencebench.evaluation.cost import calculate_request_cost
from inferencebench.evaluation.operational import (
    AggregateCostCompleteness,
    build_run_operational_summary,
    linear_percentile,
)
from inferencebench.models.domain import ModelPricing, TokenRate


MODEL_ID = "openai-gpt-oss-20b"
PRICING_ID = "controlled-pricing-v1"


def test_decimal_cost_terms_do_not_double_count_cached_or_cache_write_tokens() -> None:
    calculation = calculate_request_cost(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 10,
            },
        },
        _pricing(),
    )

    assert calculation.completeness is CostCompleteness.COMPLETE
    assert calculation.calculated_cost_usd == Decimal("0.0001")
    assert tuple(
        (term.category.value, term.token_count, term.rate_usd, term.cost_usd)
        for term in calculation.terms
    ) == (
        ("standard_input", 70, Decimal("1"), Decimal("0.00007")),
        ("cache_read_input", 20, Decimal("0.25"), Decimal("0.000005")),
        ("cache_write_input", 10, Decimal("0.5"), Decimal("0.000005")),
        ("output", 10, Decimal("2"), Decimal("0.00002")),
    )
    assert sum(
        term.token_count
        for term in calculation.terms
        if term.category.value != "output"
    ) == 100


def test_missing_usage_or_matching_price_is_unknown_not_zero() -> None:
    no_usage = calculate_request_cost({}, _pricing())
    missing_cache_rate = calculate_request_cost(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
        _pricing(cache_read_rate=None),
    )

    assert no_usage.completeness is CostCompleteness.UNKNOWN
    assert no_usage.calculated_cost_usd is None
    assert no_usage.unknown_reasons == (
        "missing prompt_tokens",
        "missing completion_tokens",
    )
    assert missing_cache_rate.completeness is CostCompleteness.UNKNOWN
    assert missing_cache_rate.calculated_cost_usd is None
    assert missing_cache_rate.unknown_reasons == (
        "no published cache_read_input rate for 20 returned tokens",
    )


def test_linear_percentile_freezes_empty_single_and_interpolated_behavior() -> None:
    assert linear_percentile((), 50) is None
    assert linear_percentile((125.0,), 95) == 125.0
    assert linear_percentile((100.0, 200.0, 300.0, 400.0), 50) == 250.0
    assert linear_percentile((100.0, 200.0, 300.0, 400.0), 95) == 385.0


def test_complete_run_reports_partial_cost_latency_throughput_and_failures() -> None:
    pricing = _pricing()
    attempts = _attempts(pricing)
    run = _run(attempts)

    summary = build_run_operational_summary(
        run,
        attempts,
        pricing,
        primary_holdout_issue_numbers=(1, 2, 3, 4, 5),
    )

    assert summary.request_costs[0].calculation.calculated_cost_usd is not None
    assert summary.request_costs[-1].calculation.calculated_cost_usd is None
    assert summary.cost.known_cost_count == 4
    assert summary.cost.unknown_cost_count == 1
    assert summary.cost.completeness is AggregateCostCompleteness.PARTIAL
    assert summary.cost.is_lower_bound is True
    assert summary.cost.cost_per_expected_classification_usd == (
        summary.cost.known_total_cost_usd / 5
    )
    assert summary.primary_holdout_cost is not None
    assert summary.primary_holdout_cost.correct_count == 2
    assert summary.primary_holdout_cost.cost_per_correct_usd == (
        summary.primary_holdout_cost.known_total_cost_usd / 2
    )
    assert summary.primary_holdout_cost.completeness is (
        AggregateCostCompleteness.PARTIAL
    )

    assert summary.latency.usable_count == 3
    assert summary.latency.expected_count == 5
    assert summary.latency.concurrency == 2
    assert summary.latency.p50_usable_request_latency_ms == 200
    assert summary.latency.p95_usable_request_latency_ms == 380
    assert summary.latency.p50_queue_wait_ms == 10
    assert summary.latency.p95_queue_wait_ms == 28
    assert summary.throughput is not None
    assert summary.throughput.run_wall_clock_seconds == Decimal("1")
    assert summary.throughput.sustained_requests_per_second == Decimal("5")
    assert summary.throughput.usable_classifications_per_second == Decimal("3")

    reliability = summary.reliability
    assert reliability.invalid_output_count == 1
    assert reliability.request_error_count == 1
    assert reliability.unusable_count == 2
    assert reliability.invalid_output_rate == 1 / 5
    assert reliability.request_error_rate == 1 / 5
    assert reliability.unusable_rate == 2 / 5
    assert reliability.timeout_count == 1
    assert reliability.rate_limit_count == 0
    assert reliability.other_request_error_count == 0
    assert reliability.detailed_request_error_counts == {ProviderOutcome.TIMEOUT: 1}
    assert attempts[0].dispatch_order == 0
    assert len(summary.request_costs) == run.expected_count


def test_zero_correct_is_undefined_and_incomplete_run_has_no_headlines() -> None:
    pricing = _pricing()
    attempts = tuple(
        attempt.model_copy(update={"scored_outcome": ScoredOutcome.INCORRECT_LABEL})
        if attempt.usable
        else attempt
        for attempt in _attempts(pricing)
    )
    complete = build_run_operational_summary(
        _run(attempts), attempts, pricing, (1, 2, 3, 4, 5)
    )
    assert complete.primary_holdout_cost is not None
    assert complete.primary_holdout_cost.correct_count == 0
    assert complete.primary_holdout_cost.cost_per_correct_usd is None
    assert (
        complete.primary_holdout_cost.cost_per_correct_status
        == "undefined_zero_correct"
    )

    incomplete_run = _run((), status=RunStatus.INCOMPLETE, expected_count=5)
    incomplete = build_run_operational_summary(incomplete_run, (), pricing)
    assert incomplete.throughput is None
    assert incomplete.latency.is_comparable is False
    assert incomplete.latency.p50_usable_request_latency_ms is None
    assert incomplete.reliability.is_comparable is False
    assert incomplete.reliability.unobserved_count == 5
    assert incomplete.cost.completeness is AggregateCostCompleteness.PARTIAL
    assert incomplete.cost.unknown_cost_count == 5


def _pricing(*, cache_read_rate: Decimal | None = Decimal("0.25")) -> ModelPricing:
    def rate(category: str, value: Decimal | None) -> TokenRate:
        return TokenRate(
            category=category,
            availability="published" if value is not None else "not_published",
            rate_usd=value,
            currency="USD",
            unit_tokens=1_000_000,
        )

    return ModelPricing(
        schema_version="model_pricing.v1",
        pricing_snapshot_id=PRICING_ID,
        model_id=MODEL_ID,
        published_catalog_name="Controlled model",
        published_model_reference_url="https://example.invalid/model",
        rates=(
            rate("standard_input", Decimal("1")),
            rate("output", Decimal("2")),
            rate("cache_read_input", cache_read_rate),
            rate("cache_write_input", Decimal("0.5")),
        ),
    )


def _attempts(pricing: ModelPricing) -> tuple[AttemptEvidence, ...]:
    artifacts = FixtureArtifacts(Settings.from_environment().fixture_root)
    base = artifacts.load_run_bundle().attempts[0]
    predictions = (
        CustomerLabel.BUG,
        CustomerLabel.ENHANCEMENT,
        CustomerLabel.QUESTION,
        None,
        None,
    )
    providers = (
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.SUCCESS,
        ProviderOutcome.TIMEOUT,
    )
    outcomes = (
        ScoredOutcome.CORRECT,
        ScoredOutcome.CORRECT,
        ScoredOutcome.INCORRECT_LABEL,
        ScoredOutcome.INVALID_OUTPUT,
        ScoredOutcome.REQUEST_ERROR,
    )
    latencies = (100.0, 200.0, 400.0, 50.0, 30_000.0)
    queue_waits = (0.0, 0.0, 10.0, 20.0, 30.0)
    rows: list[AttemptEvidence] = []
    for index in range(5):
        successful = providers[index] is ProviderOutcome.SUCCESS
        usage = (
            {
                "prompt_tokens": 100 + index,
                "completion_tokens": 10,
                "total_tokens": 110 + index,
                **(
                    {"prompt_tokens_details": {"cached_tokens": 20}}
                    if index == 1
                    else {}
                ),
            }
            if successful
            else {}
        )
        calculation = calculate_request_cost(usage, pricing)
        rows.append(
            AttemptEvidence.model_validate(
                {
                    **base.model_dump(mode="python"),
                    "schema_version": "attempt_evidence.v3",
                    "attempt_id": f"operational-attempt-{index + 1}",
                    "run_id": "operational-run",
                    "issue_number": index + 1,
                    "dispatch_order": index,
                    "provider_outcome": providers[index],
                    "http_status": 200 if successful else None,
                    "raw_response": {"result": index} if successful else None,
                    "raw_error": (
                        None
                        if successful
                        else {"type": ProviderOutcome.TIMEOUT.value}
                    ),
                    "raw_model_output": (
                        predictions[index].value
                        if successful and predictions[index] is not None
                        else ("not-a-label" if successful else None)
                    ),
                    "parsed_label": predictions[index],
                    "parse_status": (
                        ParseStatus.NORMALIZED
                        if index == 1
                        else (
                            ParseStatus.EXACT
                            if predictions[index] is not None
                            else ParseStatus.INVALID
                        )
                    ),
                    "normalizations": ("strip_quotes",) if index == 1 else (),
                    "scored_outcome": outcomes[index],
                    "usable": predictions[index] is not None,
                    "usage": usage,
                    "request_latency_ms": latencies[index],
                    "queue_wait_ms": queue_waits[index],
                    "pricing_snapshot_id": PRICING_ID,
                    "cost_formula_version": calculation.formula_version,
                    "calculated_request_cost_usd": calculation.calculated_cost_usd,
                    "cost_completeness": calculation.completeness,
                    "cost_calculation_terms": calculation.terms,
                    "cost_unknown_reasons": calculation.unknown_reasons,
                }
            )
        )
    return tuple(rows)


def _run(
    attempts: tuple[AttemptEvidence, ...],
    *,
    status: RunStatus = RunStatus.COMPLETE,
    expected_count: int | None = None,
) -> RunManifest:
    base = FixtureArtifacts(
        Settings.from_environment().fixture_root
    ).load_run_bundle().runs[0]
    expected = expected_count if expected_count is not None else len(attempts)
    usable = sum(attempt.usable for attempt in attempts)
    invalid = sum(
        attempt.provider_outcome is ProviderOutcome.SUCCESS and not attempt.usable
        for attempt in attempts
    )
    errors = sum(
        attempt.provider_outcome is not ProviderOutcome.SUCCESS for attempt in attempts
    )
    return RunManifest.model_validate(
        {
            **base.model_dump(mode="python"),
            "run_id": "operational-run",
            "model_id": MODEL_ID,
            "status": status,
            "ordered_issue_numbers": tuple(range(1, expected + 1)),
            "expected_count": expected,
            "persisted_count": len(attempts),
            "usable_count": usable,
            "normalized_count": sum(
                attempt.parse_status is ParseStatus.NORMALIZED for attempt in attempts
            ),
            "invalid_output_count": invalid,
            "request_error_count": errors,
            "pricing_snapshot_id": PRICING_ID,
            "concurrency": 2,
            "ended_at": "2026-08-20T10:00:01Z",
            "wall_clock_ms": 1000,
        }
    )
