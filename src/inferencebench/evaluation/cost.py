from __future__ import annotations

from decimal import Decimal

from pydantic import JsonValue

from inferencebench.domain import (
    CostCalculationTerm,
    CostCategory,
    CostCompleteness,
    StrictModel,
)
from inferencebench.models.domain import ModelPricing, RateAvailability, TokenRate


COST_FORMULA_VERSION = "calculated-request-cost-v1"


class RequestCostCalculation(StrictModel):
    pricing_snapshot_id: str
    model_id: str
    formula_version: str
    terms: tuple[CostCalculationTerm, ...]
    unknown_reasons: tuple[str, ...]
    calculated_cost_usd: Decimal | None
    completeness: CostCompleteness


def calculate_request_cost(
    usage: dict[str, JsonValue], pricing: ModelPricing
) -> RequestCostCalculation:
    """Apply one non-double-counting Decimal formula to provider usage."""

    reasons: list[str] = []
    prompt_tokens = _token_count(usage, "prompt_tokens", reasons)
    completion_tokens = _token_count(usage, "completion_tokens", reasons)
    cached_tokens = 0
    cache_write_tokens = 0
    details = usage.get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            reasons.append("prompt_tokens_details must be an object")
        else:
            cached_tokens = _optional_token_count(
                details, "cached_tokens", "prompt_tokens_details.cached_tokens", reasons
            )
            cache_write_tokens = _optional_token_count(
                details,
                "cache_write_tokens",
                "prompt_tokens_details.cache_write_tokens",
                reasons,
            )

    if prompt_tokens is not None:
        if cached_tokens + cache_write_tokens > prompt_tokens:
            reasons.append(
                "cached and cache-write input tokens exceed total prompt tokens"
            )
        standard_input_tokens = max(
            prompt_tokens - cached_tokens - cache_write_tokens, 0
        )
    else:
        standard_input_tokens = 0

    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        if not _is_token_count(total_tokens):
            reasons.append("total_tokens must be a non-negative integer")
        elif (
            prompt_tokens is not None
            and completion_tokens is not None
            and total_tokens != prompt_tokens + completion_tokens
        ):
            reasons.append("total_tokens disagrees with prompt plus completion tokens")

    rates = {rate.category: rate for rate in pricing.rates}
    token_categories = (
        (
            CostCategory.STANDARD_INPUT,
            "prompt_tokens minus separately priced cached/cache-write tokens",
            standard_input_tokens,
            rates[CostCategory.STANDARD_INPUT.value],
        ),
        (
            CostCategory.CACHE_READ_INPUT,
            "prompt_tokens_details.cached_tokens",
            cached_tokens,
            rates[CostCategory.CACHE_READ_INPUT.value],
        ),
        (
            CostCategory.CACHE_WRITE_INPUT,
            "prompt_tokens_details.cache_write_tokens",
            cache_write_tokens,
            rates[CostCategory.CACHE_WRITE_INPUT.value],
        ),
        (
            CostCategory.OUTPUT,
            "completion_tokens",
            completion_tokens or 0,
            rates[CostCategory.OUTPUT.value],
        ),
    )
    terms: list[CostCalculationTerm] = []
    for category, usage_path, token_count, rate in token_categories:
        if token_count == 0 and category in {
            CostCategory.CACHE_READ_INPUT,
            CostCategory.CACHE_WRITE_INPUT,
        }:
            continue
        if rate.availability is not RateAvailability.PUBLISHED or rate.rate_usd is None:
            if token_count > 0:
                reasons.append(
                    f"no published {category.value} rate for {token_count} returned tokens"
                )
            continue
        terms.append(_cost_term(category, usage_path, token_count, rate))

    unique_reasons = tuple(dict.fromkeys(reasons))
    complete = not unique_reasons
    calculated = (
        sum((term.cost_usd for term in terms), Decimal("0")) if complete else None
    )
    return RequestCostCalculation(
        pricing_snapshot_id=pricing.pricing_snapshot_id,
        model_id=pricing.model_id,
        formula_version=COST_FORMULA_VERSION,
        terms=tuple(terms),
        unknown_reasons=unique_reasons,
        calculated_cost_usd=calculated,
        completeness=(
            CostCompleteness.COMPLETE if complete else CostCompleteness.UNKNOWN
        ),
    )


def _cost_term(
    category: CostCategory,
    usage_path: str,
    token_count: int,
    rate: TokenRate,
) -> CostCalculationTerm:
    assert rate.rate_usd is not None
    cost = Decimal(token_count) * rate.rate_usd / rate.unit_tokens
    return CostCalculationTerm(
        category=category,
        usage_path=usage_path,
        token_count=token_count,
        rate_usd=rate.rate_usd,
        unit_tokens=rate.unit_tokens,
        cost_usd=cost,
    )


def _token_count(
    usage: dict[str, JsonValue], key: str, reasons: list[str]
) -> int | None:
    if key not in usage:
        reasons.append(f"missing {key}")
        return None
    value = usage[key]
    if not _is_token_count(value):
        reasons.append(f"{key} must be a non-negative integer")
        return None
    return value


def _optional_token_count(
    values: dict[str, JsonValue], key: str, path: str, reasons: list[str]
) -> int:
    value = values.get(key, 0)
    if not _is_token_count(value):
        reasons.append(f"{path} must be a non-negative integer")
        return 0
    return value


def _is_token_count(value: object) -> bool:
    return type(value) is int and value >= 0
