from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from inferencebench.artifacts import canonical_sha256
from inferencebench.evaluation.bootstrap import BootstrapInterval, PairBootstrapSummary
from inferencebench.workflows.candidate_baseline import (
    CandidateBootstrapAnalysis,
    CandidateEvidenceRow,
    CandidateEvidenceTable,
)
from inferencebench.domain import Sha256, StrictModel


class DominanceEvidence(StrictModel):
    comparator_model_id: str
    candidate_accuracy: float = Field(ge=0, le=1)
    comparator_accuracy: float = Field(ge=0, le=1)
    candidate_supported_class_macro_f1: float = Field(ge=0, le=1)
    comparator_supported_class_macro_f1: float = Field(ge=0, le=1)
    candidate_cost_per_correct_usd: Decimal
    comparator_cost_per_correct_usd: Decimal
    candidate_invalid_output_rate: float = Field(ge=0, le=1)
    comparator_invalid_output_rate: float = Field(ge=0, le=1)
    candidate_request_error_rate: float = Field(ge=0, le=1)
    comparator_request_error_rate: float = Field(ge=0, le=1)
    candidate_exact_output_rate: float = Field(ge=0, le=1)
    comparator_exact_output_rate: float = Field(ge=0, le=1)
    per_class_precision_and_recall_no_worse: Literal[True]
    bootstrap_supported_advantages: tuple[
        Literal["accuracy", "supported_class_macro_f1", "cost_per_correct"], ...
    ]


class CandidateScreeningDisposition(StrictModel):
    model_id: str
    disposition: Literal["retained", "screened_out", "not_comparable"]
    reason: str
    comparator_model_id: str | None
    dominance_evidence: DominanceEvidence | None

    @model_validator(mode="after")
    def validate_disposition(self) -> "CandidateScreeningDisposition":
        screened_out = self.disposition == "screened_out"
        if screened_out != (
            self.comparator_model_id is not None and self.dominance_evidence is not None
        ):
            raise ValueError(
                "Only a screened-out candidate may name a comparator and dominance evidence"
            )
        return self


class CandidateScreeningDispositions(StrictModel):
    schema_version: Literal["candidate_screening_dispositions.v1"]
    baseline_version: str
    plan_sha256: Sha256
    candidate_evidence_sha256: Sha256
    bootstrap_analysis_sha256: Sha256
    candidate_count: Literal[25]
    dispositions: tuple[CandidateScreeningDisposition, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_dispositions(self) -> "CandidateScreeningDispositions":
        if len(self.dispositions) != self.candidate_count:
            raise ValueError("Every active candidate requires one screening disposition")
        if len({row.model_id for row in self.dispositions}) != self.candidate_count:
            raise ValueError("Candidate screening dispositions must be unique")
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        if canonical_sha256(payload) != self.content_sha256:
            raise ValueError("Candidate screening disposition content hash is invalid")
        return self


def derive_candidate_screening_dispositions(
    evidence: CandidateEvidenceTable,
    bootstrap: CandidateBootstrapAnalysis,
) -> CandidateScreeningDispositions:
    """Apply the frozen conservative screen without calling a provider.

    Latency is deliberately not a dominance dimension here: one concurrency-4
    screen is useful context, but ADR 0009 says it is not repeated operational
    evidence. A model with incomplete cost evidence is retained but cannot be
    described as comparable or cheap.
    """

    if evidence.baseline_version != bootstrap.baseline_version:
        raise ValueError("Candidate evidence and bootstrap analysis have different versions")
    if evidence.plan_sha256 != bootstrap.baseline_plan_sha256:
        raise ValueError("Candidate evidence and bootstrap analysis reference different plans")

    bootstrap_by_model = {row.model_id: row for row in bootstrap.candidates}
    pairs = {
        frozenset((row.model_a_id, row.model_b_id)): row for row in bootstrap.pairs
    }
    comparable = {
        row.model_id: _comparable_reason(row, bootstrap_by_model.get(row.model_id))
        for row in evidence.rows
    }
    dispositions: list[CandidateScreeningDisposition] = []

    for candidate in evidence.rows:
        unavailable_reason = comparable[candidate.model_id]
        if unavailable_reason is not None:
            dispositions.append(
                CandidateScreeningDisposition(
                    model_id=candidate.model_id,
                    disposition="not_comparable",
                    reason=unavailable_reason,
                    comparator_model_id=None,
                    dominance_evidence=None,
                )
            )
            continue

        dominance: DominanceEvidence | None = None
        for comparator in evidence.rows:
            if comparator.model_id == candidate.model_id:
                continue
            if comparable[comparator.model_id] is not None:
                continue
            pair = pairs.get(frozenset((candidate.model_id, comparator.model_id)))
            if pair is None:
                raise ValueError("Candidate bootstrap analysis is missing a model pair")
            dominance = _dominance_evidence(candidate, comparator, pair)
            if dominance is not None:
                break

        if dominance is None:
            dispositions.append(
                CandidateScreeningDisposition(
                    model_id=candidate.model_id,
                    disposition="retained",
                    reason=(
                        "No single comparable model satisfies every conservative "
                        "screening safeguard with a bootstrap-supported advantage."
                    ),
                    comparator_model_id=None,
                    dominance_evidence=None,
                )
            )
        else:
            dispositions.append(
                CandidateScreeningDisposition(
                    model_id=candidate.model_id,
                    disposition="screened_out",
                    reason=(
                        "One same comparator is no worse on every frozen screening "
                        "dimension and has a bootstrap-supported advantage."
                    ),
                    comparator_model_id=dominance.comparator_model_id,
                    dominance_evidence=dominance,
                )
            )

    payload = {
        "schema_version": "candidate_screening_dispositions.v1",
        "baseline_version": evidence.baseline_version,
        "plan_sha256": evidence.plan_sha256,
        "candidate_evidence_sha256": evidence.content_sha256,
        "bootstrap_analysis_sha256": bootstrap.content_sha256,
        "candidate_count": 25,
        "dispositions": [row.model_dump(mode="json") for row in dispositions],
    }
    return CandidateScreeningDispositions.model_validate(
        {**payload, "content_sha256": canonical_sha256(payload)}
    )


def load_candidate_screening_dispositions(path: Path) -> CandidateScreeningDispositions:
    return CandidateScreeningDispositions.model_validate_json(path.read_text(encoding="utf-8"))


def _comparable_reason(
    row: CandidateEvidenceRow, bootstrap_candidate: object | None
) -> str | None:
    if not row.comparable_headlines or row.candidate_screening_holdout is None:
        return row.exclusion_reason or "Candidate lacks complete screening quality evidence."
    if row.candidate_screening_cost_completeness != "complete":
        return "Candidate cost completeness is not complete; apparent cheapness cannot dominate."
    if row.candidate_screening_cost_per_correct_usd is None:
        return "Candidate cost per correct is unavailable."
    if row.output_adherence is None:
        return "Candidate Output Adherence evidence is unavailable."
    if (
        bootstrap_candidate is None
        or getattr(bootstrap_candidate, "evidence_status", None) != "complete"
        or getattr(bootstrap_candidate, "intervals", None) is None
    ):
        return "Candidate lacks complete shared-bootstrap evidence."
    return None


def _dominance_evidence(
    candidate: CandidateEvidenceRow,
    comparator: CandidateEvidenceRow,
    pair: PairBootstrapSummary,
) -> DominanceEvidence | None:
    candidate_summary = candidate.candidate_screening_holdout
    comparator_summary = comparator.candidate_screening_holdout
    candidate_adherence = candidate.output_adherence
    comparator_adherence = comparator.output_adherence
    assert candidate_summary is not None
    assert comparator_summary is not None
    assert candidate_adherence is not None
    assert comparator_adherence is not None
    assert candidate.candidate_screening_cost_per_correct_usd is not None
    assert comparator.candidate_screening_cost_per_correct_usd is not None

    no_worse = (
        comparator_summary.accuracy >= candidate_summary.accuracy
        and comparator_summary.supported_class_macro_f1
        >= candidate_summary.supported_class_macro_f1
        and comparator.candidate_screening_cost_per_correct_usd
        <= candidate.candidate_screening_cost_per_correct_usd
        and comparator_summary.invalid_output_count <= candidate_summary.invalid_output_count
        and comparator_summary.request_error_count <= candidate_summary.request_error_count
        and comparator_adherence.exact_rate >= candidate_adherence.exact_rate
        and _per_class_no_worse(candidate, comparator)
    )
    if not no_worse or pair.evidence_status != "complete" or pair.intervals is None:
        return None

    advantages = _bootstrap_supported_advantages(
        pair, comparator.model_id, candidate.model_id
    )
    if not advantages:
        return None
    return DominanceEvidence(
        comparator_model_id=comparator.model_id,
        candidate_accuracy=candidate_summary.accuracy,
        comparator_accuracy=comparator_summary.accuracy,
        candidate_supported_class_macro_f1=candidate_summary.supported_class_macro_f1,
        comparator_supported_class_macro_f1=comparator_summary.supported_class_macro_f1,
        candidate_cost_per_correct_usd=candidate.candidate_screening_cost_per_correct_usd,
        comparator_cost_per_correct_usd=comparator.candidate_screening_cost_per_correct_usd,
        candidate_invalid_output_rate=candidate_summary.invalid_output_count
        / candidate_summary.expected_count,
        comparator_invalid_output_rate=comparator_summary.invalid_output_count
        / comparator_summary.expected_count,
        candidate_request_error_rate=candidate_summary.request_error_count
        / candidate_summary.expected_count,
        comparator_request_error_rate=comparator_summary.request_error_count
        / comparator_summary.expected_count,
        candidate_exact_output_rate=candidate_adherence.exact_rate,
        comparator_exact_output_rate=comparator_adherence.exact_rate,
        per_class_precision_and_recall_no_worse=True,
        bootstrap_supported_advantages=advantages,
    )


def _per_class_no_worse(
    candidate: CandidateEvidenceRow, comparator: CandidateEvidenceRow
) -> bool:
    candidate_classes = candidate.candidate_screening_holdout.per_class  # type: ignore[union-attr]
    comparator_classes = comparator.candidate_screening_holdout.per_class  # type: ignore[union-attr]
    for candidate_class, comparator_class in zip(
        candidate_classes, comparator_classes, strict=True
    ):
        if candidate_class.recall is not None and (
            comparator_class.recall is None
            or comparator_class.recall < candidate_class.recall
        ):
            return False
        if candidate_class.precision is not None and (
            comparator_class.precision is None
            or comparator_class.precision < candidate_class.precision
        ):
            return False
    return True


def _bootstrap_supported_advantages(
    pair: PairBootstrapSummary,
    comparator_model_id: str,
    candidate_model_id: str,
) -> tuple[Literal["accuracy", "supported_class_macro_f1", "cost_per_correct"], ...]:
    assert pair.intervals is not None
    comparator_is_model_a = pair.model_a_id == comparator_model_id
    if not comparator_is_model_a and pair.model_b_id != comparator_model_id:
        raise ValueError("Pair bootstrap evidence does not contain the comparator")
    if frozenset((pair.model_a_id, pair.model_b_id)) != frozenset(
        (comparator_model_id, candidate_model_id)
    ):
        raise ValueError("Pair bootstrap evidence does not contain the screened candidate")

    advantages: list[Literal["accuracy", "supported_class_macro_f1", "cost_per_correct"]] = []
    if _quality_favours_comparator(pair.intervals.accuracy, comparator_is_model_a):
        advantages.append("accuracy")
    if _quality_favours_comparator(
        pair.intervals.supported_class_macro_f1, comparator_is_model_a
    ):
        advantages.append("supported_class_macro_f1")
    if _cost_favours_comparator(
        pair.intervals.cost_per_correct, comparator_is_model_a
    ):
        advantages.append("cost_per_correct")
    return tuple(advantages)


def _quality_favours_comparator(
    interval: BootstrapInterval, comparator_is_model_a: bool
) -> bool:
    boundary = interval.lower_95 if comparator_is_model_a else interval.upper_95
    return (
        boundary.state == "finite"
        and boundary.value is not None
        and (boundary.value > 0 if comparator_is_model_a else boundary.value < 0)
    )


def _cost_favours_comparator(
    interval: BootstrapInterval, comparator_is_model_a: bool
) -> bool:
    boundary = interval.upper_95 if comparator_is_model_a else interval.lower_95
    return (
        boundary.state == "finite"
        and boundary.value is not None
        and (boundary.value < 0 if comparator_is_model_a else boundary.value > 0)
    )
