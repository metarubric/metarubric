#!/usr/bin/env python3
"""Core contracts for the Adaptive Compositional Rubric Evaluator (ACRE).

The external language-model evaluator reports structured observations from
multiple protocol views.  This module validates those observations and reduces
them deterministically to one reward.  It contains no network or training
framework dependencies, which keeps the scoring contract testable off GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence


class EvaluatorContractError(ValueError):
    """Raised when rubric inputs or evaluator outputs violate the contract."""


FAILURE_TAGS = frozenset(
    {
        "partial_compound",
        "implicit_as_explicit",
        "imprecise_verification",
        "omission",
        "unsupported_claim",
        "generic_response",
        "verbosity",
        "irrelevance",
    }
)

FAILURE_TAG_CODES = {
    "pc": "partial_compound",
    "ie": "implicit_as_explicit",
    "iv": "imprecise_verification",
    "om": "omission",
    "uc": "unsupported_claim",
    "gr": "generic_response",
    "ve": "verbosity",
    "ir": "irrelevance",
}

EXECUTION_CONTRACT_VERSION = "acre-execution-resolution-v1"


@dataclass(frozen=True)
class RubricItem:
    criterion_id: str
    criterion: str
    points: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluatorConfiguration:
    """Versioned execution configuration used by one or more rollout groups."""

    version: int = 0
    effective_group: int = 0
    atomization_level: int = 1
    require_explicit_support: bool = False
    full_claim_scan: bool = False
    strict_coverage: bool = False
    strict_holistic: bool = False

    def validate(self) -> None:
        if self.version < 0 or self.effective_group < 0:
            raise EvaluatorContractError("configuration counters must be non-negative")
        if self.atomization_level not in {1, 2, 3}:
            raise EvaluatorContractError("atomization_level must be 1, 2, or 3")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluatorConfiguration":
        configuration = cls(
            version=int(raw.get("version", 0)),
            effective_group=int(raw.get("effective_group", 0)),
            atomization_level=int(raw.get("atomization_level", 1)),
            require_explicit_support=bool(raw.get("require_explicit_support", False)),
            full_claim_scan=bool(raw.get("full_claim_scan", False)),
            strict_coverage=bool(raw.get("strict_coverage", False)),
            strict_holistic=bool(raw.get("strict_holistic", False)),
        )
        configuration.validate()
        return configuration


def execution_capabilities(
    configuration: EvaluatorConfiguration,
) -> tuple[str, ...]:
    """Return the behavioral checks activated by one evaluator configuration."""

    configuration.validate()
    capabilities = {"criterion_scoring"}
    if configuration.atomization_level >= 2:
        capabilities.add("limited_conjunct_enumeration")
    if configuration.atomization_level >= 3:
        capabilities.add("complete_conjunct_enumeration")
    if configuration.require_explicit_support:
        capabilities.add("explicit_support")
    if configuration.full_claim_scan:
        capabilities.add("full_claim_scan")
    if configuration.strict_coverage:
        capabilities.add("strict_coverage")
    if configuration.strict_holistic:
        capabilities.add("strict_holistic")
    return tuple(sorted(capabilities))


def compile_execution_instructions(
    configuration: EvaluatorConfiguration,
) -> tuple[str, ...]:
    """Compile configuration flags into an explicit, auditable judge procedure."""

    configuration.validate()
    if configuration.atomization_level == 1:
        atomic = (
            "Atomic procedure: treat each rubric row as one single stated obligation."
        )
    elif configuration.atomization_level == 2:
        atomic = (
            "Atomic procedure: enumerate up to two material conjuncts and use the "
            "weakest conjunct score."
        )
    else:
        atomic = (
            "Atomic procedure: enumerate every material conjunct and use the weakest "
            "conjunct score."
        )

    support = (
        "Support procedure: require direct evidence for explicit credit and do not "
        "promote implication to an explicit statement."
        if configuration.require_explicit_support
        else "Support procedure: apply only the support requirement stated by the rubric."
    )
    claim_scan = (
        "Claim-scan procedure: scan every material claim in the response for support."
        if configuration.full_claim_scan
        else "Claim-scan procedure: inspect only claims directly aligned to a rubric row."
    )
    coverage = (
        "Coverage procedure: compare every positive obligation with the response and "
        "flag omissions and generic substitutions."
        if configuration.strict_coverage
        else "Coverage procedure: check direct criterion coverage without an obligation list."
    )
    holistic = (
        "Holistic procedure: inspect the full response for verbosity and irrelevance; "
        "holistic scores may only reduce credit."
        if configuration.strict_holistic
        else "Holistic procedure: score ordinary relevance, coherence, and concision."
    )
    return (atomic, support, claim_scan, coverage, holistic)


def execution_contract_digest(configuration: EvaluatorConfiguration) -> str:
    """Hash behavioral instructions without coupling them to state counters."""

    payload = {
        "version": EXECUTION_CONTRACT_VERSION,
        "capabilities": execution_capabilities(configuration),
        "instructions": compile_execution_instructions(configuration),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CriterionEvaluation:
    criterion_id: str
    atomic_score: float
    coverage_score: float
    support_score: float
    bundle_score: float
    failure_tags: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    rubric_score: float
    global_claim_support: float
    holistic_score: float
    negative_trigger_score: float
    negative_accuracy_trigger_score: float
    criteria: tuple[CriterionEvaluation, ...]
    failure_tags: tuple[str, ...]
    configuration_version: int
    execution_contract_version: str
    execution_instruction_hash: str
    execution_instructions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rubric_score": self.rubric_score,
            "global_claim_support": self.global_claim_support,
            "holistic_score": self.holistic_score,
            "negative_trigger_score": self.negative_trigger_score,
            "negative_accuracy_trigger_score": self.negative_accuracy_trigger_score,
            "criteria": [asdict(item) for item in self.criteria],
            "failure_tags": list(self.failure_tags),
            "configuration_version": self.configuration_version,
            "execution_contract_version": self.execution_contract_version,
            "execution_instruction_hash": self.execution_instruction_hash,
            "execution_instructions": list(self.execution_instructions),
        }


def _bounded_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorContractError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise EvaluatorContractError(f"{label} must be within [0, 1]")
    return score


def normalize_rubrics(raw_rubrics: Sequence[Mapping[str, Any]]) -> tuple[RubricItem, ...]:
    if not isinstance(raw_rubrics, Sequence) or isinstance(raw_rubrics, (str, bytes)):
        raise EvaluatorContractError("rubrics must be a sequence")
    normalized: list[RubricItem] = []
    for index, raw in enumerate(raw_rubrics):
        if not isinstance(raw, Mapping):
            raise EvaluatorContractError(f"rubric {index} must be an object")
        criterion = str(raw.get("criterion", "")).strip()
        points = raw.get("points")
        if not criterion:
            raise EvaluatorContractError(f"rubric {index} has no criterion text")
        if isinstance(points, bool) or not isinstance(points, (int, float)):
            raise EvaluatorContractError(f"rubric {index} has invalid points")
        points_value = float(points)
        if not math.isfinite(points_value) or points_value == 0:
            raise EvaluatorContractError(f"rubric {index} points must be non-zero")
        raw_tags = raw.get("tags", [])
        if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)):
            raise EvaluatorContractError(f"rubric {index} tags must be a sequence")
        normalized.append(
            RubricItem(
                criterion_id=f"r{index:03d}",
                criterion=criterion,
                points=points_value,
                tags=tuple(sorted({str(tag).strip() for tag in raw_tags if str(tag).strip()})),
            )
        )
    if not normalized:
        raise EvaluatorContractError("at least one rubric is required")
    return tuple(normalized)


def rubric_digest(rubrics: Sequence[RubricItem]) -> str:
    payload = [asdict(item) for item in rubrics]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def empty_response_result(
    rubrics: Sequence[RubricItem],
    configuration: EvaluatorConfiguration,
) -> EvaluationResult:
    """Return a deterministic zero reward while preserving group accounting."""

    configuration.validate()
    if not rubrics or not any(item.points > 0 for item in rubrics):
        raise EvaluatorContractError("at least one positive-point rubric is required")
    criteria = tuple(
        CriterionEvaluation(
            criterion_id=item.criterion_id,
            atomic_score=0.0,
            coverage_score=0.0,
            support_score=0.0,
            bundle_score=0.0,
            failure_tags=("generic_response", "omission") if item.points > 0 else (),
        )
        for item in rubrics
    )
    return EvaluationResult(
        score=0.0,
        rubric_score=0.0,
        global_claim_support=0.0,
        holistic_score=0.0,
        negative_trigger_score=0.0,
        negative_accuracy_trigger_score=0.0,
        criteria=criteria,
        failure_tags=("generic_response", "omission"),
        configuration_version=configuration.version,
        execution_contract_version=EXECUTION_CONTRACT_VERSION,
        execution_instruction_hash=execution_contract_digest(configuration),
        execution_instructions=compile_execution_instructions(configuration),
    )


def build_evaluator_request(
    *,
    prompt: Sequence[Mapping[str, Any]] | str,
    response: str,
    rubrics: Sequence[RubricItem],
    configuration: EvaluatorConfiguration,
    evidence_contract: str = "prompt_and_reliable_domain_knowledge",
) -> tuple[str, dict[str, Any]]:
    """Build one joint request that exposes all protocol observations."""

    configuration.validate()
    response = str(response).strip()
    if not response:
        raise EvaluatorContractError("response must not be empty")
    if isinstance(prompt, str):
        prompt_payload: Any = prompt
    elif isinstance(prompt, Sequence):
        prompt_payload = [dict(message) for message in prompt]
    else:
        raise EvaluatorContractError("prompt must be text or chat messages")

    if evidence_contract == "supplied_abstract_only":
        evidence_instruction = (
            "Treat only the supplied abstract in q as evidence. Do not use outside "
            "biomedical knowledge to supply missing support. "
        )
    elif evidence_contract in (
        "prompt_and_reliable_domain_knowledge",
        # Alias only: identical instruction text. See note above.
        "prompt_context_and_reliable_domain_knowledge",
    ):
        evidence_instruction = (
            "Use the question and reliable domain knowledge when checking support. "
        )
    else:
        raise EvaluatorContractError(f"unsupported evidence contract: {evidence_contract}")

    execution_instructions = compile_execution_instructions(configuration)
    execution_block = " ".join(execution_instructions)

    system_prompt = (
        "Score every rubric item with A=atomic satisfaction, C=claim support, "
        "B=bidirectional coverage/support, and H=holistic relevance, "
        "coherence, concision. Positive points measure valid satisfaction; negative "
        "points measure presence of the bad condition. Do not credit topical overlap, "
        "and do not add checks that the execution contract does not activate. "
        + evidence_instruction +
        f"Execution contract {EXECUTION_CONTRACT_VERSION}: {execution_block} "
        "n is the exact required number of score rows. "
        "r rows=[id,points,text]. Return JSON only: "
        '{"s":[[A,coverage,support]],"g":global_support,'
        '"h":[relevance,coherence,concision],"f":[tag_codes]}. '
        "Keep s in r order with one score triple per row. All scores are 0..1. "
        "Tags: pc=partial_compound, "
        "ie=implicit_as_explicit, iv=imprecise_verification, om=omission, "
        "uc=unsupported_claim, gr=generic_response, ve=verbosity, ir=irrelevance."
    )
    payload = {
        "cfg": [
            configuration.atomization_level,
            int(configuration.require_explicit_support),
            int(configuration.full_claim_scan),
            int(configuration.strict_coverage),
            int(configuration.strict_holistic),
        ],
        "n": len(rubrics),
        "q": prompt_payload,
        "a": response,
        "r": [
            [item.criterion_id, item.points, item.criterion]
            for item in rubrics
        ],
    }
    return system_prompt, payload


def _expand_compact_response(
    raw: Mapping[str, Any],
    rubrics: Sequence[RubricItem],
) -> Mapping[str, Any]:
    if "s" not in raw and "c" not in raw:
        return raw
    raw_criteria = raw.get("s", raw.get("c"))
    raw_holistic = raw.get("h")
    if not isinstance(raw_criteria, list) or not isinstance(raw_holistic, list):
        raise EvaluatorContractError("compact criteria and holistic outputs must be lists")

    def expand_tags(value: Any) -> list[str]:
        if not isinstance(value, list):
            raise EvaluatorContractError("compact failure tags must be a list")
        expanded = [FAILURE_TAG_CODES.get(str(tag), str(tag)) for tag in value]
        if any(tag not in FAILURE_TAGS for tag in expanded):
            raise EvaluatorContractError("invalid compact failure tag")
        return expanded

    criteria = []
    for index, row in enumerate(raw_criteria):
        if not isinstance(row, list):
            raise EvaluatorContractError(f"compact criterion {index} must be a list")
        if "s" in raw:
            if len(row) != 3 or index >= len(rubrics):
                raise EvaluatorContractError(
                    f"compact criterion {index} must have three ordered scores"
                )
            criterion_id = rubrics[index].criterion_id
            scores = row
            tags: list[str] = []
        else:
            if len(row) != 5:
                raise EvaluatorContractError(
                    f"legacy compact criterion {index} must have five fields"
                )
            criterion_id = row[0]
            scores = row[1:4]
            tags = expand_tags(row[4])
        criteria.append(
            {
                "criterion_id": criterion_id,
                "atomic_score": scores[0],
                "coverage_score": scores[1],
                "support_score": scores[2],
                "failure_tags": tags,
            }
        )
    if len(raw_holistic) != 3:
        raise EvaluatorContractError("compact holistic output must have three scores")
    return {
        "criteria": criteria,
        "global_claim_support": raw.get("g"),
        "holistic": {
            "relevance": raw_holistic[0],
            "coherence": raw_holistic[1],
            "concision": raw_holistic[2],
        },
        "failure_tags": expand_tags(raw.get("f", [])),
    }


# --------------------------------------------------------------- aggregation

def aggregate_score(rubric_score: float, global_support: float,
                    holistic_score: float, negative_trigger: float,
                    raw_rubric_score: float | None = None) -> float:
    """MetaRubrics HealthBench scalar reward."""
    return min(rubric_score, global_support) * (0.5 + 0.5 * holistic_score)


def parse_evaluator_response(
    raw: Mapping[str, Any],
    rubrics: Sequence[RubricItem],
    configuration: EvaluatorConfiguration,
    weight_scale: Mapping[str, float] | None = None,
) -> EvaluationResult:
    """Validate multi-protocol observations and compute the scalar reward.

    ``weight_scale`` multiplies each criterion's own weight before the criteria are
    added up, and is how VAC-D runs the same rubric under three weightings. It is
    applied here rather than on the rubric itself because the judge request carries
    the rubric's points (``build_evaluator_request``): rewriting those would change
    what the judge sees and the three runs would no longer share one ruler. The
    aggregation is unchanged -- weight-average over the positive mass, clipped to
    [0, 1], then capped by the support scalar.
    """

    if not isinstance(raw, Mapping):
        raise EvaluatorContractError("evaluator response must be an object")
    raw = _expand_compact_response(raw, rubrics)
    raw_criteria = raw.get("criteria")
    if not isinstance(raw_criteria, list):
        raise EvaluatorContractError("criteria output must be a list")
    expected = {item.criterion_id: item for item in rubrics}
    observed: dict[str, CriterionEvaluation] = {}
    all_tags: set[str] = set()
    for index, record in enumerate(raw_criteria):
        if not isinstance(record, Mapping):
            raise EvaluatorContractError(f"criterion output {index} must be an object")
        criterion_id = str(record.get("criterion_id", ""))
        if criterion_id not in expected or criterion_id in observed:
            raise EvaluatorContractError(f"unexpected or duplicate criterion id: {criterion_id}")
        atomic = _bounded_score(record.get("atomic_score"), f"{criterion_id}.atomic_score")
        coverage = _bounded_score(record.get("coverage_score"), f"{criterion_id}.coverage_score")
        support = _bounded_score(record.get("support_score"), f"{criterion_id}.support_score")
        tags_raw = record.get("failure_tags", [])
        if not isinstance(tags_raw, list) or any(str(tag) not in FAILURE_TAGS for tag in tags_raw):
            raise EvaluatorContractError(f"invalid failure tags for {criterion_id}")
        tags = tuple(sorted({str(tag) for tag in tags_raw}))
        all_tags.update(tags)
        bundle_score = (
            min(atomic, coverage, support)
            if expected[criterion_id].points > 0
            else max(atomic, coverage, support)
        )
        observed[criterion_id] = CriterionEvaluation(
            criterion_id=criterion_id,
            atomic_score=atomic,
            coverage_score=coverage,
            support_score=support,
            bundle_score=bundle_score,
            failure_tags=tags,
        )
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise EvaluatorContractError(f"criterion coverage mismatch: missing={missing}, extra={extra}")

    global_support = _bounded_score(raw.get("global_claim_support"), "global_claim_support")
    holistic = raw.get("holistic")
    if not isinstance(holistic, Mapping):
        raise EvaluatorContractError("holistic output must be an object")
    holistic_scores = (
        _bounded_score(holistic.get("relevance"), "holistic.relevance"),
        _bounded_score(holistic.get("coherence"), "holistic.coherence"),
        _bounded_score(holistic.get("concision"), "holistic.concision"),
    )
    holistic_score = min(holistic_scores)
    top_tags = raw.get("failure_tags", [])
    if not isinstance(top_tags, list) or any(str(tag) not in FAILURE_TAGS for tag in top_tags):
        raise EvaluatorContractError("invalid top-level failure tags")
    all_tags.update(str(tag) for tag in top_tags)

    def _weight(item: RubricItem) -> float:
        if weight_scale is None:
            return item.points
        return item.points * float(weight_scale.get(item.criterion_id, 1.0))

    positive_points = sum(w for w in (_weight(item) for item in rubrics) if w > 0)
    if positive_points <= 0:
        raise EvaluatorContractError("at least one positive-point rubric is required")
    raw_rubric_score = sum(
        _weight(expected[criterion_id]) * observed[criterion_id].bundle_score
        for criterion_id in expected
    ) / positive_points
    rubric_score = min(1.0, max(0.0, raw_rubric_score))
    negative_items = [item for item in rubrics if item.points < 0]
    negative_weight = sum(abs(item.points) for item in negative_items)
    negative_trigger_score = (
        sum(abs(item.points) * observed[item.criterion_id].bundle_score for item in negative_items)
        / negative_weight
        if negative_weight
        else 0.0
    )
    negative_accuracy_items = [
        item for item in negative_items if "axis:accuracy" in item.tags
    ]
    negative_accuracy_weight = sum(abs(item.points) for item in negative_accuracy_items)
    negative_accuracy_trigger_score = (
        sum(
            abs(item.points) * observed[item.criterion_id].bundle_score
            for item in negative_accuracy_items
        )
        / negative_accuracy_weight
        if negative_accuracy_weight
        else 0.0
    )

    score = aggregate_score(rubric_score, global_support, holistic_score,
                            negative_trigger_score,
                            raw_rubric_score=raw_rubric_score)
    ordered = tuple(observed[item.criterion_id] for item in rubrics)
    return EvaluationResult(
        score=score,
        rubric_score=rubric_score,
        global_claim_support=global_support,
        holistic_score=holistic_score,
        negative_trigger_score=negative_trigger_score,
        negative_accuracy_trigger_score=negative_accuracy_trigger_score,
        criteria=ordered,
        failure_tags=tuple(sorted(all_tags)),
        configuration_version=configuration.version,
        execution_contract_version=EXECUTION_CONTRACT_VERSION,
        execution_instruction_hash=execution_contract_digest(configuration),
        execution_instructions=compile_execution_instructions(configuration),
    )


def advance_configuration(
    configuration: EvaluatorConfiguration,
    results: Iterable[EvaluationResult],
    *,
    effective_group: int,
    prevalence_threshold: float = 0.15,
) -> tuple[EvaluatorConfiguration, dict[str, float]]:
    """Refine execution settings from a completed window of policy failures."""

    if not 0.0 <= prevalence_threshold <= 1.0:
        raise EvaluatorContractError("prevalence_threshold must be within [0, 1]")
    result_list = list(results)
    if not result_list:
        return configuration, {tag: 0.0 for tag in sorted(FAILURE_TAGS)}
    counts: Counter[str] = Counter()
    for result in result_list:
        counts.update(set(result.failure_tags))
    prevalence = {
        tag: counts[tag] / len(result_list)
        for tag in sorted(FAILURE_TAGS)
    }

    atomization_level = configuration.atomization_level
    if max(prevalence["partial_compound"], prevalence["imprecise_verification"]) >= prevalence_threshold:
        atomization_level = min(3, atomization_level + 1)
    require_explicit_support = configuration.require_explicit_support or max(
        prevalence["implicit_as_explicit"], prevalence["unsupported_claim"]
    ) >= prevalence_threshold
    full_claim_scan = configuration.full_claim_scan or prevalence["unsupported_claim"] >= prevalence_threshold
    strict_coverage = configuration.strict_coverage or max(
        prevalence["omission"], prevalence["generic_response"]
    ) >= prevalence_threshold
    strict_holistic = configuration.strict_holistic or max(
        prevalence["verbosity"], prevalence["irrelevance"]
    ) >= prevalence_threshold

    proposed = replace(
        configuration,
        atomization_level=atomization_level,
        require_explicit_support=require_explicit_support,
        full_claim_scan=full_claim_scan,
        strict_coverage=strict_coverage,
        strict_holistic=strict_holistic,
    )
    changed = proposed != configuration
    if changed:
        proposed = replace(
            proposed,
            version=configuration.version + 1,
            effective_group=effective_group,
        )
    proposed.validate()
    return proposed, prevalence
