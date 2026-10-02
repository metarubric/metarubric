#!/usr/bin/env python3
"""verl custom reward hook for the MetaRubrics evaluator service.

Rubric weights are computed from the current snapshot before each grade, and
criterion-level results can be recorded for the outer MetaRubrics update.
"""

from __future__ import annotations
import math

import asyncio
import json
import os
import re
import uuid
from typing import Any, Mapping

import aiohttp
from openai import AsyncOpenAI


DECISION_PATTERN = re.compile(r"^\s*(yes|no|maybe)\b", re.IGNORECASE)


EXAM_EXTRACT_SYSTEM = """You are reading one candidate answer that was written for a patient.
Answer the question using ONLY what this candidate answer actually says.
Do not use outside medical knowledge and do not infer what the answer implies.

If the candidate answer gives a conditional or branching recommendation (for
example "if X do this, if not X do that") without committing to which branch
applies to this patient, then it has not stated a recommendation: choose the
option that says the answer does not say.

If the candidate answer does not state it, choose the option that says so.

The options are numbered starting at 0. Reply with that exact option number only,
as a single integer, and never renumber the options."""


def _exam_render(item: Mapping[str, Any], response: str) -> str:
    # State the zero-based choices explicitly so invalid indices remain detectable.
    opts = "\n".join(f"({i}) {o}" for i, o in enumerate(item["options"]))
    allowed = ", ".join(str(i) for i in range(len(item["options"])))
    return (f"Candidate answer:\n---\n{response}\n---\n\n{item['stem']}\n{opts}"
            f"\n\nAnswer with exactly one of these numbers: {allowed}")


def _exam_index(text: str, n: int) -> int | None:
    found = re.findall(r"-?\d+", text or "")
    if not found:
        return None
    value = int(found[-1])
    return value if 0 <= value < n else None


async def _exam_score(client, url: str, model: str, response: str,
                      items: list[Mapping[str, Any]]) -> tuple[float | None, int]:
    """Compute the signed extraction-exam score without a zero floor."""
    async def one(item):
        body = {"model": model, "max_tokens": 512, "temperature": 0.0,
                "messages": [{"role": "system", "content": EXAM_EXTRACT_SYSTEM},
                             {"role": "user", "content": _exam_render(item, response)}]}
        if model == "exam-qwen3-1.7b":
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            extra_body = None
            body["reasoning_effort"] = "low"
        completion = await client.chat.completions.create(**body, extra_body=extra_body)
        text = completion.choices[0].message.content
        return _exam_index(text, len(item["options"]))

    answers = await asyncio.gather(*(one(it) for it in items), return_exceptions=True)
    positive_mass = sum(abs(float(it["points"])) for it in items if float(it["points"]) > 0)
    if positive_mass <= 0:
        return None, 0
    total, failed = 0.0, 0
    for got, item in zip(answers, items):
        if isinstance(got, Exception):
            failed += 1
            continue
        weight = abs(float(item["points"]))
        correct = got == int(item["key"])
        if float(item["points"]) > 0:
            total += weight if correct else 0.0
        else:
            total += 0.0 if correct else -weight
    return min(1.0, total / positive_mass), failed


def _decode_contract(ground_truth: Any, extra_info: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        try:
            decoded = json.loads(ground_truth)
        except json.JSONDecodeError as error:
            raise ValueError("ACRE ground_truth must be a JSON object") from error
    elif isinstance(ground_truth, Mapping):
        decoded = dict(ground_truth)
    else:
        raise ValueError("ACRE ground_truth must be JSON text or an object")
    if not isinstance(decoded, dict):
        raise ValueError("ACRE ground_truth root must be an object")
    if extra_info and "sample_id" not in decoded and extra_info.get("sample_id"):
        decoded["sample_id"] = str(extra_info["sample_id"])
    for key in ("sample_id", "prompt", "rubrics"):
        if key not in decoded:
            raise ValueError(f"ACRE contract is missing {key}")
    return decoded



# ---- live rubric weights -------------------------------------------------------
def _anchor_table():
    env = os.environ.get("ACRE_ANCHOR_TABLE", "").strip()
    a = tuple(float(x) for x in env.split(",")) if env else (1.0, 2.0, 4.0, 8.0)
    if len(a) != 4 or any(a[i] >= a[i + 1] for i in range(3)) or a[0] <= 0:
        raise ValueError(f"ACRE_ANCHOR_TABLE must contain four increasing positive values: {env!r}")
    return {"not_applicable": 0.0, "nice_to_have": a[0], "should_have": a[1],
            "must_have": a[2], "contraindication": a[3]}


_ANCHOR = _anchor_table()
_PHI_CACHE: dict[str, Any] = {"mtime": None, "tau": {}}


def _live_tau() -> dict[str, float]:
    path = os.environ.get("ACRE_PHI_PATH", "").strip()
    if not path:
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _PHI_CACHE["tau"]
    if mtime != _PHI_CACHE["mtime"]:
        try:
            with open(path) as handle:
                _PHI_CACHE["tau"] = json.load(handle).get("tau", {})
            _PHI_CACHE["mtime"] = mtime
        except (OSError, json.JSONDecodeError):
            pass
    return _PHI_CACHE["tau"]


def _apply_live_tau(contract: dict[str, Any]) -> None:
    """Recompute criterion points from the current tau snapshot."""
    meta = contract.get("rubric_meta")
    if not meta:
        return
    if len(meta) != len(contract["rubrics"]):
        print(f"[live_tau] rubric_meta length {len(meta)} != rubrics "
              f"{len(contract['rubrics'])}, sample_id={contract.get('sample_id')}; skipping calibration",
              flush=True)
        return
    tau = _live_tau()
    rubrics = []
    for item, m in zip(contract["rubrics"], meta):
        sign = math.copysign(1.0, float(m.get("sign", item["points"])))
        anchor = _ANCHOR[m["severity"]]
        t = float(tau.get(f'{m["severity"]}|{m["edit_label"]}', 0.0))
        rubrics.append({**item, "points": round(sign * anchor * math.exp(t), 6)})
    contract["rubrics"] = rubrics


def _dump_online_scored(contract, result, response_text, reference_label) -> None:
    """Append criterion-level grades for the outer MetaRubrics update."""
    path = os.environ.get("ACRE_ONLINE_SCORED_PATH", "").strip()
    if not path or "criteria" not in result:
        return
    # Validation contracts may omit rubric metadata and therefore are not recorded.
    meta = contract.get("rubric_meta")
    if not meta or len(meta) != len(contract["rubrics"]):
        return
    record = {
        "rollout_id": f'{contract["sample_id"]}#{uuid.uuid4().hex[:8]}',
        "pair_id": contract.get("pair_id", contract["sample_id"]),
        "kind": contract.get("kind", "plain"),
        "side": contract.get("side", "orig"),
        "sample_id": contract["sample_id"],
        "reference_label": reference_label,
        "response": response_text,
        "rubric_hash": result.get("rubric_hash"),
        "configuration_version": result.get("configuration_version"),
        "score": result.get("score"), "rubric_score": result.get("rubric_score"),
        "raw_score": result.get("raw_score"),
        "label_score": result.get("label_score"),
        "reward_mode": result.get("reward_mode"),
        "global_claim_support": result.get("global_claim_support"),
        "holistic_score": result.get("holistic_score"),
        "criteria": result["criteria"],
        "rubric_meta": [{"points": r["points"], "tags": r.get("tags", []),
                         "edit_label": m["edit_label"], "severity": m["severity"]}
                        for r, m in zip(contract["rubrics"], contract["rubric_meta"])],
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a") as handle:
        handle.write(line)



def _final_answer(solution: str) -> str:
    text = str(solution).strip()
    closing = text.rfind("</think>")
    if closing >= 0:
        text = text[closing + len("</think>") :].strip()
    return text


def _decision(solution: str) -> str | None:
    match = DECISION_PATTERN.match(_final_answer(solution))
    return match.group(1).lower() if match else None


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    evaluator_url: str | None = None,
    exam_extractor_url: str | None = None,
    exam_extractor_model: str | None = None,
    exam_weight: float | None = None,
    request_timeout_seconds: float = 180.0,
    max_retries: int = 3,
    **_: Any,
) -> dict[str, Any]:
    """Score one rollout through the centralized Adaptive Rubric Evaluator."""

    if data_source not in {
        "acre_rubrichub_medical",
        "acre_pubmedqa_grounded",
        "acre_healthbench_indomain",
        "acre_healthbench_twin",
    }:
        raise ValueError(f"unsupported data source: {data_source}")
    contract = _decode_contract(ground_truth, extra_info)
    _apply_live_tau(contract)
    reward_mode = os.environ.get("ACRE_REWARD_MODE", "legacy_scalar").strip()
    reference_label = str(contract.get("reference_label", "")).strip().lower()
    if data_source == "acre_pubmedqa_grounded" and reference_label not in {
        "yes",
        "no",
        "maybe",
    }:
        raise ValueError("PubMedQA ground truth requires a valid reference_label")
    rubric_beta = float(os.environ.get("ACRE_RUBRIC_BETA", "0.25"))
    label_only_dual = reward_mode == "correctness_gated_dual" and rubric_beta == 0.0
    if reward_mode == "label_only" or label_only_dual:
        _dec = _decision(solution_str)
        _hc = float(os.environ.get("ACRE_HEDGE_CREDIT", "0"))
        if _dec == reference_label:
            label_score = 1.0
        elif _hc > 0 and _dec == "maybe" and reference_label in {"yes", "no"}:
            label_score = _hc
        else:
            label_score = 0.0
        return {
            "score": label_score,
            "rubric_score": 0.0,
            "claim_support": 0.0,
            "holistic_score": 0.0,
            "negative_trigger": 0.0,
            "negative_accuracy_trigger": 0.0,
            "evaluator_version": 0,
            "failure_count": 0,
            "label_score": label_score,
            "reference_label": reference_label,
            "matrix_advantage": 0.0,
            "rubric_advantage": 0.0,
            "rubric_valid": False,
            "protocol_weight_version": 0,
            "judge_valid": True,
            "judge_failure": "",
            "joint_gated": False,
        }
    local_label_score = float(_decision(solution_str) == reference_label)
    url = (evaluator_url or os.environ.get("ACRE_EVALUATOR_URL", "")).strip().rstrip("/")
    if not url:
        raise RuntimeError("ACRE_EVALUATOR_URL or evaluator_url is required")
    # Explicit arguments allow verl reward workers to receive their service URLs.
    exam_items = list(contract.get("exam_items") or [])
    exam_url = (exam_extractor_url
                or os.environ.get("ACRE_EXAM_EXTRACTOR_URL", "")).strip().rstrip("/")
    exam_model = (exam_extractor_model
                  or os.environ.get("ACRE_EXAM_EXTRACTOR_MODEL", "extractor")).strip()
    exam_weight = float(exam_weight if exam_weight is not None
                        else os.environ.get("ACRE_EXAM_WEIGHT", "0.3"))
    if exam_items and not exam_url:
        raise RuntimeError(
            "ground_truth carries exam_items but no extractor url was given; "
            "pass reward_kwargs.exam_extractor_url")

    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "group_id": str(contract["sample_id"]),
        "prompt": contract["prompt"],
        "response": _final_answer(solution_str),
        "rubrics": contract["rubrics"],
        "evidence_contract": contract.get(
            "evidence_contract", "prompt_and_reliable_domain_knowledge"
        ),
    }
    if reference_label:
        payload["reference_label"] = reference_label
    timeout = aiohttp.ClientTimeout(total=float(request_timeout_seconds))
    last_error: Exception | None = None
    failure_name = "UnknownError"
    for attempt in range(int(max_retries)):
        exam_client = None
        exam_task = None
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if exam_items:
                    exam_client = AsyncOpenAI(
                        api_key="local-vllm",
                        base_url=f"{exam_url}/v1",
                        timeout=float(request_timeout_seconds),
                        max_retries=0,
                    )
                    exam_task = asyncio.create_task(
                        _exam_score(exam_client, exam_url, exam_model,
                                    payload["response"], exam_items))
                async with session.post(f"{url}/score", json=payload) as response:
                    result = await response.json(content_type=None)
                if exam_task is not None:
                    exam_content, exam_failed = await exam_task
                else:
                    exam_content, exam_failed = None, 0
                    if response.status >= 400:
                        error_name = str(result.get("error", "")) if isinstance(result, dict) else ""
                        if response.status == 400 or error_name == "JudgeExhaustedError":
                            failure_name = error_name or f"HTTP{response.status}"
                            last_error = RuntimeError(error_name or f"HTTP{response.status}")
                            break
                        response.raise_for_status()
            _dump_online_scored(contract, result, payload["response"], reference_label)
            judge_score = float(result["score"])
            if exam_content is None:
                combined = judge_score
            else:
                # The extraction exam is an additive channel on top of rubric grading.
                combined = judge_score + exam_weight * exam_content
            return {
                "score": combined,
                "exam_score": float(exam_content) if exam_content is not None else 0.0,
                "exam_items": len(exam_items),
                "exam_failed": exam_failed,
                "judge_score": judge_score,
                "rubric_score": float(result["rubric_score"]),
                "claim_support": float(result["global_claim_support"]),
                "holistic_score": float(result["holistic_score"]),
                "negative_trigger": float(result["negative_trigger_score"]),
                "negative_accuracy_trigger": float(
                    result["negative_accuracy_trigger_score"]
                ),
                "evaluator_version": int(result["configuration_version"]),
                "failure_count": len(result.get("failure_tags", [])),
                "label_score": float(result.get("label_score", 0.0)),
                "reference_label": reference_label,
                "matrix_advantage": float(result.get("matrix_advantage", 0.0)),
                "rubric_advantage": float(result.get("rubric_advantage", 0.0)),
                "rubric_valid": bool(result.get("rubric_valid", True)),
                "protocol_weight_version": int(
                    result.get("protocol_weight_version", 0)
                ),
                "judge_valid": bool(result.get("judge_valid", True)),
                "judge_failure": str(result.get("judge_failure", "")),
                "joint_gated": bool(result.get("joint_gated", False)),
            }
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, TypeError, ValueError) as error:
            last_error = error
            failure_name = type(error).__name__
            if attempt + 1 < int(max_retries):
                await asyncio.sleep(min(2**attempt, 4))
        finally:
            if exam_task is not None and not exam_task.done():
                exam_task.cancel()
                await asyncio.gather(exam_task, return_exceptions=True)
            if exam_client is not None:
                await exam_client.close()
    # Keep the failure result schema identical to the success result schema.
    return {
        "score": (
            local_label_score
            if reward_mode == "correctness_gated_dual"
            else 0.0
        ),
        "exam_score": 0.0,
        "exam_items": len(exam_items),
        "exam_failed": 0,
        "judge_score": 0.0,
        "rubric_score": 0.0,
        "claim_support": 0.0,
        "holistic_score": 0.0,
        "negative_trigger": 0.0,
        "negative_accuracy_trigger": 0.0,
        "evaluator_version": -1,
        "failure_count": 0,
        "label_score": local_label_score,
        "reference_label": reference_label,
        "matrix_advantage": 0.0,
        "rubric_advantage": 0.0,
        "rubric_valid": False,
        "protocol_weight_version": 0,
        "judge_valid": reward_mode == "correctness_gated_dual",
        "judge_failure": failure_name,
        "joint_gated": False,
    }
