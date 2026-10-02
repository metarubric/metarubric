#!/usr/bin/env python3
"""HTTP service for the HealthBench MetaRubrics reward used by verl."""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, random, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping
import openai
from openai import AsyncOpenAI
from aiohttp import web
from adaptive_rubric_evaluator import (
    EvaluationResult, EvaluatorConfiguration, EvaluatorContractError,
    advance_configuration, build_evaluator_request, empty_response_result,
    normalize_rubrics, parse_evaluator_response, rubric_digest,
)


class JudgeExhaustedError(RuntimeError):
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        super().__init__("judge exhausted its retry budget")
        self.failures = failures


EVALUATOR_CONTRACT_VERSION = "acre-judge-v4-evidence-bound"
OPENAI_API_BASE_URL = "https://api.openai.com/v1"


def _judge_token_limit(payload: Mapping[str, Any]) -> int:
    raw = payload.get("r", [])
    count = len(raw) if isinstance(raw, list) else 0
    return min(1536, max(384, 192 + 16 * count))


def _judge_response_format(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("r", [])
    count = len(raw) if isinstance(raw, list) else 0
    score = {"type": "number", "minimum": 0, "maximum": 1}
    triple = {"type": "array", "items": score, "minItems": 3, "maxItems": 3}
    schema = {"type": "object", "properties": {
        "s": {"type": "array", "items": triple, "minItems": count, "maxItems": count},
        "g": score, "h": triple,
        "f": {"type": "array", "items": {"type": "string", "enum":
              ["pc", "ie", "iv", "om", "uc", "gr", "ve", "ir"]}, "maxItems": 8},
    }, "required": ["s", "g", "h", "f"], "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {
        "name": f"acre_score_n{count}", "strict": True, "schema": schema}}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


JUDGE_BACKOFF_CAP = float(os.environ.get("ACRE_JUDGE_BACKOFF_CAP", "8"))


class EvaluatorRuntime:
    def __init__(self) -> None:
        api_key = _required_env("OPENAI_API_KEY")
        self.model = os.environ.get("ACRE_JUDGE_MODEL", "gpt-5.4-mini")
        self.reward_mode = os.environ.get("ACRE_REWARD_MODE", "legacy_scalar").strip()
        if self.reward_mode != "legacy_scalar":
            raise RuntimeError("this release supports ACRE_REWARD_MODE=legacy_scalar")
        self.contract_version = EVALUATOR_CONTRACT_VERSION
        self.state_path = Path(_required_env("ACRE_STATE_PATH"))
        self.trace_path = Path(_required_env("ACRE_TRACE_PATH"))
        self.cache_dir = Path(_required_env("ACRE_CACHE_DIR")); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_n = int(os.environ.get("ACRE_ROLLOUT_N", "8"))
        self.update_interval_groups = int(os.environ.get("ACRE_UPDATE_INTERVAL_GROUPS", "32"))
        self.prevalence_threshold = float(os.environ.get("ACRE_PREVALENCE_THRESHOLD", "0.15"))
        self.max_retries = int(os.environ.get("ACRE_JUDGE_RETRIES", "2"))
        self.reasoning_effort = os.environ.get("ACRE_JUDGE_REASONING_EFFORT", "none")
        self.judge_token_headroom = int(os.environ.get("ACRE_JUDGE_TOKEN_HEADROOM", "0"))
        self.concurrency = int(os.environ.get("ACRE_JUDGE_CONCURRENCY", "32"))
        self.max_judge_attempts = int(os.environ.get("ACRE_MAX_JUDGE_ATTEMPTS", "1000000"))
        self.judge_attempts_started = 0
        if self.rollout_n < 2 or self.update_interval_groups < 1 or self.max_retries < 1:
            raise RuntimeError("invalid rollout, update, or retry settings")
        if self.judge_token_headroom < 0 or self.max_judge_attempts < 1:
            raise RuntimeError("invalid judge limit")
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENAI_API_BASE_URL,
            timeout=float(os.environ.get("ACRE_JUDGE_HTTP_TIMEOUT", "180")),
            max_retries=0,
        )
        self.lock = asyncio.Lock()
        self.request_cache: dict[str, dict[str, Any]] = {}
        self.group_assignments: dict[str, EvaluatorConfiguration] = {}
        self.group_requests: dict[str, set[str]] = defaultdict(set)
        self.group_results: dict[str, list[EvaluationResult]] = defaultdict(list)
        self.window_results: list[EvaluationResult] = []
        self.completed_groups = 0
        self.configuration = self._load_configuration()

    def _load_configuration(self) -> EvaluatorConfiguration:
        if not self.state_path.is_file():
            cfg = EvaluatorConfiguration(); self._persist(cfg, event="initialize", prevalence={}); return cfg
        raw = json.loads(self.state_path.read_text())
        if not isinstance(raw, dict) or "configuration" not in raw:
            raise RuntimeError("ACRE state file is invalid")
        self.completed_groups = int(raw.get("completed_groups", 0))
        return EvaluatorConfiguration.from_dict(raw["configuration"])

    def _persist(self, cfg: EvaluatorConfiguration, *, event: str,
                 prevalence: Mapping[str, float]) -> None:
        _atomic_json(self.state_path, {"schema_version": 1,
            "artifact_type": "ACRE-EVALUATOR-STATE", "event": event,
            "completed_groups": self.completed_groups, "reward_mode": self.reward_mode,
            "configuration": cfg.to_dict(), "protocol_controller": None,
            "failure_prevalence": dict(prevalence), "updated_at_unix": time.time()})

    async def health(self) -> dict[str, Any]:
        async with self.lock:
            return {"status": "ok", "model": self.model, "reward_mode": self.reward_mode,
                    "configuration": self.configuration.to_dict(),
                    "completed_groups": self.completed_groups,
                    "pending_groups": len(self.group_assignments),
                    "judge_attempts_started": self.judge_attempts_started,
                    "max_judge_attempts": self.max_judge_attempts}

    async def _configuration_for_group(self, group_id: str) -> EvaluatorConfiguration:
        async with self.lock:
            if group_id not in self.group_assignments:
                self.group_assignments[group_id] = self.configuration
            return self.group_assignments[group_id]

    def _cache_path(self, *, prompt: Any, response: str, rubric_hash: str,
                    configuration: EvaluatorConfiguration, evidence_contract: str) -> Path:
        payload = {"contract_version": self.contract_version, "model": self.model,
            "prompt": prompt, "response": response, "rubric_hash": rubric_hash,
            "configuration": configuration.to_dict(), "evidence_contract": evidence_contract}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    async def _call_judge(self, system_prompt: str, payload: Mapping[str, Any], *,
                          validate: Callable[[Mapping[str, Any]], Any] | None = None) -> Mapping[str, Any]:
        last_error: Exception | None = None; failures: list[dict[str, Any]] = []
        for attempt in range(self.max_retries):
            try:
                async with self.lock:
                    if self.judge_attempts_started >= self.max_judge_attempts:
                        raise JudgeExhaustedError([{"attempt": attempt + 1,
                            "error_type": "JudgeAttemptCapReached",
                            "error_message": "registered judge-attempt cap reached"}])
                    self.judge_attempts_started += 1
                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        ],
                        temperature=0.0,
                        reasoning_effort=self.reasoning_effort,
                        max_completion_tokens=(
                            _judge_token_limit(payload) + self.judge_token_headroom
                        ),
                        response_format=_judge_response_format(payload),
                    )
                content = completion.choices[0].message.content
                if not content:
                    raise EvaluatorContractError("judge returned an empty completion "
                        f"(finish_reason={completion.choices[0].finish_reason!r})")
                left, right = content.find("{"), content.rfind("}")
                if left < 0 or right < left:
                    raise EvaluatorContractError("judge response contains no JSON object")
                parsed = json.loads(content[left:right + 1])
                if not isinstance(parsed, dict):
                    raise EvaluatorContractError("judge response root must be an object")
                if validate is not None: validate(parsed)
                return parsed
            except (openai.APIError, asyncio.TimeoutError, KeyError, TypeError,
                    json.JSONDecodeError, EvaluatorContractError) as error:
                last_error = error
                failure = {"attempt": attempt + 1,
                    "error_type": type(error).__name__, "error_message": str(error)[:240]}
                status_code = getattr(error, "status_code", None)
                if status_code is not None: failure["http_status"] = status_code
                failures.append(failure)
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(2**attempt, JUDGE_BACKOFF_CAP) + random.random())
        raise JudgeExhaustedError(failures) from last_error

    async def probe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = str(payload.get("response", "")).strip()
        rubrics = normalize_rubrics(payload.get("rubrics"))
        evidence = str(payload.get("evidence_contract",
                       "prompt_and_reliable_domain_knowledge")).strip()
        if not response: raise EvaluatorContractError("probe response is required")
        async with self.lock: cfg = self.configuration
        system, judge_payload = build_evaluator_request(prompt=payload.get("prompt"),
            response=response, rubrics=rubrics, configuration=cfg, evidence_contract=evidence)
        raw = await self._call_judge(system, judge_payload,
            validate=lambda value: parse_evaluator_response(value, rubrics, cfg))
        result = parse_evaluator_response(raw, rubrics, cfg)
        return {"status": "ok", "score": result.score,
                "configuration_version": result.configuration_version}

    async def _record_result(self, *, group_id: str, request_id: str,
                             result: EvaluationResult) -> None:
        async with self.lock:
            if request_id in self.group_requests[group_id]: return
            self.group_requests[group_id].add(request_id); self.group_results[group_id].append(result)
            if len(self.group_requests[group_id]) < self.rollout_n: return
            if len(self.group_requests[group_id]) != self.rollout_n:
                raise RuntimeError(f"group {group_id} exceeded the configured rollout count")
            self.completed_groups += 1; self.window_results.extend(self.group_results[group_id])
            del self.group_assignments[group_id], self.group_requests[group_id], self.group_results[group_id]
            if self.completed_groups % self.update_interval_groups != 0:
                self._persist(self.configuration, event="group_complete", prevalence={}); return
            proposed, prevalence = advance_configuration(self.configuration, self.window_results,
                effective_group=self.completed_groups + 1,
                prevalence_threshold=self.prevalence_threshold)
            changed = proposed != self.configuration; self.configuration = proposed
            self.window_results.clear(); self._persist(self.configuration,
                event="configuration_update" if changed else "configuration_keep",
                prevalence=prevalence)

    async def score(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", "")).strip()
        group_id = str(payload.get("group_id", "")).strip()
        response = str(payload.get("response", "")).strip(); prompt = payload.get("prompt")
        evidence = str(payload.get("evidence_contract",
                       "prompt_and_reliable_domain_knowledge")).strip()
        if not request_id or not group_id:
            raise EvaluatorContractError("request_id and group_id are required")
        async with self.lock: prior = self.request_cache.get(request_id)
        if prior is not None: return prior
        rubrics = normalize_rubrics(payload.get("rubrics")); cfg = await self._configuration_for_group(group_id)
        if not response:
            result = empty_response_result(rubrics, cfg); cache_status = "deterministic_empty"
        else:
            cache_path = self._cache_path(prompt=prompt, response=response,
                rubric_hash=rubric_digest(rubrics), configuration=cfg, evidence_contract=evidence)
            if cache_path.is_file():
                raw = json.loads(cache_path.read_text()); result = parse_evaluator_response(raw, rubrics, cfg)
                cache_status = "hit"
            else:
                system, judge_payload = build_evaluator_request(prompt=prompt, response=response,
                    rubrics=rubrics, configuration=cfg, evidence_contract=evidence)
                raw = await self._call_judge(system, judge_payload,
                    validate=lambda value: parse_evaluator_response(value, rubrics, cfg))
                result = parse_evaluator_response(raw, rubrics, cfg)
                cache_path.parent.mkdir(parents=True, exist_ok=True); _atomic_json(cache_path, raw)
                cache_status = "miss"
        record = result.to_dict(); record.update({"request_id": request_id,
            "group_id": group_id, "rubric_hash": rubric_digest(rubrics),
            "cache": cache_status, "timestamp_unix": time.time()})
        _append_jsonl(self.trace_path, record)
        await self._record_result(group_id=group_id, request_id=request_id, result=result)
        async with self.lock: self.request_cache[request_id] = dict(record)
        return dict(record)


async def create_app() -> web.Application:
    runtime = EvaluatorRuntime(); app = web.Application(client_max_size=16 * 1024 * 1024)
    async def close_client(_: web.Application) -> None: await runtime.client.close()
    async def health(_: web.Request) -> web.Response: return web.json_response(await runtime.health())
    async def score(request: web.Request) -> web.Response:
        payload: Mapping[str, Any] = {}
        try:
            decoded = await request.json()
            if not isinstance(decoded, Mapping): raise EvaluatorContractError("request body must be a JSON object")
            payload = decoded; return web.json_response(await runtime.score(payload))
        except EvaluatorContractError as error: return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            details = error.failures if isinstance(error, JudgeExhaustedError) else []
            _append_jsonl(runtime.trace_path, {"status": "error", "error_type": type(error).__name__,
                "group_id": str(payload.get("group_id", "")), "judge_attempt_failures": details,
                "timestamp_unix": time.time()})
            return web.json_response({"error": type(error).__name__}, status=503)
    async def probe(request: web.Request) -> web.Response:
        try: return web.json_response(await runtime.probe(await request.json()))
        except EvaluatorContractError as error: return web.json_response({"error": str(error)}, status=400)
        except Exception as error:
            details = error.failures if isinstance(error, JudgeExhaustedError) else []
            _append_jsonl(runtime.trace_path, {"status": "probe_error",
                "error_type": type(error).__name__, "judge_attempt_failures": details,
                "timestamp_unix": time.time()})
            return web.json_response({"error": type(error).__name__}, status=503)
    app.router.add_get("/health", health); app.router.add_get("/state", health)
    app.router.add_post("/probe", probe); app.router.add_post("/score", score)
    app.on_cleanup.append(close_client)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787); args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__": main()
