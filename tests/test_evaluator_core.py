import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
import httpx
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import adaptive_rubric_evaluator as core
import evaluator_service as service


RUBRICS = [
    {"criterion": "Gives the indicated next step.", "points": 4},
    {"criterion": "Makes an unsafe absolute claim.", "points": -2},
]
JUDGE_RESPONSE = {
    "s": [[0.8, 0.6, 0.7], [0.4, 0.5, 0.3]],
    "g": 0.55,
    "h": [0.9, 0.7, 0.8],
    "f": ["om"],
}


class CoreParityTest(unittest.TestCase):
    def test_original_legacy_scalar_request_and_result(self):
        rubrics = core.normalize_rubrics(RUBRICS)
        configuration = core.EvaluatorConfiguration()
        system_prompt, payload = core.build_evaluator_request(
            prompt="What should this patient do?",
            response="Seek prompt clinical assessment.",
            rubrics=rubrics,
            configuration=configuration,
        )
        self.assertEqual(payload, {
            "cfg": [1, 0, 0, 0, 0],
            "n": 2,
            "q": "What should this patient do?",
            "a": "Seek prompt clinical assessment.",
            "r": [["r000", 4.0, RUBRICS[0]["criterion"]],
                  ["r001", -2.0, RUBRICS[1]["criterion"]]],
        })
        self.assertIn("A=atomic satisfaction", system_prompt)
        result = core.parse_evaluator_response(JUDGE_RESPONSE, rubrics, configuration)
        # Captured from the untrimmed legacy_scalar implementation: positive
        # bundle=.6, negative bundle=.5 => rubric=.35; support=.55; holistic=.7.
        self.assertEqual(result.score, 0.2975)
        self.assertEqual(result.rubric_score, 0.35)
        self.assertEqual(result.negative_trigger_score, 0.5)

    def test_empty_response_result_is_unchanged(self):
        result = core.empty_response_result(
            core.normalize_rubrics(RUBRICS), core.EvaluatorConfiguration())
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.failure_tags, ("generic_response", "omission"))


class HttpIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_openai_sdk_serializes_judge_request(self):
        captured = {}

        async def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.4-mini",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": json.dumps(JUDGE_RESPONSE)}}],
            })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
                "ACRE_STATE_PATH": str(root / "state.json"),
                "ACRE_TRACE_PATH": str(root / "traces.jsonl"),
                "ACRE_CACHE_DIR": str(root / "cache"),
            }
            with patch.dict(os.environ, environment, clear=False):
                runtime = service.EvaluatorRuntime()
                self.assertEqual(str(runtime.client.base_url), "https://api.openai.com/v1/")
                transport = httpx.MockTransport(handler)
                runtime.client = AsyncOpenAI(
                    api_key="test-key",
                    base_url="https://api.openai.com/v1",
                    max_retries=0,
                    http_client=httpx.AsyncClient(transport=transport),
                )
                rubrics = core.normalize_rubrics(RUBRICS)
                configuration = core.EvaluatorConfiguration()
                system, payload = core.build_evaluator_request(
                    prompt="Question", response="Answer", rubrics=rubrics,
                    configuration=configuration)
                result = await runtime._call_judge(system, payload)
                await runtime.client.close()
                self.assertEqual(result, JUDGE_RESPONSE)
                self.assertEqual(captured["model"], "gpt-5.4-mini")
                self.assertIn("max_completion_tokens", captured)
                self.assertEqual(captured["reasoning_effort"], "none")

    async def test_score_endpoint_group_and_cache_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
                "ACRE_REWARD_MODE": "legacy_scalar",
                "ACRE_STATE_PATH": str(root / "state.json"),
                "ACRE_TRACE_PATH": str(root / "traces.jsonl"),
                "ACRE_CACHE_DIR": str(root / "cache"),
                "ACRE_ROLLOUT_N": "2",
                "ACRE_UPDATE_INTERVAL_GROUPS": "100000",
            }

            async def fixed_judge(*_args, **_kwargs):
                return JUDGE_RESPONSE

            with patch.dict(os.environ, environment, clear=False), patch.object(
                service.EvaluatorRuntime, "_call_judge", new=fixed_judge
            ):
                client = TestClient(TestServer(await service.create_app()))
                await client.start_server()
                try:
                    payload = {"group_id": "group-1", "prompt": "Question",
                               "response": "Answer", "rubrics": RUBRICS}
                    first = await client.post("/score", json={**payload, "request_id": "r1"})
                    second = await client.post("/score", json={**payload, "request_id": "r2"})
                    repeated = await client.post("/score", json={**payload, "request_id": "r2"})
                    self.assertEqual(first.status, 200)
                    self.assertEqual((await first.json())["cache"], "miss")
                    self.assertEqual((await second.json())["cache"], "hit")
                    self.assertEqual((await repeated.json())["request_id"], "r2")
                    state = await (await client.get("/state")).json()
                    self.assertEqual(state["completed_groups"], 1)
                    self.assertEqual(state["pending_groups"], 0)
                    traces = [json.loads(line) for line in
                              (root / "traces.jsonl").read_text().splitlines()]
                    self.assertEqual(len(traces), 2)
                finally:
                    await client.close()


if __name__ == "__main__":
    unittest.main()
