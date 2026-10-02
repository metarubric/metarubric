#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${ACRE_OUTPUT_DIR:?ACRE_OUTPUT_DIR is required}"
export ACRE_JUDGE_MODEL="${METARUBRIC_OUTER_MODEL:-gpt-5.4-mini}"
export ACRE_JUDGE_REASONING_EFFORT=none
export ACRE_ROLLOUT_N="${ACRE_ROLLOUT_N:-8}"
export ACRE_REWARD_MODE=legacy_scalar
export ACRE_UPDATE_INTERVAL_GROUPS=100000
export ACRE_JUDGE_CONCURRENCY=8 ACRE_JUDGE_RETRIES=2 ACRE_JUDGE_BACKOFF_CAP=5
export ACRE_STATE_PATH="$output_dir/evaluator_state.json"
export ACRE_TRACE_PATH="$output_dir/evaluator_traces.jsonl" ACRE_CACHE_DIR="$output_dir/cache"
exec "${ACRE_PYTHON_BIN:-python}" "$package_root/evaluator/evaluator_service.py" --host 127.0.0.1 --port 19790
