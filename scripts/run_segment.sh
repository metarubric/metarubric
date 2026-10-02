#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:?usage: run_segment.sh TARGET_STEP SNAPSHOT [Hydra overrides ...]}"
snapshot="${2:?usage: run_segment.sh TARGET_STEP SNAPSHOT [Hydra overrides ...]}"
shift 2

export ACRE_PROJECT_ROOT="$package_root"
export ACRE_VERL_ROOT="${ACRE_VERL_ROOT:-$package_root/vendor/verl}"
export ACRE_REWARD_FILE="$package_root/src/metarubrics/reward.py"
export PYTHONPATH="$package_root/src:$package_root/evaluator${PYTHONPATH:+:$PYTHONPATH}"
export ACRE_REWARD_MODE=legacy_scalar
export ACRE_ANCHOR_TABLE=4,5,6,8
export ACRE_DATA_SHUFFLE=False
export ACRE_TOTAL_TRAINING_STEPS="$target"
export CUDA_VISIBLE_DEVICES="${ACRE_POLICY_CUDA_VISIBLE_DEVICES:-1,2}"

if [[ -f "$ACRE_OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    step="$(<"$ACRE_OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt")"
    export ACRE_RESUME_MODE=resume_path
    export ACRE_RESUME_FROM_PATH="$ACRE_OUTPUT_DIR/checkpoints/global_step_$step"
else
    export ACRE_RESUME_MODE=disable
fi

exec bash "$package_root/scripts/run_grpo_hb.sh" \
    reward.custom_reward_function.reward_kwargs.evaluator_url="http://127.0.0.1:19790" \
    +reward.custom_reward_function.reward_kwargs.phi_snapshot="$snapshot" \
    +reward.custom_reward_function.reward_kwargs.trace_dir="$ACRE_OUTPUT_DIR/reward_records" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra"]' \
    actor_rollout_ref.actor.optim.lr_warmup_steps=4 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    "$@"
