#!/usr/bin/env bash
set -euo pipefail

project_root="${ACRE_PROJECT_ROOT:?ACRE_PROJECT_ROOT is required}"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_root="${ACRE_VERL_ROOT:-$package_root/vendor/verl}"
model_path="${ACRE_MODEL_PATH:?ACRE_MODEL_PATH is required}"
train_file="${ACRE_TRAIN_FILE:?ACRE_TRAIN_FILE is required}"
validation_file="${ACRE_VALIDATION_FILE:?ACRE_VALIDATION_FILE is required}"
output_dir="${ACRE_OUTPUT_DIR:?ACRE_OUTPUT_DIR is required}"
reward_file="${ACRE_REWARD_FILE:-$project_root/src/metarubrics/reward.py}"
python_bin="${ACRE_PYTHON_BIN:-python}"
train_batch_size="${ACRE_TRAIN_BATCH_SIZE:-16}"
ppo_mini_batch_size="${ACRE_PPO_MINI_BATCH_SIZE:-$train_batch_size}"
rollout_n="${ACRE_ROLLOUT_N:-8}"
total_steps="${ACRE_TOTAL_TRAINING_STEPS:-100}"
save_freq="${ACRE_SAVE_FREQ:-5}"
max_actor_ckpt_to_keep="${ACRE_MAX_ACTOR_CKPT_TO_KEEP:-1000000}"   # Retain all checkpoints by default.
experiment_name="${ACRE_EXPERIMENT_NAME:-metarubric-qwen3-4b-healthbench}"
n_gpus_per_node="${ACRE_N_GPUS_PER_NODE:-2}"
reward_num_workers="${ACRE_REWARD_NUM_WORKERS:-32}"
max_prompt_length="${ACRE_MAX_PROMPT_LENGTH:-2048}"
max_response_length="${ACRE_MAX_RESPONSE_LENGTH:-512}"
resume_mode="${ACRE_RESUME_MODE:-disable}"
resume_from_path="${ACRE_RESUME_FROM_PATH:-null}"
reward_mode="${ACRE_REWARD_MODE:-legacy_scalar}"
rubric_beta="${ACRE_RUBRIC_BETA:-0.25}"
kl_loss_coef="${ACRE_KL_LOSS_COEF:-0.1}"
actor_lr="${ACRE_ACTOR_LR:-1e-6}"
seed="${ACRE_SEED:-42}"
# Twin rows must remain in the same batch for pooled advantages. Pair-level
# shuffling is performed while preparing the parquet file.
data_shuffle="${ACRE_DATA_SHUFFLE:-True}"

if [[ "$reward_mode" == "correctness_gated_dual" ]] && \
    [[ ! "$rubric_beta" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    printf 'ACRE_RUBRIC_BETA must be a nonnegative number: %s\n' "$rubric_beta" >&2
    exit 2
fi

for required_path in \
    "$verl_root/verl/trainer/main_ppo.py" \
    "$model_path/config.json" \
    "$train_file" \
    "$validation_file" \
    "$reward_file"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'required ACRE input is missing: %s\n' "$required_path" >&2
        exit 2
    fi
done
if [[ ! "$kl_loss_coef" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    printf 'ACRE_KL_LOSS_COEF must be a nonnegative number: %s\n' "$kl_loss_coef" >&2
    exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'ACRE_SEED must be a nonnegative integer: %s\n' "$seed" >&2
    exit 2
fi

export PYTHONPATH="$verl_root:$project_root/src:$project_root/evaluator${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=WARN
export ACRE_REWARD_MODE="$reward_mode"
export ACRE_RUBRIC_BETA="$rubric_beta"

mkdir -p "$output_dir" "$output_dir/rollouts" "$output_dir/evaluation"

"$python_bin" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_file" \
    data.val_files="$validation_file" \
    data.train_batch_size="$train_batch_size" \
    data.val_batch_size=64 \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation=error \
    data.shuffle="$data_shuffle" \
    data.seed="$seed" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$actor_lr" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACRE_PPO_MAX_TOKEN_LEN:-8192}" \
    actor_rollout_ref.actor.use_kl_loss="${ACRE_USE_KL_LOSS:-True}" \
    actor_rollout_ref.actor.kl_loss_coef="$kl_loss_coef" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.data_loader_seed="$seed" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.seed="$seed" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy="${ACRE_ACTOR_OFFLOAD_POLICY:-True}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ACRE_ROLLOUT_TP:-2}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ACRE_VLLM_GPU_MEMORY_UTILIZATION:-0.50}" \
    actor_rollout_ref.rollout.enforce_eager="${ACRE_VLLM_ENFORCE_EAGER:-False}" \
    actor_rollout_ref.rollout.max_model_len="$((max_prompt_length + max_response_length))" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.n="$rollout_n" \
    actor_rollout_ref.rollout.temperature="${ACRE_ROLLOUT_TEMPERATURE:-0.7}" \
    actor_rollout_ref.rollout.top_p="${ACRE_ROLLOUT_TOP_P:-0.8}" \
    actor_rollout_ref.rollout.top_k="${ACRE_ROLLOUT_TOP_K:-20}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ACRE_LOGPROB_MAX_TOKEN_LEN:-16384}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ACRE_MAX_NUM_BATCHED_TOKENS:-16384}" \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.seed="$seed" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${ACRE_LOGPROB_MAX_TOKEN_LEN:-16384}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers="$reward_num_workers" \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$reward_file" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.evaluator_url=http://127.0.0.1:8787 \
    +reward.custom_reward_function.reward_kwargs.request_timeout_seconds="${ACRE_REWARD_REQUEST_TIMEOUT_SECONDS:-1200}" \
    +reward.custom_reward_function.reward_kwargs.exam_extractor_url="${ACRE_EXAM_EXTRACTOR_URL:-}" \
    +reward.custom_reward_function.reward_kwargs.exam_extractor_model="${ACRE_EXAM_EXTRACTOR_MODEL:-extractor}" \
    +reward.custom_reward_function.reward_kwargs.exam_weight="${ACRE_EXAM_WEIGHT:-0.3}" \
    +reward.custom_reward_function.reward_kwargs.max_retries=3 \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name="${ACRE_PROJECT_NAME:-metarubric_healthbench}" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node="$n_gpus_per_node" \
    trainer.nnodes=1 \
    trainer.total_training_steps="$total_steps" \
    trainer.total_epochs="${ACRE_TOTAL_EPOCHS:-20}" \
    trainer.val_before_train="${ACRE_VAL_BEFORE_TRAIN:-False}" \
    trainer.test_freq="${ACRE_TEST_FREQ:--1}" \
    trainer.save_freq="$save_freq" \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.max_actor_ckpt_to_keep="$max_actor_ckpt_to_keep" \
    trainer.rollout_data_dir="$output_dir/rollouts" \
    trainer.validation_data_dir="$output_dir/evaluation" \
    trainer.resume_mode="$resume_mode" \
    trainer.resume_from_path="$resume_from_path" \
    ${ACRE_EXTRA_HYDRA_OVERRIDES:-} \
    "$@"
