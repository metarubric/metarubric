# MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning

**Authors:** Yuxuan Fan and Jaehong Yoon (corresponding author)

**Affiliation:** Nanyang Technological University, Singapore

[Paper (arXiv:2610.02824)](https://arxiv.org/abs/2610.02824) · [Project page](https://metarubric.github.io)

This repository contains the code for MetaRubric and the complete integrated verl source. The experiment configuration is Qwen3-4B-Instruct-2507 on HealthBench with thinking disabled.

- `vendor/verl/`: full verl source, including the applied training integration, workers, configuration, tests, and upstream documentation.
- `src/metarubrics/`: rubric revisions, weight updates, immutable snapshots, and the training reward hook.
- `evaluator/evaluator_service.py`: the MetaRubrics judge HTTP service.
- `evaluator/adaptive_rubric_evaluator.py`: rubric prompts, parsing, and scalar reward computation.
- `scripts/`: data preparation, services, training segments, and outer-loop updates.
- `.env.example`: the HealthBench experiment configuration with illustrative paths.

## Setup

Install the included source:

```bash
cd /path/to/supp_meta_rubric
python -m venv .venv
. .venv/bin/activate
pip install -e ./vendor/verl
pip install -e .
pip install -r requirements-runtime.txt
```

The included verl revision is `7aed6b230776f963fa09509c10d9c3a767d1102c` (package version `0.8.0.dev`). The modifications recorded in `framework/verl.patch` are already applied. The upstream license and attribution remain in `vendor/verl/`. Its bundled examples are upstream examples; the MetaRubrics experiment configuration is in `.env.example`.

The runtime uses PyTorch 2.8.0 with CUDA 12.8, vLLM 0.11.0, Transformers 4.57.6, and FlashAttention 2.8.3. Copy `.env.example` to `.env`, replace each `/path/to/...` value, and export it in the shell. The launch scripts use the included `vendor/verl` automatically. `ACRE_MODEL_PATH` must point to the Qwen3-4B-Instruct-2507 policy, while `ACRE_READER_MODEL_PATH` must point to the independent Qwen3-1.7B exam reader.

## Data

Train and validation inputs are Parquet tables in verl prompt-data format. Each row contains `prompt` (chat messages), `data_source` (`acre_healthbench_twin` or `acre_healthbench_indomain`), `reward_model.ground_truth` (a JSON string), and `extra_info.split`. The ground-truth contract contains `sample_id`, `pair_id`, `side`, `prompt`, `rubrics`, `rubric_meta`, and exam items used by the reward implementation.

Training rows form adjacent `orig`/`twin` pairs with the same `pair_id`. Pair-level shuffling happens before preparation; the launcher sets `data.shuffle=False` so pairs remain in one GRPO batch.

```bash
python scripts/prepare_healthbench.py /path/to/source.parquet /path/to/workspace/healthbench
```

Use an independently prepared HealthBench validation parquet for `ACRE_VALIDATION_FILE`. No data, weights, credentials, results, or logs are included.

## Services and training

The judge and outer loop use the official OpenAI Python SDK and Chat Completions interface at `https://api.openai.com/v1`. Configure only `OPENAI_API_KEY` in `.env`; the remote endpoint is fixed in code. The local Qwen3-1.7B exam reader remains an OpenAI-compatible service on localhost. See the [OpenAI SDK documentation](https://developers.openai.com/api/docs/libraries) and [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

After exporting `.env`, start the HealthBench evaluator and Qwen3-1.7B exam reader in separate shells:

```bash
set -a; . ./.env; set +a
bash scripts/start_judge.sh
bash scripts/start_exam_reader.sh
```

The reader uses GPU 0 by default; policy training uses GPUs 1 and 2.

Run a segment from the initial snapshot:

```bash
set -a; . ./.env; set +a
bash scripts/run_segment.sh 10 /path/to/workspace/healthbench/snapshot-0.json
```

The wrapper resumes from the latest verl checkpoint tracker when present. It preserves GRPO, adjacent-pair semantics, snapshot-bound reward traces, and the original dynamic token budgets. At a complete checkpoint boundary, update the outer state and use the immutable next snapshot for the following segment:

```bash
PYTHONPATH=src python scripts/outer_boundary.py healthbench 10 /path/to/workspace/healthbench/snapshot-0.json
bash scripts/run_segment.sh 20 /path/to/workspace/healthbench/snapshot-after-10.json
```

The outer step updates bounded rubric weights and may accept one model-reviewed counterfactual criterion revision using a fixed reference panel. Method formulas and prompts are preserved from the source.

## CPU checks

```bash
PYTHONPATH=src:evaluator:scripts python -m unittest discover -s tests -v
python -m compileall -q src evaluator scripts
```

The tests use synthetic fixtures and mocks, without model-service or GPU calls.

## Citation

Paper: [arXiv:2610.02824](https://arxiv.org/abs/2610.02824). Download [citation.bib](citation.bib).

```bibtex
@misc{fan2026metarubric,
  title = {MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning},
  author = {Fan, Yuxuan and Yoon, Jaehong},
  year = {2026},
  eprint = {2610.02824},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2610.02824}
}
```
