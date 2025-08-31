#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export GPU_UTIL=0.8

OUTPUT_ROOT="/nas/ucb/jjssuh/old_projs/subpop/outputs"
INPUT_ROOT="/nas/ucb/jjssuh/old_projs/subpop/subpop/train/datasets"

declare -A TASKS
TASKS["inductive_zeroshot"]="inductive_individual_zeroshot/opinionqa_0.2_val.jsonl"
TASKS["inductive_fewshot_3"]="inductive_individual_fewshot_3/opinionqa_0.2_val.jsonl"
TASKS["inductive_fewshot_5"]="inductive_individual_fewshot_5/opinionqa_0.2_val.jsonl"
TASKS["inductive_fewshot_8"]="inductive_individual_fewshot_8/opinionqa_0.2_val.jsonl"
TASKS["inductive_fewshot_13"]="inductive_individual_fewshot_13/opinionqa_0.2_val.jsonl"

MODELS=(
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.1"
  "Qwen/Qwen3-8B"
)

for task in "${!TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running $task with $model"
    python run_base_lm.py \
      --output_dir "$OUTPUT_ROOT/$task" \
      --input_paths "$INPUT_ROOT/${TASKS[$task]}" \
      --base_model_name_or_path "$model" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --is_chat
  done
done
