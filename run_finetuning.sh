export WANDB_API_KEY=8f3a80cfc6a25862eb821a30f10ea98fe02afceb
export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 \
  --nproc-per-node=1 \
  --master_port=29501 \
  scripts/experiment/run_finetune.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --fsdp_config.pure_bf16 \
  --use_peft=true \
  --use_fast_kernels \
  --checkpoint_type StateDictType.FULL_STATE_DICT \
  --peft_method='lora' \
  --use_fp16 \
  --mixed_precision \
  --batch_size_training 16 \
  --val_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --dist_checkpoint_root_folder None \
  --dist_checkpoint_folder None \
  --batching_strategy='padding' \
  --dataset_path inductive_individual_fewshot_3/opinionqa_0.2 \
  --dataset individual \
  --output_dir /nas/ucb/jjssuh/old_projs/subpop/local_training \
  --name gnn_baseline_inductive_opinionqa_0.2_mistral_instruct \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --model_nickname mistral_7b_instruct \
  --lr 2e-4 \
  --num_epochs 10 \
  --weight_decay 0 \
  --loss_function_type ce \
  --which_scheduler cosine \
  --warmup_ratio 0.1 \
  --gamma 0.95 \
  --lora_config.r 8 \
  --lora_config.lora_alpha 32 \
  --wandb_config.project steerable-pluralism \
  --wandb_config.entity ucb-steerable-pluralism \
  --use_wandb false \
  --is_chat true


# torchrun --nnodes=1 \
#     --nproc-per-node=1 \
#     --master_port=29501 \
#     scripts/experiment/run_finetune.py \
#     --enable_fsdp \
#     --low_cpu_fsdp \
#     --fsdp_config.pure_bf16 \
#     --use_peft=true \
#     --use_fast_kernels \
#     --checkpoint_type StateDictType.FULL_STATE_DICT \
#     --peft_method='lora' \
#     --use_fp16 \
#     --mixed_precision \
#     --batch_size_training 64 \
#     --val_batch_size 128 \
#     --gradient_accumulation_steps 1 \
#     --dist_checkpoint_root_folder None \
#     --dist_checkpoint_folder None \
#     --batching_strategy='padding' \
#     --dataset_path inductive_individual/opinionqa_0. \
#     --dataset individual \
#     --output_dir ${HOME}/gnn_baseline/ \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --model_nickname llama2_7b \
#     --lr 2e-4 \
#     --num_epochs 10 \
#     --weight_decay 0 \
#     --loss_function_type ce \
#     --which_scheduler cosine \
#     --warmup_ratio 0.1 \
#     --gamma 0.95 \
#     --lora_config.r 8 \
#     --lora_config.lora_alpha 32 \
#     --is_chat=false \
#     --name gnn_baseline_rnn_local \
#     --wandb_config.project steerable-pluralism \
#     --wandb_config.entity ucb-steerable-pluralism