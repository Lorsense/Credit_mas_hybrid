#!/usr/bin/env bash
# Math answer LOTO/GAE + mix entropy credit + semantic-only entropy control.
set -euo pipefail
MODE=${1:-train}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in train|eval|evaluation) ;; *) echo 'Usage: run_math_hybrid.sh [train|eval] [Hydra overrides...]' >&2; exit 2;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$SCRIPT_DIR/../.."
PYTHON=${PYTHON:-python3}
case "${DRY_RUN:-False}" in True|true|1) IS_DRY_RUN=1 ;; False|false|0) IS_DRY_RUN=0 ;; *) echo 'DRY_RUN must be True or False.' >&2; exit 2;; esac

# Independent role models share the Actor pool; value gets one further GPU.
NNODES=${NNODES:-1}
case "$NNODES" in
  1) N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}; AGENT_GPUS_PER_NODE=${AGENT_GPUS_PER_NODE:-'[15]'} ;;
  2) N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}; AGENT_GPUS_PER_NODE=${AGENT_GPUS_PER_NODE:-'[7,8]'} ;;
  *) : "${N_GPUS_PER_NODE:?Set N_GPUS_PER_NODE for a custom layout}"; : "${AGENT_GPUS_PER_NODE:?Set AGENT_GPUS_PER_NODE for a custom layout}" ;;
esac
[[ "$NNODES" =~ ^[1-9][0-9]*$ && "$N_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo 'Node and GPU counts must be positive integers.' >&2; exit 2; }
layout_text=${AGENT_GPUS_PER_NODE//[[:space:]]/}
[[ "$layout_text" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]] || { echo 'AGENT_GPUS_PER_NODE must be an integer list, e.g. [7,8].' >&2; exit 2; }
layout_text=${layout_text#\[}; layout_text=${layout_text%\]}
IFS=, read -r -a actor_counts <<< "$layout_text"
[[ ${#actor_counts[@]} -eq "$NNODES" ]] || { echo 'Actor layout must contain one entry per node.' >&2; exit 2; }
ACTOR_WORLD_SIZE=0
for count in "${actor_counts[@]}"; do
  (( count > 0 && count <= N_GPUS_PER_NODE )) || { echo 'Each Actor count must be positive and fit its physical node.' >&2; exit 2; }
  ACTOR_WORLD_SIZE=$((ACTOR_WORLD_SIZE + count))
done
ROLLOUT_TP=${ROLLOUT_TP:-1}
[[ "$ROLLOUT_TP" =~ ^[1-9][0-9]*$ ]] && (( ACTOR_WORLD_SIZE % ROLLOUT_TP == 0 )) || {
  echo 'ROLLOUT_TP must divide the actual Actor GPU count; use e.g. [6,8] for TP=2 with a separate value GPU.' >&2; exit 2;
}
if [[ "$MODE" == train ]] && (( ACTOR_WORLD_SIZE >= NNODES * N_GPUS_PER_NODE )); then
  echo 'Training needs one physical GPU outside the Actor pool for semantic value.' >&2; exit 2
fi
if (( NNODES > 1 && IS_DRY_RUN == 0 )); then
  : "${RAY_ADDRESS:?Start Ray on all nodes and set RAY_ADDRESS=auto or head:port}"
fi
SOLVER_MODEL=${SOLVER_MODEL:-Qwen/Qwen3-4B}
VERIFIER_MODEL=${VERIFIER_MODEL:-Qwen/Qwen3-4B}
VALUE_ENCODER=${VALUE_ENCODER:-Qwen/Qwen3-4B}
VALUE_CHECKPOINT=${VALUE_CHECKPOINT:-}
VALUE_INIT_MODE=${VALUE_INIT_MODE:-qualified}
case "$VALUE_INIT_MODE" in qualified|candidate) ;; *) echo 'VALUE_INIT_MODE must be qualified or candidate.' >&2; exit 2;; esac
TRAIN_DATA=${TRAIN_DATA:-$PWD/data/drmas_math/train.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((ACTOR_WORLD_SIZE * 2))}
[[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] && (( TRAIN_BATCH_SIZE % ACTOR_WORLD_SIZE == 0 )) || {
  echo 'TRAIN_BATCH_SIZE must be a positive multiple of the actual Actor world size.' >&2; exit 2;
}
RUN_NAME=${RUN_NAME:-math_hybrid_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$PWD/checkpoints/$RUN_NAME}
RESUME_FROM=${RESUME_FROM:-}
RESUME_MODE=disable
RESUME_CHECKPOINT=null
INITIAL_CHECKPOINT=null
VAL_ONLY=False
SEMANTIC_ENABLED=True
if [[ -n "$RESUME_FROM" ]]; then RESUME_MODE=resume_path; RESUME_CHECKPOINT=$RESUME_FROM; fi
if [[ "$MODE" == train ]]; then
  VAL_DATA=${VAL_DATA:-$PWD/data/drmas_math/test_sampled.parquet}
  VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-110}
  VAL_GROUP_SIZE=${VAL_GROUP_SIZE:-1}
  [[ -z "$VALUE_CHECKPOINT" ]] || INITIAL_CHECKPOINT=$VALUE_CHECKPOINT
  if (( IS_DRY_RUN == 0 )); then
    if [[ -z "$RESUME_FROM" ]]; then
      : "${VALUE_CHECKPOINT:?Set VALUE_CHECKPOINT to semantic_value.pt; candidate mode accepts unqualified semantic weights}"
      [[ -f "$VALUE_CHECKPOINT" ]] || { echo "Missing VALUE_CHECKPOINT: $VALUE_CHECKPOINT" >&2; exit 2; }
      if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo 'Fresh training requires an empty RUN_DIR; use a new path or explicit RESUME_FROM.' >&2; exit 2
      fi
    else
      [[ -f "$RESUME_FROM/semantic_value.pt" && -f "$RESUME_FROM/semantic_entropy_controller.json" && -f "$RESUME_FROM/hybrid_state.json" ]] || {
        echo 'Training resume needs semantic_value.pt, semantic_entropy_controller.json and hybrid_state.json in the hybrid checkpoint.' >&2; exit 2;
      }
    fi
  fi
else
  VAL_ONLY=True
  SEMANTIC_ENABLED=False
  VAL_DATA=${VAL_DATA:-$PWD/data/drmas_math/test.parquet}
  VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-$((ACTOR_WORLD_SIZE * 4))}
  VAL_GROUP_SIZE=${VAL_GROUP_SIZE:-1}
  : "${RESUME_FROM:?Evaluation requires RESUME_FROM pointing to the paired global_step_N Actor checkpoint}"
fi
if (( IS_DRY_RUN == 0 )); then
  [[ -f "$TRAIN_DATA" && -f "$VAL_DATA" ]] || { echo 'TRAIN_DATA and VAL_DATA must be existing parquet files.' >&2; exit 2; }
  [[ -z "$RESUME_FROM" || -d "$RESUME_FROM" ]] || { echo 'RESUME_FROM must be an existing checkpoint directory.' >&2; exit 2; }
fi

args=(
  algorithm.adv_estimator=team_event_gae algorithm.group_by_agent_id=True
  algorithm.gamma=1.0 algorithm.lam=1.0 algorithm.norm_adv_by_std_in_grpo=True
  algorithm.team_event_gae.internal_gamma=1.0 algorithm.team_event_gae.internal_lam=1.0
  algorithm.team_event_gae.include_kl_shaping=False algorithm.team_event_gae.normalize_advantages=True
  algorithm.team_event_gae.value.mode=legacy_loto algorithm.team_event_gae.agent_local.mode=off
  algorithm.team_event_gae.math_value.mode=math_answer_loto
  algorithm.team_event_gae.math_value.target=terminal_env_success
  algorithm.team_event_gae.math_value.expected_rollout_n=8
  algorithm.team_event_gae.math_value.expected_max_loop_num=3
  algorithm.team_event_gae.math_value.parser_version=latest_box_token_v1
  algorithm.team_event_gae.math_value.feature_source=executed_action_text
  algorithm.team_event_gae.math_value.state_view=team_history
  algorithm.team_event_gae.math_value.matching=same_round_then_cross
  algorithm.team_event_gae.math_value.cross_round_decay=0.5
  algorithm.team_event_gae.math_value.question_parent_strength=2.0
  algorithm.team_event_gae.math_value.beta=1.0 algorithm.team_event_gae.math_value.verifier_post=carry
  algorithm.team_event_gae.math_value.auxiliary_mode=env_only algorithm.team_event_gae.math_value.strict=True
  algorithm.team_event_gae.credit_trace.enabled=True algorithm.team_event_gae.credit_trace.every_n_steps=1
  algorithm.use_kl_in_reward=False algorithm.filter_groups.enable=False
  algorithm.entropy_credit.enable=True algorithm.entropy_credit.sparse.enable=True
  algorithm.entropy_credit.sparse.calibration_weight=response_tokens
  "algorithm.semantic_value.enable=$SEMANTIC_ENABLED"
  "algorithm.semantic_entropy_control.enabled=$SEMANTIC_ENABLED"
  "algorithm.semantic_value.model_path=$VALUE_ENCODER"
  "algorithm.semantic_value.initial_checkpoint=$INITIAL_CHECKPOINT"
  "algorithm.semantic_value.initialization_mode=$VALUE_INIT_MODE"
  algorithm.semantic_value.require_pretrained=True algorithm.semantic_value.device=cuda:0
  algorithm.semantic_value.torch_dtype=bfloat16
  "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_DATA"
  "data.train_batch_size=$TRAIN_BATCH_SIZE" "data.val_batch_size=$VAL_BATCH_SIZE" "data.seed=${DATA_SEED:-0}"
  data.max_prompt_length=8192 "data.max_response_length=${MAX_RESPONSE_LENGTH:-4096}"
  data.filter_overlong_prompts=True data.truncation=middle data.return_raw_chat=True
  +data.apply_chat_template_kwargs.enable_thinking=False
  actor_rollout_ref.model.path=null actor_rollout_ref.actor.optim.lr=null
  '+agent.agent_specific_parameters.actor.optim.lr=[1e-6,1e-6]'
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING:-True}" actor_rollout_ref.model.enable_gradient_checkpointing=True
  "actor_rollout_ref.model.attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
  "actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY:-fsdp}"
  "actor_rollout_ref.ref.strategy=${ACTOR_STRATEGY:-fsdp}"
  "actor_rollout_ref.actor.use_torch_compile=${USE_TORCH_COMPILE:-True}"
  actor_rollout_ref.model.use_fused_kernels=False actor_rollout_ref.model.use_liger=False
  actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True
  "actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH_SIZE * 8))"
  actor_rollout_ref.actor.ppo_mini_update_num=1 actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.use_invalid_action_penalty=False actor_rollout_ref.actor.invalid_action_penalty_coef=0.0
  "actor_rollout_ref.actor.entropy_control.enabled=$SEMANTIC_ENABLED"
  actor_rollout_ref.actor.entropy_control.loss_coef=0.0025
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.n=1
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1}" actor_rollout_ref.rollout.top_logprobs_num=16
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" actor_rollout_ref.rollout.enable_chunked_prefill=False
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}" actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.top_p=0.95
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6
  env.env_name=math "env.seed=${ENV_SEED:-0}" env.rollout.n=8 "env.rollout.val_n=$VAL_GROUP_SIZE"
  'agent.agent_ids=["Solver Agent","Verifier Agent"]'
  "agent.model_ids=[\"$SOLVER_MODEL\",\"$VERIFIER_MODEL\"]" agent.model_sharing=False
  agent.orchestra_type=math agent.orchestra.math.max_loop_num=3
  trainer.critic_warmup=0 "trainer.logger=${LOGGERS:-[console,wandb]}" trainer.project_name=DrMAS_math_hybrid
  "trainer.experiment_name=$RUN_NAME" "trainer.default_local_dir=$RUN_DIR"
  "trainer.rollout_data_dir=$RUN_DIR/rollouts" "trainer.validation_data_dir=$RUN_DIR/validation"
  trainer.event_trace.enabled=True trainer.event_trace.dump_train=True trainer.event_trace.dump_val=True
  trainer.event_trace.every_n_steps=1 "trainer.event_trace.output_dir=$RUN_DIR/event_traces"
  trainer.event_trace.include_token_ids=True trainer.event_trace.include_rollout_log_probs=True
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE" "trainer.nnodes=$NNODES"
  "trainer.agent_gpus_per_node=$AGENT_GPUS_PER_NODE"
  "trainer.save_freq=${SAVE_FREQ:-10}" "trainer.test_freq=${TEST_FREQ:-10}"
  "trainer.total_epochs=${TOTAL_EPOCHS:-1}" "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-50}"
  "trainer.val_only=$VAL_ONLY" trainer.val_before_train=True
  "trainer.resume_mode=$RESUME_MODE" "trainer.resume_from_path=$RESUME_CHECKPOINT"
)
if (( IS_DRY_RUN == 1 )); then
  printf '%q ' "$PYTHON" -m verl.trainer.main_ppo "${args[@]}" "$@"
  printf '\n'
else
  exec "$PYTHON" -m verl.trainer.main_ppo "${args[@]}" "$@"
fi
