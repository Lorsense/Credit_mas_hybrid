"""Parse the real launcher argument array and compose Hydra without Bash/Ray."""
from pathlib import Path
import re
import shlex

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/drmas_trainer/run_math_hybrid.sh"


def launcher_args(**changes):
    values = dict(SEMANTIC_ENABLED="True", VALUE_ENCODER="Qwen/Qwen3-4B", INITIAL_CHECKPOINT="/assets/semantic_value.pt",
                  VALUE_INIT_MODE="qualified", TRAIN_DATA="/data/train.parquet", VAL_DATA="/data/test_sampled.parquet",
                  TRAIN_BATCH_SIZE="30", VAL_BATCH_SIZE="110", VAL_GROUP_SIZE="1", SOLVER_MODEL="Qwen/Qwen3-4B",
                  VERIFIER_MODEL="Qwen/Qwen3-4B", RUN_NAME="test_hybrid", RUN_DIR="/runs/test_hybrid",
                  N_GPUS_PER_NODE="16", NNODES="1", AGENT_GPUS_PER_NODE="[15]", VAL_ONLY="False",
                  RESUME_MODE="disable", RESUME_CHECKPOINT="null")
    values.update(changes)
    source = SCRIPT.read_text(encoding="utf-8").split("args=(\n", 1)[1].split("\n)\n", 1)[0]
    source = source.replace("$((TRAIN_BATCH_SIZE * 8))", str(int(values["TRAIN_BATCH_SIZE"]) * 8))
    # Expand the launcher's scalar parameter defaults before shell tokenization.
    source = re.sub(r"\$\{([A-Z_]+):-([^}]+)\}", lambda match: values.get(match[1], match[2]), source)
    source = re.sub(r"\$([A-Z_]+)", lambda match: values[match[1]], source)
    assert "$" not in source, "new shell expressions need explicit coverage"
    return shlex.split(source)


def configuration(**changes):
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=launcher_args(**changes))


def test_training_defaults_compose_into_the_hybrid_method():
    config = configuration()
    assert config.algorithm.adv_estimator == "team_event_gae"
    assert config.algorithm.team_event_gae.math_value.mode == "math_answer_loto"
    assert config.algorithm.team_event_gae.math_value.auxiliary_mode == "env_only"
    assert config.algorithm.team_event_gae.agent_local.mode == "off"
    assert config.algorithm.gamma == config.algorithm.lam == 1
    assert config.algorithm.team_event_gae.internal_gamma == config.algorithm.team_event_gae.internal_lam == 1
    assert config.env.rollout.n == 8 and config.agent.orchestra.math.max_loop_num == 3
    assert config.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    assert config.actor_rollout_ref.actor.ppo_mini_update_num == config.actor_rollout_ref.actor.ppo_epochs == 1
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.rollout.top_logprobs_num == 16
    assert not config.actor_rollout_ref.actor.use_invalid_action_penalty
    assert not config.algorithm.use_kl_in_reward and not config.algorithm.team_event_gae.include_kl_shaping
    assert config.algorithm.entropy_credit.enable and config.algorithm.entropy_credit.sparse.enable
    assert config.algorithm.entropy_credit.sparse.calibration_weight == "response_tokens"
    assert config.algorithm.semantic_value.enable and config.algorithm.semantic_entropy_control.enabled
    assert config.actor_rollout_ref.actor.entropy_control.loss_coef == 0.0025
    assert config.trainer.resume_mode == "disable"
    assert list(config.trainer.logger) == ["console", "wandb"]
    assert config.trainer.save_freq == 10
    assert config.trainer.event_trace.enabled
    assert config.trainer.event_trace.dump_train and config.trainer.event_trace.dump_val
    assert config.trainer.event_trace.every_n_steps == 1
    assert config.trainer.event_trace.include_token_ids and config.trainer.event_trace.include_rollout_log_probs
    assert config.algorithm.team_event_gae.credit_trace.enabled
    assert config.algorithm.team_event_gae.credit_trace.every_n_steps == 1
    assert config.trainer.rollout_data_dir == "/runs/test_hybrid/rollouts"
    assert config.trainer.validation_data_dir == "/runs/test_hybrid/validation"
    assert "advantage_recovery" not in config.algorithm


def test_two_node_config_uses_fifteen_actual_actor_ranks():
    config = configuration(NNODES="2", N_GPUS_PER_NODE="8", AGENT_GPUS_PER_NODE="[7,8]")
    assert list(config.trainer.agent_gpus_per_node) == [7, 8]
    assert config.data.train_batch_size % sum(config.trainer.agent_gpus_per_node) == 0


def test_evaluation_disables_scorer_but_keeps_actor_checkpoint_partition():
    config = configuration(SEMANTIC_ENABLED="False", VAL_ONLY="True", VAL_DATA="/data/test.parquet",
                           RESUME_MODE="resume_path", RESUME_CHECKPOINT="/runs/hybrid/global_step_50")
    assert not config.algorithm.semantic_value.enable
    assert not config.algorithm.semantic_entropy_control.enabled
    assert not config.actor_rollout_ref.actor.entropy_control.enabled
    assert list(config.trainer.agent_gpus_per_node) == [15]
    assert config.trainer.val_only and config.trainer.resume_mode == "resume_path"
    assert config.data.val_files == "/data/test.parquet"


def test_qwen35_wrapper_composes_native_training_and_preserves_hybrid():
    # Read the actual wrapper exports, then compose the actual shared launcher.
    source = (SCRIPT.parent / "run_math_qwen35.sh").read_text(encoding="utf-8")
    values = {}
    for name, value in re.findall(r"^export ([A-Z_]+)=(.+)$", source, re.MULTILINE):
        default = re.fullmatch(r"\$\{[A-Z_]+:-([^}]+)\}", value)
        values[name] = default[1] if default else value
    config = configuration(**values)
    assert list(config.agent.model_ids) == ["Qwen/Qwen3.5-4B"] * 2
    assert config.algorithm.semantic_value.model_path == "Qwen/Qwen3.5-4B"
    assert config.algorithm.semantic_value.initialization_mode == "candidate"
    assert config.actor_rollout_ref.model.attn_implementation == "sdpa"
    assert not config.actor_rollout_ref.model.use_remove_padding
    assert not config.actor_rollout_ref.model.use_fused_kernels
    assert not config.actor_rollout_ref.actor.use_torch_compile
    assert config.actor_rollout_ref.actor.strategy == "fsdp2"
    assert config.actor_rollout_ref.actor.ulysses_sequence_parallel_size == 1
    assert config.actor_rollout_ref.rollout.enforce_eager
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert config.algorithm.team_event_gae.math_value.mode == "math_answer_loto"
    assert config.algorithm.entropy_credit.sparse.enable
    assert config.actor_rollout_ref.actor.entropy_control.enabled
    assert config.trainer.save_freq == 10 and config.trainer.event_trace.dump_train

    # Offline and online must resolve the same encoder and strict head settings.
    from omegaconf import OmegaConf

    pretrain_source = (SCRIPT.parent / "pretrain_semantic_value_qwen35.sh").read_text(encoding="utf-8")
    encoder_default = re.search(r"^export VALUE_ENCODER=\$\{VALUE_ENCODER:-([^}]+)\}$",
                                pretrain_source, re.MULTILINE)[1]
    assert encoder_default == config.algorithm.semantic_value.model_path
    shared_pretrain = (SCRIPT.parent / "pretrain_semantic_value.sh").read_text(encoding="utf-8")
    assert '--model-path "${VALUE_ENCODER:-' in shared_pretrain
    offline = OmegaConf.load(SCRIPT.parent / "pretrain_semantic_value.yaml").semantic_value
    for key in ("revision", "torch_dtype", "attn_implementation", "max_length", "head_hidden_dim", "dropout"):
        assert offline[key] == config.algorithm.semantic_value[key]


def test_qwen35_moe_model_and_tp_layout_are_configurable():
    config = configuration(SOLVER_MODEL="Qwen/Qwen3.5-35B-A3B", VERIFIER_MODEL="Qwen/Qwen3.5-35B-A3B",
                           ROLLOUT_TP="2", NNODES="2", N_GPUS_PER_NODE="8", AGENT_GPUS_PER_NODE="[6,8]",
                           TRAIN_BATCH_SIZE="28", USE_REMOVE_PADDING="False", ATTN_IMPLEMENTATION="sdpa")
    assert list(config.agent.model_ids) == ["Qwen/Qwen3.5-35B-A3B"] * 2
    assert sum(config.trainer.agent_gpus_per_node) % config.actor_rollout_ref.rollout.tensor_model_parallel_size == 0
    assert config.data.train_batch_size == 28


def test_worker_group_count_reaches_each_colocated_engine():
    from agent_system.agent.utils import build_wg_ids

    for sharing, expected_groups in [(False, 2), (True, 1)]:
        config = configuration()
        config.agent.model_sharing = sharing
        groups = build_wg_ids(config)
        assert len(groups) == expected_groups
        assert {entries[0]["config_actor_rollout_ref"].rollout.agent_port
                for entries in groups.values()} == set(range(expected_groups))
        assert all(entry["config_actor_rollout_ref"].rollout.agent_count == expected_groups
                   for entries in groups.values() for entry in entries)
