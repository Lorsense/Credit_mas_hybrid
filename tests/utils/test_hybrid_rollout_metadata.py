"""Exercise production rollout metadata with NumPy and lightweight batch stubs."""
import ast
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]


def definitions(path, namespace, names=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and (names is None or node.name in names)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, path, "exec"), namespace)


class Batch:
    def __init__(self, size, metadata=None):
        self.size = size
        self.non_tensor_batch = metadata or {}

    def __len__(self):
        return self.size

    @staticmethod
    def from_single_dict(data):
        return data


class Agent:
    def __init__(self, verifier=False):
        self.verifier = verifier
        self.contexts = []

    def call(self, gen_batch, env_obs, team_context, actor_rollout_wg, agent_active_mask, step):
        self.contexts.append(list(team_context))
        text = (["<verify>approve</verify>", "<verify>reject</verify>"] if self.verifier
                else ["executed answer"] * len(gen_batch))
        return Batch(len(gen_batch), {
            "agent_active_mask": agent_active_mask.copy(),
            "executed_action_text": np.asarray(text, dtype=object),
            "raw_action_text": np.asarray(["raw unprojected output"] * len(gen_batch), dtype=object),
            "is_action_valid": np.asarray([True, False]),
            "generation_finish_reason": np.asarray(["stop", "length"], dtype=object),
            "value_action_truncated": np.asarray([False, True]),
        }), text

    def update_approved_vector(self, responses, approved, active):
        return np.logical_or(approved, np.array([True, False]))


def environment():
    ns = {"np": np, "DataProto": Batch, "BaseAgent": object,
          "event_metadata_enabled": lambda config: True, "Counter": Counter,
          "defaultdict": defaultdict, "EVENT_TRACE_SCHEMA_VERSION": "math",
          "MATH_EVENT_TRACE_SCHEMA_VERSION": "math", "SEARCH_EVENT_TRACE_SCHEMA_VERSION": "search",
          "MATH_EVENT_TYPE_BY_AGENT": {"Solver Agent": "math_solution", "Verifier Agent": "math_verifier"},
          "MATH_SOLVER_EVENT_TYPE": "math_solution", "MATH_VERIFIER_EVENT_TYPE": "math_verifier",
          "MATH_EVENT_TRACE_ADAPTER_VERSION": "math", "collate_fn": lambda rows: rows}
    definitions("verl/trainer/ppo/team_event_math_value.py", ns, {"allowed_math_chains"})
    definitions("agent_system/agent/orchestra/base.py", ns)
    definitions("agent_system/agent/orchestra/math/math_orchestra.py", ns)
    definitions("agent_system/event_trace.py", ns, {"annotate_trajectory_events"})
    definitions("agent_system/math_event_trace.py", ns)
    definitions("agent_system/multi_turn_rollout/rollout_loop.py", ns)
    return ns


def orchestra(ns, enabled=True):
    cls = ns["MathMultiAgentOrchestra"]
    instance = cls.__new__(cls)
    instance.agents = {cls.SOLVER_AGENT: Agent(), cls.VERIFIER_AGENT: Agent(True)}
    instance.agents_to_wg_mapping = {role: role for role in instance.agents}
    instance.max_loop_num = 3
    instance.value_metadata_enabled = enabled
    instance.multiagent_batch_buffer = []
    instance.memory = None
    instance.config = SimpleNamespace(agent=SimpleNamespace(orchestra=SimpleNamespace(math=SimpleNamespace(max_loop_num=3))))
    return instance


def run(instance):
    gen_batch = Batch(2)
    instance.prepare_value_metadata(gen_batch, [
        {"question": "original q0", "ground_truth": "secret0"},
        {"question": "original q1", "ground_truth": "secret1"},
    ])
    actions, buffer = instance.run(gen_batch, {"text": ["truncated prompt0", "truncated prompt1"]},
                                   {role: None for role in instance.agents}, np.ones(2, dtype=bool), 1)
    return gen_batch, actions, buffer


def test_real_event_indices_and_transition_ownership_survive_metadata_merge():
    ns = environment()
    instance = orchestra(ns)
    _, actions, buffer = run(instance)
    assert len(buffer) == 5
    assert actions == ["executed answer"] * 2
    ns["annotate_math_step_events"](buffer, np.ones(2, dtype=bool), [1., 0.], [True, True])
    trajectories = [[], []]
    for entry in buffer:
        for row in range(2):
            event = {key: value[row] for key, value in entry["batch"].non_tensor_batch.items()}
            if not event["agent_active_mask"]:
                continue
            event.update(agent_id=entry["agent_id"], traj_uid=f"traj-{row}", task_type="math", active_masks=True)
            trajectories[row].append(event)
    collector = ns["TrajectoryCollector"](instance.config, None)
    result = collector.gather_rollout_data(
        trajectories, np.array([1., 0.]), np.ones(2), {"success_rate": np.array([1., 0.])},
        np.array(["traj-0", "traj-1"]), np.zeros(2))
    assert len(result) == 7
    for row, trajectory in enumerate(trajectories):
        assert [event["value_action_index"] for event in trajectory] == list(range(len(trajectory)))
        assert [event["role_turn_index"] for event in trajectory] == (
            [0, 0] if row == 0 else [0, 0, 1, 1, 2])
        assert [event["event_uid"] for event in trajectory] == [f"traj-{row}:{i}" for i in range(len(trajectory))]
        assert all(event["value_question"] == f"original q{row}" for event in trajectory)
        assert all(event["value_action_text"] == event["executed_action_text"] for event in trajectory)
        assert all(event["value_max_solver_turns"] == 3 for event in trajectory)
        assert all("ground_truth" not in event for event in trajectory)
        assert all(event["generation_finish_reason"] == ("stop" if row == 0 else "length") for event in trajectory)
        assert all(event["is_action_valid"] == (row == 0) for event in trajectory)
        assert all(event["value_action_truncated"] == (row == 1) for event in trajectory)
        assert all(event["pure_entropy_truncated"] == (row == 1) for event in trajectory)
        assert [event["is_env_action"] for event in trajectory] == [False] * (len(trajectory) - 1) + [True]
        assert [event["env_done"] for event in trajectory] == [False] * (len(trajectory) - 1) + [True]
        assert trajectory[-1]["env_reward"] == 1. - row
        assert trajectory[-1]["transition_owner_event_uid"] == trajectory[-1]["event_uid"]
        assert all(event["env_reward"] == 0 for event in trajectory[:-1])


def test_disabling_value_metadata_preserves_rollout_actions_and_contexts():
    ns = environment()
    enabled, disabled = orchestra(ns), orchestra(ns, False)
    _, enabled_actions, enabled_buffer = run(enabled)
    disabled_batch, disabled_actions, disabled_buffer = run(disabled)
    assert enabled_actions == disabled_actions
    assert len(enabled_buffer) == len(disabled_buffer)
    assert "value_question" not in disabled_batch.non_tensor_batch
    for role in enabled.agents:
        assert enabled.agents[role].contexts == disabled.agents[role].contexts
    for entry in disabled_buffer:
        # Backend truncation is a neutral event fact captured even when the
        # optional semantic question/prefix pipeline is disabled.
        assert not any(key.startswith("value_") and key != "value_action_truncated"
                       for key in entry["batch"].non_tensor_batch)


def test_original_question_required_without_answer_fallback():
    ns = environment()
    instance = orchestra(ns)
    with pytest.raises(ValueError, match="question"):
        instance.prepare_value_metadata(Batch(2), [{"ground_truth": "secret"}] * 2)
    with pytest.raises(ValueError, match="Original questions"):
        instance._save_value_metadata(Batch(2), Batch(2), ["a", "b"], [True, True])


def entropy_attachment():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hybrid_test_entropy", ROOT / "verl/utils/entropy_credit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns = {"compute_action_topk_entropy": module.compute_action_topk_entropy}
    definitions("verl/workers/rollout/sglang_rollout/sglang_rollout.py", ns, {"_attach_action_topk_entropy"})
    return ns["_attach_action_topk_entropy"]


def test_top16_retains_coverage_and_finish_reason_and_drops_distributions():
    attach = entropy_attachment()
    output = [{"meta_info": {"output_token_logprobs": [(-1., 1, None), (-2., 2, None)],
                              "output_top_logprobs": [[(-3., i, None) for i in range(16)], []],
                              "finish_reason": {"type": "length"}}}]
    attach(output, 16)
    meta = output[0]["meta_info"]
    stats = meta["top16_entropy"]
    assert stats["mean"] == pytest.approx(1.)
    assert stats["coverage"] == .5
    assert meta["finish_reason"] == {"type": "length"}
    assert "output_top_logprobs" not in meta


def test_disabled_top16_leaves_backend_output_untouched():
    attach = entropy_attachment()
    output = [{"meta_info": {"output_top_logprobs": [[(-1., 1, None)]]}}]
    attach(output, 0)
    assert "top16_entropy" not in output[0]["meta_info"]
    assert "output_top_logprobs" in output[0]["meta_info"]
