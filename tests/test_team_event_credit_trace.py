"""CPU contract tests: no Ray, OmegaConf, or DataProto installation required."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = _load("credit_trace_test_engine", "verl/trainer/ppo/team_event_boundary_value.py")
trace = _load("credit_trace_test_helper", "agent_system/team_event_credit_trace.py")


@pytest.fixture
def config(tmp_path, monkeypatch):
    # Only substitute the import location, never the production engine logic.
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.team_event_boundary_value", engine)
    asset_path = tmp_path / "idf.json"
    asset_path.write_text(json.dumps(engine.FrozenTfidfKernel.fit(["a first result", "second result"]).to_asset()), encoding="utf-8")
    return {
        "algorithm": {"adv_estimator": "team_event_gae", "gamma": 1.0, "lam": 0.95,
                      "use_kl_in_reward": False,
                      "team_event_gae": {"internal_gamma": 1.0, "internal_lam": 1.0,
                                         "agent_local": {"mode": "off"},
                                         "value": {"mode": "boundary_loto", "idf_path": str(asset_path)}}},
        "env": {"env_name": "search", "rollout": {"n": 8, "val_n": 1}},
        "agent": {"orchestra_type": "search"},
        "actor_rollout_ref": {"rollout": {"n": 1}, "actor": {"ppo_mini_update_num": 5,
                                "use_invalid_action_penalty": True, "invalid_action_penalty_coef": 0.01}},
        "trainer": {"val_only": False},
    }


def test_legacy_needs_no_asset_and_training_n_not_val_n(config):
    assert trace.validate_boundary_training_config(config) is not None
    config["env"]["rollout"]["val_n"] = 32
    assert trace.validate_boundary_training_config(config) is not None
    config["algorithm"]["team_event_gae"]["value"] = {}
    config["env"]["rollout"]["n"] = 5
    assert trace.validate_boundary_training_config(config) is None


@pytest.mark.parametrize("path,value,match", [
    ("algorithm.gamma", 0.95, "gamma"),
    ("algorithm.team_event_gae.internal_gamma", 0.95, "gamma"),
    ("algorithm.team_event_gae.agent_local.mode", "shadow", "agent_local"),
    ("env.rollout.n", 5, "env.rollout.n=8"),
    ("actor_rollout_ref.rollout.n", 8, "must remain 1"),
    ("agent.orchestra_type", "math", "Search"),
    ("algorithm.team_event_gae.value.idf_path", None, "frozen IDF"),
    ("algorithm.team_event_gae.value.target", "discounted", "terminal_env_return"),
    ("algorithm.team_event_gae.value.empty_parent_fallback", "carry", "question parent"),
    ("algorithm.team_event_gae.value.idf_sha256", "0" * 64, "SHA"),
])
def test_invalid_effective_configuration_rejected(config, path, value, match):
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        trace.validate_boundary_training_config(config)


def test_env_only_rejects_rewards_not_independent_actor_kl(config):
    config["algorithm"]["team_event_gae"]["value"]["auxiliary_mode"] = "env_only"
    with pytest.raises(ValueError, match="invalid-action"):
        trace.validate_boundary_training_config(config)
    config["actor_rollout_ref"]["actor"]["use_invalid_action_penalty"] = False
    config["actor_rollout_ref"]["actor"]["use_kl_loss"] = True
    assert trace.validate_boundary_training_config(config) is not None
    config["algorithm"]["use_kl_in_reward"] = True
    with pytest.raises(ValueError, match="reward-KL"):
        trace.validate_boundary_training_config(config)


def test_val_only_does_not_require_training_asset_or_n8(config):
    config["trainer"]["val_only"] = True
    config["env"]["rollout"]["n"] = 1
    config["algorithm"]["team_event_gae"]["value"]["idf_path"] = None
    assert trace.validate_boundary_training_config(config) is None


def test_resume_requires_same_content_and_configuration(config, tmp_path):
    manifest = trace.build_boundary_run_manifest(config, run_id="run-a")
    assert manifest["compatibility"]["idf_sha256"]
    with pytest.raises(ValueError, match="requires the checkpoint"):
        trace.check_boundary_checkpoint_manifest(manifest, tmp_path)
    trace.save_boundary_run_manifest(manifest, tmp_path)
    resumed = copy.deepcopy(manifest)
    resumed["run_id"] = "run-b"
    resumed["idf_path"] = "/different/node/same-content.json"
    trace.check_boundary_checkpoint_manifest(resumed, tmp_path)
    with pytest.raises(ValueError, match="legacy"):
        trace.check_boundary_checkpoint_manifest(None, tmp_path)
    resumed["compatibility"]["lambda_internal"] = 0.8
    with pytest.raises(ValueError, match="differs"):
        trace.check_boundary_checkpoint_manifest(resumed, tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        trace.save_boundary_run_manifest(resumed, tmp_path)


def _batch(manifest):
    nt = {"event_uid": ["e1", "e0", "e1"], "uid": ["q"] * 3,
          "traj_uid": ["t"] * 3, "event_index": [1, 0, 1], "env_step_index": [0] * 3,
          "agent_id": ["S", "V", "S"], "event_type": ["search_query", "verifier", "search_query"],
          "wg_id": ["s", "v", "s"], "env_done": [False] * 3,
          "route_target": [None, "search", None], "route_reason": [None, "verify_no", None]}
    tensors = {"event_boundary_" + suffix: torch.tensor([1., 0., 1.])
               for suffix in trace._DIAGNOSTIC_FIELDS.values()}
    tensors["response_mask"] = torch.tensor([[0, 1, 1]] * 3)
    tensors["advantages"] = torch.tensor([[0., 0.25, 0.25], [0., -0.5, -0.5], [0., 0.25, 0.25]])
    return SimpleNamespace(non_tensor_batch=nt, batch=tensors, meta_info={"boundary_loto": {
        "idf_sha256": manifest["compatibility"]["idf_sha256"],
        "details_by_event": {"e1": {"document_key_sha256": "doc-hash"}}}})


def test_credit_trace_unique_sorted_shadow_actor_and_no_overwrite(config, tmp_path):
    config["algorithm"]["team_event_gae"]["value"]["application"] = "shadow"
    manifest = trace.build_boundary_run_manifest(config, run_id="run")
    batch = _batch(manifest)
    rows = trace.build_boundary_credit_records(batch, manifest, global_step=12, split="train")
    assert [row["event_uid"] for row in rows] == ["e0", "e1"]
    assert rows[1]["candidate_advantage"] == 1
    assert rows[1]["actor_advantage"] == 0.25
    assert rows[1]["document_key_sha256"] == "doc-hash"
    path, count, part = trace.dump_boundary_credit_trace(batch, tmp_path, manifest, global_step=12)
    path2, count2, part2 = trace.dump_boundary_credit_trace(batch, tmp_path, manifest, global_step=12)
    assert (count, count2, part, part2) == (2, 2, 0, 1)
    assert Path(path).read_text() == Path(path2).read_text()
    batch.batch["event_boundary_value_post"][2] = 0.5
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        trace.build_boundary_credit_records(batch, manifest, global_step=12, split="train")


def test_credit_trace_refuses_missing_diagnostics_and_asset_mismatch(config):
    manifest = trace.build_boundary_run_manifest(config, run_id="run")
    batch = _batch(manifest)
    batch.meta_info["boundary_loto"]["idf_sha256"] = "bad"
    with pytest.raises(ValueError, match="digest"):
        trace.build_boundary_credit_records(batch, manifest, global_step=1, split="train")
    batch = _batch(manifest)
    del batch.batch["event_boundary_value_post"]
    with pytest.raises(ValueError, match="diagnostics"):
        trace.build_boundary_credit_records(batch, manifest, global_step=1, split="train")
