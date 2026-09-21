import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("semantic_worker_test", Path(__file__).resolve().parents[2] / "verl/workers/semantic_value.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.config = SimpleNamespace(hidden_size=8, model_type="qwen3", vocab_size=32)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=x.cumsum(1) / torch.arange(1, x.shape[1] + 1)[None, :, None])


class Tokenizer:
    bos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(char) % 32 for char in text]


class TinyScorer(module.PrefixValueScorer):
    def _load_backbone(self):
        with torch.random.fork_rng():
            torch.manual_seed(71)
            return Encoder(), Tokenizer()


def scorer(**kwargs):
    return TinyScorer({"model_path": "tiny-fixed", "head_hidden_dim": 8, "dropout": 0., "train_epochs": 1, **kwargs})


def record(uid="t", label=1, question="problem"):
    return {"traj_uid": uid, "question": question, "max_solver_turns": 2, "label": label,
            "actions": [{"role": "solver", "text": "one"},
                        {"role": "verifier", "text": "<verify>reject</verify>"},
                        {"role": "solver", "text": "two"}]}


def encoded(s, rec=None):
    rec = rec or record()
    row, _ = s._encode(rec)
    return {**row, "label": float(rec["label"]), "question_key": "q", "traj_uid": rec["traj_uid"]}


def test_semantic_features_ignore_entropy_labels_and_future():
    s = scorer()
    original = record()
    changed = copy.deepcopy(original)
    changed["label"] = 0
    changed["actions"][-1]["text"] = "a completely different future response"
    for action in changed["actions"]:
        action.update(entropy_mean=float("nan"), absolute=[900], temporal=[-900])
    a, b = encoded(s, original), encoded(s, changed)
    torch.testing.assert_close(a["features"][:-1], b["features"][:-1], rtol=0, atol=0)
    assert not {"absolute", "temporal", "abs_mask", "temp_mask"}.intersection(a)
    assert all(name.startswith("semantic.") for name in s.candidate_head.state_dict())
    assert bool(a["prefix_terminal"][-1])
    assert torch.all(s._flatten([a, b])["weights"] == 1)


def test_overlong_tail_preserves_only_complete_boundaries():
    s = scorer(max_length=160)
    rec = record()
    rec["actions"][-1]["text"] = "x" * 1000
    row = encoded(s, rec)
    assert 1 < len(row["features"]) < 4
    assert not row["prefix_terminal"][-1]


def test_real_cpu_bce_changes_only_candidate_head():
    s = scorer(learning_rate=.02)
    row = encoded(s)
    before_encoder, before_deployed = copy.deepcopy(s.encoder.state_dict()), copy.deepcopy(s.deployed_head.state_dict())
    old = torch.nn.functional.binary_cross_entropy(s._predict(s.candidate_head, row), torch.ones(len(row["features"])))
    loss = s._train([row], 15)
    new = torch.nn.functional.binary_cross_entropy(s._predict(s.candidate_head, row), torch.ones(len(row["features"])))
    assert loss > 0 and new < old * .5
    for key, value in before_encoder.items():
        torch.testing.assert_close(value, s.encoder.state_dict()[key], atol=0, rtol=0)
    for key, value in before_deployed.items():
        torch.testing.assert_close(value, s.deployed_head.state_dict()[key], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in s.encoder.parameters())
    assert any(parameter.grad is not None for parameter in s.candidate_head.parameters())


def test_prepare_does_not_train_on_current_labels_and_terminal_is_probability():
    s = scorer()
    s.ready = True
    s.reliability = {"solver": 1., "verifier": 1.}
    first = s.prepare([record("one", 0)])["values"]["one"]
    prepared = s.prepare([record("two", 1)])
    second = prepared["values"]["two"]
    assert prepared["metrics"]["reliability_solver"] == 1.
    assert prepared["metrics"]["reliability_verifier"] == 1.
    assert first == second
    assert 0 < first[-1]["sem"] < 1 and first[-1]["terminal"]
    assert len(s._pending) == 2
    s.update()
    assert not s._pending


def test_split_uses_original_question_not_caller_key_or_outcome():
    s = scorer()
    a, b = record("a", 0), record("b", 1)
    a["question_key"], b["question_key"] = "train", "validation"
    s.prepare([a, b])
    keys = {row["question_key"] for row in s._pending.values()}
    assert keys == {hashlib.sha256(b"problem").hexdigest()}
    train, val = s._ingest()
    assert (len(train), len(val)) in ((2, 0), (0, 2))


@pytest.mark.parametrize("mutation", ["missing_question", "partial", "early_stop", "duplicate"])
def test_malformed_histories_do_not_enter_replay(mutation):
    s = scorer()
    rec = record()
    if mutation == "missing_question":
        rec["question"] = ""
    elif mutation == "partial":
        rec["actions"] = rec["actions"][:-1]
    elif mutation == "early_stop":
        rec["actions"][1]["text"] = "<verify>approve</verify>"
    else:
        rec["actions"].insert(1, rec["actions"][0])
    result = s.prepare([rec])
    assert result["metrics"]["skipped_invalid"] == 1
    assert not s._pending


class SignalScorer(TinyScorer):
    """Analytic frozen encoder for a controlled, separable CPU qualification test."""
    def _encode(self, record):
        signal = 3. if record["question"].startswith("good") else -3.
        features = torch.zeros((4, self.feature_dim))
        features[:, 0] = signal
        features[:, 1] = -signal
        return {"features": features, "prefix_roles": ["initial", "solver", "verifier", "solver"],
                "prefix_terminal": torch.tensor([False, False, False, True]),
                "prefix_valid": torch.ones(4, dtype=torch.bool)}, 4


def signal_scorer(**kwargs):
    return SignalScorer({"model_path": "tiny-fixed", "head_hidden_dim": 8, "dropout": 0.,
                         "learning_rate": .02, "train_epochs": 20, "train_batch_size": 512,
                         "min_train_trajectories": 2, "min_val_trajectories": 2,
                         "min_train_questions": 2, "min_val_questions": 2, "min_val_per_class": 1,
                         "min_role_val_prefixes": 2, "max_ece": .2, **kwargs})


def signal_records():
    return [record(str(i), i % 2, ("good" if i % 2 else "bad") + str(i)) for i in range(100)]


def test_semantic_alone_qualifies_and_only_later_prepare_uses_new_head():
    s = signal_scorer()
    prepared = s.prepare(signal_records())
    assert prepared["values"] == {} and not prepared["ready"]
    update = s.update()
    assert update["deployed"] == 1 and s.ready
    assert s.reliability == {"solver": 1., "verifier": 1.}
    later = s.prepare(signal_records()[:2])
    assert later["version"] > prepared["version"]
    assert later["values"]["0"][1]["sem"] < .2
    assert later["values"]["1"][1]["sem"] > .8


def test_terminal_predictions_do_not_count_as_role_qualification():
    s = signal_scorer()
    rows = [encoded(s, rec) for rec in signal_records()[:8]]
    metrics = s._evaluate(s.candidate_head, rows, .5)
    assert metrics["solver_prefixes"] == 8
    assert metrics["verifier_prefixes"] == 8
    assert metrics["control_prefixes"] == 16


def test_role_quality_requires_its_own_classes_auc_and_calibration():
    s = signal_scorer()
    good = {"solver_prefixes": 10, "solver_positive": 5, "solver_negative": 5,
            "solver_auc": .9, "solver_brier": .1, "solver_prior_brier": .25, "solver_ece": .05}
    assert s._quality(good, "solver")
    for key, value in (("solver_auc", .4), ("solver_ece", .9), ("solver_positive", 0), ("solver_brier", .3)):
        assert not s._quality({**good, key: value}, "solver")


def test_miscalibration_revokes_after_new_validation_windows():
    s = signal_scorer(miscalibration_patience=2)
    s.prepare(signal_records())
    s.update()
    assert s.ready
    for parameter in s.deployed_head.parameters():
        parameter.data.zero_()
    for parameter in s.candidate_head.parameters():
        parameter.data.zero_()
    train, val = s._ingest()
    s._validate_deploy(train, val, {})
    assert s.ready
    # Replaying the same validation set cannot consume another patience window.
    s._validate_deploy(train, val, {})
    assert s.ready
    changed = copy.deepcopy(val)
    changed[0]["traj_uid"] += "new-policy-window"
    s._validate_deploy(train, changed, {})
    assert not s.ready and not any(s.reliability.values())


def test_checkpoint_resume_preserves_optimizer_rng_pending_and_next_update(tmp_path):
    s = signal_scorer(train_epochs=2)
    s.prepare(signal_records())
    s.update()
    s.prepare(signal_records()[:3])
    path = tmp_path / "semantic.pt"
    s.save(path)
    other = signal_scorer(train_epochs=2)
    other.load(path, resume=True)
    assert len(other._pending) == 3 and other.step == s.step
    a, b = s.update(), other.update()
    assert a["train_bce"] == b["train_bce"]
    for key, value in s.candidate_head.state_dict().items():
        torch.testing.assert_close(value, other.candidate_head.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="configuration mismatch"):
        signal_scorer(train_epochs=3).load(path, resume=True)


def test_qualified_load_is_fresh_and_warm_start_uses_candidate(tmp_path):
    s = signal_scorer()
    s.prepare(signal_records())
    s.pretrain()
    assert s.pretrained
    path = tmp_path / "value.pt"
    s.save(path)
    fresh = signal_scorer(replay_max_trajectories=1000)
    fresh.load(path)
    assert fresh.ready and not fresh._replay and fresh.step == 0
    warm = signal_scorer()
    warm.load(path, warm_start=True)
    assert not warm.ready and warm.warm_started
    assert not warm._replay and not warm.optimizer.state


def test_old_mix_requires_explicit_warm_start_and_uses_candidate_semantics(tmp_path):
    s = scorer()
    path = tmp_path / "mix.pt"
    candidate = {key: torch.ones_like(value) * .123 for key, value in s.candidate_head.state_dict().items()}
    torch.save({"checkpoint_version": 2, "schema": "causal-role-mean-top16-v2", "config": s.config,
                "encoder_identity": {**s.encoder_identity, "entropy_schema": "old"},
                "candidate_head": {**candidate, "absolute.0.weight": torch.ones(1)},
                "deployed_head": s.deployed_head.state_dict(), "ready": False}, path)
    with pytest.raises(ValueError, match="explicit warm_start"):
        s.load(path)
    s.load(path, warm_start=True)
    assert not s.ready and s.warm_started
    for key, value in candidate.items():
        torch.testing.assert_close(value, s.candidate_head.state_dict()[key])
    assert all(name.startswith("semantic.") for name in s.candidate_head.state_dict())


@pytest.mark.parametrize("config", [{"model_type": "bert"}, {"model_type": "qwen3", "rope_scaling": {"type": "dynamic"}}])
def test_noncausal_or_sequence_length_dependent_encoder_is_rejected(config):
    with pytest.raises(ValueError):
        module._validate_backbone_config(SimpleNamespace(**config))


def test_ray_bridge_startup_serializable_results_and_checkpoint(monkeypatch, tmp_path):
    import pickle
    monkeypatch.setattr(module, "PrefixValueScorer", TinyScorer)
    worker = module.SemanticValueWorker({"model_path": "tiny-fixed", "head_hidden_dim": 8,
                                       "train_epochs": 1, "dropout": 0.})
    startup = worker.ready()
    assert startup["initialized"] and not startup["ready"]
    assert worker.scorer.step == 0 and not worker.scorer._pending
    result = worker.prepare([record()])
    assert pickle.loads(pickle.dumps(result)) == result
    assert worker.update()["version"] == 0
    path = tmp_path / "worker.pt"
    assert worker.save(path)["path"] == str(path)
    assert not worker.load(path, resume=True)["ready"]
    assert worker.ready()["initialized"]


def test_ray_bridge_rejects_nonlocal_cuda_index():
    with pytest.raises(ValueError, match="dedicated semantic Ray worker"):
        module.SemanticValueWorker({"model_path": "unused", "device": "cuda:3"})
