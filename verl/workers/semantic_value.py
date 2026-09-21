"""Semantic-only frozen causal encoder, candidate/deployed success-probability heads.

No entropy statistic enters encoding, supervision or deployment qualification.
Prepare uses only an already deployed head. Update runs after both Actors, so
labels from the current batch can affect entropy control only in a later batch.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

_DEFAULTS = {
    "device": "cpu", "torch_dtype": "float32", "attn_implementation": "sdpa",
    "trust_remote_code": False, "revision": None, "max_length": 16384,
    "head_hidden_dim": 256, "dropout": .1, "learning_rate": .001,
    "weight_decay": .0001, "train_epochs": 5, "train_batch_size": 256,
    "replay_max_trajectories": 2048, "validation_fraction": .2,
    "holdout_salt": "semantic-prefix-v1", "seed": 0,
    "min_train_trajectories": 64, "min_val_trajectories": 32,
    "min_val_per_class": 8, "min_train_questions": 8, "min_val_questions": 8,
    "min_val_auc": .55, "min_brier_improvement": 0.,
    "candidate_brier_tolerance": 0., "max_ece": .15,
    "min_role_val_prefixes": 32, "miscalibration_patience": 3,
    "disable_miscalibrated": True, "disable_ready_when_insufficient": False,
    "cpu_num_threads": 1,
}
_STRUCTURAL_DIM = 8
_CHECKPOINT_VERSION = 1
FEATURE_SCHEMA = "semantic-prefix-v1"
ROLES = ("solver", "verifier")
_QWEN35_BACKBONES = {
    "qwen3_5": ("qwen3_5_text", "Qwen3_5ForConditionalGeneration"),
    "qwen3_5_moe": ("qwen3_5_moe_text", "Qwen3_5MoeForConditionalGeneration"),
}

class SemanticValueHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout):
        super().__init__()
        self.semantic = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                                      nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, features):
        return self.semantic(features).squeeze(-1)

class _InvalidRecord(ValueError):
    pass

def _config_field(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _validate_backbone_config(config):
    """Resolve the causal text config and reject length-dependent prefix states.

    Qwen3.5's public checkpoints have a composite vision/text config. Checking
    only its root would miss text hidden dimensions and the new rope_parameters
    field. Linear recurrent attention is causal too; this does not require all
    layers to be conventional self-attention.
    """
    model_type = _config_field(config, "model_type")
    text_config = config
    if model_type in _QWEN35_BACKBONES:
        text_config = _config_field(config, "text_config")
        expected = _QWEN35_BACKBONES[model_type][0]
        if text_config is None or _config_field(text_config, "model_type") != expected:
            raise ValueError(f"{model_type} requires a matching {expected} text_config")
    elif model_type not in {"qwen3", *(item[0] for item in _QWEN35_BACKBONES.values())}:
        raise ValueError("prefix-value encoder requires a supported causal Qwen3 or Qwen3.5 text backbone")

    def check_rope(value):
        if not isinstance(value, Mapping):
            return
        rope_type = str(value.get("rope_type", value.get("type", "default"))).lower()
        if "dynamic" in rope_type or rope_type == "longrope":
            raise ValueError("sequence-length-dependent RoPE is incompatible with exact causal-prefix feature reuse")
        # New Transformers configs may specify per-layer-type RoPE settings.
        for child in value.values():
            if isinstance(child, Mapping):
                check_rope(child)

    for owner in (config, text_config):
        for name in ("rope_scaling", "rope_parameters"):
            check_rope(_config_field(owner, name, {}))
    return text_config


def _semantic_config_fingerprint(config):
    """Fingerprint architecture fields, excluding loader/runtime annotations."""
    values = config.to_dict() if hasattr(config, "to_dict") else dict(config) if isinstance(config, Mapping) else dict(vars(config))
    ignored = {"transformers_version", "dtype", "torch_dtype", "architectures"}

    def canonical(value):
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in value.items()
                    if not str(key).startswith("_") and str(key) not in ignored}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        if isinstance(value, set):
            return sorted(canonical(item) for item in value)
        return value

    return hashlib.sha256(json.dumps(canonical(values), sort_keys=True, default=str).encode()).hexdigest()


def _tokenizer_fingerprint(tokenizer):
    if not callable(getattr(tokenizer, "get_vocab", None)):
        raise ValueError("Qwen3.5 semantic tokenizer must expose its vocabulary identity")
    values = {"class": type(tokenizer).__name__, "vocab": tokenizer.get_vocab(),
              "special_tokens": getattr(tokenizer, "special_tokens_map", {})}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if callable(getattr(backend, "to_str", None)):
        values["backend"] = backend.to_str()
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def _encoder_weights_fingerprint(encoder):
    """Identify actual frozen weights, including local files replaced in place.

    Hash before transferring the encoder to its device. Bounded byte chunks
    avoid a second whole-model copy, including for BF16 tensors.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        raw = tensor.detach().reshape(-1).contiguous().view(torch.uint8)
        for start in range(0, raw.numel(), 16 * 1024 * 1024):
            digest.update(memoryview(raw[start:start + 16 * 1024 * 1024].cpu().numpy()))
    return digest.hexdigest()


def _check_complete_backbone_load(loading_info):
    """Do not silently use freshly initialized text weights from a bad mapping."""
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading_info.get(key):
            raise ValueError(f"Qwen3.5 semantic checkpoint did not load completely: {key}={loading_info[key]}")


def _extract_qwen35_language_model(model, root_config):
    expected = _QWEN35_BACKBONES[_config_field(root_config, "model_type")][0]
    encoder = getattr(getattr(model, "model", None), "language_model", None)
    if encoder is None or _config_field(encoder.config, "model_type") != expected:
        raise ValueError("Qwen3.5 semantic loader requires the loaded model.language_model text backbone")
    _validate_backbone_config(encoder.config)
    if any(parameter.is_meta for parameter in encoder.parameters()):
        raise ValueError("Qwen3.5 semantic text backbone still contains uninitialized meta parameters")
    return encoder

def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)

def _weighted_auc(scores: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> float:
    """Weighted Mann-Whitney AUROC, with half credit for tied predictions."""
    order = torch.argsort(scores)
    scores, labels, weights = scores[order], labels[order], weights[order]
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive == 0 or negative == 0:
        return float("nan")
    cumulative_negative = 0.0
    concordant = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group_labels = labels[start:end]
        group_weights = weights[start:end]
        group_positive = float(group_weights[group_labels == 1].sum())
        group_negative = float(group_weights[group_labels == 0].sum())
        concordant += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        start = end
    return concordant / (positive * negative)

class PrefixValueScorer:
    """Frozen independent causal text encoder with one trainable semantic MLP."""

    def __init__(self, config):
        self.config = {**_DEFAULTS, **dict(config)}
        self._validate_config()
        self.device = torch.device(self.config["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("semantic value requested CUDA but CUDA is unavailable")
        if self.device.type == "cpu":
            torch.set_num_threads(int(self.config["cpu_num_threads"]))
        self.encoder, self.tokenizer = self._load_backbone()
        text_config = _validate_backbone_config(self.encoder.config)
        weights_fingerprint = (_encoder_weights_fingerprint(self.encoder)
                               if _config_field(text_config, "model_type") != "qwen3" else None)
        self.encoder.to(self.device).eval().requires_grad_(False)
        hidden = int(_config_field(text_config, "hidden_size"))
        self.feature_dim = hidden + _STRUCTURAL_DIM
        self.encoder_identity = {
            "model_path": str(self.config["model_path"]), "revision": self.config["revision"],
            "model_type": getattr(self.encoder.config, "model_type", None),
            "commit_hash": getattr(self.encoder.config, "_commit_hash", None),
            "hidden_size": hidden, "vocab_size": getattr(self.encoder.config, "vocab_size", None),
            "serialization": "math-prefix-segments-v2", "structural_dim": _STRUCTURAL_DIM,
        }
        # Preserve the original Qwen3 identity exactly, so existing qualified
        # checkpoints remain loadable. New backbones record their text model
        # and tokenization protocol and can never load an old Qwen3 head.
        if _config_field(text_config, "model_type") != "qwen3":
            self.encoder_identity.update(getattr(self, "_source_backbone_identity", {}))
            self.encoder_identity.update(
                text_config_sha256=_semantic_config_fingerprint(text_config),
                tokenizer_sha256=_tokenizer_fingerprint(self.tokenizer),
                text_backbone_class=type(self.encoder).__name__,
                text_weights_sha256=weights_fingerprint,
                extraction="qwen3.5-loaded-model.language_model-v1",
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.config["seed"]))
            self.candidate_head = SemanticValueHead(self.feature_dim, int(self.config["head_hidden_dim"]),
                                                     float(self.config["dropout"])).to(self.device)
        self.deployed_head = copy.deepcopy(self.candidate_head).eval().requires_grad_(False)
        self._reset_training_state()

    def _reset_training_state(self):
        self.optimizer = torch.optim.AdamW(self.candidate_head.parameters(),
                                           lr=float(self.config["learning_rate"]),
                                           weight_decay=float(self.config["weight_decay"]))
        self._rng = torch.Generator(device="cpu").manual_seed(int(self.config["seed"]))
        self._replay, self._pending = OrderedDict(), OrderedDict()
        self.ready = self.pretrained = self.warm_started = False
        self.version = self.step = 0
        self.reliability = dict.fromkeys(ROLES, 0.0)
        self.role_bad_windows = dict.fromkeys(ROLES, 0)
        self.last_validation_fingerprint = None
        self.last_metrics = {}

    def _validate_config(self):
        if not self.config.get("model_path"):
            raise ValueError("semantic_value.model_path must identify the fixed causal encoder")
        if self.config["torch_dtype"] not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported encoder torch_dtype")
        for key in ("max_length", "head_hidden_dim", "train_epochs", "train_batch_size",
                    "replay_max_trajectories", "min_train_trajectories", "min_val_trajectories",
                    "min_val_per_class", "min_train_questions", "min_val_questions",
                    "cpu_num_threads", "min_role_val_prefixes", "miscalibration_patience"):
            value = self.config[key]
            if isinstance(value, bool) or int(value) != float(value) or int(value) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("validation_fraction", "dropout", "min_val_auc", "max_ece"):
            value = float(self.config[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not 0 < float(self.config["validation_fraction"]) < 1 or float(self.config["dropout"]) == 1:
            raise ValueError("validation_fraction must be interior and dropout less than one")
        for key in ("learning_rate", "weight_decay", "min_brier_improvement", "candidate_brier_tolerance"):
            if not math.isfinite(float(self.config[key])) or float(self.config[key]) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if float(self.config["learning_rate"]) == 0:
            raise ValueError("learning_rate must be positive")

    @staticmethod
    def _validate_record(record):
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise _InvalidRecord("complete original question text is required")
        if not record.get("traj_uid") or float(record["label"]) not in (0., 1.):
            raise _InvalidRecord("trajectory identity and binary outcome are required")
        budget = record["max_solver_turns"]
        if isinstance(budget, bool) or int(budget) != float(budget) or int(budget) < 1:
            raise _InvalidRecord("actual positive solver budget is required")
        actions = record["actions"]
        if not isinstance(actions, list) or not actions:
            raise _InvalidRecord("complete action chain is required")
        for i, action in enumerate(actions):
            role = "solver" if i % 2 == 0 else "verifier"
            if action["role"] != role or not isinstance(action["text"], str):
                raise _InvalidRecord("action chain must follow executed Solver/Verifier order")
            if int(action.get("role_turn_index", i // 2)) != i // 2:
                raise _InvalidRecord("role turn does not match action order")
            if role == "solver":
                terminal = i // 2 + 1 >= int(budget)
            else:
                terminal = "<verify>approve</verify>" in action["text"] or "<verify>reject</verify>" not in action["text"]
            if terminal != (i == len(actions) - 1):
                raise _InvalidRecord("incomplete trajectory or action after termination")

    @staticmethod
    def _flatten(rows):
        result = {key: torch.cat([row[key] for row in rows])
                  for key in ("features", "prefix_terminal", "prefix_valid")}
        result["labels"] = torch.cat([torch.full((len(row["features"]),), row["label"]) for row in rows])
        # Every observed prefix has the same probability target. Weighting by
        # final trajectory length would let future stopping alter that target.
        result["weights"] = torch.ones_like(result["labels"])
        result["roles"] = sum([row["prefix_roles"] for row in rows], [])
        return result

    def _predict(self, head, data):
        head.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(data["features"]), int(self.config["train_batch_size"])):
                features = data["features"][start:start + int(self.config["train_batch_size"])].to(self.device).float()
                scores.append(torch.sigmoid(head(features)).cpu())
        return torch.cat(scores).clone()

    def prepare(self, records):
        started = time.perf_counter()
        values, seen = {}, set()
        invalid = overlong = partial = encoded = 0
        for record in records:
            try:
                self._validate_record(record)
                uid = str(record["traj_uid"])
                if uid in seen:
                    raise _InvalidRecord("duplicate trajectory identity")
                seen.add(uid)
                row, _ = self._encode(record)
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid += 1
                continue
            if row is None:
                overlong += 1
                continue
            partial += int(len(row["features"]) < len(record["actions"]) + 1)
            if self.ready:
                prediction = self._predict(self.deployed_head, row)
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError("nonfinite deployed semantic probabilities")
                values[uid] = [{"sem": float(p), "terminal": bool(row["prefix_terminal"][i])}
                               for i, p in enumerate(prediction)]
            # Text, never caller-supplied ID or outcome, determines held-out split.
            if record.get("train_eligible", True):
                question_key = hashlib.sha256(record["question"].strip().encode("utf-8")).hexdigest()
                self._pending[uid] = {**row, "traj_uid": uid, "question_key": question_key,
                                      "label": float(record["label"])}
                self._pending.move_to_end(uid)
                while len(self._pending) > int(self.config["replay_max_trajectories"]):
                    self._pending.popitem(last=False)
            encoded += 1
        metrics = {"ready": float(self.ready), "version": self.version, "records": len(records),
                   "encoded_trajectories": encoded, "skipped_invalid": invalid,
                   "skipped_overlong": overlong, "partial_trajectories": partial,
                   "pending_trajectories": len(self._pending),
                   "coverage": encoded / len(records) if records else 0.,
                   "prepare_seconds": time.perf_counter() - started,
                   "frozen_parameters": sum(p.numel() for p in self.encoder.parameters()),
                   "head_parameters": sum(p.numel() for p in self.candidate_head.parameters())}
        # These permissions describe the deployed head used by this batch;
        # update() separately reports permissions for the next batch.
        metrics.update({"reliability_" + role: float(value) for role, value in self.reliability.items()})
        return {"ready": self.ready, "version": self.version, "values": values,
                "reliability": self.reliability.copy(), "metrics": metrics}

    def _train(self, rows, epochs):
        data = self._flatten(rows)
        seed = int(torch.randint(0, 2 ** 31, (1,), generator=self._rng).item())
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        loss_sum = count = 0.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            self.candidate_head.train().requires_grad_(True)
            for _ in range(int(epochs)):
                indices = torch.randperm(len(data["labels"]), generator=self._rng)
                for start in range(0, len(indices), int(self.config["train_batch_size"])):
                    ix = indices[start:start + int(self.config["train_batch_size"])]
                    logits = self.candidate_head(data["features"][ix].to(self.device).float())
                    loss = F.binary_cross_entropy_with_logits(logits, data["labels"][ix].to(self.device))
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("nonfinite semantic BCE loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.candidate_head.parameters(), 1.)
                    self.optimizer.step()
                    loss_sum += float(loss.detach()) * len(ix)
                    count += len(ix)
            self.candidate_head.eval()
        return loss_sum / max(count, 1.)

    def _evaluate(self, head, rows, prior):
        data = self._flatten(rows)
        scores, labels = self._predict(head, data).double(), data["labels"].double()
        roles = data["roles"]
        base = ~data["prefix_terminal"].bool() & data["prefix_valid"].bool()
        control = base & torch.tensor([role in ROLES for role in roles])
        result = {}
        for name, mask in [("control", control)] + [(role, base & torch.tensor([r == role for r in roles])) for role in ROLES]:
            p, y = scores[mask], labels[mask]
            result[name + "_prefixes"] = len(y)
            result[name + "_positive"] = int((y == 1).sum())
            result[name + "_negative"] = int((y == 0).sum())
            result[name + "_brier"] = float((p - y).square().mean()) if len(y) else float("nan")
            result[name + "_prior_brier"] = float((y - prior).square().mean()) if len(y) else float("nan")
            result[name + "_auc"] = _weighted_auc(p, y, torch.ones_like(y))
            ece = 0.
            for i in range(10):
                bucket = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
                if bool(bucket.any()):
                    ece += float(bucket.sum()) * float((p[bucket].mean() - y[bucket].mean()).abs())
            result[name + "_ece"] = ece / len(y) if len(y) else float("nan")
        result["all_prefix_brier"] = float((scores - labels).square().mean())
        return result

    def _quality(self, metrics, scope):
        return (metrics[scope + "_prefixes"] >= int(self.config["min_role_val_prefixes"])
                and min(metrics[scope + "_positive"], metrics[scope + "_negative"]) >= int(self.config["min_val_per_class"])
                and math.isfinite(metrics[scope + "_auc"])
                and metrics[scope + "_auc"] >= float(self.config["min_val_auc"])
                and metrics[scope + "_brier"] < metrics[scope + "_prior_brier"] - float(self.config["min_brier_improvement"])
                and metrics[scope + "_ece"] <= float(self.config["max_ece"]))

    def _ingest(self):
        for uid, row in self._pending.items():
            self._replay[uid] = row
            self._replay.move_to_end(uid)
        self._pending.clear()
        while len(self._replay) > int(self.config["replay_max_trajectories"]):
            self._replay.popitem(last=False)
        train, val = [], []
        for row in self._replay.values():
            (val if self._is_validation(row["question_key"]) else train).append(row)
        return train, val

    def _enough(self, train, val):
        return (len(train) >= int(self.config["min_train_trajectories"])
                and len(val) >= int(self.config["min_val_trajectories"])
                and len({r["question_key"] for r in train}) >= int(self.config["min_train_questions"])
                and len({r["question_key"] for r in val}) >= int(self.config["min_val_questions"])
                and min(sum(r["label"] == 1 for r in val), sum(r["label"] == 0 for r in val)) >= int(self.config["min_val_per_class"])
                and len({r["label"] for r in train}) == 2)

    def _validate_deploy(self, train, val, metrics):
        prior = float(self._flatten(train)["labels"].mean())
        candidate = self._evaluate(self.candidate_head, val, prior)
        metrics.update({"candidate_" + key: value for key, value in candidate.items()})
        permissions = {role: float(self._quality(candidate, "control") and self._quality(candidate, role)) for role in ROLES}
        deployed = self._evaluate(self.deployed_head, val, prior) if self.ready else None
        # Shared semantic weights can be replaced only when already-qualified
        # roles remain qualified and no role's Brier regresses beyond tolerance.
        preserve = deployed is None or all(
            not self.reliability[role] or (permissions[role] and candidate[role + "_brier"] <=
                deployed[role + "_brier"] + float(self.config["candidate_brier_tolerance"])) for role in ROLES)
        if any(permissions.values()) and preserve:
            self.deployed_head.load_state_dict(self.candidate_head.state_dict())
            self.deployed_head.eval().requires_grad_(False)
            self.reliability = permissions
            self.ready = True
            self.version += 1
            self.role_bad_windows = dict.fromkeys(ROLES, 0)
            metrics["deployed"] = 1.
        elif deployed is not None and self.config["disable_miscalibrated"]:
            # Re-evaluating unchanged replay is not new evidence of drift.
            fingerprint = hashlib.sha256(repr(sorted((r["traj_uid"], r["question_key"], r["label"]) for r in val)).encode()).hexdigest()
            if fingerprint != self.last_validation_fingerprint:
                for role in ROLES:
                    if self.reliability[role]:
                        good = self._quality(deployed, "control") and self._quality(deployed, role)
                        self.role_bad_windows[role] = 0 if good else self.role_bad_windows[role] + 1
                        if self.role_bad_windows[role] >= int(self.config["miscalibration_patience"]):
                            self.reliability[role] = 0.
                            metrics["disabled"] = 1.
            self.last_validation_fingerprint = fingerprint
            self.ready = any(self.reliability.values())
        if deployed is not None:
            metrics.update({"deployed_" + key: value for key, value in deployed.items()})

    def update(self):
        started = time.perf_counter()
        self.step += 1
        train, val = self._ingest()
        metrics = {"train_trajectories": len(train), "val_trajectories": len(val), "deployed": 0., "disabled": 0.}
        if train:
            metrics["train_bce"] = self._train(train, self.config["train_epochs"])
        enough = self._enough(train, val)
        metrics["validation_sufficient"] = float(enough)
        if enough:
            self._validate_deploy(train, val, metrics)
        elif self.config["disable_ready_when_insufficient"]:
            metrics["disabled"] = float(self.ready)
            self.ready = False
            self.reliability = dict.fromkeys(ROLES, 0.)
        metrics.update(ready=float(self.ready), version=self.version, update_seconds=time.perf_counter() - started)
        metrics.update({"reliability_" + role: value for role, value in self.reliability.items()})
        self.last_metrics = metrics
        return metrics

    def pretrain(self, rounds=1, epochs=None):
        if int(rounds) < 1 or (epochs is not None and int(epochs) < 1):
            raise ValueError("pretrain rounds and epochs must be positive")
        original = self.config["train_epochs"]
        if epochs is not None:
            self.config["train_epochs"] = int(epochs)
        try:
            for _ in range(int(rounds)):
                metrics = self.update()
            self.pretrained = bool(self.ready)
            return metrics
        finally:
            self.config["train_epochs"] = original

    def save(self, path):
        target = Path(path)
        if target.is_dir():
            raise ValueError("semantic checkpoint path must be a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {"checkpoint_version": _CHECKPOINT_VERSION, "schema": FEATURE_SCHEMA,
                 "config": self.config, "encoder_identity": self.encoder_identity,
                 "candidate_head": self.candidate_head.state_dict(), "deployed_head": self.deployed_head.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "ready": self.ready, "pretrained": self.pretrained,
                 "warm_started": self.warm_started, "version": self.version, "step": self.step,
                 "reliability": self.reliability, "role_bad_windows": self.role_bad_windows,
                 "last_validation_fingerprint": self.last_validation_fingerprint, "last_metrics": self.last_metrics,
                 "rng_state": self._rng.get_state(), "replay": list(self._replay.items()), "pending": list(self._pending.items())}
        temporary = target.with_name(target.name + ".tmp")
        torch.save(_cpu_copy(state), temporary)
        os.replace(temporary, target)

    def load(self, path, *, resume=False, warm_start=False):
        """Trusted local checkpoint; legacy mix import requires explicit warm start.

        Warm start extracts candidate semantic weights (the old deployed head
        may never have qualified), resets optimizers/replay/permissions, and
        requires new semantic-only validation before use.
        """
        if resume and warm_start:
            raise ValueError("resume and warm_start are mutually exclusive")
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        semantic = state.get("schema") == FEATURE_SCHEMA and state.get("checkpoint_version") == _CHECKPOINT_VERSION
        legacy = state.get("schema") == "causal-role-mean-top16-v2" and state.get("checkpoint_version") == 2
        if not semantic and not (warm_start and legacy):
            raise ValueError("semantic checkpoint required; old mix needs explicit warm_start")
        saved_identity = state.get("encoder_identity", {})
        if any(saved_identity.get(key) != value for key, value in self.encoder_identity.items()):
            raise ValueError("checkpoint encoder identity/serialization does not match")
        if semantic and saved_identity != self.encoder_identity:
            raise ValueError("checkpoint encoder identity contains incompatible fields")
        strict = {"head_hidden_dim", "max_length", "dropout", "torch_dtype", "attn_implementation"}
        if not warm_start:
            strict |= {"holdout_salt", "validation_fraction", "min_val_auc", "min_brier_improvement", "max_ece",
                       "min_role_val_prefixes", "min_val_per_class", "min_val_questions", "min_val_trajectories"}
        if resume:
            strict |= set(_DEFAULTS) - {"device", "cpu_num_threads"}
        for key in strict:
            if state["config"].get(key, _DEFAULTS.get(key)) != self.config[key]:
                raise ValueError(f"checkpoint configuration mismatch: {key}")
        if warm_start:
            weights = {key: value for key, value in state["candidate_head"].items() if key.startswith("semantic.")}
            self.candidate_head.load_state_dict(weights, strict=True)
            self._reset_training_state()
            self.warm_started = True
        else:
            permissions = state.get("reliability", {})
            if set(permissions) != set(ROLES) or any(value not in (0., 1.) for value in permissions.values()):
                raise ValueError("checkpoint role permissions are invalid")
            if bool(state["ready"]) != any(permissions.values()):
                raise ValueError("checkpoint readiness disagrees with role permissions")
            if not resume and (not state["ready"] or not state["pretrained"]):
                raise ValueError("qualified initialization requires a pretrained semantic-qualified checkpoint")
            self._reset_training_state()
            self.deployed_head.load_state_dict(state["deployed_head"], strict=True)
            self.candidate_head.load_state_dict(state["candidate_head"] if resume else state["deployed_head"], strict=True)
            self.ready, self.pretrained = bool(state["ready"]), bool(state["pretrained"])
            self.warm_started = bool(state.get("warm_started", False))
            self.version = int(state["version"])
            self.reliability = dict(permissions)
            if resume:
                self.optimizer.load_state_dict(state["optimizer"])
                for row in self.optimizer.state.values():
                    for key, value in row.items():
                        if torch.is_tensor(value):
                            row[key] = value.to(self.device)
                self._replay, self._pending = OrderedDict(state["replay"]), OrderedDict(state["pending"])
                self._rng.set_state(state["rng_state"])
                self.step = int(state["step"])
                self.role_bad_windows = dict(state["role_bad_windows"])
                self.last_validation_fingerprint = state["last_validation_fingerprint"]
                self.last_metrics = copy.deepcopy(state["last_metrics"])
        self.encoder.eval().requires_grad_(False)
        self.deployed_head.eval().requires_grad_(False)
        self.candidate_head.eval().requires_grad_(True)
        return {"ready": self.ready, "pretrained": self.pretrained, "version": self.version,
                "warm_started": self.warm_started, "reliability": self.reliability.copy()}

    def _load_backbone(self):
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        kwargs = {"trust_remote_code": bool(self.config["trust_remote_code"])}
        if self.config.get("revision") is not None:
            kwargs["revision"] = self.config["revision"]
        model_config = AutoConfig.from_pretrained(self.config["model_path"], **kwargs)
        _validate_backbone_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], **kwargs)
        model_type = model_config.model_type
        if model_type in _QWEN35_BACKBONES:
            class_name = _QWEN35_BACKBONES[model_type][1]
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                raise RuntimeError(f"{class_name} requires a Transformers release with Qwen3.5 support")
            # Load the entire official checkpoint with its original key layout,
            # check all loading diagnostics, then retain only the causal text
            # backbone. Never create an uninitialized TextModel from config or
            # reinterpret multimodal model.* keys as text-only model.* keys.
            complete, loading_info = model_class.from_pretrained(
                self.config["model_path"], config=model_config,
                dtype=getattr(torch, self.config["torch_dtype"]),
                attn_implementation=self.config["attn_implementation"],
                output_loading_info=True, **kwargs,
            )
            _check_complete_backbone_load(loading_info)
            encoder = _extract_qwen35_language_model(complete, model_config)
            from verl.models.transformers.qwen3_5 import install_qwen3_5_padding_fix
            install_qwen3_5_padding_fix(model_config)
            self._source_backbone_identity = {
                "source_model_type": model_type,
                "source_architecture": class_name,
                "source_commit_hash": getattr(model_config, "_commit_hash", None),
            }
            # The vision tower and LM head are released with the wrapper; no
            # vision weights are moved to the dedicated semantic GPU.
            del complete
            return encoder, tokenizer
        if model_type != "qwen3":
            raise ValueError("Use a complete official Qwen3.5 dense/MoE checkpoint for semantic initialization")
        encoder = AutoModel.from_pretrained(
            self.config["model_path"],
            torch_dtype=getattr(torch, self.config["torch_dtype"]),
            attn_implementation=self.config["attn_implementation"],
            **kwargs,
        )
        return encoder, tokenizer

    def _is_validation(self, question_key: str) -> bool:
        payload = (str(self.config["holdout_salt"]) + "\0" + str(question_key)).encode("utf-8")
        fraction = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (1 << 64)
        return fraction < float(self.config["validation_fraction"])

    @staticmethod
    def _state_features(solver_count: int, verifier_count: int, budget: int,
                        next_role: str, last_valid: bool) -> list[float]:
        remaining = max(0, budget - solver_count)
        return [
            math.log1p(solver_count), math.log1p(verifier_count),
            math.log1p(remaining), math.log1p(budget),
            float(next_role == "solver"), float(next_role == "verifier"),
            float(next_role == "terminal"), float(last_valid),
        ]

    def _segments_and_states(self, record: dict):
        question = str(record["question"])
        budget = int(record["max_solver_turns"])
        if budget < 1:
            raise ValueError("max_solver_turns must be positive")
        actions = record["actions"]
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")
        segments = ["[PROBLEM]\n" + question + "\n[INTERACTION]\n"]
        states = [self._state_features(0, 0, budget, "solver", True)]
        solver_count = verifier_count = 0
        for action in actions:
            role = str(action["role"]).lower()
            text = str(action["text"])
            valid = bool(action.get("valid", True))
            if role == "solver":
                solver_count += 1
                next_role = "terminal" if solver_count >= budget else "verifier"
            elif role == "verifier":
                verifier_count += 1
                verdict = text
                # Match the math controller: approve takes priority, unrecognized
                # verdicts stop, and only reject asks for another solver action.
                if "<verify>approve</verify>" in verdict:
                    next_role = "terminal"
                elif "<verify>reject</verify>" in verdict and solver_count < budget:
                    next_role = "solver"
                else:
                    next_role = "terminal"
            else:
                raise ValueError(f"unknown action role: {role!r}")
            segments.append(f"\n[{role.upper()}]\n{text}\n[END_ACTION]\n")
            states.append(self._state_features(solver_count, verifier_count, budget, next_role, valid))
        return segments, states

    def _encode(self, record):
        try:
            segments, states = self._segments_and_states(record)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise _InvalidRecord(str(error)) from error
        token_ids, boundaries = [], []
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None:
            token_ids.append(int(bos))
        max_length = int(self.config["max_length"])
        total = len(token_ids)
        for segment_index, segment in enumerate(segments):
            ids = self.tokenizer.encode(segment, add_special_tokens=False)
            if not ids:
                raise _InvalidRecord("empty tokenized prefix segment")
            total += len(ids)
            # Never score the interior of a truncated action. Earlier complete
            # boundaries still receive labels/predictions and keep their state.
            if len(token_ids) + len(ids) <= max_length and len(boundaries) == segment_index:
                token_ids.extend(ids)
                boundaries.append(len(token_ids) - 1)
        count = len(boundaries)
        if not count:
            return None, total
        self.encoder.eval()
        with torch.inference_mode():
            ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            output = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            semantic = output.last_hidden_state[0, boundaries].float().cpu()
        features = torch.cat((semantic.clone(), torch.tensor(states[:count], dtype=torch.float32)), -1).float().contiguous()
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("nonfinite encoder features")
        return {"features": features,
                "prefix_roles": ["initial"] + [a["role"].lower() for a in record["actions"][:count - 1]],
                "prefix_terminal": torch.tensor([bool(s[6]) for s in states[:count]]),
                "prefix_valid": torch.tensor([True] + [bool(a.get("valid", True)) and not a.get("truncated", False) for a in record["actions"][:count - 1]])}, total


class SemanticValueWorker:
    """Small synchronous Ray actor bridge without importing Ray into the scorer.

    The coordinator wraps this class with ``ray.remote`` and reserves one GPU
    with ``.options(num_gpus=1, scheduling_strategy=...)`` for CUDA operation.
    Ray remaps that dedicated GPU to ``cuda:0`` inside this process. Calling
    ``ready.remote()`` after construction provides a startup barrier without
    touching the replay buffer or training step. Ordinary Ray actors serialize
    method calls, preserving prepare -> both policy updates -> update ordering
    when the coordinator waits at these boundaries.
    """

    def __init__(self, config):
        config = dict(config)
        device = str(config.get("device", "cpu"))
        if device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("dedicated semantic Ray worker device must be cpu or cuda:0")
        if device == "cuda":
            config["device"] = "cuda:0"
        self.scorer = PrefixValueScorer(config)

    def ready(self):
        return {"initialized": True, "ready": bool(self.scorer.ready),
                "version": int(self.scorer.version), "device": str(self.scorer.device),
                "schema": FEATURE_SCHEMA}

    def prepare(self, records):
        return self.scorer.prepare(records)

    def update(self):
        return self.scorer.update()

    def save(self, path):
        self.scorer.save(path)
        return {"path": str(path), "ready": bool(self.scorer.ready), "version": int(self.scorer.version)}

    def load(self, path, resume=False, warm_start=False):
        return self.scorer.load(path, resume=resume, warm_start=warm_start)

