#!/usr/bin/env python3
"""Pretrain semantic prefix success probabilities from complete historical Math runs.

Accept canonical trajectory records (question, max_solver_turns, actions, label,
traj_uid), canonical action rows, columnar non_tensor_batch, or trainer exports
with trajectories/steps and the complete original value_question or env question.
Entropy statistics are neither required nor read. An unqualified checkpoint is
saved for explicit candidate warm start and causes exit status 2.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path):
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".json":
        config = json.loads(text)
    else:
        import yaml
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("configuration must contain a mapping")
    if "algorithm" in config:
        config = config["algorithm"]["semantic_value"]
    elif "semantic_value" in config:
        config = config["semantic_value"]
    return dict(config)


def _documents(path):
    text = path.read_text(encoding="utf-8-sig")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        documents = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        documents = document if isinstance(document, list) else [document]
    if any(not isinstance(document, dict) for document in documents):
        raise ValueError(f"{path}: expected JSON objects")
    return documents


def _canonical_rows(document, budget=None):
    if "non_tensor_batch" in document:
        document = document["non_tensor_batch"]
    if isinstance(document.get("traj_uid"), list):
        size = len(document["traj_uid"])
        if any(not isinstance(values, list) or len(values) != size for values in document.values()):
            raise ValueError("columnar action fields must have equal lengths")
        return [{key: values[i] for key, values in document.items()} for i in range(size)]
    if "trajectories" in document:
        rows = []
        for trajectory in document["trajectories"]:
            question = (trajectory.get("value_question") or (trajectory.get("env_kwargs") or {}).get("question")
                        or trajectory.get("anchor_obs"))
            actions = trajectory.get("steps", [])
            if not actions:
                raise ValueError("historical trajectory has no action steps")
            indices = [action.get("step_idx") for action in actions]
            if any(not isinstance(index, int) or index < 0 for index in indices) or sorted(set(indices)) != list(range(len(set(indices)))):
                raise ValueError("historical steps need contiguous explicit step_idx identities")
            actual_budget = trajectory.get("value_max_solver_turns", trajectory.get("max_solver_turns",
                            document.get("value_max_solver_turns", document.get("max_solver_turns", budget))))
            for action in sorted(actions, key=lambda row: row["step_idx"]):
                index = action.get("value_action_index", action["step_idx"])
                rows.append({"traj_uid": trajectory["traj_uid"], "uid": trajectory.get("uid", question),
                             "value_question": question, "value_max_solver_turns": actual_budget,
                             "value_action_text": action.get("value_action_text", action.get("response")),
                             "value_action_index": index, "role_turn_index": action.get("role_turn_index", index // 2),
                             "agent_id": action["agent_id"], "is_action_valid": action.get("is_action_valid", True),
                             "pass": trajectory["pass"], "value_action_truncated": action.get("truncated", False),
                             "step": document.get("global_step", document.get("step", 0))})
        return rows
    if isinstance(document.get("actions"), list):
        return [{"traj_uid": document["traj_uid"], "uid": document.get("uid", document["question"]),
                 "value_question": document["question"], "value_max_solver_turns": document.get("max_solver_turns", budget),
                 "value_action_index": i, "value_action_text": action["text"], "agent_id": action["role"],
                 "role_turn_index": action.get("role_turn_index", i // 2), "is_action_valid": action.get("valid", True),
                 "pass": document["label"], "value_action_truncated": action.get("truncated", False)}
                for i, action in enumerate(document["actions"])]
    return [document]


def load_records(patterns, max_solver_turns=None):
    utility = _module("semantic_offline_metadata", "verl/utils/semantic_credit.py")
    files, rows = [], []
    for pattern in patterns:
        path = Path(pattern)
        matches = sorted(path.rglob("*.json*")) if path.is_dir() else sorted(Path(p) for p in glob.glob(pattern, recursive=True))
        if not matches:
            raise ValueError(f"no input files match {pattern}")
        for path in matches:
            path = path.resolve()
            if path in files:
                raise ValueError(f"input file selected more than once: {path}")
            files.append(path)
            for document in _documents(path):
                for original in _canonical_rows(document, max_solver_turns):
                    row = dict(original)
                    row.setdefault("value_max_solver_turns", max_solver_turns)
                    # Multiple historical files may reuse rollout identifiers.
                    namespace = hashlib.sha256((str(path) + "\0" + str(row.get("step", 0))).encode()).hexdigest()[:20]
                    row["traj_uid"] = namespace + ":" + str(row["traj_uid"])
                    rows.append(row)
    required = utility._RECORD_FIELDS + ("value_max_solver_turns",)
    for row in rows:
        if any(row.get(key) is None for key in required):
            raise ValueError("complete original question, real historical budget and canonical action metadata are required")
    columns = {key: [row[key] for row in rows] for key in required}
    columns["value_action_truncated"] = [row.get("value_action_truncated", False) for row in rows]
    records, metrics = utility.build_trajectory_records(columns, max_solver_turns)
    if not records or metrics["value_credit/skipped_trajectories"]:
        raise ValueError(f"historical trajectories are incomplete/conflicting: {metrics}")
    return records, {"files": [str(path) for path in files], "trajectories": len(records),
                     "unique_questions": len({record["question_key"] for record in records}), "reconstruction": metrics}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", action="extend", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--model-path", "--encoder-model", dest="model_path")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-solver-turns", type=int)
    parser.add_argument("--warm-start", help="explicitly initialize from candidate semantic weights, including old mix v2")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.rounds, args.batch_size or 1, args.max_solver_turns or 1) < 1:
        parser.error("epochs, rounds, batch size and historical budget must be positive")
    try:
        records, provenance = load_records(args.input, args.max_solver_turns)
        if args.validate_only:
            print(json.dumps(provenance, ensure_ascii=False), flush=True)
            return 0
        config = load_config(args.config)
        for key in ("model_path", "device", "seed"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.batch_size:
            config["train_batch_size"] = args.batch_size
        config["replay_max_trajectories"] = max(len(records), int(config.get("replay_max_trajectories", 2048)))
        worker = _module("offline_semantic_worker", "verl/workers/semantic_value.py")
        scorer = worker.PrefixValueScorer(config)
        if args.warm_start:
            scorer.load(args.warm_start, warm_start=True)
        prepared = scorer.prepare(records)
        if prepared["metrics"]["skipped_invalid"]:
            raise ValueError("semantic encoder rejected historical records")
        metrics = scorer.pretrain(rounds=args.rounds, epochs=args.epochs)
        scorer.save(args.output)
        report = _json_safe({"ready": scorer.ready, "preparation": prepared["metrics"], "training": metrics,
                             "provenance": provenance, "schema": worker.FEATURE_SCHEMA})
        Path(args.output + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if scorer.ready else 2
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
