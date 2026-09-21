"""Math answer LOTO boundary values for the fixed three-Solver-loop profile.

This module intentionally imports neither torch nor verl.  Its float64 (Python
double) reference path is shared by the trainer and offline replay.  It estimates
*environment* terminal-success values from peer answers, never a learned critic,
and never reads the current trajectory's own terminal reward, gold answer, or
any post-action information of the event being valued.

Contract: CLAUDE_CODE_MATH_STAGE1_IMPLEMENTATION_AND_TRAINING_GUIDE.md (v2) §4/§5.
The answer parser is the exact literal default path (``numeric=False``,
``missing_state=False``) of the audited reference ``hardened_value.py``; the
optional historical-audit parser options are deliberately not exposed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any, Mapping, Optional

MATH_VALUE_PARSER_VERSION = "latest_box_token_v1"
MATH_KINDS = frozenset(("math_solution", "math_verifier"))
MATH_SOLVER_EVENT_TYPE = "math_solution"
MATH_VERIFIER_EVENT_TYPE = "math_verifier"


# ---------------------------------------------------------------------------
# Answer parsing (exact port of the audited literal path)
# ---------------------------------------------------------------------------

def _escaped(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and text[j] == "\\":
        j -= 1
    return (i - 1 - j) % 2 == 1


def _closing_brace(text: str, start: int) -> Optional[int]:
    depth = 0
    for i in range(start, len(text)):
        if _escaped(text, i):
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_latest_box(text: str) -> dict[str, Any]:
    """Find the last complete ``\\boxed``/``\\fbox`` command; never fall back."""

    if not isinstance(text, str):
        raise ValueError("answer source must be text")
    matches = list(re.finditer(r"\\(?:boxed|fbox)(?![A-Za-z])", text))
    if not matches:
        return {"status": "no_box", "raw": "", "commands": 0}
    m = matches[-1]
    start = m.end()
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text) or text[start] != "{":
        return {"status": "missing_open_brace", "raw": "", "commands": len(matches)}
    end = _closing_brace(text, start)
    if end is None:
        return {"status": "unclosed_last_box", "raw": "", "commands": len(matches)}
    inner = text[start + 1:end]
    return {"status": "ok" if inner.strip() else "empty_box", "raw": inner, "commands": len(matches)}


def canonical_answer_key(raw: str) -> str:
    """Token-aware literal key; preserve minus signs, case, units, boundaries.

    Only explicit display commands are removed and ``dfrac``/``tfrac`` are
    normalized to ``frac``.  No eval/sympy/LLM grading, no unit/sign stripping,
    and no merging of ``\\sin x`` into ``\\sinx``.
    """

    tok = []
    i = 0
    while i < len(raw):
        if raw[i].isspace():
            i += 1
            continue
        m = re.match(r"\\[A-Za-z]+|\\[^A-Za-z]|\d+(?:\.\d*)?|\.\d+|[^\s]", raw[i:])
        if m is None:
            return ""
        x = m.group()
        i += len(x)
        if x in ("\\left", "\\right", "\\displaystyle", "\\quad", "\\qquad", "\\,", "\\;", "\\!"):
            continue
        if x in ("\\dfrac", "\\tfrac"):
            x = "\\frac"
        if x == "\\text":
            while i < len(raw) and raw[i].isspace():
                i += 1
            if i >= len(raw) or raw[i] != "{":
                return ""
            end = _closing_brace(raw, i)
            if end is None:
                return ""
            tok.append(("text", " ".join(raw[i + 1:end].split())))
            i = end + 1
        else:
            tok.append(("token", x))
    if not tok:
        return ""
    return "literal:" + json.dumps(tok, ensure_ascii=True, separators=(",", ":"))


def parse_answer_feature(text: str) -> dict[str, Any]:
    """Latest-box strict parse of one executed Solver response."""

    box = extract_latest_box(text)
    key = canonical_answer_key(box["raw"]) if box["status"] == "ok" else ""
    return {"key": key, "status": box["status"], "raw": box["raw"], "commands": box["commands"]}


class BoundedAnswerCache:
    """Bounded parse cache keyed by exact text; never caches rewards or values."""

    def __init__(self, size: int = 8192) -> None:
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.size = int(size)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def feature(self, text: str) -> dict[str, Any]:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        feature = parse_answer_feature(text)
        self._cache[text] = feature
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return feature


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MathValueConfig:
    """Immutable, fail-closed Math answer-LOTO configuration."""

    mode: str = "off"
    target: str = "terminal_env_success"
    expected_rollout_n: int = 8
    expected_max_loop_num: int = 3
    parser_version: str = MATH_VALUE_PARSER_VERSION
    feature_source: str = "executed_action_text"
    state_view: str = "team_history"
    matching: str = "same_round_then_cross"
    cross_round_decay: float = 0.5
    question_parent_strength: float = 2.0
    beta: float = 1.0
    verifier_post: str = "carry"
    auxiliary_mode: str = "env_only"
    strict: bool = True
    cache_size: int = 8192
    record_matched_peers: bool = False

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]] = None) -> "MathValueConfig":
        if isinstance(mapping, cls):
            return mapping
        values = dict(mapping or {})
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown team_event_gae.math_value configuration keys: {sorted(unknown)}")
        return cls(**values)

    def __post_init__(self) -> None:
        if self.mode not in ("off", "math_answer_loto"):
            raise ValueError(f"Unsupported math_value.mode={self.mode!r}")
        if self.target != "terminal_env_success":
            raise ValueError("math answer LOTO supports only target=terminal_env_success")
        if self.parser_version != MATH_VALUE_PARSER_VERSION:
            raise ValueError(f"Unsupported parser_version={self.parser_version!r}")
        if self.feature_source != "executed_action_text":
            raise ValueError("math answer LOTO uses executed_action_text only")
        if self.state_view != "team_history":
            raise ValueError("math answer LOTO uses the centralized team-history state view")
        if self.matching != "same_round_then_cross":
            raise ValueError("math answer LOTO supports only same_round_then_cross matching")
        if self.verifier_post != "carry":
            raise ValueError("Only the explicit nonterminal Verifier carry is supported")
        if self.auxiliary_mode not in ("env_only", "keep_existing_local"):
            raise ValueError(f"Unknown auxiliary_mode={self.auxiliary_mode!r}")
        for name in ("expected_rollout_n", "expected_max_loop_num", "cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.expected_rollout_n < 2:
            raise ValueError("Require expected_rollout_n>=2 for leave-one-trajectory-out values")
        if self.expected_max_loop_num != 3:
            raise ValueError("The audited Math profile is max_loop_num=3 (S0,V0,S1,V1,S2)")
        for name in ("cross_round_decay", "question_parent_strength", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.cross_round_decay <= 1.0:
            raise ValueError("cross_round_decay must be in [0,1]")
        if self.question_parent_strength < 0:
            raise ValueError("question_parent_strength must be >= 0")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")
        for name in ("strict", "record_matched_peers"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean, not a string or number")
        if not self.strict:
            raise ValueError("Non-strict math answer LOTO is not implemented; incomplete chains must fail")

    def validate_runtime(self, gamma: float, internal_gamma: float, agent_local_mode: str = "off") -> None:
        if self.mode == "off":
            return
        if gamma != 1.0 or internal_gamma != 1.0:
            raise ValueError(
                "math_answer_loto requires gamma=internal_gamma=1; "
                "discounted peer targets are not implemented"
            )
        if agent_local_mode != "off":
            raise ValueError("math_answer_loto requires agent_local.mode=off; G1/SMDP is not part of this profile")


# ---------------------------------------------------------------------------
# Chain validation
# ---------------------------------------------------------------------------

def _integer(record: Mapping[str, Any], name: str) -> int:
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, Integral):
        if isinstance(value, Real) and float(value).is_integer():
            return int(value)
        raise ValueError(f"{record.get('event_uid')}: {name} must be an integer")
    if value < 0:
        raise ValueError(f"{record.get('event_uid')}: {name} must be nonnegative")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if value is None:
        return False
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        value = value.item()
    if not isinstance(value, (bool, Integral)) or value not in (0, 1):
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(value)


def allowed_math_chains(max_loop: int) -> tuple[tuple[str, ...], ...]:
    """Legal deduplicated chains for L solver rounds (early stop after V, or full).

    The orchestra runs Verifier after every Solver round except the last loop,
    so a chain ending in a bare Solver is legal only at the full horizon, and
    a Verifier can never follow the final Solver round.
    """

    chains = []
    for solver_rounds in range(1, max_loop + 1):
        stopped_after_verifier = solver_rounds < max_loop
        chain = []
        for k in range(solver_rounds):
            chain.append(MATH_SOLVER_EVENT_TYPE)
            if k < solver_rounds - 1 or stopped_after_verifier:
                chain.append(MATH_VERIFIER_EVENT_TYPE)
        chains.append(tuple(chain))
    return tuple(chains)


def _validated_chains(events: Mapping[str, Mapping[str, Any]],
                      trajectories: Mapping[str, list[str]],
                      expected_n: int,
                      max_loop: int):
    if not events or not trajectories:
        raise ValueError("math_answer_loto requires nonempty complete rollout groups")
    required = {"event_uid", "uid", "traj_uid", "event_index", "event_count", "role_event_index",
                "role_event_count", "agent_id", "event_type", "env_step_index", "is_env_action",
                "env_action_owner", "env_reward", "env_done", "is_action_valid", "executed_action_text"}
    normalized = {}
    producer_types: dict[str, str] = {}
    type_producers: dict[str, str] = {}
    for key, source in events.items():
        missing = required - source.keys()
        if missing:
            raise ValueError(f"Event {key} missing required fields: {sorted(missing)}")
        event = dict(source)
        for name in ("event_uid", "uid", "traj_uid", "agent_id"):
            if not isinstance(event[name], str) or not event[name].strip():
                raise ValueError(f"Event {key}: {name} must be a nonempty string")
        if key != event["event_uid"]:
            raise ValueError(f"Event dictionary key {key!r} does not match event_uid")
        if event["event_type"] not in MATH_KINDS:
            raise ValueError(f"Event {key}: unsupported Math event_type {event['event_type']!r}")
        if "task_type" in event and str(event["task_type"]).lower() != "math":
            raise ValueError("math_answer_loto currently supports task_type=Math only")
        if not isinstance(event["executed_action_text"], str):
            raise ValueError(f"Event {key}: executed_action_text must be text")
        for name in ("event_index", "event_count", "role_event_index", "role_event_count", "env_step_index"):
            event[name] = _integer(event, name)
        for name in ("is_env_action", "env_done", "is_action_valid"):
            event[name] = _boolean(event[name], name)
        reward = event["env_reward"]
        if reward is None:
            reward = 0.0
        if not isinstance(reward, Real) or not math.isfinite(reward):
            raise ValueError(f"Event {key}: env_reward must be finite")
        event["env_reward"] = float(reward)
        owner = event["env_action_owner"]
        if owner is not None and (not isinstance(owner, str) or not owner.strip()):
            raise ValueError(f"Event {key}: env_action_owner must be a nonempty string or None")
        if event["is_env_action"] != (owner == event["agent_id"]):
            raise ValueError(f"Event {key}: ownership mismatch")
        if not event["is_env_action"] and (event["env_done"] or abs(event["env_reward"]) > 1e-12):
            raise ValueError(f"Event {key}: non-owner cannot carry reward or done")
        if not event["env_done"] and abs(event["env_reward"]) > 1e-12:
            raise ValueError(f"Event {key}: terminal_env_success requires zero nonterminal reward")
        if event["env_done"] and reward not in (0.0, 1.0):
            raise ValueError(f"Event {key}: terminal_env_success requires binary terminal env reward")
        role, kind = event["agent_id"], event["event_type"]
        if producer_types.setdefault(role, kind) != kind or type_producers.setdefault(kind, role) != role:
            raise ValueError("Math requires one consistent producer per solution/verifier event type")
        normalized[key] = event
    chains, groups, returns, seen = {}, defaultdict(list), {}, set()
    legal = set(allowed_math_chains(max_loop))
    for trajectory in sorted(trajectories):
        keys = list(trajectories[trajectory])
        if not keys or any(key not in normalized for key in keys):
            raise ValueError(f"Trajectory {trajectory}: empty chain or unknown event UID")
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError(f"Trajectory {trajectory}: duplicate event UID across chains")
        seen.update(keys)
        keys.sort(key=lambda key: normalized[key]["event_index"])
        records = [normalized[key] for key in keys]
        if any(event["traj_uid"] != trajectory for event in records):
            raise ValueError(f"Trajectory {trajectory}: mismatched traj_uid")
        group = records[0]["uid"]
        if any(event["uid"] != group for event in records):
            raise ValueError(f"Trajectory {trajectory} crosses question groups")
        if any(event["event_count"] != len(keys) for event in records) or [event["event_index"] for event in records] != list(range(len(keys))):
            raise ValueError(f"Trajectory {trajectory}: incomplete event_index/event_count chain")
        if any(event["env_done"] for event in records[:-1]) or not records[-1]["env_done"]:
            raise ValueError(f"Trajectory {trajectory}: events after done or missing true terminal boundary")
        if any(event["env_step_index"] != 0 for event in records):
            raise ValueError(f"Trajectory {trajectory}: Math executes exactly one env.step (env_step_index=0)")
        kinds = tuple(event["event_type"] for event in records)
        if kinds not in legal:
            raise ValueError(
                f"Trajectory {trajectory}: invalid Math event chain {list(kinds)} for max_loop={max_loop}"
            )
        if not records[-1]["is_env_action"]:
            raise ValueError(f"Trajectory {trajectory}: the true terminal event must own the env action")
        by_role = defaultdict(list)
        for event in records:
            by_role[event["agent_id"]].append(event)
        for role, role_events in by_role.items():
            if any(event["role_event_count"] != len(role_events) for event in role_events) or [event["role_event_index"] for event in role_events] != list(range(len(role_events))):
                raise ValueError(f"Trajectory {trajectory}/{role}: incomplete role indices/counts")
        chains[trajectory] = keys
        groups[group].append(trajectory)
        returns[trajectory] = sum(event["env_reward"] for event in records)
    if seen != set(normalized):
        raise ValueError("Event collection contains orphan events not present in trajectories")
    for group, members in groups.items():
        if len(members) != expected_n:
            raise ValueError(f"Question group {group} has {len(members)} trajectories; expected_rollout_n={expected_n}")
    return normalized, chains, groups, returns


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MathValueResult:
    pre: dict[str, float]
    post: dict[str, float]
    details: dict[str, dict[str, Any]]
    returns: dict[str, float]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_math_values(events: Mapping[str, Mapping[str, Any]],
                         trajectories: Mapping[str, list[str]],
                         config: MathValueConfig) -> MathValueResult:
    """Compute pre/post terminal-success values over complete same-question groups.

    Caller must deduplicate copied training rows *with conflict detection* first.
    Immutable IDs, chain completeness, ownership and the single binary terminal
    reward contract are re-checked here.  Input records are never mutated.

    Per guide §5: the question mean ``Q`` is an internal shrinkage prior (not a
    separate training mode); the same-round gate is whole-query (fallback to
    cross-round weights only when *no* peer matches the current round); each
    peer contributes at most one maximal weight; the terminal Solver S2 of a
    full chain is excluded from every non-terminal peer pool.
    """

    config = MathValueConfig.from_mapping(config)
    if config.mode != "math_answer_loto":
        raise ValueError("estimate_math_values is the math_answer_loto engine, not the legacy estimator")
    records, chains, groups, returns = _validated_chains(
        events, trajectories, config.expected_rollout_n, config.expected_max_loop_num
    )
    cache = BoundedAnswerCache(config.cache_size)

    # Non-terminal solver events per trajectory, keyed by solver round, plus the
    # parsed answer feature of each.  The terminal event of a full chain (S2)
    # never enters a peer pool; early-stop terminals are Verifiers, so every
    # Solver event of an early-stopped peer stays eligible.
    solver_rounds: dict[str, dict[int, dict[str, Any]]] = {}
    for trajectory, keys in chains.items():
        rounds = {}
        terminal_uid = keys[-1]
        for key in keys:
            event = records[key]
            if event["event_type"] != MATH_SOLVER_EVENT_TYPE:
                continue
            round_index = event["role_event_index"]
            if key == terminal_uid:
                # The terminal S2 is a fact of this trajectory only; it carries
                # no post value and is not comparable peer state.
                feature = cache.feature(event["executed_action_text"])
                rounds[round_index] = {"event_uid": key, "feature": feature, "terminal": True}
            else:
                feature = cache.feature(event["executed_action_text"])
                rounds[round_index] = {"event_uid": key, "feature": feature, "terminal": False}
        solver_rounds[trajectory] = rounds

    n = config.expected_rollout_n
    peers_per: dict[str, list[str]] = {}
    for members in groups.values():
        if len(members) != n:
            raise ValueError(f"Question group has {len(members)} trajectories; expected {n}")
        for own in members:
            peers_per[own] = [other for other in members if other != own]

    pre: dict[str, float] = {}
    post: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    for own, peers in peers_per.items():
        qparent = sum(returns[other] for other in peers) / len(peers)
        value = qparent
        for key in chains[own]:
            event = records[key]
            kind = event["event_type"]
            pre[key] = value
            detail: dict[str, Any] = {
                "question_parent": qparent,
                "parent": qparent,
                "same_round_peer_count": 0,
                "cross_round_peer_count": 0,
                "used_peer_count": 0,
                "kernel_mass": 0.0,
                "kernel_ess": 0.0,
                "effective_ess": 0.0,
                "max_effective_coefficient": 0.0,
                "answer_key_sha256": None,
                "parser_status": None,
                "role_event_index": event["role_event_index"],
            }
            if config.record_matched_peers:
                detail["matched_peers"] = []
            if event["env_done"]:
                value = 0.0
                detail["branch"] = "terminal"
            elif kind == MATH_VERIFIER_EVENT_TYPE:
                # Non-terminal Verifier carry: post == pre.
                detail["branch"] = "carry"
            else:
                k = event["role_event_index"]
                feature = solver_rounds[own][k]["feature"]
                detail["answer_key_sha256"] = _sha256(feature["key"]) if feature["key"] else None
                detail["parser_status"] = feature["status"]
                key_text = feature["key"]

                # Round parent: peers whose same-round Solver event exists and
                # is not terminal.  Existence only; never filtered by answer or
                # success.
                same_round = [
                    other for other in peers
                    if solver_rounds[other].get(k, {"terminal": True})["terminal"] is False
                ]
                m = len(same_round)
                parent = (sum(returns[other] for other in same_round)
                          + config.question_parent_strength * qparent) / (m + config.question_parent_strength)
                detail["parent"] = parent
                detail["same_round_peer_count"] = m

                same_flags: list[float] = []
                cross_best: list[float] = []
                for other in peers:
                    matches_same = False
                    best_cross = 0.0
                    for round_index, entry in solver_rounds[other].items():
                        if entry["terminal"]:
                            continue
                        peer_feature = entry["feature"]
                        if not key_text or peer_feature["key"] != key_text:
                            continue
                        if round_index == k:
                            matches_same = True
                        elif config.cross_round_decay ** abs(round_index - k) > best_cross:
                            best_cross = config.cross_round_decay ** abs(round_index - k)
                    same_flags.append(1.0 if matches_same else 0.0)
                    cross_best.append(best_cross)
                total_same = sum(same_flags)
                # Whole-query gating: cross weights only when no peer matches
                # the current round at all.
                used_cross = total_same <= 0
                weights = same_flags if total_same > 0 else cross_best
                mass = sum(weights)
                beta = config.beta
                denom = mass + beta
                # Effective label coefficients, including parent shrinkage.
                peer_count = len(peers)
                parent_coeff = [
                    ((1.0 if other in same_round else 0.0) + config.question_parent_strength / peer_count)
                    / (m + config.question_parent_strength)
                    for other in peers
                ]
                coefficients = [(w + beta * p) / denom for w, p in zip(weights, parent_coeff)]
                coefficient_sum = sum(coefficients)
                if abs(coefficient_sum - 1.0) > 1e-12 or any(c < -1e-12 or c > 1.0 + 1e-12 for c in coefficients):
                    raise ValueError(f"Invalid effective peer coefficients at {key}: sum={coefficient_sum}")
                value = sum(c * returns[other] for c, other in zip(coefficients, peers))
                detail["branch"] = "same" if total_same > 0 else ("cross" if mass > 0 else "parent")
                detail["kernel_mass"] = mass
                detail["used_peer_count"] = sum(1 for w in weights if w > 0)
                detail["cross_round_peer_count"] = (
                    sum(1 for w in cross_best if w > 0) if used_cross else 0
                )
                squares = [w * w for w in weights]
                detail["kernel_ess"] = mass * mass / sum(squares) if mass > 0 else 0.0
                detail["effective_ess"] = 1.0 / sum(c * c for c in coefficients)
                detail["max_effective_coefficient"] = max(coefficients)
                if config.record_matched_peers:
                    for other, w, s, c in zip(peers, weights, same_flags, cross_best):
                        if w > 0:
                            detail["matched_peers"].append({
                                "traj_uid": other,
                                "same_round": bool(s),
                                "weight": w,
                                "cross_weight": c,
                                "reward": returns[other],
                            })
            if not math.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
                raise ValueError(f"Invalid environment boundary value {value} at {key}")
            post[key] = float(value)
            details[key] = detail
    for keys in chains.values():
        previous_post = None
        for position, key in enumerate(keys):
            if previous_post is not None and pre[key] != previous_post:
                raise ValueError(f"Boundary identity violated at {key}: pre != previous post")
            previous_post = post[key]
        if chains and post[keys[-1]] != 0.0:
            raise ValueError(f"Boundary identity violated at {keys[-1]}: terminal post != 0")
    return MathValueResult(pre, post, details, returns)


__all__ = [
    "MATH_VALUE_PARSER_VERSION",
    "MathValueConfig",
    "MathValueResult",
    "allowed_math_chains",
    "BoundedAnswerCache",
    "canonical_answer_key",
    "estimate_math_values",
    "extract_latest_box",
    "parse_answer_feature",
]
