"""Frozen document kernels and whole-trajectory-out Search boundary values.

This module intentionally imports neither torch nor verl.  Its float64 (Python
double) reference path is shared by the trainer and offline replay.  It estimates
*environment* values, not shaped rewards, and never computes a learned critic.
See CLAUDE_CODE_STAGE1_BOUNDARY_LOTO_IMPLEMENTATION_GUIDE.md for the contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional


STOP_WORDS = frozenset(
    "a an the and or of to in on for with by from as is are was were be been being "
    "it its that this these those at into also which who whom whose have has had "
    "do does did not but than then so such their his her he she they them we you "
    "i our your can could would should may might will shall about over under "
    "between through during after before more most some any all each both other "
    "only up out if when where what how why there here no yes".split()
)
NORMALIZATION_VERSION = "search-docs-ascii-v1"
ASSET_SCHEMA_VERSION = 1
_KINDS = frozenset(("verifier", "search_query", "final_answer"))


def plain_docs(raw: Optional[str]) -> str:
    """Reproduce the historical information/JSON/escape parser exactly.

    None is a valid empty result. Structured new observations must be explicitly
    adapted upstream, rather than silently changing their serialization here.
    """
    if raw is not None and not isinstance(raw, str):
        raise ValueError("tool_observation must be a string or None; adapt structured results explicitly")
    text = raw or ""
    match = re.search(r"<information>(.*?)</information>", text, re.S)
    if match:
        text = match.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            text = str(obj.get("result", obj.get("results", text)))
    except (ValueError, TypeError):
        pass
    return text.replace("\\n", "\n").replace('\\"', '"')


def canonical(text: str) -> str:
    """Canonicalize an already unwrapped document; preserve order and repeats."""
    text = re.sub(r"Doc\s+\d+:", " ", text, flags=re.I).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def canonical_document_key(raw: Optional[str]) -> str:
    return canonical(plain_docs(raw))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass(frozen=True)
class BoundaryValueConfig:
    """Immutable, fail-closed configuration; an omitted section stays legacy."""

    mode: str = "legacy_loto"
    application: str = "active"
    target: str = "terminal_env_return"
    expected_rollout_n: int = 8
    similarity: str = "tfidf_cosine"
    similarity_power: float = 4.0
    depth_window: int = 1
    cross_depth_decay: float = 0.5
    parent_alpha: float = 1.0
    question_parent_beta: float = 2.0
    verifier_same_route: bool = True
    verifier_cross_depth_same_reason: bool = True
    verifier_same_depth_same_reason: bool = False
    nonterminal_answer: str = "carry"
    idf_path: Optional[str] = None
    idf_sha256: Optional[str] = None
    auxiliary_mode: str = "keep_existing_local"
    strict: bool = True
    cache_size: int = 8192
    record_matched_peers: bool = False
    # Explicitly for the historical same_round offline reference. Production
    # boundary_loto uses question fallback, including when beta is set to zero.
    empty_parent_fallback: str = "question"

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]] = None) -> "BoundaryValueConfig":
        if isinstance(mapping, cls):
            return mapping
        values = dict(mapping or {})
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown team_event_gae.value configuration keys: {sorted(unknown)}")
        return cls(**values)

    def __post_init__(self) -> None:
        if self.mode not in ("legacy_loto", "boundary_loto"):
            raise ValueError(f"Unsupported value.mode={self.mode!r}")
        if self.application not in ("active", "shadow"):
            raise ValueError(f"Unsupported value.application={self.application!r}")
        if self.target != "terminal_env_return" or self.similarity != "tfidf_cosine":
            raise ValueError("boundary LOTO supports only terminal_env_return with tfidf_cosine")
        if self.nonterminal_answer != "carry":
            raise ValueError("Only the explicit nonterminal Answer carry approximation is supported")
        if self.auxiliary_mode not in ("env_only", "keep_existing_local"):
            raise ValueError(f"Unknown auxiliary_mode={self.auxiliary_mode!r}")
        if self.empty_parent_fallback not in ("question", "carry"):
            raise ValueError("empty_parent_fallback must be question or carry")
        for name in ("expected_rollout_n", "depth_window", "cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.expected_rollout_n < 2 or self.depth_window < 0 or self.cache_size < 1:
            raise ValueError("Require expected_rollout_n>=2, depth_window>=0, cache_size>=1")
        for name in ("similarity_power", "cross_depth_decay", "parent_alpha", "question_parent_beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.similarity_power <= 0 or self.parent_alpha <= 0 or self.question_parent_beta < 0:
            raise ValueError("Require similarity_power>0, parent_alpha>0, question_parent_beta>=0")
        if not 0 <= self.cross_depth_decay <= 1:
            raise ValueError("cross_depth_decay must be in [0,1]")
        for name in ("strict", "record_matched_peers", "verifier_same_route",
                     "verifier_cross_depth_same_reason", "verifier_same_depth_same_reason"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean, not a string or number")
        if not self.strict:
            raise ValueError("Non-strict boundary LOTO is not implemented; do not silently accept incomplete chains")
        if not self.verifier_same_route or not self.verifier_cross_depth_same_reason:
            raise ValueError("The approved boundary estimator requires same route and cross-depth same reason")
        if self.verifier_same_depth_same_reason:
            raise ValueError("The approved estimator does not hard-split same-depth Verifier reasons")
        if self.idf_path is not None and (not isinstance(self.idf_path, str) or not self.idf_path.strip()):
            raise ValueError("idf_path must be a nonempty path string or None")
        if self.idf_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", self.idf_sha256):
            raise ValueError("idf_sha256 must be a 64-character SHA-256 hexadecimal digest")

    def validate_runtime(self, gamma: float, internal_gamma: float, agent_local_mode: str = "off") -> None:
        if self.mode == "legacy_loto":
            return
        if gamma != 1.0 or internal_gamma != 1.0:
            raise ValueError("boundary_loto requires gamma=internal_gamma=1; discounted peer post-return/parent targets are not implemented")
        if agent_local_mode != "off":
            raise ValueError("boundary_loto requires agent_local.mode=off; G1/SMDP is not part of this profile")
        if self.empty_parent_fallback != "question":
            raise ValueError("boundary_loto training requires question parent fallback; carry is only an offline historical reference")


class FrozenTfidfKernel:
    """Frozen corpus frequencies, OOV-preserving vectors, bounded LRU caches."""

    def __init__(self, M: int, df: Mapping[str, int], *, cache_size: int = 8192,
                 metadata: Optional[Mapping[str, Any]] = None) -> None:
        if isinstance(M, bool) or not isinstance(M, Integral) or M < 0:
            raise ValueError("IDF M must be a nonnegative integer")
        if isinstance(cache_size, bool) or not isinstance(cache_size, Integral) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        frequencies = {}
        for word, count in df.items():
            if not isinstance(word, str) or re.fullmatch(r"[a-z0-9]+", word) is None:
                raise ValueError(f"Invalid IDF token {word!r}")
            if word in STOP_WORDS:
                raise ValueError(f"IDF df must exclude fixed stop word {word!r}")
            if isinstance(count, bool) or not isinstance(count, Integral) or not 1 <= count <= M:
                raise ValueError(f"IDF df[{word!r}] must lie in [1,M]")
            frequencies[word] = int(count)
        self.M = int(M)
        self.df = MappingProxyType(frequencies)
        self.cache_size = int(cache_size)
        self.metadata = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
        self._vectors: OrderedDict[str, Mapping[str, float]] = OrderedDict()
        self._pairs: OrderedDict[tuple[str, str, float], float] = OrderedDict()
        self.idf_sha256 = hashlib.sha256(_json_bytes(self.to_asset())).hexdigest()

    @classmethod
    def fit(cls, raw_documents: Iterable[Optional[str]], **kwargs: Any) -> "FrozenTfidfKernel":
        return cls.fit_keys((canonical_document_key(raw) for raw in raw_documents), **kwargs)

    @classmethod
    def fit_keys(cls, keys: Iterable[str], **kwargs: Any) -> "FrozenTfidfKernel":
        unique = set(keys)
        unique.discard("")
        if any(not isinstance(key, str) or canonical(key) != key for key in unique):
            raise ValueError("fit_keys requires canonical document keys")
        frequency: Counter = Counter()
        for key in unique:
            frequency.update(set(key.split()) - STOP_WORDS)
        return cls(len(unique), frequency, **kwargs)

    def to_asset(self) -> dict[str, Any]:
        return {"schema_version": ASSET_SCHEMA_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "M": self.M, "df": dict(self.df), "stop_words": sorted(STOP_WORDS),
                "metadata": self.metadata}

    @classmethod
    def from_asset(cls, asset: Mapping[str, Any], *, cache_size: int = 8192) -> "FrozenTfidfKernel":
        if not isinstance(asset, Mapping):
            raise ValueError("IDF asset must be a JSON object")
        if asset.get("schema_version") != ASSET_SCHEMA_VERSION:
            raise ValueError("Unsupported or missing IDF schema_version")
        if asset.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError("Unsupported or missing IDF normalization_version")
        stop = asset.get("stop_words")
        if not isinstance(stop, list) or len(stop) != len(STOP_WORDS) or set(stop) != STOP_WORDS:
            raise ValueError("IDF asset stop_words must exactly match the frozen reference")
        if not isinstance(asset.get("df"), Mapping):
            raise ValueError("IDF asset must contain complete df mapping")
        return cls(asset.get("M"), asset["df"], cache_size=cache_size, metadata=asset.get("metadata"))

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: Optional[str] = None,
             cache_size: int = 8192) -> "FrozenTfidfKernel":
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ValueError(f"IDF SHA-256 mismatch for {path}: got {digest}")
        try:
            asset = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid IDF JSON asset {path}") from exc
        kernel = cls.from_asset(asset, cache_size=cache_size)
        kernel.idf_sha256 = digest  # Exactly the bytes recorded in the run manifest.
        return kernel

    def export(self, path: str | Path, *, overwrite: bool = False) -> str:
        """Explicitly export generated assets; never overwrite by default."""
        data = _json_bytes(self.to_asset())
        with Path(path).open("wb" if overwrite else "xb") as stream:
            stream.write(data)
        return hashlib.sha256(data).hexdigest()

    @property
    def cache_info(self) -> dict[str, int]:
        return {"vectors": len(self._vectors), "pairs": len(self._pairs), "limit": self.cache_size}

    def vector(self, key: str) -> Mapping[str, float]:
        if key in self._vectors:
            self._vectors.move_to_end(key)
            return self._vectors[key]
        counts = Counter(word for word in key.split() if word not in STOP_WORDS)
        vector = {word: (1.0 + math.log(count)) *
                  (1.0 + math.log((1.0 + self.M) / (1.0 + self.df.get(word, 0))))
                  for word, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            vector = {word: value / norm for word, value in vector.items()}
        result = MappingProxyType(vector)
        self._vectors[key] = result
        if len(self._vectors) > self.cache_size:
            self._vectors.popitem(last=False)
        return result

    def similarity(self, left: str, right: str, power: float = 4.0) -> float:
        if not math.isfinite(power) or power <= 0:
            raise ValueError("similarity power must be finite and positive")
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        pair = (left, right, float(power)) if left <= right else (right, left, float(power))
        if pair in self._pairs:
            self._pairs.move_to_end(pair)
            return self._pairs[pair]
        lv, rv = self.vector(left), self.vector(right)
        if len(lv) > len(rv):
            lv, rv = rv, lv
        cosine = min(1.0, max(0.0, sum(weight * rv.get(word, 0.0) for word, weight in lv.items())))
        result = cosine ** power
        self._pairs[pair] = result
        if len(self._pairs) > self.cache_size:
            self._pairs.popitem(last=False)
        return result

    __call__ = similarity


# At most four frozen assets are retained. No rewards/values are cached here.
_ASSET_CACHE: OrderedDict[tuple[Any, ...], FrozenTfidfKernel] = OrderedDict()


def load_frozen_kernel(config: BoundaryValueConfig) -> FrozenTfidfKernel:
    if not config.idf_path:
        raise ValueError("boundary_loto requires a frozen IDF asset (value.idf_path)")
    path = Path(config.idf_path).resolve(strict=True)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, config.idf_sha256, config.cache_size)
    if key not in _ASSET_CACHE:
        _ASSET_CACHE[key] = FrozenTfidfKernel.load(path, expected_sha256=config.idf_sha256,
                                                 cache_size=config.cache_size)
        if len(_ASSET_CACHE) > 4:
            _ASSET_CACHE.popitem(last=False)
    _ASSET_CACHE.move_to_end(key)
    return _ASSET_CACHE[key]


@dataclass(frozen=True)
class BoundaryValueResult:
    pre: dict[str, float]
    post: dict[str, float]
    details: dict[str, dict[str, Any]]
    returns: dict[str, float]
    idf_sha256: str


def _integer(record: Mapping[str, Any], name: str) -> int:
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{record.get('event_uid')}: {name} must be a nonnegative integer")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if value is None:
        return False
    # np.bool_ is neither bool nor numbers.Integral, but has scalar item().
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        value = value.item()
    if not isinstance(value, (bool, Integral)) or value not in (0, 1):
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(value)


def _validated_chains(events: Mapping[str, Mapping[str, Any]],
                      trajectories: Mapping[str, list[str]], expected_n: int):
    if not events or not trajectories:
        raise ValueError("boundary_loto requires nonempty complete rollout groups")
    required = {"event_uid", "uid", "traj_uid", "event_index", "event_count", "role_event_index",
                "role_event_count", "agent_id", "event_type", "env_step_index", "is_env_action",
                "env_action_owner", "env_reward", "env_done", "is_action_valid"}
    normalized = {}
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
        if event["event_type"] not in _KINDS:
            raise ValueError(f"Event {key}: unsupported Search event_type {event['event_type']!r}")
        if "task_type" in event and str(event["task_type"]).lower() != "search":
            raise ValueError("boundary_loto currently supports task_type=Search only")
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
        if not event["is_env_action"] and (event["env_done"] or abs(reward) > 1e-12):
            raise ValueError(f"Event {key}: non-owner cannot carry reward or done")
        if not event["env_done"] and abs(reward) > 1e-12:
            raise ValueError(f"Event {key}: terminal_env_return requires zero nonterminal reward")
        if event["env_done"] and reward not in (0.0, 1.0):
            raise ValueError(f"Event {key}: terminal_env_return requires binary terminal env reward")
        if event["event_type"] == "verifier":
            for name in ("route_target", "route_reason"):
                if name not in event or not isinstance(event[name], str) or not event[name].strip():
                    raise ValueError(f"Verifier {key}: missing/invalid {name}; use executed route metadata")
        if event["event_type"] == "search_query" and "tool_observation" not in event:
            raise ValueError(f"Search {key}: missing tool_observation field (None is a valid empty result)")
        normalized[key] = event
    chains, groups, returns, seen = {}, defaultdict(list), {}, set()
    producer_types: dict[str, str] = {}
    type_producers: dict[str, str] = {}
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
        by_step, by_role = defaultdict(list), defaultdict(list)
        for event in records:
            by_step[event["env_step_index"]].append(event)
            by_role[event["agent_id"]].append(event)
            role, kind = event["agent_id"], event["event_type"]
            if producer_types.setdefault(role, kind) != kind or type_producers.setdefault(kind, role) != role:
                raise ValueError("Search requires one consistent producer per Verifier/Search/Answer event type")
        rounds = [event["env_step_index"] for event in records]
        if rounds != sorted(rounds) or sorted(by_step) != list(range(len(by_step))):
            raise ValueError(f"Trajectory {trajectory}: env_step_index must be contiguous chronological rounds from zero")
        for step, step_events in by_step.items():
            if len(step_events) != 2 or step_events[0]["event_type"] != "verifier" or step_events[1]["event_type"] not in ("search_query", "final_answer"):
                raise ValueError(f"Trajectory {trajectory} round {step}: expected Verifier then Search/Answer")
            verifier, owner = step_events
            if verifier["is_env_action"] or not owner["is_env_action"]:
                raise ValueError(f"Trajectory {trajectory} round {step}: must end with exactly one owner")
            if verifier["route_target"] != owner["agent_id"]:
                raise ValueError(f"Verifier {verifier['event_uid']}: route_target does not equal the executed owner")
            if verifier["env_action_owner"] not in (None, owner["agent_id"]):
                raise ValueError(f"Verifier {verifier['event_uid']}: inconsistent env_action_owner")
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


def estimate_boundary_values(events: Mapping[str, Mapping[str, Any]],
                             trajectories: Mapping[str, list[str]],
                             config: BoundaryValueConfig,
                             kernel: Optional[FrozenTfidfKernel] = None) -> BoundaryValueResult:
    """Compute pre/post values using complete same-question rollout groups.

    Caller must deduplicate copied training rows *with conflict detection* first.
    The immutable IDs, chain completeness and owner reward contract are checked
    again here. Neither input records nor the caller's chain ordering is mutated.
    """
    config = BoundaryValueConfig.from_mapping(config)
    if config.mode != "boundary_loto":
        raise ValueError("estimate_boundary_values is the boundary_loto engine, not the legacy estimator")
    kernel = load_frozen_kernel(config) if kernel is None else kernel
    if config.idf_sha256 and kernel.idf_sha256.lower() != config.idf_sha256.lower():
        raise ValueError("Provided kernel does not match configured IDF SHA-256")
    records, chains, groups, returns = _validated_chains(events, trajectories, config.expected_rollout_n)
    docs, previous_docs, by_kind = {}, {}, {}
    for trajectory, keys in chains.items():
        latest = ""
        kinds = defaultdict(list)
        for key in keys:
            event = records[key]
            previous_docs[key] = latest
            if event["event_type"] == "search_query":
                docs[key] = canonical_document_key(event["tool_observation"])
                latest = docs[key]  # A genuine empty Search must clear old docs.
            elif event["event_type"] == "verifier":
                docs[key] = latest
            else:
                docs[key] = ""
            kinds[event["event_type"]].append(key)
        by_kind[trajectory] = kinds
    pre, post, details = {}, {}, {}
    for members in groups.values():
        for own in members:
            peers = [other for other in members if other != own]
            qparent = sum(returns[other] for other in peers) / len(peers)
            value = qparent
            for key in chains[own]:
                event = records[key]
                kind, step = event["event_type"], event["env_step_index"]
                pre[key] = value
                detail = {"qparent": qparent, "parent": qparent, "n_same": 0,
                          "mass": 0.0, "ess": 0.0, "positive_peer_count": 0,
                          "cross_round_peer_count": 0,
                          "document_key_sha256": _sha256(docs[key]),
                          "previous_document_key_sha256": _sha256(previous_docs[key]),
                          "route_target": event.get("route_target"),
                          "route_reason": event.get("route_reason")}
                if config.record_matched_peers:
                    detail["matched_peers"] = []
                if event["env_done"]:
                    value = 0.0
                elif kind == "final_answer":
                    pass  # Explicit nonterminal Answer carry approximation.
                else:
                    same = [other for other in peers if any(records[ref]["env_step_index"] == step
                                                            for ref in by_kind[other][kind])]
                    if same:
                        parent = (sum(returns[other] for other in same) + config.question_parent_beta * qparent) / (len(same) + config.question_parent_beta)
                    else:
                        parent = pre[key] if config.empty_parent_fallback == "carry" else qparent
                    detail["parent"], detail["n_same"] = parent, len(same)
                    weights, weighted_returns, squares = [], [], []
                    for other in peers:
                        best, selected = 0.0, None
                        for ref in by_kind[other][kind]:
                            candidate = records[ref]
                            ref_step = candidate["env_step_index"]
                            distance = abs(ref_step - step)
                            if distance > config.depth_window:
                                continue
                            if kind == "verifier":
                                if candidate["route_target"] != event["route_target"]:
                                    continue
                                if (step == 0) != (ref_step == 0):
                                    continue
                                if distance and candidate["route_reason"] != event["route_reason"]:
                                    continue
                            score = (1.0 if kind == "verifier" and step == ref_step == 0
                                     else kernel.similarity(docs[key], docs[ref], config.similarity_power))
                            score *= config.cross_depth_decay ** distance
                            # Keys are chronological; strict > keeps first on ties.
                            if score > best:
                                best, selected = score, ref
                        if selected is not None:
                            weights.append(best)
                            weighted_returns.append(best * returns[other])
                            squares.append(best * best)
                            is_cross = records[selected]["env_step_index"] != step
                            detail["positive_peer_count"] += 1
                            detail["cross_round_peer_count"] += int(is_cross)
                            if config.record_matched_peers:
                                detail["matched_peers"].append({"traj_uid": other, "event_uid": selected,
                                                               "env_step_index": records[selected]["env_step_index"],
                                                               "weight": best, "reward": returns[other]})
                    mass = sum(weights)
                    value = (sum(weighted_returns) + config.parent_alpha * parent) / (mass + config.parent_alpha)
                    detail["mass"] = mass
                    detail["ess"] = mass * mass / sum(squares) if squares else 0.0
                if not math.isfinite(value) or not -1e-12 <= value <= 1.0 + 1e-12:
                    raise ValueError(f"Invalid environment boundary value {value} at {key}")
                post[key], details[key] = float(value), detail
    return BoundaryValueResult(pre, post, details, returns, kernel.idf_sha256)


__all__ = ["BoundaryValueConfig", "BoundaryValueResult", "FrozenTfidfKernel", "STOP_WORDS",
           "NORMALIZATION_VERSION", "canonical", "plain_docs", "canonical_document_key",
           "estimate_boundary_values", "load_frozen_kernel"]
