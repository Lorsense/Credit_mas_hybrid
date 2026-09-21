"""Offline hindsight outcome-sensitivity scorer for Dr.MAS Search (S5).

This module implements the leakage-free, frozen-policy hindsight scorer specified
by ``S5_Claude_Next_Step_Directive.md``. It tests *outcome sensitivity*: does
conditioning the frozen generating policy on the true terminal-outcome label make
the actually-taken action more likely than conditioning on the counterfactual
opposite label, after controlling for the prompt-structure perturbation?

It does NOT claim causal credit assignment. Terminology: "outcome sensitivity",
"event relevance", "outcome compatibility".

Three per-event forward conditions (label-only certificate appended as a user
turn to the actor's pre-action chat state):
  - TRUE     the event's real terminal_success
  - FLIP     1 - terminal_success
  - UNKNOWN  success=unknown  (structural control)

SHUFFLE is NOT a fourth forward condition -- a shuffled binary label is exactly
TRUE or FLIP for that event. It is implemented as a matched permutation null
(``matched_permutation_labels``), reusing the TRUE/FLIP likelihoods.

Contrasts:
  guide_delta      = ell_TRUE - ell_0        (guide-compatible; basis of rho)
  structural_shift = ell_UNKNOWN - ell_0     (the cost of appending a turn)
  semantic_margin  = ell_TRUE - ell_FLIP     (core directional diagnostic)
  true_vs_neutral  = ell_TRUE - ell_UNKNOWN

ell is the per-token MEAN log-prob of the saved response (response_mask is all-1
in the trace). Both ell_0 and ell_H come from the SAME HF teacher-forcing
backend; saved SGLang rollout log-probs are audit-only and never subtracted.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

T_H_DEFAULT = 5.0
RHO_MIN_DEFAULT = 0.8
RHO_MAX_DEFAULT = 1.2

CERT_TAG_OPEN = "[POSTHOC_OUTCOME]"
CERT_TAG_CLOSE = "[/POSTHOC_OUTCOME]"

# Three actual forward-pass conditions. SHUFFLE is a permutation null, not a
# forward condition (see module docstring).
CERTIFICATE_TYPES = ("TRUE", "FLIP", "UNKNOWN")

# Event fields whose values must NEVER appear in a certificate / rendered prompt
# (beyond the terminal_success bit, which is the intended intervention). Only
# UNIQUE LONG future-bearing text is forbidden here; short categorical values
# (verifier_decision: yes/no/invalid/inactive) and action-format strings
# (raw/executed_action_text like "<verify>yes</verify>") are EXCLUDED because they
# coincide with the task instructions/examples in the original prompt and would
# false-positive. The current event's action is still covered by the decoded
# response-token check below.
FORBIDDEN_LEAK_FIELDS = (
    "final_answer_text",
    "terminal_action_text",
    "tool_observation",
    "env_observation_text",
    "offline_eval_info",
)


# --------------------------------------------------------------------------- #
# Certificate construction
# --------------------------------------------------------------------------- #

def certificate_text(success: bool | int | str) -> str:
    """Render the label-only certificate body.

    ``success`` may be a bool/int (0/1) for TRUE/FLIP/SHUFFLE-true-label, or the
    string ``"unknown"`` for UNKNOWN. The body is the ONLY text the certificate
    ever carries.
    """
    if isinstance(success, str):
        label = success
    else:
        label = "1" if bool(success) else "0"
    return f"{CERT_TAG_OPEN}\nsuccess={label}\n{CERT_TAG_CLOSE}"


def build_conditional_chat(hcapo_state_chat: Sequence[Mapping[str, str]], cert_body: str) -> list[dict]:
    """Return a NEW chat list with the certificate appended as a user turn.

    The original ``hcapo_state_chat`` is never mutated. Re-templating the result
    with ``apply_chat_template(add_generation_prompt=True)`` yields the
    conditional prompt.
    """
    return [dict(msg) for msg in hcapo_state_chat] + [{"role": "user", "content": cert_body}]


def build_certificate_chat(
    hcapo_state_chat: Sequence[Mapping[str, str]],
    certificate_type: str,
    *,
    event_terminal_success: bool,
    shuffle_label: bool | None = None,
) -> list[dict]:
    """Dispatch the certificate condition -> conditional chat list.

    TRUE -> real label; FLIP -> flipped label; UNKNOWN -> 'unknown'.
    (SHUFFLE is handled via permutation, not here; if a per-event shuffle label
    is recorded for audit, pass shuffle_label through certificate_text directly.)
    """
    if certificate_type == "TRUE":
        body = certificate_text(event_terminal_success)
    elif certificate_type == "FLIP":
        body = certificate_text(not event_terminal_success)
    elif certificate_type == "UNKNOWN":
        body = certificate_text("unknown")
    elif certificate_type == "SHUFFLE":
        if shuffle_label is None:
            raise ValueError("SHUFFLE requires shuffle_label")
        body = certificate_text(shuffle_label)
    else:
        raise ValueError(f"unknown certificate type: {certificate_type!r}")
    return build_conditional_chat(hcapo_state_chat, body)


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #

def _normalized_substrings(event: Mapping[str, Any], tokenizer=None) -> list[str]:
    """Forbidden substrings drawn from the event's future/action/answer fields."""
    out: list[str] = []
    for key in FORBIDDEN_LEAK_FIELDS:
        val = event.get(key)
        if isinstance(val, str):
            s = val.strip()
            if len(s) >= 4:  # skip trivially short / whitespace
                out.append(s)
    # decoded current response text (the action actually taken)
    rids = event.get("response_token_ids")
    if tokenizer is not None and isinstance(rids, list) and rids:
        try:
            decoded = tokenizer.decode(rids).strip()
            if len(decoded) >= 8:
                out.append(decoded)
        except Exception:
            pass
    return out


import re

_CERT_BODY_RE = re.compile(
    r"^\[POSTHOC_OUTCOME\]\nsuccess=(0|1|unknown)\n\[/POSTHOC_OUTCOME\]$"
)


def assert_no_leakage(
    certificate_body: str,
    event: Mapping[str, Any],
    *,
    rendered_prompt: str | None = None,
    tokenizer=None,
) -> None:
    """Assert the certificate is label-only and carries no forbidden future info.

    The leakage invariant is enforced on the CERTIFICATE BODY, which is the only
    new content added to the pre-action state (``build_conditional_chat`` is
    append-only, verified by tests). Two checks:
      1. The certificate body MUST exactly match the label-only format (regex).
      2. The certificate body must contain none of the forbidden event-specific
         future text (decoded response, final answer, retrieved docs, gold).

    Note: we intentionally do NOT substring-scan the *rendered conditional
    prompt* (the original ``hcapo_state_chat`` is the legit pre-action state and
    legitimately contains accumulated history -- previous retrieved docs that are
    re-observed, action-format examples in the instructions, etc. -- so a prompt
    substring scan false-positives on that legitimate history). Since the cert is
    label-only and the build is append-only, the rendered prompt cannot carry
    future info beyond the original pre-action state.
    """
    del rendered_prompt  # accepted for signature compatibility; not scanned (see docstring)
    if not _CERT_BODY_RE.match(certificate_body):
        raise AssertionError(
            f"certificate body is not label-only: {certificate_body!r}"
        )
    forbidden = _normalized_substrings(event, tokenizer=tokenizer)
    nlow_body = certificate_body.lower()
    for needle in forbidden:
        if needle.lower() in nlow_body:
            raise AssertionError(
                f"leakage: forbidden event text fragment {needle[:40]!r} found in certificate body"
            )


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #

def _has_complete_tokens(event: Mapping[str, Any]) -> bool:
    rids = event.get("response_token_ids")
    pids = event.get("prompt_token_ids")
    return (
        isinstance(rids, list) and len(rids) >= 1
        and isinstance(pids, list) and len(pids) >= 1
    )


def compute_eligibility(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Split into ``scoreable`` events (by role) and record exclusion reasons.

    Scoreable (directive §3.4):
      - complete prompt/response token representation
      - prompt_was_truncated is False
      - response has >=1 scored token
    Role/tokenizer/checkpoint availability is enforced by the CLI at load time.
    """
    eligible_by_role: dict[str, list[dict]] = defaultdict(list)
    exclusions: dict[str, str] = {}
    for ev in events:
        uid = ev.get("event_uid")
        if not _has_complete_tokens(ev):
            exclusions[uid] = "incomplete_tokens"
        elif ev.get("prompt_was_truncated"):
            exclusions[uid] = "prompt_truncated"
        else:
            role = ev.get("agent_id") or ev.get("role")
            eligible_by_role[role].append(ev)
    return dict(eligible_by_role), exclusions


def compute_diagnostic_matched(events: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return event_uids that are *diagnostic-matched* (directive §3.4):
    their ``(task_uid, role)`` cohort has >=2 distinct traj_uid AND contains both
    a success and a failure trajectory. Used for paired/permutation significance.
    """
    # traj-level outcome per (task, role), deduplicated by traj_uid
    cohort_trajs: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for ev in events:
        key = (ev.get("task_uid"), ev.get("agent_id") or ev.get("role"))
        tu = ev.get("traj_uid")
        if tu is not None and tu not in cohort_trajs[key]:
            cohort_trajs[key][tu] = bool(ev.get("terminal_success"))
    matched_cohorts = set()
    for key, traj_succ in cohort_trajs.items():
        if len(traj_succ) >= 2 and len(set(traj_succ.values())) == 2:
            matched_cohorts.add(key)
    return {
        ev.get("event_uid")
        for ev in events
        if (ev.get("task_uid"), ev.get("agent_id") or ev.get("role")) in matched_cohorts
    }


# --------------------------------------------------------------------------- #
# Matched permutation null (replaces per-event SHUFFLE forward pass)
# --------------------------------------------------------------------------- #

def build_matched_cohorts(events: Sequence[Mapping[str, Any]]) -> dict[tuple, list[dict]]:
    """Group events by ``(task_uid, role)`` cohort. A cohort is permutation-eligible
    if it has >=2 distinct traj_uid with both outcomes (compute_diagnostic_matched).
    """
    cohorts: dict[tuple, list[dict]] = defaultdict(list)
    for ev in events:
        cohorts[(ev.get("task_uid"), ev.get("agent_id") or ev.get("role"))].append(ev)
    return cohorts


def matched_permutation_labels(
    cohort_events: Sequence[Mapping[str, Any]],
    n_perms: int,
    seed: int,
) -> list[list[bool]]:
    """Return ``n_perms`` permuted outcome-label assignments over the cohort's
    DISTINCT trajectories. Each assignment is a list of bools aligned to the
    sorted distinct ``traj_uid`` order. Labels are permuted (never identity-fixed
    to self by construction -- a trajectory may receive its own label only as part
    of a valid random permutation, which is correct exchangeability under the null).

    Deterministic via ``numpy.default_rng(seed)``.
    """
    import numpy as np

    traj_succ: dict[str, bool] = {}
    for ev in cohort_events:
        tu = ev.get("traj_uid")
        if tu is not None and tu not in traj_succ:
            traj_succ[tu] = bool(ev.get("terminal_success"))
    trajs = sorted(traj_succ)
    labels = np.array([traj_succ[t] for t in trajs], dtype=bool)
    rng = np.random.default_rng(seed)
    perms: list[list[bool]] = []
    for _ in range(n_perms):
        perm = rng.permutation(labels)
        perms.append(bool_list(perm))
    return perms


def bool_list(arr) -> list[bool]:
    return [bool(x) for x in arr]


# --------------------------------------------------------------------------- #
# Contrasts and rho
# --------------------------------------------------------------------------- #

def compute_contrasts(
    ell_0: float,
    ell_true: float,
    ell_flip: float,
    ell_unknown: float,
) -> dict[str, float]:
    """All four directive contrasts (ell is per-token mean log-prob)."""
    return {
        "guide_delta": ell_true - ell_0,
        "structural_shift": ell_unknown - ell_0,
        "semantic_margin": ell_true - ell_flip,
        "true_vs_neutral": ell_true - ell_unknown,
    }


def compute_d(ell_0: float, ell_true: float, T_H: float = T_H_DEFAULT) -> float:
    """guide-compatible d = (ell_TRUE - ell_0) / T_H. ell is already a per-token
    mean; do NOT divide by token count again."""
    return (ell_true - ell_0) / T_H


def compute_rho(
    ell_0: float,
    ell_true: float,
    *,
    T_H: float = T_H_DEFAULT,
    rho_min: float = RHO_MIN_DEFAULT,
    rho_max: float = RHO_MAX_DEFAULT,
) -> tuple[float, float, bool]:
    """Return (d, rho, clipped). rho = clip(exp(d), rho_min, rho_max)."""
    d = compute_d(ell_0, ell_true, T_H)
    rho = math.exp(d)
    clipped = False
    if rho <= rho_min:
        rho = rho_min
        clipped = True
    elif rho >= rho_max:
        rho = rho_max
        clipped = True
    return d, rho, clipped


# --------------------------------------------------------------------------- #
# HF teacher-forcing scorer
# --------------------------------------------------------------------------- #

def _per_token_logp(logits, labels):
    """log_softmax(logits).gather(labels) per position (bf16-safe enough for
    relative ell_0/ell_H with the same kernel). Mirrors the semantics of
    verl.utils.torch_functional.logprobs_from_logits without importing verl,
    so this module stays unit-testable on CPU without the verl stack."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


class HFTeacherForcingScorer:
    """Frozen HF model + tokenizer; computes mean per-token log p(response|prompt).

    Batch-invariant by construction: prompts are left-padded, responses
    right-padded and right-aligned, with attention_mask + position_ids so a
    sequence scores identically alone or inside a heterogeneous batch.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", pad_id: int | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
        if self.pad_id is None:
            raise ValueError("tokenizer has no pad_token_id")

    @staticmethod
    def _build_batch(prompt_ids, response_ids, pad_id):
        import torch

        bsz = len(prompt_ids)
        p_len = max(len(p) for p in prompt_ids)
        r_len = max(len(r) for r in response_ids)
        seq_len = p_len + r_len
        input_ids = torch.full((bsz, seq_len), pad_id, dtype=torch.long)
        resp_tgt = torch.full((bsz, r_len), pad_id, dtype=torch.long)
        resp_mask = torch.zeros((bsz, r_len), dtype=torch.float)
        attn = torch.zeros((bsz, seq_len), dtype=torch.long)
        for i, (p, r) in enumerate(zip(prompt_ids, response_ids)):
            lp, lr = len(p), len(r)
            # left-pad prompt -> real prompt occupies [p_len-lp : p_len]
            input_ids[i, p_len - lp:p_len] = torch.tensor(p, dtype=torch.long)
            # response occupies [p_len : p_len+lr] (right-aligned within the batch)
            input_ids[i, p_len:p_len + lr] = torch.tensor(r, dtype=torch.long)
            resp_tgt[i, :lr] = torch.tensor(r, dtype=torch.long)   # right-pad target
            resp_mask[i, :lr] = 1.0
            attn[i, p_len - lp:p_len] = 1
            attn[i, p_len:p_len + lr] = 1
        # position_ids: real positions 0..L-1, padded positions clamped to 0 (masked)
        pos = torch.cumsum(attn, dim=-1) - 1
        pos = pos.clamp(min=0)
        return input_ids, attn, pos, resp_tgt, resp_mask, r_len

    def ell_batch(self, prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]]) -> list[float]:
        """Mean per-token log p of each response under teacher forcing."""
        import torch

        if not prompt_ids:
            return []
        input_ids, attn, pos, resp_tgt, resp_mask, r_len = self._build_batch(
            prompt_ids, response_ids, self.pad_id
        )
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        pos = pos.to(self.device)
        resp_tgt = resp_tgt.to(self.device)
        resp_mask = resp_mask.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                position_ids=pos,
                use_cache=False,
            )
            logits = out.logits  # (bsz, seq_len, vocab)
            # predicting logits for the response region [p_len : p_len+r_len):
            # positions [-r_len-1 : -1]
            logits = logits[:, -r_len - 1:-1, :]  # (bsz, r_len, vocab)
            logp = _per_token_logp(logits, resp_tgt)  # (bsz, r_len)
            ell = (logp * resp_mask).sum(-1) / resp_mask.sum(-1).clamp(min=1.0)
        return ell.float().tolist()


def identity_check(tokenizer, hcapo_state_chat, apply_kwargs: Mapping[str, Any] | None = None) -> bool:
    """Re-template the SAME chat twice; must be byte-identical. (The empty-cert
    identity is modeled as NOT appending a turn, so the conditional prompt is
    produced by re-templating the original chat -- this checks that path is
    deterministic.)"""
    kw = dict(apply_kwargs or {})
    a = tokenizer.apply_chat_template(hcapo_state_chat, add_generation_prompt=True, tokenize=True, **kw)
    b = tokenizer.apply_chat_template(hcapo_state_chat, add_generation_prompt=True, tokenize=True, **kw)
    return list(a) == list(b)


def chat_template_sha256(tokenizer) -> str:
    return hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()


def conditional_prompt_ids(tokenizer, hcapo_state_chat, cert_body: str, apply_kwargs: Mapping[str, Any] | None = None) -> list[int]:
    """Build the conditional prompt token ids (state + certificate)."""
    kw = dict(apply_kwargs or {})
    chat = build_conditional_chat(hcapo_state_chat, cert_body)
    return list(tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, **kw))
