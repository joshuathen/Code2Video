#!/usr/bin/env python3
"""Build and update an LLM-driven MAS belief library from run artifacts.

This utility analyses one MAS run or a full pipeline, prepares condensed run
evidence, asks Gemini for structured belief assessments, and persists the
resulting belief library over repeated runs.

The deterministic parts of the script only:
- collect run artifacts into a stable summary
- maintain belief evidence counts and Bayesian confidence
- write analysis/library JSON files

Belief creation and belief evaluation are delegated to Gemini.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from google.genai import Client
from pydantic import BaseModel, Field, ValidationError, model_validator
from belief_bbn import (
    BBNParameters,
    BeliefEmbeddingIndex,
    TransitionEvidence,
    update_beta_posterior,
)
from mas_interactions import create_interaction, response_usage_dict
from code_normalization import normalize_code_to_code2video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = Path(__file__).with_name("api_config.json")


def cfg(service: str, key: str, default: Any = None) -> Any:
    """Read only the lightweight configuration needed by this utility."""
    env_value = os.getenv(f"{service}_{key}".upper())
    if env_value is not None:
        return env_value
    try:
        with _CFG_PATH.open("r", encoding="utf-8") as config_file:
            config_payload = json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
    service_payload = config_payload.get(service, {})
    if not isinstance(service_payload, dict):
        return default
    return service_payload.get(key, default)


DEFAULT_LOGS_ROOT = PROJECT_ROOT / "mas_logs"
DEFAULT_LIBRARY_FILENAME = "belief_library.json"
DEFAULT_ANALYSIS_FILENAME = "belief_analysis.json"
DEFAULT_EVIDENCE_FILENAME = "belief_evidence.jsonl"
DEFAULT_BBN_PARAMETERS_FILENAME = "bbn_parameters.json"
DEFAULT_PROGRESS_FILENAME = "belief_generation_progress.json"
DEFAULT_REFLECTION_MODEL = cfg("gemini", "model", "gemini-3-flash-preview")

ACTION_ADD = "ADD"
ACTION_SUPPORT = "SUPPORT"
ACTION_CONTRADICT = "CONTRADICT"
ACTION_OBSERVE = "OBSERVE"
ACTION_IRRELEVANT = "IRRELEVANT"
ACTION_REVISE = "REVISE"
ACTION_MERGE = "MERGE"
VALID_ACTIONS = {
    ACTION_ADD,
    ACTION_SUPPORT,
    ACTION_CONTRADICT,
    ACTION_OBSERVE,
    ACTION_IRRELEVANT,
    ACTION_REVISE,
    ACTION_MERGE,
}

STATUS_ACTIVE = "active"
STATUS_PROBATION = "probation"
STATUS_DEPRECATED = "deprecated"

IMPACT_LEVELS = {"low": 0.25, "medium": 0.5, "high": 0.8, "critical": 1.0}
BELIEF_TYPES = {"confirmed", "precaution", "hypothesis", "quality"}
BELIEF_TIMINGS = {"preventative", "reactive", "both"}

STRATEGY_APPLICATION_PROBABILITIES = {
    "full": 0.95,
    "partial": 0.65,
    "unclear": 0.35,
    "none": 0.05,
}
ATTRIBUTION_PROBABILITIES = {
    "strong": 0.90,
    "moderate": 0.65,
    "weak": 0.30,
    "unclear": 0.20,
    "none": 0.05,
}
EVIDENCE_RELIABILITY_PROBABILITIES = {
    "direct": 1.00,
    "corroborated": 0.85,
    "indirect": 0.65,
    "inferred": 0.40,
    "unverifiable": 0.00,
}
OUTCOME_IMPROVEMENT_VALUES = {
    "resolved": 1.00,
    "improved": 0.75,
    "unchanged": 0.50,
    "worsened": 0.00,
    "unclear": 0.50,
}


def _evidence_categories_to_values(
    strategy_application: str,
    attribution_strength: str,
    evidence_reliability: str,
    outcome_improvement: str,
) -> Dict[str, float]:
    """Map auditable model classifications to deterministic BBN inputs."""
    return {
        "strategy_applied_probability": STRATEGY_APPLICATION_PROBABILITIES[
            strategy_application
        ],
        "attribution_probability": ATTRIBUTION_PROBABILITIES[attribution_strength],
        "reliability_probability": EVIDENCE_RELIABILITY_PROBABILITIES[
            evidence_reliability
        ],
        "improvement": OUTCOME_IMPROVEMENT_VALUES[outcome_improvement],
    }


def _extract_structured_json(response_text: str) -> str:
    """Accept plain JSON and the fenced JSON some Gemini models still emit."""
    stripped = response_text.strip()
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced_match.group(1).strip() if fenced_match else stripped


def _normalize_impact(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in IMPACT_LEVELS else "medium"


def _normalize_belief_type(value: Any) -> str:
    normalized = str(value or "confirmed").strip().lower()
    return normalized if normalized in BELIEF_TYPES else "confirmed"


def _normalize_belief_timing(value: Any) -> str:
    # Legacy libraries predate timing classification. Treating those beliefs
    # as usable in both contexts preserves their previous behaviour.
    normalized = str(value or "both").strip().lower()
    return normalized if normalized in BELIEF_TIMINGS else "both"


def _impact_max(first: str, second: str) -> str:
    return max((_normalize_impact(first), _normalize_impact(second)), key=lambda item: IMPACT_LEVELS[item])


def _belief_type_max(first: str, second: str) -> str:
    order = {"hypothesis": 0, "quality": 1, "precaution": 2, "confirmed": 3}
    return max((_normalize_belief_type(first), _normalize_belief_type(second)), key=lambda item: order[item])


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, path)


def _coerce_evidence_payload(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    if isinstance(value, (str, int, float, bool)):
        return {"value": value}
    return {"value": str(value)}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    collected = [value for value in values if value is not None]
    return mean(collected) if collected else None


def _normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_role_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    normalized = str(label).strip()
    if normalized == "OrchestratorAgent":
        return "Orchestrator"
    normalized = re.sub(r"\d+$", "", normalized)
    return normalized or None


def _truncate_text(text: str, max_chars: int = 800) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "[code omitted]", text, flags=re.DOTALL)


def _compact_jsonish(value: Any, max_chars: int = 300) -> str:
    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            text = str(value)
    except Exception:
        text = str(value)
    return _truncate_text(_strip_code_blocks(text), max_chars=max_chars)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _status_from_counts(
    *,
    contradiction_count: int,
    confidence: float,
) -> str:
    if contradiction_count >= 5 and confidence < 0.35:
        return STATUS_DEPRECATED
    if contradiction_count >= 3 and confidence < 0.5:
        return STATUS_PROBATION
    return STATUS_ACTIVE


@dataclass
class BeliefScope:
    roles: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)
    problem_description: str = ""
    context_conditions: List[str] = field(default_factory=list)


@dataclass
class BeliefRecord:
    belief_id: str
    instruction: str
    scope: BeliefScope
    status: str = STATUS_ACTIVE
    alpha: float = 1.0
    beta: float = 1.0
    support_count: int = 0
    contradiction_count: int = 0
    relevant_count: int = 0
    irrelevant_count: int = 0
    weighted_support: float = 0.0
    weighted_contradiction: float = 0.0
    confidence: float = 0.5
    impact: str = "medium"
    priority: float = 0.0
    belief_type: str = "confirmed"
    timing: str = "both"
    created_from_run: Optional[str] = None
    observed_runs: List[str] = field(default_factory=list)
    independent_topics: List[str] = field(default_factory=list)
    topic_evidence_weights: Dict[str, float] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def update_confidence(self) -> None:
        self.confidence = round(self.alpha / (self.alpha + self.beta), 4)
        support_signal = min(1.0, math.log1p(self.support_count) / math.log(11.0))
        self.impact = _normalize_impact(self.impact)
        self.belief_type = _normalize_belief_type(self.belief_type)
        self.timing = _normalize_belief_timing(self.timing)
        self.priority = round(
            0.5 * self.confidence + 0.3 * IMPACT_LEVELS[self.impact] + 0.2 * support_signal,
            4,
        )
        self.status = _status_from_counts(
            contradiction_count=self.contradiction_count,
            confidence=self.confidence,
        )


@dataclass
class BeliefAssessment:
    run_id: str
    belief_id: Optional[str]
    instruction: str
    applicable: bool
    compliance: str
    outcome: str
    action: str
    weight: float
    scope: BeliefScope
    evidence_confidence: float = 0.5
    impact: str = "medium"
    belief_type: str = "confirmed"
    timing: str = "both"
    agent: Optional[str] = None
    section_id: Optional[str] = None
    reason: str = ""
    evidence_event_ids: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    merge_belief_ids: List[str] = field(default_factory=list)
    topic: str = ""
    strategy_application: str = "unclear"
    attribution_strength: str = "unclear"
    evidence_reliability: str = "unverifiable"
    outcome_improvement: str = "unclear"
    strategy_applied_probability: float = 0.5
    attribution_probability: float = 0.5
    reliability_probability: float = 0.5
    improvement: float = 0.5
    reflection_call: str = ""


class ReflectionScopePayload(BaseModel):
    roles: List[Literal["Orchestrator", "Coder", "AnimationPlanner", "ScriptWriter"]] = Field(
        min_length=1
    )
    stages: List[str] = Field(min_length=1, max_length=5)
    problem_description: str = Field(min_length=10, max_length=500)
    context_conditions: List[str] = Field(min_length=1, max_length=8)


class ReflectionAssessmentPayload(BaseModel):
    belief_id: Optional[str] = None
    instruction: str = Field(min_length=10, max_length=500)
    scope: ReflectionScopePayload
    applicable: bool
    compliance: Literal["followed", "violated", "mixed", "unclear", "observed_pattern"]
    outcome: Literal["positive", "negative", "mixed", "unclear"]
    action: Literal["ADD", "SUPPORT", "CONTRADICT", "IRRELEVANT", "REVISE", "MERGE"]
    merge_belief_ids: List[str] = Field(default_factory=list)
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    strategy_application: Literal["full", "partial", "unclear", "none"]
    attribution_strength: Literal["strong", "moderate", "weak", "unclear", "none"]
    evidence_reliability: Literal[
        "direct", "corroborated", "indirect", "inferred", "unverifiable"
    ]
    outcome_improvement: Literal["resolved", "improved", "unchanged", "worsened", "unclear"]
    impact: Literal["low", "medium", "high", "critical"]
    belief_type: Literal["confirmed", "precaution", "hypothesis", "quality"]
    timing: Literal["preventative", "reactive", "both"]
    agent: Optional[Literal["Orchestrator", "Coder", "AnimationPlanner", "ScriptWriter"]] = None
    section_id: Optional[str] = None
    reason: str = Field(min_length=10, max_length=500)
    # Gemini structured output does not support free-form object schemas because
    # Pydantic represents them with JSON Schema `additionalProperties`.
    evidence: str = Field(min_length=10, max_length=500)

    @model_validator(mode="after")
    def validate_action_target(self) -> "ReflectionAssessmentPayload":
        if self.action == ACTION_ADD and self.belief_id is not None:
            raise ValueError("ADD assessments must set belief_id to null")
        if self.action in {
            ACTION_SUPPORT,
            ACTION_CONTRADICT,
            ACTION_REVISE,
            ACTION_MERGE,
        } and not self.belief_id:
            raise ValueError(f"{self.action} assessments must identify an existing belief_id")
        return self


class ReflectionResponsePayload(BaseModel):
    assessments: List[ReflectionAssessmentPayload]


@dataclass
class SectionAttemptSummary:
    section_id: str
    attempts: int = 0
    scope_refine_attempts: int = 0
    infrastructure_attempts: int = 0
    successful: bool = False
    elapsed_seconds: float = 0.0
    render_errors: List[str] = field(default_factory=list)
    attempt_evidence: List[Dict[str, Any]] = field(default_factory=list)


def _is_manim_cache_race(render_error: str) -> bool:
    return (
        "FileExistsError" in render_error
        and ("media/texts" in render_error or "media/Tex" in render_error)
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _summarize_code_calls(code: str, *, max_calls: int = 20) -> List[Dict[str, Any]]:
    """Extract bounded call-site evidence without placing whole source files in prompts."""
    if not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    calls: Dict[str, Dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func) or "unknown"
        entry = calls.setdefault(name, {"name": name, "count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 1:
            source = ast.get_source_segment(code, node) or name
            entry["examples"].append(_truncate_text(source, max_chars=200))

    return sorted(calls.values(), key=lambda item: (-item["count"], item["name"]))[:max_calls]


def _bounded_code_diff(code_before: str, code_after: str, *, max_chars: int = 5000) -> str:
    if not code_before or not code_after or code_before == code_after:
        return ""
    diff = "\n".join(
        difflib.unified_diff(
            code_before.splitlines(),
            code_after.splitlines(),
            fromfile="code_before.py",
            tofile="code_after.py",
            lineterm="",
            n=3,
        )
    )
    return diff if len(diff) <= max_chars else diff[: max_chars - 3] + "..."


def _code_fingerprint(code: str, *, include_full_source: bool = False) -> Dict[str, Any]:
    calls = _summarize_code_calls(code, max_calls=12)
    fingerprint = {
        "sha1": hashlib.sha1(code.encode("utf-8")).hexdigest() if code else None,
        "char_count": len(code),
        "line_count": len(code.splitlines()),
        "calls": [
            {"name": call.get("name"), "count": call.get("count")}
            for call in calls
        ],
    }
    if include_full_source and code:
        fingerprint["full_source"] = code
    return fingerprint


def _state_code_by_section(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    storyboard = payload.get("storyboard") or []
    code = payload.get("code") or []
    return {
        str(section.get("id")): str(code[index] or "")
        for index, section in enumerate(storyboard)
        if isinstance(section, dict) and section.get("id") and index < len(code)
    }


def _phase_turn(phase: str) -> Optional[str]:
    match = re.match(r"turn_(\d+)", phase)
    if match:
        return str(int(match.group(1)))
    if phase.startswith("final_render"):
        return "final_render"
    return None


def _turn_state_codes(run_dir: Path, turn: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    state_dir = run_dir / "mas_state"
    if turn == "final_render":
        numeric_turns = []
        for path in state_dir.glob("turn_*_video_state.json"):
            match = re.fullmatch(r"turn_(\d+)_video_state\.json", path.name)
            if match:
                numeric_turns.append(int(match.group(1)))
        numeric_turns.sort()
        start = (
            _state_code_by_section(state_dir / f"turn_{numeric_turns[-1]:02d}_video_state.json")
            if numeric_turns
            else {}
        )
        end = _state_code_by_section(state_dir / "final_render_pass_video_state.json")
        return start, end

    turn_number = int(turn)
    end = _state_code_by_section(state_dir / f"turn_{turn_number:02d}_video_state.json")
    if turn_number <= 1:
        return {}, end
    previous = turn_number - 1
    start_path = state_dir / f"turn_{previous:02d}_orchestrator_video_state.json"
    if not start_path.exists():
        start_path = state_dir / f"turn_{previous:02d}_video_state.json"
    return _state_code_by_section(start_path), end


def _build_turn_code_trajectories(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Reconstruct compact per-turn code histories without sending whole source files."""
    final_state_path = _find_final_state_json(run_dir)
    final_state = _read_json(final_state_path) if final_state_path is not None else {}
    section_metadata = {
        str(section.get("id")): {
            "title": str(section.get("title") or ""),
            "lecture_lines": [
                str(line) for line in (section.get("lecture_lines") or [])
            ],
        }
        for section in (final_state.get("storyboard") or [])
        if isinstance(section, dict) and section.get("id")
    }

    replacements: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for sequence, event in enumerate(_read_jsonl(run_dir / "agent_function_calls.jsonl"), start=1):
        if event.get("tool_name") != "replace_code" or event.get("status") != "success":
            continue
        turn_number = event.get("turn_number")
        section_id = str((event.get("output") or {}).get("section_id") or "")
        raw_code = str((event.get("input") or {}).get("code") or "")
        if turn_number is None or not section_id or not raw_code:
            continue
        metadata = section_metadata.get(section_id, {})
        normalized_code = normalize_code_to_code2video(
            raw_code,
            section_id=section_id,
            section_title=str(metadata.get("title") or ""),
            lecture_lines=list(metadata.get("lecture_lines") or []),
        )
        replacements.setdefault((str(int(turn_number)), section_id), []).append(
            {
                "sequence": sequence,
                "timestamp": event.get("timestamp"),
                "agent": _normalize_role_label(event.get("agent")),
                "raw_code_sha1": hashlib.sha1(raw_code.encode("utf-8")).hexdigest(),
                "normalization_changed_code": normalized_code != raw_code,
                "code": normalized_code,
            }
        )

    attempts: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for attempt_path in sorted(run_dir.glob("coder_debugger/section_*/attempt_*/attempt.json")):
        payload = _read_json(attempt_path)
        phase = str(payload.get("phase") or "")
        turn = _phase_turn(phase)
        if turn is None:
            continue
        section_id = attempt_path.parent.parent.name
        code = str(payload.get("code_before") or "")
        render_outcome = payload.get("render_outcome") or {}
        render_error = str(payload.get("render_error") or "")
        attempts.setdefault((turn, section_id), []).append(
            {
                "attempt_number": payload.get("attempt_number"),
                "phase": phase,
                "code_sha1": hashlib.sha1(code.encode("utf-8")).hexdigest() if code else None,
                "outcome": {
                    "success": bool(
                        render_outcome.get("success", render_outcome.get("ok", False))
                    ),
                    "returncode": render_outcome.get("returncode"),
                    "timed_out": bool(render_outcome.get("timed_out", False)),
                    "elapsed_seconds": round(
                        float(payload.get("elapsed_seconds", 0.0) or 0.0), 3
                    ),
                },
                "error": _truncate_text(render_error, max_chars=1800),
                "error_code_excerpt": _error_code_excerpt(code, render_error),
                "repair_strategy": str(payload.get("repair_strategy") or ""),
                "incident_type": (
                    "infrastructure" if _is_manim_cache_race(render_error) else "code_or_render"
                ),
            }
        )

    keys = set(replacements) | set(attempts)
    trajectories: Dict[str, Dict[str, Any]] = {}
    sections_with_full_source: set[str] = set()
    for turn, section_id in sorted(
        keys, key=lambda item: (999999 if item[0] == "final_render" else int(item[0]), item[1])
    ):
        start_codes, end_codes = _turn_state_codes(run_dir, turn)
        start_code = start_codes.get(section_id, "")
        end_code = end_codes.get(section_id, "")
        current_code = start_code
        include_start_full_source = (
            bool(start_code)
            and turn != "final_render"
            and section_id not in sections_with_full_source
        )
        if include_start_full_source:
            sections_with_full_source.add(section_id)
        modifications: List[Dict[str, Any]] = []
        for replacement in replacements.get((turn, section_id), []):
            next_code = replacement["code"]
            if next_code == current_code:
                continue
            is_initial_generation = not current_code.strip()
            include_initial_full_source = (
                is_initial_generation and section_id not in sections_with_full_source
            )
            modification = {
                "sequence": len(modifications) + 1,
                "source_event_sequence": replacement["sequence"],
                "timestamp": replacement["timestamp"],
                "agent": replacement["agent"],
                "change_type": "initial_generation" if is_initial_generation else "agent_edit",
                "raw_agent_code_sha1": replacement["raw_code_sha1"],
                "normalization_changed_code": replacement["normalization_changed_code"],
                "before_sha1": _code_fingerprint(current_code)["sha1"],
                "after": _code_fingerprint(
                    next_code,
                    include_full_source=include_initial_full_source,
                ),
                "diff": _bounded_code_diff(current_code, next_code, max_chars=2500),
            }
            if include_initial_full_source:
                sections_with_full_source.add(section_id)
                modification["diff"] = ""
                modification["note"] = (
                    "Initial code generation; this section's complete normalized source is included once in after.full_source."
                )
            modifications.append(modification)
            current_code = next_code

        if end_code and end_code != current_code:
            modifications.append(
                {
                    "sequence": len(modifications) + 1,
                    "agent": "System/Unattributed",
                    "change_type": "turn_boundary_reconciliation",
                    "before_sha1": _code_fingerprint(current_code)["sha1"],
                    "after": _code_fingerprint(end_code),
                    "diff": _bounded_code_diff(current_code, end_code, max_chars=2500),
                    "note": "Observed in end-of-turn state but not attributable to a replace_code event.",
                }
            )

        turn_attempts = sorted(
            attempts.get((turn, section_id), []),
            key=lambda item: int(item.get("attempt_number") or 0),
        )
        trajectories.setdefault(turn, {})[section_id] = {
            "start_code": _code_fingerprint(
                start_code,
                include_full_source=include_start_full_source,
            ),
            "modifications": modifications,
            "render_attempts": turn_attempts,
            "end_code": _code_fingerprint(end_code or current_code),
            "association_note": (
                "Render attempts are linked to code versions exactly by code_sha1. "
                "A chronological edit after an error is not automatically claimed to be caused by it."
            ),
        }
    return trajectories


def _error_code_excerpt(code: str, render_error: str, *, radius: int = 10) -> str:
    """Include source around the final traceback line when one is available."""
    if not code or not render_error:
        return ""
    line_numbers = [int(value) for value in re.findall(r"\bline\s+(\d+)\b", render_error)]
    line_numbers.extend(int(value) for value in re.findall(r"\.py:(\d+)\b", render_error))
    if not line_numbers:
        return ""
    lines = code.splitlines()
    line_number = line_numbers[-1]
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    return "\n".join(
        f"{idx + 1}: {lines[idx] if len(lines[idx]) <= 300 else lines[idx][:297] + '...'}"
        for idx in range(start, end)
    )


@dataclass
class RunSummary:
    run_id: str
    run_dir: str
    topic: str
    pipeline_id: Optional[str]
    aes_overall: Optional[float]
    tq_learning_gain: Optional[float]
    combined_score: Optional[float]
    baseline_combined_score: Optional[float]
    relative_score_delta: Optional[float]
    turns_run: int
    total_issues: int
    unresolved_issue_count: int
    resolved_issue_count: int
    render_ok_count: int
    render_total: int
    token_total: Optional[int]
    call_count: Optional[int]
    scope_refine_attempts: int
    total_debug_attempts: int
    video_review_issue_count: int
    issue_counts_by_role: Dict[str, int] = field(default_factory=dict)
    section_attempts: Dict[str, SectionAttemptSummary] = field(default_factory=dict)
    belief_assessments: List[BeliefAssessment] = field(default_factory=list)
    reflection_model: Optional[str] = None
    reflection_usage: Dict[str, int] = field(default_factory=dict)
    reflection_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def render_success_rate(self) -> float:
        return (self.render_ok_count / self.render_total) if self.render_total else 0.0


def _belief_scope_from_dict(payload: Dict[str, Any]) -> BeliefScope:
    roles = [str(item) for item in payload.get("roles", []) if str(item).strip()]
    # Backward compatibility for libraries written before explicit multi-role
    # scope was required. Legacy general beliefs with no roles applied to all
    # MAS generation roles.
    if not roles and str(payload.get("level") or "").strip() == "general":
        roles = ["Orchestrator", "Coder", "AnimationPlanner", "ScriptWriter"]
    return BeliefScope(
        roles=roles,
        stages=[str(item) for item in payload.get("stages", []) if str(item).strip()],
        problem_description=str(payload.get("problem_description") or ""),
        context_conditions=[
            str(item) for item in payload.get("context_conditions", []) if str(item).strip()
        ],
    )


def _record_from_dict(payload: Dict[str, Any]) -> BeliefRecord:
    record = BeliefRecord(
        belief_id=str(payload.get("belief_id", payload.get("lesson_id"))),
        instruction=str(payload["instruction"]),
        scope=_belief_scope_from_dict(payload.get("scope", {})),
        status=str(payload.get("status", STATUS_ACTIVE)),
        alpha=float(payload.get("alpha", 1.0)),
        beta=float(payload.get("beta", 1.0)),
        support_count=int(payload.get("support_count", 0)),
        contradiction_count=int(payload.get("contradiction_count", 0)),
        relevant_count=int(payload.get("relevant_count", 0)),
        irrelevant_count=int(payload.get("irrelevant_count", 0)),
        weighted_support=float(payload.get("weighted_support", 0.0)),
        weighted_contradiction=float(payload.get("weighted_contradiction", 0.0)),
        confidence=float(payload.get("confidence", 0.5)),
        impact=_normalize_impact(payload.get("impact", "medium")),
        priority=float(payload.get("priority", 0.0)),
        belief_type=_normalize_belief_type(
            payload.get("belief_type", payload.get("lesson_type", "confirmed"))
        ),
        timing=_normalize_belief_timing(payload.get("timing", "both")),
        created_from_run=payload.get("created_from_run"),
        observed_runs=[str(item) for item in payload.get("observed_runs", [])],
        independent_topics=[str(item) for item in payload.get("independent_topics", [])],
        topic_evidence_weights={
            str(key): float(value)
            for key, value in (payload.get("topic_evidence_weights") or {}).items()
        },
        evidence=list(payload.get("evidence", [])),
    )
    record.update_confidence()
    return record


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_library(path: Path) -> Dict[str, BeliefRecord]:
    if path.exists():
        payload = _read_json(path)
        beliefs = payload.get("beliefs", payload.get("lessons", payload))
        records = [_record_from_dict(item) for item in beliefs]
        return {record.belief_id: record for record in records}
    return {}


def save_library(path: Path, beliefs: Dict[str, BeliefRecord], metadata: Dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "beliefs": [asdict(belief) for belief in sorted(beliefs.values(), key=lambda item: item.belief_id)],
    }
    _write_json(path, payload)


def save_evidence(path: Path, beliefs: Dict[str, BeliefRecord]) -> None:
    """Persist denormalised immutable-style evidence rows for audit and analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for belief in sorted(beliefs.values(), key=lambda item: item.belief_id):
            for index, evidence in enumerate(belief.evidence, start=1):
                row = {
                    "evidence_id": f"{belief.belief_id}-E{index:04d}",
                    "belief_id": belief.belief_id,
                    **evidence,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _iter_run_dirs_from_pipeline(pipeline_dir: Path) -> List[Path]:
    return sorted([path for path in pipeline_dir.iterdir() if path.is_dir()])


def _find_eval_json(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("*_eval.json"))
    return candidates[0] if candidates else None


def _find_final_state_json(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "mas_state" / "final_video_state.json",
        run_dir / "mas_state" / "final_render_pass_video_state.json",
    ]
    return _first_existing(candidates)


def _load_timeout_diagnostics(run_dir: Path, render_outcome: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(render_outcome.get("timed_out", False)):
        return {}
    artifacts = render_outcome.get("timeout_artifacts") or {}
    metadata_value = artifacts.get("metadata_path")
    if not metadata_value:
        return {}
    metadata_path = Path(str(metadata_value)).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = run_dir / metadata_path
    try:
        metadata_path = metadata_path.resolve()
        metadata_path.relative_to(run_dir.resolve())
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "timeout_seconds": metadata.get("timeout_seconds"),
        "command": metadata.get("command"),
        "stdout_tail": _truncate_text(str(metadata.get("stdout") or "")[-4000:], max_chars=4000),
        "stderr_tail": _truncate_text(str(metadata.get("stderr") or "")[-6000:], max_chars=6000),
    }


def _load_section_attempts(run_dir: Path) -> Dict[str, SectionAttemptSummary]:
    summaries: Dict[str, SectionAttemptSummary] = {}
    for attempt_path in sorted(run_dir.glob("coder_debugger/section_*/attempt_*/attempt.json")):
        payload = _read_json(attempt_path)
        section_id = attempt_path.parent.parent.name
        summary = summaries.setdefault(section_id, SectionAttemptSummary(section_id=section_id))
        summary.attempts += 1
        summary.elapsed_seconds += float(payload.get("elapsed_seconds", 0.0) or 0.0)
        strategy = str(payload.get("repair_strategy") or "")
        if "scope_refine" in strategy:
            summary.scope_refine_attempts += 1
        render_error = str(payload.get("render_error") or "").strip()
        is_infrastructure_incident = _is_manim_cache_race(render_error)
        if is_infrastructure_incident:
            summary.infrastructure_attempts += 1
        if render_error:
            summary.render_errors.append(render_error)
        render_outcome = payload.get("render_outcome") or {}
        if isinstance(render_outcome, dict):
            succeeded = bool(render_outcome.get("success", render_outcome.get("ok", False)))
            summary.successful = summary.successful or succeeded
        else:
            render_outcome = {}

        timeout_diagnostics = _load_timeout_diagnostics(run_dir, render_outcome)

        code_before = str(payload.get("code_before") or "")
        code_after = str(payload.get("code_after") or "")
        changed_code = bool(payload.get("changed_code", code_before != code_after))
        attempt_evidence = {
            "attempt_number": payload.get("attempt_number"),
            "phase": str(payload.get("phase") or "unspecified"),
            "elapsed_seconds": round(float(payload.get("elapsed_seconds", 0.0) or 0.0), 3),
            "render_outcome": {
                "success": bool(render_outcome.get("success", render_outcome.get("ok", False))),
                "returncode": render_outcome.get("returncode"),
                "timed_out": bool(render_outcome.get("timed_out", False)),
            },
            "render_error": _truncate_text(render_error, max_chars=1800),
            "repair_strategy": str(payload.get("repair_strategy") or ""),
            "repair_reason": _truncate_text(str(payload.get("repair_reason") or ""), max_chars=600),
            "changed_code": changed_code,
            "incident_type": "infrastructure" if is_infrastructure_incident else "code_or_render",
        }
        if timeout_diagnostics:
            attempt_evidence["timeout_diagnostics"] = timeout_diagnostics
        if not is_infrastructure_incident:
            attempt_evidence["calls_before"] = _summarize_code_calls(code_before)
            diagnostic_error = "\n".join(
                value
                for value in (
                    render_error,
                    str(timeout_diagnostics.get("stderr_tail") or ""),
                )
                if value
            )
            attempt_evidence["error_code_excerpt"] = _error_code_excerpt(code_before, diagnostic_error)
        if changed_code and not is_infrastructure_incident:
            attempt_evidence["calls_after"] = _summarize_code_calls(code_after)
            attempt_evidence["code_diff"] = _bounded_code_diff(code_before, code_after)
        summary.attempt_evidence.append(attempt_evidence)
    return summaries


def _load_pipeline_paper_average(pipeline_dir: Path) -> Optional[float]:
    summary_path = pipeline_dir / "pipeline_summary.json"
    if not summary_path.exists():
        return None
    payload = _read_json(summary_path)
    paper_row = payload.get("paper_row") or {}
    avg_value = paper_row.get("Avg")
    return float(avg_value) if avg_value is not None else None


def _combined_score(aes_overall: Optional[float], tq_learning_gain: Optional[float]) -> Optional[float]:
    values: List[float] = []
    if aes_overall is not None:
        values.append(float(aes_overall))
    if tq_learning_gain is not None:
        values.append(float(tq_learning_gain) * 100.0)
    return round(mean(values), 3) if values else None


def _parse_video_review_issue_count(video_review: Any) -> int:
    if not isinstance(video_review, list):
        return 0

    count = 0
    for section_review in video_review:
        analysis = (section_review or {}).get("analysis") or {}
        for category_payload in analysis.values():
            if not isinstance(category_payload, dict):
                continue
            if category_payload.get("has_issues"):
                improvements = category_payload.get("improvements") or []
                count += max(1, len(improvements))
    return count


def _summarize_video_review(video_review: Any) -> List[Dict[str, Any]]:
    if not isinstance(video_review, list):
        return []

    summary: List[Dict[str, Any]] = []
    for section_review in video_review:
        section_id = str((section_review or {}).get("section_id") or "")
        section_entry: Dict[str, Any] = {
            "section_id": section_id,
            "status": (section_review or {}).get("status"),
            "issues": [],
        }
        analysis = (section_review or {}).get("analysis") or {}
        for category_name, category_payload in analysis.items():
            if not isinstance(category_payload, dict) or not category_payload.get("has_issues"):
                continue
            improvements = category_payload.get("improvements") or []
            for improvement in improvements[:6]:
                section_entry["issues"].append(
                    {
                        "category": category_name,
                        "problem": improvement.get("problem"),
                        "solution": improvement.get("solution"),
                        "line_number": improvement.get("line_number"),
                    }
                )
        if section_entry["issues"]:
            summary.append(section_entry)
    return summary


def _summarize_storyboard(storyboard: Any) -> Dict[str, Any]:
    if not isinstance(storyboard, list):
        return {"section_count": 0, "empty_sections": [], "count_mismatches": [], "sections": []}

    sections: List[Dict[str, Any]] = []
    empty_sections: List[str] = []
    count_mismatches: List[Dict[str, Any]] = []
    for raw_section in storyboard:
        if not isinstance(raw_section, dict):
            continue
        section_id = str(raw_section.get("id") or "")
        lecture_lines = [str(value) for value in (raw_section.get("lecture_lines") or [])]
        animations = [str(value) for value in (raw_section.get("animations") or [])]
        if not lecture_lines or not animations:
            empty_sections.append(section_id)
        if len(lecture_lines) != len(animations):
            count_mismatches.append(
                {
                    "section_id": section_id,
                    "lecture_line_count": len(lecture_lines),
                    "animation_count": len(animations),
                }
            )
        paired_steps = []
        for index in range(max(len(lecture_lines), len(animations))):
            paired_steps.append(
                {
                    "step": index + 1,
                    "lecture_line": _truncate_text(lecture_lines[index], 220) if index < len(lecture_lines) else None,
                    "animation": _truncate_text(animations[index], 280) if index < len(animations) else None,
                }
            )
        sections.append(
            {
                "section_id": section_id,
                "title": _truncate_text(str(raw_section.get("title") or ""), 160),
                "lecture_line_count": len(lecture_lines),
                "animation_count": len(animations),
                "paired_steps": paired_steps,
            }
        )
    return {
        "section_count": len(sections),
        "empty_sections": empty_sections,
        "count_mismatches": count_mismatches,
        "sections": sections,
    }


def _summarize_evaluation(eval_payload: Dict[str, Any]) -> Dict[str, Any]:
    aes = eval_payload.get("aes") or {}
    tq = eval_payload.get("tq") or {}
    aes_result = aes.get("result") or {}
    tq_result = tq.get("result") or {}

    aes_scores = {
        key: value
        for key, value in aes_result.items()
        if key not in {"detailed_feedback", "knowledge_point"} and isinstance(value, (int, float))
    }
    detailed_responses: Dict[str, List[str]] = {}
    for stage, responses in (tq_result.get("detailed_responses") or {}).items():
        if not isinstance(responses, list):
            continue
        detailed_responses[str(stage)] = [_truncate_text(str(value), 500) for value in responses[:10]]

    return {
        "success": eval_payload.get("success"),
        "aes": {
            "ok": aes.get("ok"),
            "scores": aes_scores,
            "knowledge_point": aes_result.get("knowledge_point"),
            "detailed_feedback": _truncate_text(str(aes_result.get("detailed_feedback") or ""), 6000),
            "error": _truncate_text(str(aes.get("error") or ""), 1000),
        },
        "tq": {
            "ok": tq.get("ok"),
            "concept": tq_result.get("concept") or eval_payload.get("tq_concept"),
            "pre_unlearning_score": tq_result.get("pre_unlearning_score"),
            "post_unlearning_score": tq_result.get("post_unlearning_score"),
            "post_video_score": tq_result.get("post_video_score"),
            "learning_gain": tq_result.get("learning_gain"),
            "unlearning_success": tq_result.get("unlearning_success"),
            "detailed_responses": detailed_responses,
            "report": _truncate_text(str(tq.get("report") or ""), 3000),
            "error": _truncate_text(str(tq.get("error") or ""), 1000),
        },
    }


def _summarize_function_trace(
    run_dir: Path,
    max_events: Optional[int] = None,
) -> List[Dict[str, Any]]:
    trace_path = run_dir / "agent_function_calls.jsonl"
    events = _read_jsonl(trace_path)
    if not events:
        return []

    informative_events = [
        event
        for event in events
        if (
            event.get("text_parts")
            or event.get("function_calls")
            or event.get("function_responses")
            or event.get("error")
            or event.get("error_message")
        )
    ]
    selected = informative_events
    if max_events is not None and len(informative_events) > max_events:
        selected = (
            informative_events[: max_events // 2]
            + informative_events[-(max_events - max_events // 2) :]
        )
    trace_summary: List[Dict[str, Any]] = []
    for event in selected:
        text_parts = event.get("text_parts") or []
        joined_text = " ".join(str(part) for part in text_parts)
        cleaned_text = _truncate_text(_strip_code_blocks(joined_text), max_chars=500)
        function_calls = event.get("function_calls") or []
        function_responses = event.get("function_responses") or []
        summary_item = {
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "agent": _normalize_role_label(event.get("agent")),
                "model": event.get("model"),
                "summary": cleaned_text,
                "usage": event.get("usage") or {},
                "function_call_count": len(function_calls),
                "function_calls": [
                    {
                        "name": (
                            call.get("name")
                            or call.get("function")
                            or ((call.get("function") or {}).get("name") if isinstance(call.get("function"), dict) else None)
                            or "unknown"
                        ),
                        "arguments_summary": _compact_jsonish(
                            call.get("arguments")
                            or ((call.get("function") or {}).get("arguments") if isinstance(call.get("function"), dict) else None)
                            or call.get("args")
                            or {}
                        ),
                    }
                    for call in function_calls[:12]
                ],
                "function_responses": [
                    {
                        "name": (
                            response.get("name")
                            or response.get("function")
                            or response.get("tool_name")
                            or "unknown"
                        ),
                        "response_summary": _compact_jsonish(
                            response.get("response")
                            or response.get("content")
                            or response.get("result")
                            or response
                        ),
                    }
                    for response in function_responses[:12]
                ],
        }
        error = event.get("error") or event.get("error_message")
        if error:
            summary_item["error"] = _compact_jsonish(error, max_chars=500)
            summary_item["retrying"] = bool(event.get("retrying"))
        trace_summary.append(summary_item)
    return trace_summary


def _summarize_issues(issues: List[Dict[str, Any]], *, max_items: int = 20) -> Dict[str, Any]:
    unresolved: List[Dict[str, Any]] = []
    resolved: List[Dict[str, Any]] = []
    for issue in issues:
        item = {
            "id": issue.get("id"),
            "from_role": _normalize_role_label(issue.get("fromAgent")),
            "to_role": _normalize_role_label(issue.get("toAgent")),
            "section_id": issue.get("section_id"),
            "description": _truncate_text(str(issue.get("description") or ""), 240),
            "resolution_note": _truncate_text(str(issue.get("resolution_note") or ""), 240),
        }
        if issue.get("resolved"):
            if len(resolved) < max_items:
                resolved.append(item)
        elif len(unresolved) < max_items:
            unresolved.append(item)
    return {"unresolved": unresolved, "resolved": resolved}


def _build_reflection_context(run: RunSummary, state_payload: Dict[str, Any], beliefs: Dict[str, BeliefRecord]) -> Dict[str, Any]:
    existing_beliefs = [
        {
            "belief_id": belief.belief_id,
            "instruction": belief.instruction,
            "scope": asdict(belief.scope),
            "status": belief.status,
            "confidence": belief.confidence,
            "impact": belief.impact,
            "priority": belief.priority,
            "belief_type": belief.belief_type,
            "timing": belief.timing,
            "support_count": belief.support_count,
            "contradiction_count": belief.contradiction_count,
            "relevant_count": belief.relevant_count,
        }
        for belief in sorted(beliefs.values(), key=lambda item: item.belief_id)
    ]

    section_outcomes = {}
    for section_id, summary in sorted(run.section_attempts.items()):
        section_outcomes[section_id] = {
            "attempts": summary.attempts,
            "scope_refine_attempts": summary.scope_refine_attempts,
            "infrastructure_attempts": summary.infrastructure_attempts,
            "successful": summary.successful,
            "elapsed_seconds": round(summary.elapsed_seconds, 3),
            "render_errors": summary.render_errors[:5],
        }

    return {
        "run_summary": {
            "run_id": run.run_id,
            "topic": run.topic,
            "pipeline_id": run.pipeline_id,
            "aes_overall": run.aes_overall,
            "tq_learning_gain": run.tq_learning_gain,
            "combined_score": run.combined_score,
            "baseline_combined_score": run.baseline_combined_score,
            "relative_score_delta": run.relative_score_delta,
            "turns_run": run.turns_run,
            "render_ok_count": run.render_ok_count,
            "render_total": run.render_total,
            "render_success_rate": round(run.render_success_rate, 4),
            "total_issues": run.total_issues,
            "resolved_issue_count": run.resolved_issue_count,
            "unresolved_issue_count": run.unresolved_issue_count,
            "token_total": run.token_total,
            "call_count": run.call_count,
            "scope_refine_attempts": run.scope_refine_attempts,
            "total_debug_attempts": run.total_debug_attempts,
            "video_review_issue_count": run.video_review_issue_count,
            "issue_counts_by_role": run.issue_counts_by_role,
        },
        "coder_assignments": {
            section_id: _normalize_role_label(role_name)
            for section_id, role_name in (state_payload.get("coder_assignments") or {}).items()
        },
        "issues": _summarize_issues(state_payload.get("issues") or []),
        "storyboard_summary": _summarize_storyboard(state_payload.get("storyboard")),
        "evaluation_summary": _summarize_evaluation(
            _read_json(eval_path) if (eval_path := _find_eval_json(Path(run.run_dir))) is not None else {}
        ),
        "section_outcomes": section_outcomes,
        "turn_code_trajectories": _build_turn_code_trajectories(Path(run.run_dir)),
        "video_review_summary": _summarize_video_review(state_payload.get("video_review")),
        "function_trace_summary": _global_trace_events(
            _summarize_function_trace(Path(run.run_dir))
        ),
        "existing_beliefs": existing_beliefs,
    }


def _issues_for_section(issues: Dict[str, Any], section_id: str) -> Dict[str, Any]:
    return {
        status: [
            item
            for item in (issues.get(status) or [])
            if item.get("section_id") == section_id
        ]
        for status in ("unresolved", "resolved")
    }


def _compact_section_attempts(section_summary: Dict[str, Any]) -> Dict[str, Any]:
    attempts = list(section_summary.get("attempt_evidence") or [])
    if len(attempts) > 6:
        attempts = attempts[:2] + attempts[-4:]

    compact_attempts = []
    for attempt in attempts:
        compact = dict(attempt)
        for key in ("calls_before", "calls_after"):
            if isinstance(compact.get(key), list):
                compact[key] = compact[key][:10]
        diagnostics = compact.get("timeout_diagnostics")
        if isinstance(diagnostics, dict):
            compact["timeout_diagnostics"] = {
                "elapsed_seconds": diagnostics.get("elapsed_seconds"),
                "timeout_seconds": diagnostics.get("timeout_seconds"),
                "command": diagnostics.get("command"),
                "stdout_tail": _truncate_text(
                    str(diagnostics.get("stdout_tail") or ""), 500
                ),
                "stderr_tail": _truncate_text(
                    str(diagnostics.get("stderr_tail") or ""), 1400
                ),
            }
        for key in ("code_diff", "repair_diff", "error_code_excerpt"):
            if compact.get(key):
                compact[key] = _truncate_text(str(compact[key]), 2500)
        compact_attempts.append(compact)

    result = {
        key: value
        for key, value in section_summary.items()
        if key != "attempt_evidence"
    }
    result["attempt_evidence"] = compact_attempts
    result["split_prompt_omitted_attempt_count"] = max(
        0, len(section_summary.get("attempt_evidence") or []) - len(compact_attempts)
    )
    return result


def _section_reflection_context(
    full_context: Dict[str, Any],
    section_id: str,
) -> Dict[str, Any]:
    storyboard = full_context.get("storyboard_summary") or {}
    section_storyboards = [
        item
        for item in (storyboard.get("sections") or [])
        if item.get("section_id") == section_id
    ]
    section_reviews = [
        item
        for item in (full_context.get("video_review_summary") or [])
        if item.get("section_id") == section_id
    ]
    section_trajectories = {
        turn: sections[section_id]
        for turn, sections in (full_context.get("turn_code_trajectories") or {}).items()
        if section_id in sections
    }
    return {
        "reflection_scope": {
            "type": "section",
            "section_id": section_id,
            "instruction": (
                "Assess every distinct reusable mechanism evidenced in this section only. "
                "Do not infer topic-wide pedagogical conclusions from section evidence."
            ),
        },
        "run_summary": full_context.get("run_summary") or {},
        "coder_assignments": {
            section_id: (full_context.get("coder_assignments") or {}).get(section_id)
        },
        "issues": _issues_for_section(full_context.get("issues") or {}, section_id),
        "storyboard_summary": {
            "section_count": len(section_storyboards),
            "sections": section_storyboards,
        },
        "section_outcome": {
            key: value
            for key, value in (
                (full_context.get("section_outcomes") or {}).get(section_id, {})
            ).items()
        },
        "turn_code_trajectories": section_trajectories,
        "video_review_summary": section_reviews,
        # The turn trajectories replace raw Coder source, duplicate agent
        # responses, and isolated attempt evidence with ordered changes and
        # the render outcomes for the exact code hashes.
        "existing_beliefs": full_context.get("existing_beliefs") or [],
    }


def _global_trace_events(events: List[Dict[str, Any]], *, max_events: int = 60) -> List[Dict[str, Any]]:
    allowed_roles = {"ScriptWriter", "AnimationPlanner", "Orchestrator"}
    informative = [
        event
        for event in events
        if event.get("agent") in allowed_roles
        and (
            event.get("summary")
            or event.get("function_calls")
            or event.get("function_responses")
        )
    ]
    if len(informative) <= max_events:
        return informative
    split = max_events // 2
    return informative[:split] + informative[-(max_events - split) :]


def _topic_reflection_context(full_context: Dict[str, Any]) -> Dict[str, Any]:
    issues = full_context.get("issues") or {}
    global_issues = {
        status: [
            item
            for item in (issues.get(status) or [])
            if not item.get("section_id") or item.get("to_role") != "Coder"
        ]
        for status in ("unresolved", "resolved")
    }
    section_outcomes = {
        section_id: {
            key: value
            for key, value in summary.items()
            if key != "attempt_evidence"
        }
        for section_id, summary in (full_context.get("section_outcomes") or {}).items()
    }
    return {
        "reflection_scope": {
            "type": "topic",
            "instruction": (
                "Assess topic-wide ScriptWriter, AnimationPlanner, Orchestrator, "
                "pedagogical, evaluation, consistency, and cross-section mechanisms. "
                "Do not repeat section-specific Coder repair mechanisms."
            ),
        },
        "run_summary": full_context.get("run_summary") or {},
        "issues": global_issues,
        "storyboard_summary": full_context.get("storyboard_summary") or {},
        "evaluation_summary": full_context.get("evaluation_summary") or {},
        "section_outcomes": section_outcomes,
        "function_trace_summary": _global_trace_events(
            full_context.get("function_trace_summary") or []
        ),
        "existing_beliefs": full_context.get("existing_beliefs") or [],
    }


def _build_reflection_prompt(run: RunSummary, context_payload: Dict[str, Any]) -> str:
    # Gemini can parse compact JSON; pretty-print whitespace materially inflates
    # large turn trajectories without adding evidence.
    context_json = json.dumps(
        context_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""
You are maintaining a reusable belief library for a multi-agent MAS pipeline.

Your job:
1. Read the run evidence carefully.
2. Evaluate existing beliefs when relevant.
3. Create new beliefs organically when the run contains reusable evidence.

Important rules:
- All belief generation and belief management must be evidence-based.
- Do not rely on predefined starter beliefs.
- Scope beliefs by role, not by agent instance. Example: use "Coder", never "Coder3".
- Every belief must list one or more explicitly applicable roles. A cross-role belief lists every applicable role; there is no "general" scope.
- Scope every belief with roles, stages, a reusable problem description, and machine-readable context conditions.
- Do not leave roles empty. Use lowercase stage labels such as "planning", "coding", "debugging", "rendering", or "evaluation".
- Each belief must be actionable, general, concise, and reusable across runs.
- Base every belief on explicit evidence in the supplied context. Clearly distinguish observed facts from inferred causes.
- Only claim that an action fixed a problem when the evidence shows that action occurred and was followed by an improved outcome. A later success alone does not prove causation.
- Prefer beliefs that identify the underlying reusable mechanism rather than a topic-specific symptom, fixed numerical threshold, section number, or isolated example.
- Generalize only as far as the evidence supports. Preserve uncertainty when multiple explanations remain plausible.
- Before adding a belief, compare it with existing beliefs. SUPPORT, REVISE, MERGE, or CONTRADICT an existing belief when appropriate instead of creating a near-duplicate.
- When evidence contradicts part of an existing belief, narrow or revise the belief rather than supporting it unchanged.
- Evaluate belief importance separately from evidence confidence. Consider failure severity, recovery cost, repeated work, and downstream impact, not only how frequently a pattern appears.
- Do not dismiss an event solely because it occurred once. For a rare high-impact event with uncertain causality, create a cautiously worded precaution or hypothesis rather than asserting an unproven universal rule.
- Use belief_type "confirmed" for demonstrated mechanisms, "precaution" for safe mitigations to rare costly risks, "hypothesis" for unresolved patterns that should not yet guide agents, and "quality" for non-failing output improvements.
- Classify every belief's timing as "preventative", "reactive", or "both":
  - preventative: guidance primarily useful before execution to avoid a foreseeable failure or quality defect;
  - reactive: recovery or fallback guidance that should only be applied after a matching observed failure;
  - both: guidance that is safe and useful before execution and is also a direct repair for a matching failure.
- Prefer reactive for fallbacks, degradation strategies, retries, and recovery actions whose premature use could reduce quality. Do not choose both merely because a belief could theoretically be remembered before a failure.
- Before assigning timing, apply this counterfactual test to the exact instruction as written:
  1. Does following it require an observed failure, timeout, unsuccessful attempt, or diagnosed quality defect?
  2. Would following it before such evidence unnecessarily constrain a valid approach or reduce quality/capability?
  3. Is the exact same action, without changing its conditions or strength, appropriate both before execution and after a matching failure?
- Choose reactive when either of the first two answers is yes. Choose both only when the third answer is clearly yes. Otherwise choose preventative.
- Set evidence_confidence from the strength of the causal evidence. Set impact from the likely consequence if the event recurs. The system calculates final injection priority separately.
- Classify strategy_application as "full", "partial", "none", or "unclear" according to whether the observed action implemented the stated strategy.
- Classify attribution_strength as "strong", "moderate", "weak", "none", or "unclear" after accounting for simultaneous unrelated changes.
- Classify evidence_reliability as "direct", "corroborated", "indirect", "inferred", or "unverifiable". Exact matching render/error/code evidence is direct; semantic interpretation without direct verification is inferred.
- Classify outcome_improvement as "resolved", "improved", "unchanged", "worsened", or "unclear" for the targeted problem.
- Do not output numeric values for these four classifications. Deterministic local code maps the options to BBN inputs and performs the Bayesian update.
- Assign impact using these consequence-based definitions:
  - critical: can terminate the pipeline, corrupt or lose outputs, make a run unrecoverable, or create an uncontrolled external effect;
  - high: causes a timeout, exhausts or is likely to exhaust retries, repeatedly blocks progress, or creates substantial compute or API cost;
  - medium: causes a deterministic but recoverable failure that normally requires a repair or retry;
  - low: affects presentation quality, style, clarity, or minor efficiency without preventing successful completion.
- Do not mark a belief high or critical merely because an exception occurred. Base impact on the observed or reasonably expected operational consequence, independently of confidence and frequency.
- Prefer a small set of distinct, high-value beliefs over many overlapping low-impact beliefs.
- Keep instruction, reason, evidence, and problem_description concise.
- Perform a consolidation pass before returning assessments. Compare proposed beliefs with every existing belief and with each other by underlying mechanism and mitigation, not merely by wording.
- If an existing belief already covers the same mechanism, use SUPPORT instead of ADD.
- If new evidence changes the appropriate scope, specificity, or wording of an existing belief, use REVISE instead of adding a variant.
- If multiple existing beliefs describe the same underlying mechanism or substantially overlapping mitigations, use MERGE and provide one complete consolidated instruction.
- Prefer one adaptable principle over separate beliefs for different examples, fixed coordinates, numeric thresholds, API instances, or topic-specific manifestations of the same mechanism.
- Do not merge beliefs that have genuinely different causes, consequences, roles, or required mitigations merely to reduce the belief count.
- Produce at most one ADD assessment for each distinct underlying mechanism found in the current run.
- Consider evidence and potential beliefs for every participating role. Do not default to the role with the most detailed logs.
- For existing beliefs, choose one action: SUPPORT, CONTRADICT, IRRELEVANT, REVISE, or MERGE.
- For new beliefs, choose action ADD and set belief_id to null.
- For REVISE, belief_id must identify the belief to rewrite and instruction must contain the complete replacement instruction.
- For MERGE, belief_id must identify the surviving belief, merge_belief_ids must identify the other existing beliefs to absorb, and instruction must contain the complete consolidated instruction.
- Evidence confidence must be between 0 and 1.
- Keep evidence short and grounded in the supplied run context only.
- Inspect turn_code_trajectories for code-specific patterns. Each section is organised by turn as start code metadata, ordered modifications, renders of exact code hashes, and end code metadata.
- The complete normalized source appears only once per section across the run: in the earliest start_code.full_source, or in the first initial_generation modification's after.full_source when that section starts with no code. Later versions and later turns are represented by continuous diffs and boundary hashes only.
- Use modification diffs, call summaries, error excerpts, and matching code_sha1 values together. Do not treat chronological proximity alone as proof that an edit fixed an error.
- Prefer actionable Coder beliefs that name the relevant Manim/Python API or coding pattern when repeated evidence or a successful repair supports that conclusion.
- Do not claim that a function caused a timeout merely because it appears in the code. Require repeated association, traceback evidence, or a before/after change followed by a successful render.
- Treat a repair as successful evidence only when a later attempt for the same section renders successfully.
- Never derive a Coder belief from attempt evidence whose incident_type is "infrastructure"; it describes the runtime environment rather than generated code.

Before returning each assessment, verify:
1. What was directly observed?
2. What causal relationship, if any, was demonstrated?
3. Is the belief more general than the evidence permits?
4. Does an existing belief already cover it?
5. Is its proposed importance proportional to its likely impact?
6. Does it overlap with another proposed or existing belief that should be supported, revised, or merged instead?
7. Is its timing genuinely preventative, reactive, or useful in both contexts?

Return JSON only in this exact shape:
{{
  "assessments": [
    {{
      "belief_id": "B001" or null,
      "instruction": "Actionable belief text",
      "scope": {{
        "roles": ["Orchestrator"] or ["Coder"] or ["AnimationPlanner"] or ["ScriptWriter"] or a multi-role list,
        "stages": ["debugging", "rendering"],
        "problem_description": "Reusable description of the problem this belief addresses",
        "context_conditions": ["short_machine_readable_condition"]
      }},
      "applicable": true,
      "compliance": "followed" or "violated" or "mixed" or "unclear" or "observed_pattern",
      "outcome": "positive" or "negative" or "mixed" or "unclear",
      "action": "ADD" or "SUPPORT" or "CONTRADICT" or "IRRELEVANT" or "REVISE" or "MERGE",
      "merge_belief_ids": ["L002", "L003"] or [],
      "evidence_confidence": 0.0,
      "strategy_application": "full" or "partial" or "none" or "unclear",
      "attribution_strength": "strong" or "moderate" or "weak" or "none" or "unclear",
      "evidence_reliability": "direct" or "corroborated" or "indirect" or "inferred" or "unverifiable",
      "outcome_improvement": "resolved" or "improved" or "unchanged" or "worsened" or "unclear",
      "impact": "low" or "medium" or "high" or "critical",
      "belief_type": "confirmed" or "precaution" or "hypothesis" or "quality",
      "timing": "preventative" or "reactive" or "both",
      "agent": "Coder" or null,
      "section_id": "section_1" or null,
      "reason": "Short explanation",
      "evidence": "Short concrete evidence summary"
    }}
  ]
}}

Run being analysed: {run.run_id}
Topic: {run.topic}

Run context:
```json
{context_json}
```
""".strip()


def _llm_reflect_context(
    run: RunSummary,
    *,
    context_payload: Dict[str, Any],
    client: Client,
    call_label: str,
) -> Tuple[List[BeliefAssessment], Dict[str, int], Dict[str, Any]]:
    prompt = _build_reflection_prompt(run, context_payload)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    parsed: Optional[ReflectionResponsePayload] = None
    last_error: Optional[ValidationError] = None
    attempts_made = 0
    for attempt in range(1, 3):
        attempts_made = attempt
        attempt_prompt = prompt
        if attempt > 1:
            validation_feedback = ""
            if last_error is not None:
                compact_errors = [
                    {
                        "location": ".".join(str(value) for value in error.get("loc", [])),
                        "message": str(error.get("msg") or "")[:240],
                        "type": error.get("type"),
                    }
                    for error in last_error.errors()[:12]
                ]
                validation_feedback = (
                    "\nValidation errors to correct:\n"
                    + json.dumps(compact_errors, ensure_ascii=False, separators=(",", ":"))
                )
            attempt_prompt += """

RETRY REQUIREMENT:
The previous structured response was invalid or truncated. Return a fresh,
complete JSON object from the beginning. Do not omit relevant distinct
assessments, but consolidate duplicates and keep every prose field under 400
characters. Use only the listed categorical evidence options. Do not include
commentary outside the JSON object.
""".rstrip() + validation_feedback
        response = create_interaction(
            client,
            model=DEFAULT_REFLECTION_MODEL,
            input_value=attempt_prompt,
            response_schema=ReflectionResponsePayload,
            max_output_tokens=12000,
        )
        attempt_usage = response_usage_dict(response)
        for key in usage:
            usage[key] += attempt_usage.get(key, 0)
        try:
            response_text = _extract_structured_json(response.output_text)
            parsed = ReflectionResponsePayload.model_validate_json(response_text)
            break
        except ValidationError as exc:
            last_error = exc
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_label)
            invalid_path = (
                Path(run.run_dir)
                / f"belief_reflection_{safe_label}_invalid_response_{attempt}.txt"
            )
            invalid_path.write_text(response.output_text, encoding="utf-8")

    if parsed is None:
        raise ValueError(
            "Gemini returned invalid structured belief reflection JSON twice. "
            f"Raw responses were saved under {run.run_dir}."
        ) from last_error

    assessments: List[BeliefAssessment] = []
    for item in parsed.assessments:
        action = item.action
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported belief assessment action: {action}")
        evidence_confidence = item.evidence_confidence
        evidence_values = _evidence_categories_to_values(
            item.strategy_application,
            item.attribution_strength,
            item.evidence_reliability,
            item.outcome_improvement,
        )
        assessments.append(
            BeliefAssessment(
                run_id=run.run_id,
                belief_id=item.belief_id,
                instruction=item.instruction,
                applicable=item.applicable,
                compliance=item.compliance,
                outcome=item.outcome,
                action=action,
                weight=evidence_confidence,
                scope=BeliefScope(
                    roles=list(item.scope.roles),
                    stages=list(item.scope.stages),
                    problem_description=item.scope.problem_description,
                    context_conditions=list(item.scope.context_conditions),
                ),
                evidence_confidence=evidence_confidence,
                impact=item.impact,
                belief_type=item.belief_type,
                timing=item.timing,
                agent=item.agent,
                section_id=item.section_id,
                reason=item.reason,
                evidence=_coerce_evidence_payload(item.evidence),
                merge_belief_ids=list(item.merge_belief_ids),
                topic=run.topic,
                strategy_application=item.strategy_application,
                attribution_strength=item.attribution_strength,
                evidence_reliability=item.evidence_reliability,
                outcome_improvement=item.outcome_improvement,
                **evidence_values,
                reflection_call=call_label,
            )
        )
    call_metrics = {
        "call_label": call_label,
        "scope_type": (context_payload.get("reflection_scope") or {}).get("type"),
        "section_id": (context_payload.get("reflection_scope") or {}).get("section_id"),
        "prompt_characters": len(prompt),
        "prompt_words": len(prompt.split()),
        "api_attempts": attempts_made,
        "assessment_count": len(assessments),
        **usage,
    }
    return assessments, usage, call_metrics


def _llm_reflect_run(
    run: RunSummary,
    beliefs: Dict[str, BeliefRecord],
    *,
    state_payload: Dict[str, Any],
) -> Tuple[List[BeliefAssessment], Dict[str, int], List[Dict[str, Any]]]:
    context_payload = _build_reflection_context(run, state_payload, beliefs)
    api_key = cfg("gemini", "api_key")
    if not api_key:
        raise ValueError("Missing gemini.api_key in api_config.json or GEMINI_API_KEY")
    client = Client(api_key=api_key)

    assessments, usage, metrics = _llm_reflect_context(
        run,
        context_payload=context_payload,
        client=client,
        call_label="topic",
    )
    return assessments, usage, [metrics]


def _find_belief_by_instruction(
    beliefs: Dict[str, BeliefRecord],
    instruction: str,
) -> Optional[BeliefRecord]:
    target = _normalize_instruction(instruction)
    for record in beliefs.values():
        if _normalize_instruction(record.instruction) == target:
            return record
    return None


def _next_belief_id(beliefs: Dict[str, BeliefRecord]) -> str:
    max_numeric = 0
    for belief_id in beliefs:
        match = re.fullmatch(r"[BL](\d+)", belief_id)
        if match:
            max_numeric = max(max_numeric, int(match.group(1)))
    return f"B{max_numeric + 1:03d}"


def summarise_run(run_dir: Path, baseline_combined_score: Optional[float]) -> RunSummary:
    state_path = _find_final_state_json(run_dir)
    if state_path is None:
        raise FileNotFoundError(f"Could not find final MAS state in {run_dir}")

    state = _read_json(state_path)
    eval_payload: Dict[str, Any] = {}
    eval_path = _find_eval_json(run_dir)
    if eval_path is not None:
        eval_payload = _read_json(eval_path)

    aes_result = ((eval_payload.get("aes") or {}).get("result") or {})
    tq_result = ((eval_payload.get("tq") or {}).get("result") or {})
    aes_overall = aes_result.get("overall_score")
    tq_learning_gain = tq_result.get("learning_gain")
    combined = _combined_score(
        float(aes_overall) if aes_overall is not None else None,
        float(tq_learning_gain) if tq_learning_gain is not None else None,
    )

    render_status = state.get("render_status") or []
    render_ok_count = sum(1 for item in render_status if str(item).lower() == "ok")
    render_total = len(render_status)

    issues = state.get("issues") or []
    unresolved_issue_count = sum(1 for item in issues if not item.get("resolved"))
    resolved_issue_count = sum(1 for item in issues if item.get("resolved"))

    issue_counts_by_role = Counter(_normalize_role_label(item.get("toAgent")) or "unknown" for item in issues)
    token_totals = ((state.get("token_usage") or {}).get("totals") or {})
    section_attempts = _load_section_attempts(run_dir)

    relative_delta = None
    if combined is not None and baseline_combined_score is not None:
        relative_delta = round(combined - baseline_combined_score, 4)

    return RunSummary(
        run_id=str(run_dir.relative_to(DEFAULT_LOGS_ROOT)),
        run_dir=str(run_dir),
        topic=str(state.get("topic") or run_dir.name),
        pipeline_id=run_dir.parent.name if run_dir.parent.name.startswith("pipeline_") else None,
        aes_overall=float(aes_overall) if aes_overall is not None else None,
        tq_learning_gain=float(tq_learning_gain) if tq_learning_gain is not None else None,
        combined_score=combined,
        baseline_combined_score=baseline_combined_score,
        relative_score_delta=relative_delta,
        turns_run=int(state.get("turns_run") or 0),
        total_issues=len(issues),
        unresolved_issue_count=unresolved_issue_count,
        resolved_issue_count=resolved_issue_count,
        render_ok_count=render_ok_count,
        render_total=render_total,
        token_total=token_totals.get("total_tokens"),
        call_count=token_totals.get("call_count"),
        scope_refine_attempts=sum(item.scope_refine_attempts for item in section_attempts.values()),
        total_debug_attempts=sum(item.attempts for item in section_attempts.values()),
        video_review_issue_count=_parse_video_review_issue_count(state.get("video_review")),
        issue_counts_by_role=dict(issue_counts_by_role),
        section_attempts=section_attempts,
    )


def apply_assessments(
    beliefs: Dict[str, BeliefRecord],
    assessments: List[BeliefAssessment],
) -> Dict[str, Any]:
    summary = Counter()

    for assessment in assessments:
        summary[f"action:{assessment.action.lower()}"] += 1

        if assessment.action == ACTION_IRRELEVANT:
            if assessment.belief_id and assessment.belief_id in beliefs:
                beliefs[assessment.belief_id].irrelevant_count += 1
            continue

        record: Optional[BeliefRecord] = None
        if assessment.belief_id:
            record = beliefs.get(assessment.belief_id)
        elif assessment.action == ACTION_ADD:
            record = _find_belief_by_instruction(beliefs, assessment.instruction)

        if record is None and assessment.action == ACTION_ADD:
            belief_id = _next_belief_id(beliefs)
            record = BeliefRecord(
                belief_id=belief_id,
                instruction=assessment.instruction,
                scope=assessment.scope,
                impact=assessment.impact,
                belief_type=assessment.belief_type,
                timing=assessment.timing,
                created_from_run=assessment.run_id,
            )
            beliefs[belief_id] = record
            assessment.belief_id = belief_id
            summary["beliefs_created"] += 1

        if record is None:
            continue

        previous_instruction: Optional[str] = None
        if assessment.action in {ACTION_REVISE, ACTION_MERGE}:
            previous_instruction = record.instruction
            record.instruction = assessment.instruction
            record.scope = assessment.scope
            record.belief_type = assessment.belief_type
            record.timing = assessment.timing
            summary["beliefs_revised"] += 1

        record.impact = _impact_max(record.impact, assessment.impact)
        if assessment.action not in {ACTION_REVISE, ACTION_MERGE}:
            record.belief_type = _belief_type_max(record.belief_type, assessment.belief_type)
        if assessment.action in {ACTION_ADD, ACTION_SUPPORT}:
            # Timing describes when the current instruction should be used,
            # rather than an accumulating severity signal. Let a fresh
            # assessment classify legacy records that previously defaulted to
            # ``both``.
            record.timing = assessment.timing

        if assessment.action == ACTION_MERGE:
            merged_ids: List[str] = []
            for merge_id in dict.fromkeys(assessment.merge_belief_ids):
                if merge_id == record.belief_id:
                    continue
                merged = beliefs.get(merge_id)
                if merged is None:
                    continue
                record.alpha += merged.alpha - 1.0
                record.beta += merged.beta - 1.0
                record.support_count += merged.support_count
                record.contradiction_count += merged.contradiction_count
                record.relevant_count += merged.relevant_count
                record.irrelevant_count += merged.irrelevant_count
                record.weighted_support += merged.weighted_support
                record.weighted_contradiction += merged.weighted_contradiction
                record.impact = _impact_max(record.impact, merged.impact)
                record.belief_type = _belief_type_max(record.belief_type, merged.belief_type)
                record.observed_runs.extend(merged.observed_runs)
                record.evidence.extend(merged.evidence)
                merged_ids.append(merge_id)
                del beliefs[merge_id]
            summary["beliefs_merged"] += len(merged_ids)
            assessment.merge_belief_ids = merged_ids

        record.relevant_count += 1
        record.observed_runs.append(assessment.run_id)
        record.evidence.append(
            {
                "run_id": assessment.run_id,
                "action": assessment.action,
                "weight": assessment.weight,
                "evidence_confidence": assessment.evidence_confidence,
                "impact": assessment.impact,
                "belief_type": assessment.belief_type,
                "timing": assessment.timing,
                "reason": assessment.reason,
                "compliance": assessment.compliance,
                "outcome": assessment.outcome,
                "agent": assessment.agent,
                "section_id": assessment.section_id,
                "topic": assessment.topic,
                "strategy_application": assessment.strategy_application,
                "attribution_strength": assessment.attribution_strength,
                "evidence_reliability": assessment.evidence_reliability,
                "outcome_improvement": assessment.outcome_improvement,
                "strategy_applied_probability": assessment.strategy_applied_probability,
                "attribution_probability": assessment.attribution_probability,
                "reliability_probability": assessment.reliability_probability,
                "improvement": assessment.improvement,
                "reflection_call": assessment.reflection_call,
                "previous_instruction": previous_instruction,
                "merge_belief_ids": assessment.merge_belief_ids,
                "evidence": assessment.evidence,
            }
        )

        if assessment.action in {
            ACTION_ADD,
            ACTION_SUPPORT,
            ACTION_REVISE,
            ACTION_MERGE,
            ACTION_CONTRADICT,
        } and assessment.outcome_improvement != "unclear":
            topic_key = assessment.topic or assessment.run_id
            consumed_topic_weight = record.topic_evidence_weights.get(topic_key, 0.0)
            transition = TransitionEvidence(
                p_applicable=1.0 if assessment.applicable else 0.0,
                p_strategy_applied=assessment.strategy_applied_probability,
                p_attributable=assessment.attribution_probability,
                p_reliable=assessment.reliability_probability,
                improvement=assessment.improvement,
            )
            posterior = update_beta_posterior(
                record.alpha,
                record.beta,
                transition,
                remaining_topic_weight=max(0.0, 1.0 - consumed_topic_weight),
            )
            record.alpha = posterior["alpha"]
            record.beta = posterior["beta"]
            record.topic_evidence_weights[topic_key] = (
                consumed_topic_weight + posterior["evidence_weight"]
            )
            if topic_key not in record.independent_topics:
                record.independent_topics.append(topic_key)

            if assessment.action == ACTION_CONTRADICT or assessment.improvement < 0.5:
                record.contradiction_count += 1
                record.weighted_contradiction += posterior["beta_increment"]
            else:
                record.support_count += 1
                record.weighted_support += posterior["alpha_increment"]

            record.evidence[-1]["bayesian_update"] = posterior

        record.update_confidence()

    summary["total_assessments"] = len(assessments)
    return dict(summary)


def _state_for_run(run_dir: Path) -> Dict[str, Any]:
    state_path = _find_final_state_json(run_dir)
    if state_path is None:
        return {}
    return _read_json(state_path)


def analyse_single_run(run_dir: Path, beliefs: Dict[str, BeliefRecord]) -> Dict[str, Any]:
    pipeline_id = run_dir.parent.name if run_dir.parent.name.startswith("pipeline_") else None
    baseline = _load_pipeline_paper_average(run_dir.parent) if pipeline_id else None
    run_summary = summarise_run(run_dir, baseline)
    state_payload = _state_for_run(run_dir)
    assessments, usage, call_metrics = _llm_reflect_run(
        run_summary, beliefs, state_payload=state_payload
    )
    run_summary.belief_assessments = assessments
    run_summary.reflection_model = DEFAULT_REFLECTION_MODEL
    run_summary.reflection_usage = usage
    run_summary.reflection_calls = call_metrics
    return {
        "mode": "single_run",
        "run_summary": _run_summary_to_json(run_summary),
    }


def _pipeline_analysis_payload(
    pipeline_dir: Path,
    runs: List[RunSummary],
    *,
    expected_run_count: int,
    complete: bool,
) -> Dict[str, Any]:
    pipeline_scores = [run.combined_score for run in runs if run.combined_score is not None]
    return {
        "mode": "pipeline",
        "pipeline_dir": str(pipeline_dir),
        "run_count": len(runs),
        "expected_run_count": expected_run_count,
        "complete": complete,
        "average_combined_score": round(mean(pipeline_scores), 3) if pipeline_scores else None,
        "average_aes_overall": _safe_mean(run.aes_overall for run in runs),
        "average_tq_learning_gain": _safe_mean(run.tq_learning_gain for run in runs),
        "runs": [_run_summary_to_json(run) for run in runs],
    }


def analyse_pipeline(
    pipeline_dir: Path,
    beliefs: Dict[str, BeliefRecord],
    *,
    checkpoint_callback: Optional[
        Callable[[List[RunSummary], Dict[str, BeliefRecord], Dict[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    run_dirs = _iter_run_dirs_from_pipeline(pipeline_dir)
    eligible_run_dirs = [
        run_dir for run_dir in run_dirs if _find_final_state_json(run_dir)
    ]
    provisional_runs = [summarise_run(run_dir, None) for run_dir in eligible_run_dirs]
    baseline = _safe_mean(
        run.combined_score for run in provisional_runs if run.combined_score is not None
    )

    final_runs: List[RunSummary] = []
    working_beliefs = {belief_id: deepcopy(record) for belief_id, record in beliefs.items()}
    total_runs = len(eligible_run_dirs)
    print(
        f"[belief-pipeline] Found {total_runs} eligible topic runs. "
        f"Starting with {len(working_beliefs)} beliefs.",
        flush=True,
    )
    for run_index, run_dir in enumerate(eligible_run_dirs, start=1):
        run_summary = summarise_run(run_dir, baseline)
        topic_started_at = time.perf_counter()
        progress_prefix = f"[belief-pipeline][{run_index}/{total_runs}]"
        print(
            f"{progress_prefix} Starting: {run_summary.topic} "
            f"(run={run_summary.run_id})",
            flush=True,
        )
        state_payload = _state_for_run(run_dir)
        try:
            assessments, usage, call_metrics = _llm_reflect_run(
                run_summary, working_beliefs, state_payload=state_payload
            )
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - topic_started_at
            print(
                f"{progress_prefix} FAILED after {elapsed_seconds:.1f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        run_summary.belief_assessments = assessments
        run_summary.reflection_model = DEFAULT_REFLECTION_MODEL
        run_summary.reflection_usage = usage
        run_summary.reflection_calls = call_metrics
        topic_update_summary = apply_assessments(working_beliefs, assessments)
        final_runs.append(run_summary)
        if checkpoint_callback is not None:
            checkpoint_callback(final_runs, working_beliefs, topic_update_summary)
        elapsed_seconds = time.perf_counter() - topic_started_at
        prompt_characters = sum(
            int(call.get("prompt_characters", 0) or 0) for call in call_metrics
        )
        api_attempts = sum(
            int(call.get("api_attempts", 0) or 0) for call in call_metrics
        )
        print(
            f"{progress_prefix} Completed in {elapsed_seconds:.1f}s | "
            f"prompt_chars={prompt_characters} | "
            f"input_tokens={int(usage.get('prompt_tokens', 0) or 0)} | "
            f"output_tokens={int(usage.get('completion_tokens', 0) or 0)} | "
            f"api_attempts={api_attempts} | "
            f"assessments={len(assessments)} | "
            f"beliefs={len(working_beliefs)} | "
            f"updates={json.dumps(topic_update_summary, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )

    return _pipeline_analysis_payload(
        pipeline_dir,
        final_runs,
        expected_run_count=total_runs,
        complete=True,
    )


def _run_summary_to_json(run: RunSummary) -> Dict[str, Any]:
    payload = asdict(run)
    payload["section_attempts"] = {
        key: asdict(value) for key, value in sorted(run.section_attempts.items())
    }
    payload["belief_assessments"] = [asdict(item) for item in run.belief_assessments]
    payload["render_success_rate"] = round(run.render_success_rate, 4)
    return payload


def _resolve_analysis_target(args: argparse.Namespace) -> Tuple[str, Path]:
    if args.run_dir and args.pipeline_dir:
        raise ValueError("Pass either --run-dir or --pipeline-dir, not both.")
    if args.run_dir:
        return "single_run", Path(args.run_dir).expanduser().resolve()
    if args.pipeline_dir:
        return "pipeline", Path(args.pipeline_dir).expanduser().resolve()
    raise ValueError("One of --run-dir or --pipeline-dir is required.")


def _default_output_dir(mode: str, target_path: Path) -> Path:
    return target_path if mode == "pipeline" else target_path


def _resolve_output_path(
    *,
    requested_path: Optional[str],
    mode: str,
    target_path: Path,
    default_filename: str,
) -> Path:
    if requested_path:
        return Path(requested_path).expanduser().resolve()
    return (_default_output_dir(mode, target_path) / default_filename).resolve()


def _clear_pipeline_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse MAS run(s) and update an LLM-driven belief library."
    )
    parser.add_argument("--run-dir", default=None, help="Path to one MAS run directory.")
    parser.add_argument("--pipeline-dir", default=None, help="Path to a MAS pipeline directory.")
    parser.add_argument(
        "--library-path",
        default=None,
        help="JSON path for the persistent belief library. Defaults to the target run/pipeline folder.",
    )
    parser.add_argument(
        "--analysis-output",
        default=None,
        help="JSON path for run/pipeline analysis output. Defaults to the target run/pipeline folder.",
    )
    parser.add_argument(
        "--evidence-output",
        default=None,
        help="JSONL evidence path. Defaults to belief_evidence.jsonl beside the library.",
    )
    parser.add_argument(
        "--bbn-parameters-output",
        default=None,
        help="Shared BBN parameter path. Defaults to bbn_parameters.json beside the library.",
    )
    parser.add_argument(
        "--belief-embedding-model",
        default=None,
        help="Optional BGE sentence-transformers model/path used to persist belief embeddings.",
    )
    parser.add_argument(
        "--write-library",
        action="store_true",
        help="Apply generated assessments and write the updated belief library.",
    )
    parser.add_argument(
        "--fresh-library",
        action="store_true",
        help=(
            "Start from an empty belief bank and remove prior analysis, evidence, "
            "BBN, and embedding outputs at the selected paths before processing."
        ),
    )
    args = parser.parse_args()

    mode, target_path = _resolve_analysis_target(args)
    library_path = _resolve_output_path(
        requested_path=args.library_path,
        mode=mode,
        target_path=target_path,
        default_filename=DEFAULT_LIBRARY_FILENAME,
    )
    analysis_output_path = _resolve_output_path(
        requested_path=args.analysis_output,
        mode=mode,
        target_path=target_path,
        default_filename=DEFAULT_ANALYSIS_FILENAME,
    )
    evidence_output_path = _resolve_output_path(
        requested_path=args.evidence_output,
        mode=mode,
        target_path=target_path,
        default_filename=DEFAULT_EVIDENCE_FILENAME,
    )
    bbn_parameters_output_path = _resolve_output_path(
        requested_path=args.bbn_parameters_output,
        mode=mode,
        target_path=target_path,
        default_filename=DEFAULT_BBN_PARAMETERS_FILENAME,
    )
    embeddings_output_path = library_path.with_name("belief_embeddings.npz")
    embedding_metadata_output_path = library_path.with_name(
        "belief_embedding_metadata.json"
    )
    progress_output_path = library_path.with_name(DEFAULT_PROGRESS_FILENAME)
    if args.fresh_library:
        _clear_pipeline_outputs(
            analysis_output_path,
            library_path,
            evidence_output_path,
            bbn_parameters_output_path,
            embeddings_output_path,
            embedding_metadata_output_path,
            progress_output_path,
        )
        beliefs: Dict[str, BeliefRecord] = {}
    else:
        beliefs = load_library(library_path)

    if mode == "single_run":
        analysis = analyse_single_run(target_path, beliefs)
        run_payload = analysis["run_summary"]
        assessment_payloads = run_payload.get("belief_assessments", [])
    else:
        expected_run_count = sum(
            1
            for run_dir in _iter_run_dirs_from_pipeline(target_path)
            if _find_final_state_json(run_dir) is not None
        )

        def _checkpoint_pipeline(
            completed_runs: List[RunSummary],
            working_beliefs: Dict[str, BeliefRecord],
            topic_update_summary: Dict[str, Any],
        ) -> None:
            if not args.write_library:
                return
            completed_count = len(completed_runs)
            last_run = completed_runs[-1]
            save_library(
                library_path,
                working_beliefs,
                metadata={
                    "source_mode": mode,
                    "source_path": str(target_path),
                    "belief_count": len(working_beliefs),
                    "checkpoint": True,
                    "completed_run_count": completed_count,
                    "expected_run_count": expected_run_count,
                    "last_completed_run_id": last_run.run_id,
                    "last_topic_update_summary": topic_update_summary,
                },
            )
            save_evidence(evidence_output_path, working_beliefs)
            if not bbn_parameters_output_path.exists():
                _write_json(
                    bbn_parameters_output_path,
                    BBNParameters().to_payload(),
                )
            _write_json(
                analysis_output_path,
                _pipeline_analysis_payload(
                    target_path,
                    completed_runs,
                    expected_run_count=expected_run_count,
                    complete=False,
                ),
            )
            _write_json(
                progress_output_path,
                {
                    "status": "running",
                    "completed_run_count": completed_count,
                    "expected_run_count": expected_run_count,
                    "completed_run_ids": [run.run_id for run in completed_runs],
                    "last_completed_run_id": last_run.run_id,
                    "last_completed_topic": last_run.topic,
                    "belief_count": len(working_beliefs),
                    "updated_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
            )
            print(
                f"[belief-pipeline][{completed_count}/{expected_run_count}] "
                f"Checkpoint saved: {library_path}",
                flush=True,
            )

        analysis = analyse_pipeline(
            target_path,
            beliefs,
            checkpoint_callback=_checkpoint_pipeline,
        )
        assessment_payloads = []
        for run_payload in analysis["runs"]:
            assessment_payloads.extend(run_payload.get("belief_assessments", []))

    _write_json(analysis_output_path, analysis)

    update_summary: Dict[str, Any] = {}
    if args.write_library:
        parsed_assessments = [
            BeliefAssessment(
                run_id=item["run_id"],
                belief_id=item.get("belief_id"),
                instruction=item["instruction"],
                applicable=bool(item["applicable"]),
                compliance=str(item["compliance"]),
                outcome=str(item["outcome"]),
                action=str(item["action"]).upper(),
                weight=float(item.get("weight", item.get("evidence_confidence", 0.5))),
                scope=_belief_scope_from_dict(item.get("scope", {})),
                evidence_confidence=float(
                    item.get("evidence_confidence", item.get("weight", 0.5))
                ),
                impact=_normalize_impact(item.get("impact", "medium")),
                belief_type=_normalize_belief_type(item.get("belief_type", "confirmed")),
                timing=_normalize_belief_timing(item.get("timing", "both")),
                agent=item.get("agent"),
                section_id=item.get("section_id"),
                reason=str(item.get("reason", "")),
                evidence_event_ids=[str(value) for value in item.get("evidence_event_ids", [])],
                evidence=_coerce_evidence_payload(item.get("evidence")),
                merge_belief_ids=[str(value) for value in item.get("merge_belief_ids", [])],
                topic=str(item.get("topic") or ""),
                strategy_application=str(item.get("strategy_application", "unclear")),
                attribution_strength=str(item.get("attribution_strength", "unclear")),
                evidence_reliability=str(item.get("evidence_reliability", "unverifiable")),
                outcome_improvement=str(item.get("outcome_improvement", "unclear")),
                strategy_applied_probability=float(
                    item.get("strategy_applied_probability", 0.5)
                ),
                attribution_probability=float(item.get("attribution_probability", 0.5)),
                reliability_probability=float(item.get("reliability_probability", 0.5)),
                improvement=float(item.get("improvement", 0.5)),
                reflection_call=str(item.get("reflection_call") or ""),
            )
            for item in assessment_payloads
        ]
        update_summary = apply_assessments(beliefs, parsed_assessments)
        save_library(
            library_path,
            beliefs,
            metadata={
                "source_mode": mode,
                "source_path": str(target_path),
                "belief_count": len(beliefs),
                "update_summary": update_summary,
            },
        )
        save_evidence(evidence_output_path, beliefs)
        if not bbn_parameters_output_path.exists():
            _write_json(
                bbn_parameters_output_path,
                BBNParameters().to_payload(),
            )
        if args.belief_embedding_model:
            BeliefEmbeddingIndex.build(
                [asdict(belief) for belief in beliefs.values()],
                model_name_or_path=args.belief_embedding_model,
                embeddings_path=embeddings_output_path,
                metadata_path=embedding_metadata_output_path,
            )
        if mode == "pipeline":
            _write_json(
                progress_output_path,
                {
                    "status": "complete",
                    "completed_run_count": analysis["run_count"],
                    "expected_run_count": analysis.get(
                        "expected_run_count", analysis["run_count"]
                    ),
                    "completed_run_ids": [
                        run["run_id"] for run in analysis.get("runs", [])
                    ],
                    "last_completed_run_id": (
                        analysis["runs"][-1]["run_id"] if analysis.get("runs") else None
                    ),
                    "last_completed_topic": (
                        analysis["runs"][-1]["topic"] if analysis.get("runs") else None
                    ),
                    "belief_count": len(beliefs),
                    "updated_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                },
            )

    if mode == "single_run":
        run = analysis["run_summary"]
        print(f"Run: {run['run_id']}")
        print(f"Topic: {run['topic']}")
        print(f"Combined score: {run['combined_score']}")
        print(f"Baseline combined score: {run['baseline_combined_score']}")
        print(f"Assessments generated: {len(run['belief_assessments'])}")
        for call in run.get("reflection_calls", []):
            print(
                "Reflection call "
                f"{call['call_label']}: chars={call['prompt_characters']}, "
                f"input_tokens={call['prompt_tokens']}, "
                f"output_tokens={call['completion_tokens']}, "
                f"assessments={call['assessment_count']}, "
                f"attempts={call['api_attempts']}"
            )
    else:
        print(f"Pipeline: {analysis['pipeline_dir']}")
        print(f"Runs analysed: {analysis['run_count']}")
        print(f"Average combined score: {analysis['average_combined_score']}")
        print(f"Assessments generated: {sum(len(run['belief_assessments']) for run in analysis['runs'])}")
        print(
            "Reflection API calls: "
            f"{sum(len(run.get('reflection_calls', [])) for run in analysis['runs'])}"
        )

    if args.write_library:
        print(f"Belief library written to: {library_path}")
        print(f"Belief evidence written to: {evidence_output_path}")
        print(f"BBN parameters written to: {bbn_parameters_output_path}")
        if args.belief_embedding_model:
            print(
                "Belief embeddings written to: "
                f"{embeddings_output_path}"
            )
        if mode == "pipeline":
            print(f"Belief generation progress written to: {progress_output_path}")
        print(f"Update summary: {json.dumps(update_summary, ensure_ascii=False)}")

    print(f"Analysis output written to: {analysis_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
