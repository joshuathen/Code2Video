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
import json
import math
import os
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from google.genai import Client
from pydantic import BaseModel, Field
from mas_interactions import create_interaction, response_usage_dict

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
DEFAULT_REFLECTION_MODEL = cfg("gemini", "model", "gemini-3-flash-preview")

ACTION_ADD = "ADD"
ACTION_SUPPORT = "SUPPORT"
ACTION_CONTRADICT = "CONTRADICT"
ACTION_IRRELEVANT = "IRRELEVANT"
ACTION_REVISE = "REVISE"
ACTION_MERGE = "MERGE"
VALID_ACTIONS = {
    ACTION_ADD,
    ACTION_SUPPORT,
    ACTION_CONTRADICT,
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


class ReflectionScopePayload(BaseModel):
    roles: List[Literal["Orchestrator", "Coder", "AnimationPlanner", "ScriptWriter"]] = Field(
        min_length=1
    )


class ReflectionAssessmentPayload(BaseModel):
    belief_id: Optional[str] = None
    instruction: str
    scope: ReflectionScopePayload
    applicable: bool
    compliance: Literal["followed", "violated", "mixed", "unclear", "observed_pattern"]
    outcome: Literal["positive", "negative", "mixed", "unclear"]
    action: Literal["ADD", "SUPPORT", "CONTRADICT", "IRRELEVANT", "REVISE", "MERGE"]
    merge_belief_ids: List[str] = Field(default_factory=list)
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    impact: Literal["low", "medium", "high", "critical"]
    belief_type: Literal["confirmed", "precaution", "hypothesis", "quality"]
    timing: Literal["preventative", "reactive", "both"]
    agent: Optional[Literal["Orchestrator", "Coder", "AnimationPlanner", "ScriptWriter"]] = None
    section_id: Optional[str] = None
    reason: str
    # Gemini structured output does not support free-form object schemas because
    # Pydantic represents them with JSON Schema `additionalProperties`.
    evidence: str = ""


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
    return BeliefScope(roles=roles)


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

    selected = events
    if max_events is not None and len(events) > max_events:
        selected = events[: max_events // 2] + events[-(max_events - max_events // 2) :]
    trace_summary: List[Dict[str, Any]] = []
    for event in selected:
        text_parts = event.get("text_parts") or []
        joined_text = " ".join(str(part) for part in text_parts)
        cleaned_text = _truncate_text(_strip_code_blocks(joined_text), max_chars=500)
        function_calls = event.get("function_calls") or []
        function_responses = event.get("function_responses") or []
        trace_summary.append(
            {
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "agent": _normalize_role_label(event.get("agent")),
                "model": event.get("model"),
                "summary": cleaned_text,
                "usage": event.get("usage") or {},
                "function_call_count": len(event.get("function_calls") or []),
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
        )
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

    section_attempts = {}
    for section_id, summary in sorted(run.section_attempts.items()):
        # Preserve both early and late evidence when a section has a long repair
        # history while keeping the reflection prompt bounded.
        if len(summary.attempt_evidence) > 12:
            selected_attempts = summary.attempt_evidence[:4] + summary.attempt_evidence[-8:]
        else:
            selected_attempts = summary.attempt_evidence
        section_attempts[section_id] = {
            "attempts": summary.attempts,
            "scope_refine_attempts": summary.scope_refine_attempts,
            "infrastructure_attempts": summary.infrastructure_attempts,
            "successful": summary.successful,
            "elapsed_seconds": round(summary.elapsed_seconds, 3),
            "render_errors": summary.render_errors[:5],
            "attempt_evidence": selected_attempts,
            "omitted_attempt_evidence_count": summary.attempts - len(selected_attempts),
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
        "section_attempts": section_attempts,
        "video_review_summary": _summarize_video_review(state_payload.get("video_review")),
        "function_trace_summary": _summarize_function_trace(Path(run.run_dir)),
        "existing_beliefs": existing_beliefs,
    }


def _build_reflection_prompt(run: RunSummary, context_payload: Dict[str, Any]) -> str:
    context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
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
- Scope schema must be exactly: {{"roles": ["Coder"]}} or, for a cross-role belief, {{"roles": ["ScriptWriter", "AnimationPlanner"]}}.
- Do not leave roles empty and do not add extra scope fields.
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
- Assign impact using these consequence-based definitions:
  - critical: can terminate the pipeline, corrupt or lose outputs, make a run unrecoverable, or create an uncontrolled external effect;
  - high: causes a timeout, exhausts or is likely to exhaust retries, repeatedly blocks progress, or creates substantial compute or API cost;
  - medium: causes a deterministic but recoverable failure that normally requires a repair or retry;
  - low: affects presentation quality, style, clarity, or minor efficiency without preventing successful completion.
- Do not mark a belief high or critical merely because an exception occurred. Base impact on the observed or reasonably expected operational consequence, independently of confidence and frequency.
- Prefer a small set of distinct, high-value beliefs over many overlapping low-impact beliefs.
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
- Inspect section_attempts.attempt_evidence for code-specific patterns, including call sites, traceback-adjacent source, timeouts, and before/after repair diffs.
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
        "roles": ["Orchestrator"] or ["Coder"] or ["AnimationPlanner"] or ["ScriptWriter"] or a multi-role list
      }},
      "applicable": true,
      "compliance": "followed" or "violated" or "mixed" or "unclear" or "observed_pattern",
      "outcome": "positive" or "negative" or "mixed" or "unclear",
      "action": "ADD" or "SUPPORT" or "CONTRADICT" or "IRRELEVANT" or "REVISE" or "MERGE",
      "merge_belief_ids": ["L002", "L003"] or [],
      "evidence_confidence": 0.0,
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


def _llm_reflect_run(
    run: RunSummary,
    beliefs: Dict[str, BeliefRecord],
    *,
    state_payload: Dict[str, Any],
) -> Tuple[List[BeliefAssessment], Dict[str, int]]:
    context_payload = _build_reflection_context(run, state_payload, beliefs)
    prompt = _build_reflection_prompt(run, context_payload)
    api_key = cfg("gemini", "api_key")
    if not api_key:
        raise ValueError("Missing gemini.api_key in api_config.json or GEMINI_API_KEY")
    client = Client(api_key=api_key)
    response = create_interaction(
        client,
        model=DEFAULT_REFLECTION_MODEL,
        input_value=prompt,
        response_schema=ReflectionResponsePayload,
        max_output_tokens=12000,
    )
    parsed = ReflectionResponsePayload.model_validate_json(response.output_text)
    usage = response_usage_dict(response)

    assessments: List[BeliefAssessment] = []
    for item in parsed.assessments:
        action = item.action
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported belief assessment action: {action}")
        evidence_confidence = item.evidence_confidence
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
                scope=BeliefScope(roles=list(item.scope.roles)),
                evidence_confidence=evidence_confidence,
                impact=item.impact,
                belief_type=item.belief_type,
                timing=item.timing,
                agent=item.agent,
                section_id=item.section_id,
                reason=item.reason,
                evidence=_coerce_evidence_payload(item.evidence),
                merge_belief_ids=list(item.merge_belief_ids),
            )
        )
    return assessments, usage


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
                "agent": assessment.agent,
                "section_id": assessment.section_id,
                "previous_instruction": previous_instruction,
                "merge_belief_ids": assessment.merge_belief_ids,
                "evidence": assessment.evidence,
            }
        )

        if assessment.action in {ACTION_ADD, ACTION_SUPPORT, ACTION_REVISE, ACTION_MERGE}:
            record.support_count += 1
            record.weighted_support += assessment.evidence_confidence
            record.alpha += assessment.evidence_confidence
        elif assessment.action == ACTION_CONTRADICT:
            record.contradiction_count += 1
            record.weighted_contradiction += assessment.evidence_confidence
            record.beta += assessment.evidence_confidence

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
    assessments, usage = _llm_reflect_run(run_summary, beliefs, state_payload=state_payload)
    run_summary.belief_assessments = assessments
    run_summary.reflection_model = DEFAULT_REFLECTION_MODEL
    run_summary.reflection_usage = usage
    return {
        "mode": "single_run",
        "run_summary": _run_summary_to_json(run_summary),
    }


def analyse_pipeline(pipeline_dir: Path, beliefs: Dict[str, BeliefRecord]) -> Dict[str, Any]:
    run_dirs = _iter_run_dirs_from_pipeline(pipeline_dir)
    provisional_runs = [summarise_run(run_dir, None) for run_dir in run_dirs if _find_final_state_json(run_dir)]
    baseline = _safe_mean(
        run.combined_score for run in provisional_runs if run.combined_score is not None
    )

    final_runs: List[RunSummary] = []
    working_beliefs = {belief_id: deepcopy(record) for belief_id, record in beliefs.items()}
    for run_dir in run_dirs:
        if _find_final_state_json(run_dir) is None:
            continue
        run_summary = summarise_run(run_dir, baseline)
        state_payload = _state_for_run(run_dir)
        assessments, usage = _llm_reflect_run(run_summary, working_beliefs, state_payload=state_payload)
        run_summary.belief_assessments = assessments
        run_summary.reflection_model = DEFAULT_REFLECTION_MODEL
        run_summary.reflection_usage = usage
        apply_assessments(working_beliefs, assessments)
        final_runs.append(run_summary)

    pipeline_scores = [run.combined_score for run in final_runs if run.combined_score is not None]
    payload = {
        "mode": "pipeline",
        "pipeline_dir": str(pipeline_dir),
        "run_count": len(final_runs),
        "average_combined_score": round(mean(pipeline_scores), 3) if pipeline_scores else None,
        "average_aes_overall": _safe_mean(run.aes_overall for run in final_runs),
        "average_tq_learning_gain": _safe_mean(run.tq_learning_gain for run in final_runs),
        "runs": [_run_summary_to_json(run) for run in final_runs],
    }
    return payload


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
        "--write-library",
        action="store_true",
        help="Apply generated assessments and write the updated belief library.",
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
    if mode == "pipeline":
        _clear_pipeline_outputs(analysis_output_path, library_path)
        beliefs: Dict[str, BeliefRecord] = {}
    else:
        beliefs = load_library(library_path)

    if mode == "single_run":
        analysis = analyse_single_run(target_path, beliefs)
        run_payload = analysis["run_summary"]
        assessment_payloads = run_payload.get("belief_assessments", [])
    else:
        analysis = analyse_pipeline(target_path, beliefs)
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

    if mode == "single_run":
        run = analysis["run_summary"]
        print(f"Run: {run['run_id']}")
        print(f"Topic: {run['topic']}")
        print(f"Combined score: {run['combined_score']}")
        print(f"Baseline combined score: {run['baseline_combined_score']}")
        print(f"Assessments generated: {len(run['belief_assessments'])}")
    else:
        print(f"Pipeline: {analysis['pipeline_dir']}")
        print(f"Runs analysed: {analysis['run_count']}")
        print(f"Average combined score: {analysis['average_combined_score']}")
        print(f"Assessments generated: {sum(len(run['belief_assessments']) for run in analysis['runs'])}")

    if args.write_library:
        print(f"Belief library written to: {library_path}")
        print(f"Update summary: {json.dumps(update_summary, ensure_ascii=False)}")

    print(f"Analysis output written to: {analysis_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
