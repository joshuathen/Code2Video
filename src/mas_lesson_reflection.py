#!/usr/bin/env python3
"""Build and update an LLM-driven MAS lesson library from run artifacts.

This utility analyses one MAS run or a full pipeline, prepares condensed run
evidence, asks Gemini for structured lesson assessments, and persists the
resulting lesson library over repeated runs.

The deterministic parts of the script only:
- collect run artifacts into a stable summary
- maintain lesson evidence counts and Bayesian confidence
- write analysis/library JSON files

Lesson creation and lesson evaluation are delegated to Gemini.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gpt_request import cfg, request_gemini_token
from utils import extract_answer_from_response, extract_json_from_markdown


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "mas_logs"
DEFAULT_LIBRARY_FILENAME = "lesson_library.json"
DEFAULT_ANALYSIS_FILENAME = "lesson_analysis.json"
DEFAULT_REFLECTION_MODEL = cfg("gemini", "model", "gemini-3-flash-preview")

ACTION_ADD = "ADD"
ACTION_SUPPORT = "SUPPORT"
ACTION_CONTRADICT = "CONTRADICT"
ACTION_IRRELEVANT = "IRRELEVANT"

STATUS_ACTIVE = "active"
STATUS_PROBATION = "probation"
STATUS_DEPRECATED = "deprecated"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_first_json_object(text: str) -> Dict[str, Any]:
    cleaned = extract_json_from_markdown(text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(cleaned[idx:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse JSON object from model response: {cleaned[:1000]}")


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
class LessonScope:
    level: str = "general"
    roles: List[str] = field(default_factory=list)


@dataclass
class LessonRecord:
    lesson_id: str
    instruction: str
    scope: LessonScope
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
    created_from_run: Optional[str] = None
    observed_runs: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def update_confidence(self) -> None:
        self.confidence = round(self.alpha / (self.alpha + self.beta), 4)
        self.status = _status_from_counts(
            contradiction_count=self.contradiction_count,
            confidence=self.confidence,
        )


@dataclass
class LessonAssessment:
    run_id: str
    lesson_id: Optional[str]
    instruction: str
    applicable: bool
    compliance: str
    outcome: str
    action: str
    weight: float
    scope: LessonScope
    agent: Optional[str] = None
    section_id: Optional[str] = None
    reason: str = ""
    evidence_event_ids: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionAttemptSummary:
    section_id: str
    attempts: int = 0
    scope_refine_attempts: int = 0
    successful: bool = False
    elapsed_seconds: float = 0.0
    render_errors: List[str] = field(default_factory=list)


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
    lesson_assessments: List[LessonAssessment] = field(default_factory=list)
    reflection_model: Optional[str] = None
    reflection_usage: Dict[str, int] = field(default_factory=dict)

    @property
    def render_success_rate(self) -> float:
        return (self.render_ok_count / self.render_total) if self.render_total else 0.0


def _lesson_scope_from_dict(payload: Dict[str, Any]) -> LessonScope:
    return LessonScope(
        level=str(payload.get("level", "general")),
        roles=[str(item) for item in payload.get("roles", [])],
    )


def _record_from_dict(payload: Dict[str, Any]) -> LessonRecord:
    record = LessonRecord(
        lesson_id=str(payload["lesson_id"]),
        instruction=str(payload["instruction"]),
        scope=_lesson_scope_from_dict(payload.get("scope", {})),
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


def load_library(path: Path) -> Dict[str, LessonRecord]:
    if path.exists():
        payload = _read_json(path)
        lessons = payload.get("lessons", payload)
        records = [_record_from_dict(item) for item in lessons]
        return {record.lesson_id: record for record in records}
    return {}


def save_library(path: Path, lessons: Dict[str, LessonRecord], metadata: Dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "lessons": [asdict(lesson) for lesson in sorted(lessons.values(), key=lambda item: item.lesson_id)],
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
        if render_error:
            summary.render_errors.append(render_error)
        render_outcome = payload.get("render_outcome") or {}
        if isinstance(render_outcome, dict):
            succeeded = bool(render_outcome.get("success"))
            summary.successful = summary.successful or succeeded
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


def _build_reflection_context(run: RunSummary, state_payload: Dict[str, Any], lessons: Dict[str, LessonRecord]) -> Dict[str, Any]:
    existing_lessons = [
        {
            "lesson_id": lesson.lesson_id,
            "instruction": lesson.instruction,
            "scope": asdict(lesson.scope),
            "status": lesson.status,
            "confidence": lesson.confidence,
            "support_count": lesson.support_count,
            "contradiction_count": lesson.contradiction_count,
            "relevant_count": lesson.relevant_count,
        }
        for lesson in sorted(lessons.values(), key=lambda item: item.lesson_id)
    ]

    section_attempts = {
        section_id: {
            "attempts": summary.attempts,
            "scope_refine_attempts": summary.scope_refine_attempts,
            "successful": summary.successful,
            "elapsed_seconds": round(summary.elapsed_seconds, 3),
            "render_errors": summary.render_errors[:5],
        }
        for section_id, summary in sorted(run.section_attempts.items())
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
        "section_attempts": section_attempts,
        "video_review_summary": _summarize_video_review(state_payload.get("video_review")),
        "function_trace_summary": _summarize_function_trace(Path(run.run_dir)),
        "existing_lessons": existing_lessons,
    }


def _build_reflection_prompt(run: RunSummary, context_payload: Dict[str, Any]) -> str:
    context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
    return f"""
You are maintaining a reusable lesson library for a multi-agent MAS pipeline.

Your job:
1. Read the run evidence carefully.
2. Evaluate existing lessons when relevant.
3. Create new lessons organically when the run contains reusable evidence.

Important rules:
- All lesson generation and lesson management must be evidence-based.
- Do not rely on predefined starter lessons.
- Scope lessons by role, not by agent instance. Example: use "Coder", never "Coder3".
- Scope schema must be simple: {{"level": "general" or "role_specific", "roles": ["Coder"]}}
- Do not add extra scope fields beyond level and roles.
- Each lesson must be actionable, general, concise, and reusable across runs.
- Avoid duplicates or near-duplicates of existing lessons.
- For existing lessons, choose one action: SUPPORT, CONTRADICT, IRRELEVANT.
- For new lessons, choose action ADD and set lesson_id to null.
- Weight must be between 0 and 1.
- Keep evidence short and grounded in the supplied run context only.

Return JSON only in this exact shape:
{{
  "assessments": [
    {{
      "lesson_id": "L001" or null,
      "instruction": "Actionable lesson text",
      "scope": {{
        "level": "general" or "role_specific",
        "roles": ["Orchestrator"] or ["Coder"] or ["AnimationPlanner"] or ["ScriptWriter"]
      }},
      "applicable": true,
      "compliance": "followed" or "violated" or "mixed" or "unclear" or "observed_pattern",
      "outcome": "positive" or "negative" or "mixed" or "unclear",
      "action": "ADD" or "SUPPORT" or "CONTRADICT" or "IRRELEVANT",
      "weight": 0.0,
      "agent": "Coder" or null,
      "section_id": "section_1" or null,
      "reason": "Short explanation",
      "evidence": {{}}
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
    lessons: Dict[str, LessonRecord],
    *,
    state_payload: Dict[str, Any],
) -> Tuple[List[LessonAssessment], Dict[str, int]]:
    context_payload = _build_reflection_context(run, state_payload, lessons)
    prompt = _build_reflection_prompt(run, context_payload)
    response, usage = request_gemini_token(
        prompt,
        max_tokens=12000,
        model_name=DEFAULT_REFLECTION_MODEL,
        response_format={"type": "json_object"},
    )
    content = extract_answer_from_response(response)
    payload = _parse_first_json_object(content)

    assessments: List[LessonAssessment] = []
    for item in payload.get("assessments", []):
        assessments.append(
            LessonAssessment(
                run_id=run.run_id,
                lesson_id=item.get("lesson_id"),
                instruction=str(item["instruction"]),
                applicable=bool(item.get("applicable", True)),
                compliance=str(item.get("compliance", "unclear")),
                outcome=str(item.get("outcome", "unclear")),
                action=str(item["action"]),
                weight=float(_clamp(float(item.get("weight", 0.5)), 0.0, 1.0)),
                scope=_lesson_scope_from_dict(item.get("scope", {})),
                agent=item.get("agent"),
                section_id=item.get("section_id"),
                reason=str(item.get("reason", "")),
                evidence=_coerce_evidence_payload(item.get("evidence")),
            )
        )
    return assessments, usage


def _find_lesson_by_instruction(
    lessons: Dict[str, LessonRecord],
    instruction: str,
) -> Optional[LessonRecord]:
    target = _normalize_instruction(instruction)
    for record in lessons.values():
        if _normalize_instruction(record.instruction) == target:
            return record
    return None


def _next_lesson_id(lessons: Dict[str, LessonRecord]) -> str:
    max_numeric = 0
    for lesson_id in lessons:
        match = re.fullmatch(r"L(\d+)", lesson_id)
        if match:
            max_numeric = max(max_numeric, int(match.group(1)))
    return f"L{max_numeric + 1:03d}"


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
    lessons: Dict[str, LessonRecord],
    assessments: List[LessonAssessment],
) -> Dict[str, Any]:
    summary = Counter()

    for assessment in assessments:
        summary[f"action:{assessment.action.lower()}"] += 1

        if assessment.action == ACTION_IRRELEVANT:
            if assessment.lesson_id and assessment.lesson_id in lessons:
                lessons[assessment.lesson_id].irrelevant_count += 1
            continue

        record: Optional[LessonRecord] = None
        if assessment.lesson_id:
            record = lessons.get(assessment.lesson_id)
        elif assessment.action == ACTION_ADD:
            record = _find_lesson_by_instruction(lessons, assessment.instruction)

        if record is None and assessment.action == ACTION_ADD:
            lesson_id = _next_lesson_id(lessons)
            record = LessonRecord(
                lesson_id=lesson_id,
                instruction=assessment.instruction,
                scope=assessment.scope,
                created_from_run=assessment.run_id,
            )
            lessons[lesson_id] = record
            assessment.lesson_id = lesson_id
            summary["lessons_created"] += 1

        if record is None:
            continue

        record.relevant_count += 1
        record.observed_runs.append(assessment.run_id)
        record.evidence.append(
            {
                "run_id": assessment.run_id,
                "action": assessment.action,
                "weight": assessment.weight,
                "reason": assessment.reason,
                "agent": assessment.agent,
                "section_id": assessment.section_id,
                "evidence": assessment.evidence,
            }
        )

        if assessment.action in {ACTION_ADD, ACTION_SUPPORT}:
            record.support_count += 1
            record.weighted_support += assessment.weight
            record.alpha += assessment.weight
        elif assessment.action == ACTION_CONTRADICT:
            record.contradiction_count += 1
            record.weighted_contradiction += assessment.weight
            record.beta += assessment.weight

        record.update_confidence()

    summary["total_assessments"] = len(assessments)
    return dict(summary)


def _state_for_run(run_dir: Path) -> Dict[str, Any]:
    state_path = _find_final_state_json(run_dir)
    if state_path is None:
        return {}
    return _read_json(state_path)


def analyse_single_run(run_dir: Path, lessons: Dict[str, LessonRecord]) -> Dict[str, Any]:
    pipeline_id = run_dir.parent.name if run_dir.parent.name.startswith("pipeline_") else None
    baseline = _load_pipeline_paper_average(run_dir.parent) if pipeline_id else None
    run_summary = summarise_run(run_dir, baseline)
    state_payload = _state_for_run(run_dir)
    assessments, usage = _llm_reflect_run(run_summary, lessons, state_payload=state_payload)
    run_summary.lesson_assessments = assessments
    run_summary.reflection_model = DEFAULT_REFLECTION_MODEL
    run_summary.reflection_usage = usage
    return {
        "mode": "single_run",
        "run_summary": _run_summary_to_json(run_summary),
    }


def analyse_pipeline(pipeline_dir: Path, lessons: Dict[str, LessonRecord]) -> Dict[str, Any]:
    run_dirs = _iter_run_dirs_from_pipeline(pipeline_dir)
    provisional_runs = [summarise_run(run_dir, None) for run_dir in run_dirs if _find_final_state_json(run_dir)]
    baseline = _safe_mean(
        run.combined_score for run in provisional_runs if run.combined_score is not None
    )

    final_runs: List[RunSummary] = []
    working_lessons = {lesson_id: deepcopy(record) for lesson_id, record in lessons.items()}
    for run_dir in run_dirs:
        if _find_final_state_json(run_dir) is None:
            continue
        run_summary = summarise_run(run_dir, baseline)
        state_payload = _state_for_run(run_dir)
        assessments, usage = _llm_reflect_run(run_summary, working_lessons, state_payload=state_payload)
        run_summary.lesson_assessments = assessments
        run_summary.reflection_model = DEFAULT_REFLECTION_MODEL
        run_summary.reflection_usage = usage
        apply_assessments(working_lessons, assessments)
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
    payload["lesson_assessments"] = [asdict(item) for item in run.lesson_assessments]
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
        description="Analyse MAS run(s) and update an LLM-driven lesson library."
    )
    parser.add_argument("--run-dir", default=None, help="Path to one MAS run directory.")
    parser.add_argument("--pipeline-dir", default=None, help="Path to a MAS pipeline directory.")
    parser.add_argument(
        "--library-path",
        default=None,
        help="JSON path for the persistent lesson library. Defaults to the target run/pipeline folder.",
    )
    parser.add_argument(
        "--analysis-output",
        default=None,
        help="JSON path for run/pipeline analysis output. Defaults to the target run/pipeline folder.",
    )
    parser.add_argument(
        "--write-library",
        action="store_true",
        help="Apply generated assessments and write the updated lesson library.",
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
        lessons: Dict[str, LessonRecord] = {}
    else:
        lessons = load_library(library_path)

    if mode == "single_run":
        analysis = analyse_single_run(target_path, lessons)
        run_payload = analysis["run_summary"]
        assessment_payloads = run_payload.get("lesson_assessments", [])
    else:
        analysis = analyse_pipeline(target_path, lessons)
        assessment_payloads = []
        for run_payload in analysis["runs"]:
            assessment_payloads.extend(run_payload.get("lesson_assessments", []))

    _write_json(analysis_output_path, analysis)

    update_summary: Dict[str, Any] = {}
    if args.write_library:
        parsed_assessments = [
            LessonAssessment(
                run_id=item["run_id"],
                lesson_id=item.get("lesson_id"),
                instruction=item["instruction"],
                applicable=bool(item["applicable"]),
                compliance=str(item["compliance"]),
                outcome=str(item["outcome"]),
                action=str(item["action"]),
                weight=float(item["weight"]),
                scope=_lesson_scope_from_dict(item.get("scope", {})),
                agent=item.get("agent"),
                section_id=item.get("section_id"),
                reason=str(item.get("reason", "")),
                evidence_event_ids=[str(value) for value in item.get("evidence_event_ids", [])],
                evidence=_coerce_evidence_payload(item.get("evidence")),
            )
            for item in assessment_payloads
        ]
        update_summary = apply_assessments(lessons, parsed_assessments)
        save_library(
            library_path,
            lessons,
            metadata={
                "source_mode": mode,
                "source_path": str(target_path),
                "lesson_count": len(lessons),
                "update_summary": update_summary,
            },
        )

    if mode == "single_run":
        run = analysis["run_summary"]
        print(f"Run: {run['run_id']}")
        print(f"Topic: {run['topic']}")
        print(f"Combined score: {run['combined_score']}")
        print(f"Baseline combined score: {run['baseline_combined_score']}")
        print(f"Assessments generated: {len(run['lesson_assessments'])}")
    else:
        print(f"Pipeline: {analysis['pipeline_dir']}")
        print(f"Runs analysed: {analysis['run_count']}")
        print(f"Average combined score: {analysis['average_combined_score']}")
        print(f"Assessments generated: {sum(len(run['lesson_assessments']) for run in analysis['runs'])}")

    if args.write_library:
        print(f"Lesson library written to: {library_path}")
        print(f"Update summary: {json.dumps(update_summary, ensure_ascii=False)}")

    print(f"Analysis output written to: {analysis_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
