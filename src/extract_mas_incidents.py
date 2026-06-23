#!/usr/bin/env python3
"""Build an initial MAS incident dataset from existing mas_logs artifacts.

This extractor creates:
1. A rich JSONL master dataset with per-incident metadata.
2. A flattened CSV for Model 1 training.
3. A summary JSON and lightweight README.

The primary training export uses the following input template:

Topic: ...
Section: ...
Exception: ...
Render status: ...
Timed out: ...

Error:
...

Code excerpt:
...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MODEL1_LABELS = [
    "environment_error",
    "performance_or_timeout",
    "name_error",
    "type_error",
]

REPAIR_ACTION_LABELS = [
    "fix_environment",
    "scope_refine_repair",
    "retry_with_timeout_or_perf_fix",
    "full_regenerate",
]

TRAINING_CSV_FIELDS = [
    "incident_id",
    "source_type",
    "quality_tier",
    "run_id",
    "topic",
    "section_id",
    "section_title",
    "attempt_number",
    "snapshot_name",
    "turn_index",
    "phase",
    "label_coarse",
    "label_fine",
    "exception_type",
    "render_status",
    "timed_out",
    "elapsed_seconds",
    "failing_line_number",
    "input_text",
    "normalized_error_text",
    "code_excerpt",
    "repair_strategy",
    "repair_reason",
    "later_attempt_succeeded",
    "resolved_later_in_run",
    "final_section_status",
]

REPAIR_ACTION_CSV_FIELDS = [
    "incident_id",
    "base_incident_id",
    "source_type",
    "quality_tier",
    "run_id",
    "topic",
    "section_id",
    "section_title",
    "attempt_number",
    "phase",
    "repair_action",
    "repair_action_source",
    "view_name",
    "input_text",
    "label_coarse",
    "label_fine",
    "exception_type",
    "render_status",
    "timed_out",
    "elapsed_seconds",
    "failing_line_number",
    "normalized_error_text",
    "code_excerpt",
    "repair_strategy",
    "repair_reason",
    "later_attempt_succeeded",
    "resolved_later_in_run",
    "final_section_status",
]

ASSET_PATH_PATTERN = re.compile(r"\.(?:png|svg|jpe?g|webp)\b", re.IGNORECASE)


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    topic: str
    coder_assignments: Dict[str, str]
    section_titles: Dict[str, str]
    final_render_status: Dict[str, Optional[str]]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _iter_state_files(run_dir: Path) -> List[Path]:
    mas_state_dir = run_dir / "mas_state"
    if not mas_state_dir.exists():
        return []

    candidates = [
        path
        for path in mas_state_dir.glob("*.json")
        if re.match(r"turn_\d+_video_state\.json$", path.name)
        or path.name in {"final_render_pass_video_state.json", "final_video_state.json"}
    ]

    def sort_key(path: Path) -> Tuple[int, int, str]:
        match = re.match(r"turn_(\d+)_video_state\.json$", path.name)
        if match:
            return (0, int(match.group(1)), path.name)
        if path.name == "final_render_pass_video_state.json":
            return (1, 0, path.name)
        if path.name == "final_video_state.json":
            return (2, 0, path.name)
        return (3, 0, path.name)

    return sorted(candidates, key=sort_key)


def _fallback_topic_from_dirname(run_dir: Path) -> str:
    slug = run_dir.name
    if "-" in slug:
        slug = slug.split("-", 1)[1]
    return slug.replace("_", " ").strip()


def _load_run_context(run_dir: Path, logs_root: Path) -> RunContext:
    state_files = list(reversed(_iter_state_files(run_dir)))
    topic = ""
    coder_assignments: Dict[str, str] = {}
    section_titles: Dict[str, str] = {}
    final_render_status: Dict[str, Optional[str]] = {}

    for state_path in state_files:
        state = _read_json(state_path)
        topic = str(state.get("topic") or topic)

        storyboard = state.get("storyboard")
        if isinstance(storyboard, list):
            for item in storyboard:
                if not isinstance(item, dict):
                    continue
                section_id = str(item.get("id") or "")
                if section_id:
                    section_titles[section_id] = str(item.get("title") or section_titles.get(section_id, ""))

        assignments = state.get("coder_assignments")
        if isinstance(assignments, dict):
            coder_assignments.update({str(k): str(v) for k, v in assignments.items()})

        render_status = state.get("render_status")
        if isinstance(render_status, list):
            section_ids = list(section_titles.keys())
            if len(section_ids) == len(render_status):
                for idx, section_id in enumerate(section_ids):
                    value = render_status[idx]
                    final_render_status[section_id] = str(value) if value is not None else None
        if topic and coder_assignments and section_titles:
            break

    if not section_titles:
        storyboard_path = run_dir / "storyboard.json"
        if storyboard_path.exists():
            storyboard = _read_json(storyboard_path)
            for item in storyboard.get("sections", []):
                if isinstance(item, dict) and item.get("id"):
                    section_titles[str(item["id"])] = str(item.get("title") or "")

    if not topic:
        outline_path = run_dir / "outline.json"
        if outline_path.exists():
            outline = _read_json(outline_path)
            topic = str(outline.get("topic") or "")

    topic = topic or _fallback_topic_from_dirname(run_dir)

    return RunContext(
        run_dir=run_dir,
        run_id=str(run_dir.relative_to(logs_root)),
        topic=topic,
        coder_assignments=coder_assignments,
        section_titles=section_titles,
        final_render_status=final_render_status,
    )


def _strip_progress_lines(text: str) -> str:
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if "Animation " in line and "%|" in line:
            continue
        if "it/s]" in line:
            continue
        if re.fullmatch(r"[▁▂▃▄▅▆▇█\s]+", line):
            continue
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


def _normalize_paths(text: str, run_dir: Path) -> str:
    normalized = text.replace(str(run_dir), "<RUN_DIR>")
    normalized = re.sub(r"/home/[^/\s]+/", "/home/<USER>/", normalized)
    normalized = re.sub(r"/mmfs1/data/home/[^/\s]+/", "/mmfs1/data/home/<USER>/", normalized)
    normalized = re.sub(r"line \d+", "line <N>", normalized)
    normalized = re.sub(r":\d+\b", ":<N>", normalized)
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_error_text(raw_text: str, run_dir: Path) -> str:
    if not raw_text.strip():
        return ""
    cleaned = _strip_progress_lines(raw_text)
    return _normalize_paths(cleaned, run_dir)


def _error_digest(text: str) -> str:
    if not text.strip():
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _extract_exception_type(text: str) -> str:
    if not text.strip():
        return ""
    matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):", text)
    if matches:
        return matches[-1]
    return ""


def _classify_incident(
    *,
    precheck_error: str,
    raw_error_text: str,
    normalized_error_text: str,
    timed_out: bool,
    render_status: Optional[str],
) -> Tuple[str, str]:
    text = "\n".join(
        part for part in [precheck_error, raw_error_text, normalized_error_text, render_status or ""] if part
    ).lower()

    normalized_lower = normalized_error_text.lower()

    if timed_out or normalized_lower.startswith("render timed out") or "render timed out after" in normalized_lower:
        return "performance_or_timeout", "timeout"

    if "videocritic" in text or "layout issue" in text or "overlap" in text or "off-screen" in text:
        return "layout_error", "layout_issue"

    if "filenotfounderror" in text and "'latex'" in text:
        return "environment_error", "missing_latex_binary"
    if "filenotfounderror" in text and "ffmpeg" in text:
        return "environment_error", "missing_ffmpeg_binary"
    if "filenotfounderror" in text and ASSET_PATH_PATTERN.search(text):
        return "asset_error", "asset_not_found"
    if "filenotfounderror" in text or "no such file or directory" in text:
        return "environment_error", "missing_external_binary"

    if "modulenotfounderror" in text or "importerror" in text:
        return "import_error", "module_import_error"

    if "indentationerror" in text:
        return "syntax_error", "indentation_error"
    if "syntaxerror" in text:
        return "syntax_error", "syntax_error"

    if "attributeerror" in text:
        return "attribute_error", "attribute_error"
    if "nameerror" in text:
        return "name_error", "name_error"
    if "typeerror" in text:
        return "type_error", "type_error"
    if "valueerror" in text:
        if ASSET_PATH_PATTERN.search(text):
            return "asset_error", "bad_asset_path"
        return "value_error", "value_error"

    if "always_redraw" in text or "updater" in text or "zero dimension" in text:
        return "performance_or_timeout", "runtime_performance_issue"
    if "svgmobject" in text or "image" in text or "[asset:" in text:
        return "asset_error", "asset_usage_error"

    if (render_status or "").lower() == "failed" and normalized_error_text.strip():
        return "generic_render_error", "generic_render_error"
    if normalized_error_text.strip():
        return "generic_render_error", "generic_render_error"
    return "unknown", "unknown"


def _map_to_model1_label(label_coarse: str) -> Optional[str]:
    if label_coarse == "attribute_error":
        return "type_error"
    if label_coarse in MODEL1_LABELS:
        return label_coarse
    return None


def _extract_failing_line_number(raw_error_text: str, section_id: str) -> Optional[int]:
    match = re.search(rf"{re.escape(section_id)}\.py:(\d+)\b", raw_error_text)
    if match:
        return int(match.group(1))
    return None


def _extract_code_excerpt(code: str, line_number: Optional[int], radius: int = 3) -> str:
    if not code.strip():
        return ""
    lines = code.splitlines()
    if line_number is None or line_number <= 0 or line_number > len(lines):
        return "\n".join(lines[: min(20, len(lines))]).strip()

    start = max(0, line_number - 1 - radius)
    end = min(len(lines), line_number + radius)
    excerpt: List[str] = []
    for idx in range(start, end):
        excerpt.append(f"{idx + 1}: {lines[idx]}")
    return "\n".join(excerpt).strip()


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def _build_model1_input_text(record: Dict[str, object]) -> str:
    return (
        f"Topic: {record.get('topic', '')}\n"
        f"Section: {record.get('section_title', '')}\n"
        f"Exception: {record.get('exception_type', '')}\n"
        f"Render status: {record.get('render_status', '')}\n"
        f"Timed out: {_bool_text(record.get('timed_out', False))}\n\n"
        f"Error:\n{record.get('normalized_error_text', '')}\n\n"
        f"Code excerpt:\n{record.get('code_excerpt', '')}"
    ).strip()


def _build_repair_action_input_text(
    record: Dict[str, object],
    *,
    include_topic: bool = True,
    include_exception: bool = True,
    include_code_excerpt: bool = True,
) -> str:
    lines: List[str] = []
    if include_topic:
        lines.append(f"Topic: {record.get('topic', '')}")
    lines.append(f"Section: {record.get('section_title', '')}")
    if include_exception:
        lines.append(f"Exception: {record.get('exception_type', '')}")
    lines.append(f"Render status: {record.get('render_status', '')}")
    lines.append(f"Timed out: {_bool_text(record.get('timed_out', False))}")

    error_text = str(record.get("normalized_error_text", "") or "").strip()
    if error_text:
        lines.extend(["", "Error:", error_text])

    if include_code_excerpt:
        code_excerpt = str(record.get("code_excerpt", "") or "").strip()
        if code_excerpt:
            lines.extend(["", "Code excerpt:", code_excerpt])

    return "\n".join(lines).strip()


def _derive_repair_action(record: Dict[str, object]) -> Tuple[str, str]:
    label_coarse = str(record.get("label_coarse") or "")
    label_fine = str(record.get("label_fine") or "")
    repair_strategy = str(record.get("repair_strategy") or "")
    later_attempt_succeeded = record.get("later_attempt_succeeded")
    resolved_later_in_run = bool(record.get("resolved_later_in_run"))

    if label_coarse == "environment_error":
        return "fix_environment", "heuristic:environment_error"

    if label_coarse == "performance_or_timeout" or label_fine == "timeout" or bool(record.get("timed_out")):
        return "retry_with_timeout_or_perf_fix", "heuristic:timeout_or_performance"

    if label_coarse in {"name_error", "type_error"} and repair_strategy == "scope_refine":
        return "scope_refine_repair", "heuristic:label_plus_scope_refine"

    if label_coarse in {"name_error", "type_error"}:
        if later_attempt_succeeded is False or not resolved_later_in_run:
            return "full_regenerate", "heuristic:unresolved_codegen_failure"
        return "scope_refine_repair", "heuristic:label_only"

    if repair_strategy == "scope_refine":
        return "scope_refine_repair", "heuristic:repair_strategy"

    if later_attempt_succeeded is False or not resolved_later_in_run:
        return "full_regenerate", "heuristic:unresolved_fallback"

    return "scope_refine_repair", "heuristic:default"


def _relative(path: Optional[Path], base: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _load_context_cache(logs_root: Path) -> Dict[Path, RunContext]:
    cache: Dict[Path, RunContext] = {}
    for state_path in logs_root.glob("**/mas_state/final_video_state.json"):
        run_dir = state_path.parent.parent
        cache[run_dir] = _load_run_context(run_dir, logs_root)

    for debugger_dir in logs_root.glob("**/coder_debugger"):
        run_dir = debugger_dir.parent
        cache.setdefault(run_dir, _load_run_context(run_dir, logs_root))

    for timeout_dir in logs_root.glob("**/coder_runtime_timeouts"):
        run_dir = timeout_dir.parent
        cache.setdefault(run_dir, _load_run_context(run_dir, logs_root))

    return cache


def _group_attempt_paths(logs_root: Path) -> Dict[Tuple[Path, str], List[Path]]:
    grouped: Dict[Tuple[Path, str], List[Path]] = defaultdict(list)
    for attempt_path in logs_root.glob("**/coder_debugger/*/attempt_*/attempt.json"):
        section_dir = attempt_path.parent.parent
        run_dir = section_dir.parent.parent
        grouped[(run_dir, section_dir.name)].append(attempt_path)

    for paths in grouped.values():
        paths.sort(key=lambda path: int(re.search(r"attempt_(\d+)", str(path.parent)).group(1)))
    return grouped


def _extract_gold_incidents(logs_root: Path, context_cache: Dict[Path, RunContext]) -> List[Dict[str, object]]:
    incidents: List[Dict[str, object]] = []
    grouped_paths = _group_attempt_paths(logs_root)

    for (run_dir, section_id), attempt_paths in sorted(grouped_paths.items()):
        run_context = context_cache.setdefault(run_dir, _load_run_context(run_dir, logs_root))
        attempts_payload = [_read_json(path) for path in attempt_paths]
        later_success_lookup: Dict[int, bool] = {}
        future_success = False

        for path, payload in reversed(list(zip(attempt_paths, attempts_payload))):
            attempt_number = int(payload.get("attempt_number") or 0)
            render_outcome = payload.get("render_outcome") or {}
            is_success = bool(isinstance(render_outcome, dict) and render_outcome.get("ok"))
            later_success_lookup[attempt_number] = future_success or is_success
            future_success = future_success or is_success

        for attempt_path, payload in zip(attempt_paths, attempts_payload):
            render_outcome = payload.get("render_outcome") or {}
            if not isinstance(render_outcome, dict):
                render_outcome = {}

            precheck_error = str(payload.get("precheck_error") or "")
            render_error = str(payload.get("render_error") or "")
            timed_out = bool(render_outcome.get("timed_out"))
            is_failure = bool(precheck_error.strip() or render_error.strip() or not render_outcome.get("ok", False))
            if not is_failure:
                continue

            attempt_dir = attempt_path.parent
            input_code_path = attempt_dir / "input.py"
            repaired_code_path = attempt_dir / "repaired.py"
            stderr_path = attempt_dir / "render.stderr.txt"

            input_code = _read_text(input_code_path)
            repaired_code = _read_text(repaired_code_path)
            raw_error_text = precheck_error or render_error or _read_text(stderr_path)
            normalized_error = _normalize_error_text(raw_error_text, run_dir)
            exception_type = _extract_exception_type(normalized_error)
            label_coarse_raw, label_fine = _classify_incident(
                precheck_error=precheck_error,
                raw_error_text=raw_error_text,
                normalized_error_text=normalized_error,
                timed_out=timed_out,
                render_status="failed",
            )
            label_coarse = _map_to_model1_label(label_coarse_raw)
            if label_coarse is None:
                continue
            attempt_number = int(payload.get("attempt_number") or 0)
            line_number = _extract_failing_line_number(raw_error_text, section_id)
            code_excerpt = _extract_code_excerpt(input_code, line_number)

            record: Dict[str, object] = {
                "incident_id": hashlib.sha1(
                    f"gold::{run_context.run_id}::{section_id}::{attempt_number}".encode("utf-8")
                ).hexdigest()[:16],
                "source_type": "coder_debugger",
                "quality_tier": "gold",
                "run_id": run_context.run_id,
                "topic": run_context.topic,
                "section_id": section_id,
                "section_title": run_context.section_titles.get(section_id, ""),
                "coder_agent": run_context.coder_assignments.get(section_id, ""),
                "attempt_number": attempt_number,
                "snapshot_name": "",
                "turn_index": None,
                "phase": "coder_debugger",
                "label_coarse": label_coarse,
                "label_fine": label_fine,
                "exception_type": exception_type,
                "render_status": "failed",
                "timed_out": timed_out,
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "failing_line_number": line_number,
                "precheck_error": precheck_error,
                "render_error": render_error,
                "raw_error_text": raw_error_text,
                "raw_error_digest": _error_digest(raw_error_text),
                "normalized_error_text": normalized_error,
                "input_code": input_code,
                "repaired_code": repaired_code,
                "code_excerpt": code_excerpt,
                "repair_strategy": (payload.get("repair_outcome") or {}).get("strategy", "")
                if isinstance(payload.get("repair_outcome"), dict)
                else str(payload.get("repair_strategy") or ""),
                "repair_reason": (payload.get("repair_outcome") or {}).get("reason", "")
                if isinstance(payload.get("repair_outcome"), dict)
                else str(payload.get("repair_reason") or ""),
                "repair_changed": (payload.get("repair_outcome") or {}).get("changed", False)
                if isinstance(payload.get("repair_outcome"), dict)
                else bool(payload.get("changed_code")),
                "later_attempt_succeeded": later_success_lookup.get(attempt_number, False),
                "resolved_later_in_run": run_context.final_render_status.get(section_id) == "ok",
                "final_section_status": run_context.final_render_status.get(section_id),
                "artifact_paths": {
                    "attempt_json": _relative(attempt_path, logs_root),
                    "input_code": _relative(input_code_path if input_code_path.exists() else None, logs_root),
                    "repaired_code": _relative(repaired_code_path if repaired_code_path.exists() else None, logs_root),
                    "render_stderr": _relative(stderr_path if stderr_path.exists() else None, logs_root),
                },
            }
            record["input_text"] = _build_model1_input_text(record)
            incidents.append(record)

    return incidents


def _extract_timeout_incidents(logs_root: Path, context_cache: Dict[Path, RunContext]) -> List[Dict[str, object]]:
    incidents: List[Dict[str, object]] = []

    for timeout_path in sorted(logs_root.glob("**/coder_runtime_timeouts/*/*.json")):
        payload = _read_json(timeout_path)
        run_dir = timeout_path.parents[2]
        run_context = context_cache.setdefault(run_dir, _load_run_context(run_dir, logs_root))
        section_id = str(payload.get("section_id") or timeout_path.parent.name)
        raw_error_text = str(payload.get("stderr") or payload.get("stdout") or "")
        raw_error_text = raw_error_text or f"Render timed out after {payload.get('elapsed_seconds')}s."
        normalized_error = _normalize_error_text(raw_error_text, run_dir)
        code_snapshot_path = Path(str(payload.get("code_snapshot_path") or ""))
        input_code = _read_text(code_snapshot_path) if code_snapshot_path.exists() else ""
        line_number = _extract_failing_line_number(raw_error_text, section_id)
        code_excerpt = _extract_code_excerpt(input_code, line_number)

        record: Dict[str, object] = {
            "incident_id": hashlib.sha1(f"timeout::{run_context.run_id}::{timeout_path}".encode("utf-8")).hexdigest()[:16],
            "source_type": "coder_runtime_timeout",
            "quality_tier": "gold",
            "run_id": run_context.run_id,
            "topic": run_context.topic,
            "section_id": section_id,
            "section_title": run_context.section_titles.get(section_id, ""),
            "coder_agent": run_context.coder_assignments.get(section_id, ""),
            "attempt_number": int(payload.get("attempt_number") or 0) or None,
            "snapshot_name": "",
            "turn_index": None,
            "phase": "coder_runtime_timeout",
            "label_coarse": "performance_or_timeout",
            "label_fine": "timeout",
            "exception_type": _extract_exception_type(normalized_error),
            "render_status": "failed",
            "timed_out": True,
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "failing_line_number": line_number,
            "precheck_error": "",
            "render_error": raw_error_text,
            "raw_error_text": raw_error_text,
            "raw_error_digest": _error_digest(raw_error_text),
            "normalized_error_text": normalized_error or "Render timed out.",
            "input_code": input_code,
            "repaired_code": "",
            "code_excerpt": code_excerpt,
            "repair_strategy": "",
            "repair_reason": "",
            "repair_changed": False,
            "later_attempt_succeeded": None,
            "resolved_later_in_run": run_context.final_render_status.get(section_id) == "ok",
            "final_section_status": run_context.final_render_status.get(section_id),
            "artifact_paths": {
                "timeout_json": _relative(timeout_path, logs_root),
                "code_snapshot": _relative(code_snapshot_path if code_snapshot_path.exists() else None, logs_root),
            },
        }
        record["input_text"] = _build_model1_input_text(record)
        incidents.append(record)

    return incidents


def _dedupe_incidents(incidents: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    priority = {
        "coder_debugger": 0,
        "coder_runtime_timeout": 1,
    }
    best_by_key: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for record in incidents:
        key = (
            str(record.get("run_id") or ""),
            str(record.get("section_id") or ""),
            str(record.get("raw_error_digest") or ""),
            hashlib.sha1(str(record.get("code_excerpt") or "").encode("utf-8")).hexdigest()[:12],
        )
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = record
            continue

        current_priority = priority.get(str(record.get("source_type") or ""), 99)
        existing_priority = priority.get(str(existing.get("source_type") or ""), 99)
        if current_priority < existing_priority:
            best_by_key[key] = record

    return list(best_by_key.values())


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_training_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TRAINING_CSV_FIELDS})


def _build_repair_action_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    repair_rows: List[Dict[str, object]] = []
    for row in rows:
        action_label, action_source = _derive_repair_action(row)
        base_payload = {
            "base_incident_id": row.get("incident_id", ""),
            "source_type": row.get("source_type", ""),
            "quality_tier": row.get("quality_tier", ""),
            "run_id": row.get("run_id", ""),
            "topic": row.get("topic", ""),
            "section_id": row.get("section_id", ""),
            "section_title": row.get("section_title", ""),
            "attempt_number": row.get("attempt_number", ""),
            "phase": row.get("phase", ""),
            "repair_action": action_label,
            "repair_action_source": action_source,
            "label_coarse": row.get("label_coarse", ""),
            "label_fine": row.get("label_fine", ""),
            "exception_type": row.get("exception_type", ""),
            "render_status": row.get("render_status", ""),
            "timed_out": row.get("timed_out", ""),
            "elapsed_seconds": row.get("elapsed_seconds", ""),
            "failing_line_number": row.get("failing_line_number", ""),
            "normalized_error_text": row.get("normalized_error_text", ""),
            "code_excerpt": row.get("code_excerpt", ""),
            "repair_strategy": row.get("repair_strategy", ""),
            "repair_reason": row.get("repair_reason", ""),
            "later_attempt_succeeded": row.get("later_attempt_succeeded", ""),
            "resolved_later_in_run": row.get("resolved_later_in_run", ""),
            "final_section_status": row.get("final_section_status", ""),
        }

        view_specs = [
            ("full_context", dict(include_topic=True, include_exception=True, include_code_excerpt=True)),
            ("masked_exception", dict(include_topic=True, include_exception=False, include_code_excerpt=True)),
            ("error_only", dict(include_topic=False, include_exception=False, include_code_excerpt=False)),
        ]
        for view_name, view_kwargs in view_specs:
            input_text = _build_repair_action_input_text(row, **view_kwargs)
            repair_row = dict(base_payload)
            repair_row["view_name"] = view_name
            repair_row["input_text"] = input_text
            suffix = hashlib.sha1(f"{row.get('incident_id')}::{view_name}".encode("utf-8")).hexdigest()[:8]
            repair_row["incident_id"] = f"{row.get('incident_id', '')}-{suffix}"
            repair_rows.append(repair_row)
    return repair_rows


def _write_repair_action_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPAIR_ACTION_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPAIR_ACTION_CSV_FIELDS})


def _summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_source_type = Counter(str(row.get("source_type") or "") for row in rows)
    by_quality_tier = Counter(str(row.get("quality_tier") or "") for row in rows)
    by_label_coarse = Counter(str(row.get("label_coarse") or "") for row in rows)
    by_label_fine = Counter(str(row.get("label_fine") or "") for row in rows)
    by_exception_type = Counter(str(row.get("exception_type") or "") for row in rows if row.get("exception_type"))
    top_topics = Counter(str(row.get("topic") or "") for row in rows if row.get("topic"))

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "incident_count": len(rows),
        "by_source_type": dict(by_source_type.most_common()),
        "by_quality_tier": dict(by_quality_tier.most_common()),
        "by_label_coarse": dict(by_label_coarse.most_common()),
        "by_label_fine": dict(by_label_fine.most_common()),
        "by_exception_type": dict(by_exception_type.most_common()),
        "top_topics": dict(top_topics.most_common(20)),
        "label_space": MODEL1_LABELS,
    }


def _summarize_repair_actions(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_action = Counter(str(row.get("repair_action") or "") for row in rows)
    by_action_source = Counter(str(row.get("repair_action_source") or "") for row in rows)
    by_view = Counter(str(row.get("view_name") or "") for row in rows)
    by_base_incident = Counter(str(row.get("base_incident_id") or "") for row in rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": len(rows),
        "base_incident_count": len(by_base_incident),
        "views_per_incident": dict(by_view.most_common()),
        "by_repair_action": dict(by_action.most_common()),
        "by_repair_action_source": dict(by_action_source.most_common()),
        "label_space": REPAIR_ACTION_LABELS,
    }


def _write_readme(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# MAS Incident Dataset",
        "",
        f"Generated from `mas_logs` on `{summary['generated_at_utc']}`.",
        "",
        "## Files",
        "",
        "- `all_incidents.jsonl`: Rich master dataset with metadata, labels, code excerpts, and training input text.",
        "- `incidents_train.csv`: Flattened training export for Model 1.",
        "- `repair_actions_train.csv`: One row per incident with a derived repair-action label.",
        "- `repair_actions_train_augmented.csv`: Multi-view repair-action export for ablation/augmentation experiments.",
        "- `summary.json`: Aggregate counts by source, label, and exception type.",
        "- `repair_actions_summary.json`: Aggregate counts for derived repair actions and view variants.",
        "",
        "## Counts",
        "",
        f"- Total incidents: `{summary['incident_count']}`",
    ]

    by_quality = summary.get("by_quality_tier", {})
    if isinstance(by_quality, dict):
        for tier, count in by_quality.items():
            lines.append(f"- {tier.title()} incidents: `{count}`")

    lines.extend(
        [
            "",
            "## Model 1 Input",
            "",
            "Each training row includes an `input_text` field with this format:",
            "",
            "```text",
            "Topic: ...",
            "Section: ...",
            "Exception: ...",
            "Render status: ...",
            "Timed out: ...",
            "",
            "Error:",
            "...",
            "",
            "Code excerpt:",
            "...",
            "```",
            "",
            "## Coarse Labels",
            "",
        ]
    )
    for label in MODEL1_LABELS:
        lines.append(f"- `{label}`")

    lines.extend(
        [
            "",
            "## Repair Action Labels",
            "",
        ]
    )
    for label in REPAIR_ACTION_LABELS:
        lines.append(f"- `{label}`")

    lines.extend(
        [
            "",
            "## Repair Action Augmentation Views",
            "",
            "- `full_context`: topic + exception + code excerpt",
            "- `masked_exception`: hides the explicit exception field",
            "- `error_only`: only render status/timed-out/error text",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=repo_root / "mas_logs",
        help="Directory containing MAS run logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "datasets" / "mas_incidents",
        help="Directory to write dataset artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = args.logs_root.resolve()
    output_dir = args.output_dir.resolve()

    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root does not exist: {logs_root}")

    context_cache = _load_context_cache(logs_root)
    gold_incidents = _extract_gold_incidents(logs_root, context_cache)
    timeout_incidents = _extract_timeout_incidents(logs_root, context_cache)
    all_incidents = _dedupe_incidents([*gold_incidents, *timeout_incidents])

    all_incidents.sort(
        key=lambda row: (
            str(row.get("run_id") or ""),
            str(row.get("section_id") or ""),
            str(row.get("source_type") or ""),
            int(row.get("attempt_number") or 0),
            int(row.get("turn_index") or 0),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "all_incidents.jsonl", all_incidents)
    _write_training_csv(output_dir / "incidents_train.csv", all_incidents)
    repair_action_rows = _build_repair_action_rows(all_incidents)
    _write_repair_action_csv(output_dir / "repair_actions_train_augmented.csv", repair_action_rows)
    repair_action_base_rows = [row for row in repair_action_rows if row.get("view_name") == "full_context"]
    _write_repair_action_csv(output_dir / "repair_actions_train.csv", repair_action_base_rows)
    summary = _summarize(all_incidents)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "repair_actions_summary.json", _summarize_repair_actions(repair_action_rows))
    _write_readme(output_dir / "README.md", summary)

    print(f"Wrote {len(all_incidents)} incidents to {output_dir}")
    print(f"Wrote {len(repair_action_base_rows)} repair-action rows to {output_dir / 'repair_actions_train.csv'}")
    print(
        f"Wrote {len(repair_action_rows)} augmented repair-action rows to "
        f"{output_dir / 'repair_actions_train_augmented.csv'}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
