"""Shared, lightweight normalisation for generated Code2Video section code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


def _load_teaching_scene_base_class() -> str:
    base_class_path = Path(__file__).resolve().parent.parent / "prompts" / "base_class.py"
    module_scope = {}
    exec(base_class_path.read_text(encoding="utf-8"), module_scope)
    base_class = module_scope.get("base_class")
    if not isinstance(base_class, str) or not base_class.strip():
        raise ValueError(f"No base_class string found in {base_class_path}")
    return base_class.strip()


def _replace_teaching_scene_base_class(code: str, new_class_def: str) -> str:
    lines = code.splitlines(keepends=True)
    class_start = None
    class_end = None

    for index, line in enumerate(lines):
        if re.match(r"^\s*class\s+TeachingScene\s*\(Scene\)\s*:", line):
            class_start = index
            break

    if class_start is not None:
        base_indent = len(lines[class_start]) - len(lines[class_start].lstrip())
        class_end = class_start + 1
        while class_end < len(lines):
            line = lines[class_end]
            if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                break
            class_end += 1
        new_block = new_class_def.strip() + "\n\n"
        return "".join(lines[:class_start]) + new_block + "".join(lines[class_end:])

    insert_position = 0
    for index, line in enumerate(lines):
        if re.match(r"^\s*class\s+\w+", line):
            insert_position = index
            break
    new_block = new_class_def.strip() + "\n\n"
    return "".join(lines[:insert_position]) + new_block + "".join(lines[insert_position:])


def normalize_code_to_code2video(
    code: str,
    section_id: str,
    section_title: str,
    lecture_lines: List[str],
    *,
    teaching_scene_base_class: Optional[str] = None,
) -> str:
    """Apply the exact deterministic transformation used before shared-state storage."""
    del section_id, section_title, lecture_lines  # Reserved for future normalisation rules.
    normalized = (code or "").strip()
    if not normalized:
        return normalized

    if "```python" in normalized:
        normalized = normalized.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in normalized:
        normalized = normalized.split("```", 1)[1].split("```", 1)[0].strip()

    base_class = teaching_scene_base_class or _load_teaching_scene_base_class()
    normalized = _replace_teaching_scene_base_class(normalized, base_class)
    return normalized.strip() + "\n"
