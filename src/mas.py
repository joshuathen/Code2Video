from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Lock
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from type_utils import *
from external_assets import process_storyboard_with_assets
from gpt_request import request_gemini_token, request_gemini_video_img_token
from scope_refine import ScopeRefineFixer, GridPositionExtractor
from utils import extract_answer_from_response
from utils import extract_json_from_markdown
from utils import replace_base_class
from utils import topic_to_safe_name

try:
    from prompts import get_prompt1_outline, get_prompt3_code, get_prompt4_layout_feedback, get_regenerate_note
except ModuleNotFoundError:
    # Allow running `python src/mas.py` without manually setting PYTHONPATH.
    import sys

    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from prompts import get_prompt1_outline, get_prompt3_code, get_prompt4_layout_feedback, get_regenerate_note

from google.genai import types, Client
from google.genai.errors import ClientError, ServerError

# MAS Team
ORCHESTRATOR = "Orchestrator"
CODER = "Coder"
ANIMATION_PLANNER = "AnimationPlanner"
SCRIPT_WRITER = "ScriptWriter"

# Team role summaries for clearer cross-agent issue routing.
CODER_ROLE = "Builds and debugs Manim code; ensures animation plans are technically feasible."
ANIMATION_PLANNER_ROLE = "Designs animation steps and visual flow; keeps stages clear and codable."
SCRIPT_WRITER_ROLE = "Writes concise lecture lines; aligns each line with a matching animation stage."


# Global Settings
GLOBAL_MAX_TURNS = 3
MAX_RETRIES = 3
DEFAULT_TURN_CODER_FIX_ATTEMPTS = 3
DEFAULT_FINAL_RENDER_FIX_ATTEMPTS = 10
DEFAULT_CODER_REGENERATE_TRIES = 10
# Default topic for standalone `python src/mas.py` runs.
TOPIC = "Implicit Differentiation"
OUTLINE_DURATION_MINUTES = 5
OUTLINE_MAX_REGENERATE_TRIES = 10
OUTLINE_MODEL = "gemini-3-flash-preview"
# ScopeRefine/bug-fix model used by MAS coder runtime.
DEBUG_FIX_MODEL = "gemini-3-flash-preview"
# Debugger thinking controls:
# - thinking_budget: 0 = disabled, -1 = automatic, positive int = token budget.
# - include_thoughts: include thought parts in response payload.
DEBUG_FIX_THINKING_BUDGET: Optional[int] = None
DEBUG_FIX_INCLUDE_THOUGHTS: Optional[bool] = None

# Prompt blocks aligned to original Code2Video stage2/stage3 templates.
STAGE2_STORYBOARD_REQUIREMENTS = """
## Storyboard Requirements

### Content Structure
- For key sections (max 3 sections), use up to 5 lecture lines along with their corresponding 5 animations to provide a logically coherent explanation. Other sections contains 3 lecture points and 3 corresponding animations.
- In key sections, assets not forbiddened.
- Must keep each lecture line brief [NO MORE THAN 10 WORDS FOR ONE LINE].
- Animation steps must closely correspond to lecture points.
- Do not apply any animation to lecture lines except for changing the color of corresponding line when its related animation is presented.

### Visual Design
- Colors: Background fixed at #000000, use ligt color for contrast.
- IMPORTANT: Provide hexadecimal codes for colors.
- Element Labeling: Assign clear colors and labels near all elements (formulas, etc.).

### Animation Effects
- Basic Animations: Appearance, movement, color changes, fade in/out, scaling.
- Emphasis Effects: Flashing, color changes, bolding to highlight key knowledge points.

### Constraints
- No panels or 3D methods.
- Avoid coordinate axes unless absolutely necessary.
- Focus animations on visualizing concepts that are difficult to grasp from lecture lines alone.
- Ensure that all animations are easy to understand.
- Do not introduce new external elements on your own (such as SVGs or downloaded assets) unless they are already provided as existing [Asset: ...] references.
- If existing [Asset: ...] references are present in the storyboard, preserve them in the animation descriptions and do not remove them.
""".strip()

STAGE3_CODE_REQUIREMENTS = """
1. Basic Requirements:
- Use the provided TeachingScene base class without modification.
- Each lecture line must have a matching color with its corresponding animation elements.
- Apply ONLY color changes to lecture lines - no scaling, translation, or Transform animations.

2. Visual Anchor System (MANDATORY):
- Use 6x6 grid system (A1-F6) for precise positioning.
- Pay attention to the positioning of elements to avoid occlusions (e.g., labels and formulas).
- All labels must be positioned within 1 grid unit of their corresponding objects
- Grid layout (right side only):
```
lecture |  A1  A2  A3  A4  A5  A6
        |  B1  B2  B3  B4  B5  B6
        |  C1  C2  C3  C4  C5  C6
        |  D1  D2  D3  D4  D5  D6
        |  E1  E2  E3  E4  E5  E6
        |  F1  F2  F3  F4  F5  F6
```

3. POSITIONING METHODS:
- Point example: self.place_at_grid(obj, 'B2', scale_factor=0.8)
- Area example: self.place_in_area(obj, 'A1', 'C3', scale_factor=0.7)
- NEVER use .to_edge(), .move_to(), or manual positioning!

5. STRUCTURE FOR CODE:
Use the following comment format to indicate which block corresponds to which line:
```python
# === Animation for Lecture Line 1 ===
```

6. EXAMPLE STRUCTURE:
```python
from manim import *

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("SECTION_TITLE", ["LECTURE_LINE_1", "LECTURE_LINE_2"])

        # === Animation for Lecture Line 1 ===
        ...
```

7. MANDATORY CONSTRAINTS:
- Colors: Use light, distinguishable hexadecimal colors.
- Scaling: Maintain appropriate font sizes and object scales for readability.
- Consistency: Do not apply any animation to the lecture lines except for color changes; The lecture lines and title's size and position must remain unchanged.
- Assets: If provided, MUST use the elements in the Animation Description formatted as [Asset: XXX/XXX.png] (abstract path).
- Assets: Existing [Asset: ...] references are mandatory inputs. Do not remove, ignore, or replace them unless an issue explicitly instructs you to do so.
- Simplicity: Avoid 3D functions, complex panels, or external dependencies except for filenames in Animation Description.
""".strip()

CODER_GUIDELINES = STAGE3_CODE_REQUIREMENTS

ANIMATION_PLANNER_GUIDELINES = STAGE2_STORYBOARD_REQUIREMENTS

SCRIPT_WRITER_GUIDELINES = """
Follow the original Code2Video Stage-2 storyboard requirements and only update `lecture_lines`:

### Content Structure
- For key sections (max 3 sections), use up to 5 lecture lines along with their corresponding 5 animations to provide a logically coherent explanation. Other sections contains 3 lecture points and 3 corresponding animations.
- Must keep each lecture line brief [NO MORE THAN 10 WORDS FOR ONE LINE].
- Animation steps must closely correspond to lecture points.
- Do not apply any animation to lecture lines except for changing the color of corresponding line when its related animation is presented.

### Constraints
- No panels or 3D methods.
- Avoid coordinate axes unless absolutely necessary.
- Ensure that all animations are easy to understand.
- Only update lecture_lines. Do not remove or rewrite existing [Asset: ...] references in animations.
""".strip()

def _load_api_key() -> str:
    # Prefer environment variable overrides.
    env_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    # Fallback to local config file: src/api_config.json -> gemini.api_key
    cfg_path = Path(__file__).with_name("api_config.json")
    if not cfg_path.exists():
        raise ValueError(f"api_config.json not found at {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    key = cfg.get("gemini", {}).get("api_key")
    if not key:
        raise ValueError("Missing gemini.api_key in src/api_config.json")
    return key


def _load_iconfinder_api_key() -> str:
    env_key = os.getenv("ICONFINDER_API_KEY")
    if env_key:
        return env_key

    cfg_path = Path(__file__).with_name("api_config.json")
    if not cfg_path.exists():
        return ""

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return str(cfg.get("iconfinder", {}).get("api_key", "") or "")


API_KEY = _load_api_key()
client = Client(api_key=API_KEY)


def _empty_token_usage_bucket() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
    }


def _empty_token_usage_summary() -> Dict[str, object]:
    return {
        "totals": _empty_token_usage_bucket(),
        "by_source": {},
        "by_model": {},
    }


def _extract_token_usage(response: object = None, usage: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    bucket = _empty_token_usage_bucket()

    if isinstance(usage, dict):
        bucket["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        bucket["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
        bucket["total_tokens"] = int(
            usage.get(
                "total_tokens",
                bucket["prompt_tokens"] + bucket["completion_tokens"],
            )
            or 0
        )
        return bucket

    usage_md = getattr(response, "usage_metadata", None)
    if usage_md is not None:
        bucket["prompt_tokens"] = int(getattr(usage_md, "prompt_token_count", 0) or 0)
        bucket["completion_tokens"] = int(getattr(usage_md, "candidates_token_count", 0) or 0)
        bucket["total_tokens"] = int(
            getattr(
                usage_md,
                "total_token_count",
                bucket["prompt_tokens"] + bucket["completion_tokens"],
            )
            or 0
        )
        return bucket

    completion_usage = getattr(response, "usage", None)
    if completion_usage is not None:
        bucket["prompt_tokens"] = int(getattr(completion_usage, "prompt_tokens", 0) or 0)
        bucket["completion_tokens"] = int(getattr(completion_usage, "completion_tokens", 0) or 0)
        bucket["total_tokens"] = int(
            getattr(
                completion_usage,
                "total_tokens",
                bucket["prompt_tokens"] + bucket["completion_tokens"],
            )
            or 0
        )
        return bucket
    
    else:
        print("[Warning] Unable to extract token usage from response. No 'usage' or 'usage_metadata' found.")
        return bucket


class MASTokenTracker:
    def __init__(self, initial_summary: Optional[Dict[str, object]] = None):
        self._lock = Lock()
        self._summary = _empty_token_usage_summary()
        if isinstance(initial_summary, dict):
            self._merge_existing_summary(initial_summary)

    def _add_bucket_values(self, target: Dict[str, int], source: Dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count"):
            target[key] += int(source.get(key, 0) or 0)

    def _merge_existing_summary(self, summary: Dict[str, object]) -> None:
        totals = summary.get("totals")
        if isinstance(totals, dict):
            self._add_bucket_values(self._summary["totals"], totals)

        for group_name in ("by_source", "by_model"):
            raw_group = summary.get(group_name)
            if not isinstance(raw_group, dict):
                continue
            for key, bucket in raw_group.items():
                if not isinstance(bucket, dict):
                    continue
                group = self._summary[group_name]
                entry = group.setdefault(key, _empty_token_usage_bucket())
                self._add_bucket_values(entry, bucket)

    def record(
        self,
        source: str,
        response: object = None,
        usage: Optional[Dict[str, int]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, object]:
        usage_bucket = _extract_token_usage(response=response, usage=usage)
        usage_bucket["call_count"] = 1
        model_name = model or getattr(response, "model", None) or "unknown"

        with self._lock:
            self._add_bucket_values(self._summary["totals"], usage_bucket)

            by_source = self._summary["by_source"]
            source_entry = by_source.setdefault(source, _empty_token_usage_bucket())
            self._add_bucket_values(source_entry, usage_bucket)

            by_model = self._summary["by_model"]
            model_entry = by_model.setdefault(model_name, _empty_token_usage_bucket())
            self._add_bucket_values(model_entry, usage_bucket)

            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> Dict[str, object]:
        return {
            "totals": dict(self._summary["totals"]),
            "by_source": {name: dict(bucket) for name, bucket in self._summary["by_source"].items()},
            "by_model": {name: dict(bucket) for name, bucket in self._summary["by_model"].items()},
        }

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return self._snapshot_unlocked()

    def merge_summary(self, summary: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(summary, dict):
            return self.snapshot()
        with self._lock:
            self._merge_existing_summary(summary)
            return self._snapshot_unlocked()


def _format_token_usage_summary(summary: Dict[str, object]) -> str:
    totals = summary.get("totals", {})
    lines = [
        (
            "[TokenUsage] prompt={prompt_tokens} completion={completion_tokens} "
            "total={total_tokens} calls={call_count}"
        ).format(
            prompt_tokens=totals.get("prompt_tokens", 0),
            completion_tokens=totals.get("completion_tokens", 0),
            total_tokens=totals.get("total_tokens", 0),
            call_count=totals.get("call_count", 0),
        )
    ]

    by_source = summary.get("by_source", {})
    if isinstance(by_source, dict) and by_source:
        lines.append("[TokenUsage] Top sources:")
        ranked = sorted(
            by_source.items(),
            key=lambda item: item[1].get("total_tokens", 0),
            reverse=True,
        )
        for source_name, bucket in ranked[:8]:
            lines.append(
                (
                    "  - {source}: total={total_tokens} prompt={prompt_tokens} "
                    "completion={completion_tokens} calls={call_count}"
                ).format(source=source_name, **bucket)
            )

    return "\n".join(lines)


def _total_tokens_from_summary(summary: Dict[str, object]) -> int:
    totals = summary.get("totals", {})
    if not isinstance(totals, dict):
        return 0
    return int(totals.get("total_tokens", 0) or 0)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump())
        except Exception:
            pass

    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            return _json_safe(to_json_dict())
        except Exception:
            pass

    raw_dict = getattr(value, "__dict__", None)
    if isinstance(raw_dict, dict) and raw_dict:
        return {str(key): _json_safe(val) for key, val in raw_dict.items() if not str(key).startswith("_")}

    return str(value)


def _response_tool_trace(response: object) -> Dict[str, object]:
    function_calls: List[Dict[str, object]] = []
    function_responses: List[Dict[str, object]] = []
    text_parts: List[str] = []

    candidates = getattr(response, "candidates", None) or []
    for candidate_index, candidate in enumerate(candidates):
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part_index, part in enumerate(parts):
            text_value = getattr(part, "text", None)
            if text_value:
                text_parts.append(str(text_value))

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                function_calls.append(
                    {
                        "candidate_index": candidate_index,
                        "part_index": part_index,
                        "name": getattr(function_call, "name", None),
                        "args": _json_safe(getattr(function_call, "args", None)),
                        "id": getattr(function_call, "id", None),
                    }
                )

            function_response = getattr(part, "function_response", None)
            if function_response is not None:
                response_payload = getattr(function_response, "response", None)
                function_responses.append(
                    {
                        "candidate_index": candidate_index,
                        "part_index": part_index,
                        "name": getattr(function_response, "name", None),
                        "id": getattr(function_response, "id", None),
                        "response": _json_safe(response_payload if response_payload is not None else function_response),
                    }
                )

    return {
        "function_calls": function_calls,
        "function_responses": function_responses,
        "text_parts": text_parts,
    }


class MASAgentCallLogger:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._lock = Lock()

    def _append(self, event: Dict[str, object]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_response(self, agent_name: str, model: str, response: object) -> None:
        trace = _response_tool_trace(response)
        self._append(
            {
                "event_type": "agent_response",
                "agent": agent_name,
                "model": model,
                "usage": _extract_token_usage(response=response),
                "function_calls": trace["function_calls"],
                "function_responses": trace["function_responses"],
                "text_parts": trace["text_parts"],
            }
        )

    def record_error(
        self,
        agent_name: str,
        model: str,
        error: Exception,
        *,
        stage: str,
        attempt: int,
        max_retries: int,
        retrying: bool,
    ) -> None:
        self._append(
            {
                "event_type": "agent_error",
                "agent": agent_name,
                "model": model,
                "stage": stage,
                "attempt": attempt,
                "max_retries": max_retries,
                "retrying": retrying,
                "error_type": error.__class__.__name__,
                "message": str(error),
            }
        )


def _final_video_output_path(video_state: "VideoMASState") -> Optional[Path]:
    if not video_state.run_output_dir:
        return None
    safe_name = topic_to_safe_name(video_state.topic) or "final_video"
    return Path(video_state.run_output_dir) / f"{safe_name}.mp4"


def _default_teaching_scene_base_class() -> str:
    return """
class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]

        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])

        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject
""".strip()


def _load_teaching_scene_base_class() -> str:
    base_class_path = Path(__file__).resolve().parent.parent / "prompts" / "base_class.py"
    try:
        module_scope: Dict[str, object] = {}
        exec(base_class_path.read_text(encoding="utf-8"), module_scope)
        loaded = module_scope.get("base_class")
        if isinstance(loaded, str) and loaded.strip():
            return loaded.strip()
    except Exception:
        pass
    return _default_teaching_scene_base_class()


TEACHING_SCENE_BASE_CLASS = _load_teaching_scene_base_class()


@dataclass
class Issue:
    id: int
    fromAgent: str
    toAgent: str
    description: str
    isActive: bool
    section_id: Optional[str] = None
    resolved: bool = False
    under_review: bool = False
    resolution_note: str = ""


@dataclass
class VideoMASState:
    topic: str
    target_audience: str
    code: List[str]
    outline: TeachingOutline
    storyboard: List[Section]
    turns_run: int = 0
    issues: List[Issue] = field(default_factory=list)
    finalised: bool = False
    coder_assignments: Dict[str, str] = field(default_factory=dict)
    section_id_to_index: Dict[str, int] = field(default_factory=dict)
    rendered_video_path: List[Optional[str]] = field(default_factory=list)
    render_status: List[Optional[str]] = field(default_factory=list)
    render_error: List[str] = field(default_factory=list)
    video_review: List[Optional[Dict[str, object]]] = field(default_factory=list)
    run_output_dir: str = ""
    token_usage: Dict[str, object] = field(default_factory=_empty_token_usage_summary)

    def __post_init__(self) -> None:
        self.topic = self.outline.topic
        self.target_audience = self.outline.target_audience

        section_ids = [section.id for section in self.storyboard]
        if not section_ids:
            raise ValueError("VideoMASState.storyboard must contain at least one section.")
        if len(section_ids) != len(set(section_ids)):
            raise ValueError(f"Duplicate section ids detected in storyboard: {section_ids}")

        self.section_id_to_index = {sid: idx for idx, sid in enumerate(section_ids)}
        section_count = len(section_ids)

        if len(self.code) < section_count:
            self.code.extend([""] * (section_count - len(self.code)))
        elif len(self.code) > section_count:
            self.code = self.code[:section_count]

        if not self.coder_assignments:
            self.coder_assignments = {sid: f"Coder{idx + 1}" for idx, sid in enumerate(section_ids)}

        def _normalise_len(values: List[object], fill_value: object) -> List[object]:
            if len(values) < section_count:
                values.extend([fill_value] * (section_count - len(values)))
            elif len(values) > section_count:
                del values[section_count:]
            return values

        self.rendered_video_path = _normalise_len(self.rendered_video_path, None)
        self.render_status = _normalise_len(self.render_status, None)
        self.render_error = _normalise_len(self.render_error, "")
        self.video_review = _normalise_len(self.video_review, None)
        if not isinstance(self.token_usage, dict):
            self.token_usage = _empty_token_usage_summary()

    def unresolved_issues(self) -> List[Issue]:
        return [x for x in self.issues if not x.resolved]

    def active_issues(self) -> List[Issue]:
        return [x for x in self.issues if x.isActive and not x.resolved]

    def section_ids(self) -> List[str]:
        return [section.id for section in self.storyboard]

    def section_index(self, section_id: str) -> int:
        if section_id not in self.section_id_to_index:
            raise ValueError(f"Unknown section_id '{section_id}'. Valid ids: {self.section_ids()}")
        return self.section_id_to_index[section_id]

    def section_outline(self, section_id: str) -> Dict[str, object]:
        for section_data in self.outline.sections:
            if section_data["id"] == section_id:
                return section_data
        raise ValueError(f"Section '{section_id}' not found in outline.")

    def code_for(self, section_id: str) -> str:
        return self.code[self.section_index(section_id)]

    def set_code_for(self, section_id: str, code: str) -> None:
        self.code[self.section_index(section_id)] = code

    def section_summaries(self) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        for idx, section in enumerate(self.storyboard):
            summaries.append(
                {
                    "id": section.id,
                    "title": section.title,
                    "lecture_lines_count": len(section.lecture_lines),
                    "animations_count": len(section.animations),
                    "has_code": bool((self.code[idx] or "").strip()),
                    "render_status": self.render_status[idx],
                    "rendered_video_path": self.rendered_video_path[idx],
                    "render_error": self.render_error[idx],
                }
            )
        return summaries


def _save_video_state_json(video_state: VideoMASState, logs_dir: Path, filename: str) -> Path:
    output_path = logs_dir / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(video_state), f, ensure_ascii=False, indent=2)
    return output_path


def _scene_name_from_section_id(section_id: str) -> str:
    return f"{section_id.title().replace('_', '')}Scene"


def normalize_code_to_code2video(
    code: str,
    section_id: str,
    section_title: str,
    lecture_lines: List[str],
) -> str:
    normalized = (code or "").strip()
    if not normalized:
        return normalized

    if "```python" in normalized:
        normalized = normalized.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in normalized:
        normalized = normalized.split("```", 1)[1].split("```", 1)[0].strip()

    normalized = replace_base_class(normalized, TEACHING_SCENE_BASE_CLASS)
    return normalized.strip() + "\n"


def _extract_text_from_gemini_response(response) -> str:
    """
    Extract concatenated text parts without calling `response.text` to avoid
    warnings when non-text parts (e.g., thought_signature) are present.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []

    text_parts: List[str] = []
    for part in parts:
        text_value = getattr(part, "text", None)
        if text_value:
            text_parts.append(text_value)

    return "\n".join(text_parts).strip()


def _extract_python_code_from_response(response: object) -> str:
    text = _extract_text_from_gemini_response(response)
    if not text:
        try:
            text = str(response.choices[0].message.content or "").strip()
        except Exception:
            text = str(response)

    if not text:
        return ""

    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    # Fallback for models that return bare code without fences.
    if re.search(r"^\s*class\s+\w+Scene\b", text, re.MULTILINE) and re.search(
        r"^\s*def\s+construct\s*\(",
        text,
        re.MULTILINE,
    ):
        return text.strip()

    return ""


def _scope_refine_request_with_client(
    request_client: Client,
    prompt: str,
    max_tokens: int = 8000,
    token_tracker: Optional[MASTokenTracker] = None,
    usage_source: str = "scope_refine",
):
    """
    ScopeRefine request adapter aligned with agent.py's Gemini wrapper.

    The `request_client` parameter is kept for compatibility with older call
    sites, but the actual request path intentionally mirrors agent.py by
    going through `request_gemini_token(...)`.
    """
    del request_client
    response, usage = request_gemini_token(prompt, max_tokens=max_tokens)
    if token_tracker is not None:
        token_tracker.record(
            source=usage_source,
            usage=usage,
            response=response,
        )

    return response, usage


class CoderRuntime:
    """
    Post-codegen runtime for MAS sections.

    The scheduler in MAS decides *which* sections should be debugged.
    Once a section reaches this runtime, the repair loop intentionally
    mirrors the classic `agent.py` `debug_and_fix_code(...)` workflow as
    closely as possible so ScopeRefine behavior stays aligned.
    """

    def __init__(
        self,
        runtime_dir: Path,
        request_fn: Callable,
        max_code_token_length: int = 10000,
    ):
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.request_fn = request_fn
        self.max_code_token_length = max_code_token_length

    def _scene_name(self, section_id: str) -> str:
        return _scene_name_from_section_id(section_id)

    def _find_video_path(self, section_id: str, scene_name: str) -> Optional[Path]:
        video_patterns = [
            self.runtime_dir / "media" / "videos" / section_id / "480p15" / f"{scene_name}.mp4",
            self.runtime_dir / "media" / "videos" / "480p15" / f"{scene_name}.mp4",
        ]
        for path in video_patterns:
            if path.exists():
                return path
        return None

    def debug_and_fix(
        self,
        section_id: str,
        code: str,
        max_fix_attempts: int = 3,
        section_title: str = "",
        lecture_lines: Optional[List[str]] = None,
        topic: str = "",
    ) -> Dict[str, object]:
        if not code or not code.strip():
            return {"success": False, "status": "skipped", "reason": "No code available for section"}

        scene_name = self._scene_name(section_id)
        code_file = self.runtime_dir / f"{section_id}.py"
        current_code = code
        code_file.write_text(current_code, encoding="utf-8")

        last_error = ""
        topic_prefix = f"{topic} " if topic else ""
        for fix_attempt in range(max_fix_attempts):
            print(
                f"🔧 {topic_prefix}Debugging {section_id} (attempt {fix_attempt + 1}/{max_fix_attempts})",
                flush=True,
            )
            try:
                result = subprocess.run(
                    ["manim", "-ql", code_file.name, scene_name],
                    capture_output=True,
                    text=True,
                    cwd=self.runtime_dir,
                    timeout=180,
                )

                if result.returncode == 0:
                    video_patterns = [
                        self.runtime_dir / "media" / "videos" / f"{code_file.name.replace('.py', '')}" / "480p15" / f"{scene_name}.mp4",
                        self.runtime_dir / "media" / "videos" / "480p15" / f"{scene_name}.mp4",
                    ]
                    for video_path in video_patterns:
                        if video_path.exists():
                            print(f"✅ {topic_prefix}{section_id} finished", flush=True)
                            return {
                                "success": True,
                                "status": "ok",
                                "attempts": fix_attempt + 1,
                                "code": current_code,
                                "video_path": str(video_path),
                            }

                last_error = (result.stderr or "").strip()
                scope_refine_fixer = ScopeRefineFixer(self.request_fn, self.max_code_token_length)
                fixed_code = scope_refine_fixer.fix_code_smart(section_id, current_code, result.stderr, self.runtime_dir)

                if fixed_code:
                    current_code = fixed_code
                    code_file.write_text(fixed_code, encoding="utf-8")
                else:
                    break
            except subprocess.TimeoutExpired:
                print(f"❌ {topic_prefix}{section_id} timed out", flush=True)
                break
            except Exception as e:
                print(f"❌ {topic_prefix}{section_id} failed with exception: {e}", flush=True)
                break

        return {
            "success": False,
            "status": "failed",
            "attempts": max_fix_attempts,
            "code": current_code,
            "error": last_error,
        }


def _run_coder_runtime_worker(
    *,
    runtime_dir: str,
    topic: str,
    section_id: str,
    code: str,
    max_fix_attempts: int,
    max_code_token_length: int,
    section_title: str,
    lecture_lines: List[str],
) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass

    local_token_tracker = MASTokenTracker()
    runtime_path = Path(runtime_dir)
    normalized_code = normalize_code_to_code2video(
        code=code,
        section_id=section_id,
        section_title=section_title,
        lecture_lines=lecture_lines,
    )

    runtime = CoderRuntime(
        runtime_dir=runtime_path,
        request_fn=lambda prompt, max_tokens=8000, section_id=section_id: _scope_refine_request_with_client(
            client,
            prompt,
            max_tokens=max_tokens,
            token_tracker=local_token_tracker,
            usage_source=f"scope_refine:{section_id}",
        ),
        max_code_token_length=max_code_token_length,
    )

    try:
        result = runtime.debug_and_fix(
            section_id=section_id,
            code=normalized_code,
            max_fix_attempts=max_fix_attempts,
            section_title=section_title,
            lecture_lines=lecture_lines,
            topic=topic,
        )
    except Exception as e:
        result = {
            "success": False,
            "status": "failed",
            "attempts": max_fix_attempts,
            "code": normalized_code,
            "error": str(e),
        }

    return section_id, result, local_token_tracker.snapshot()


class MASAgent:
    def __init__(
        self,
        name: str,
        role: str,
        guidelines: str,
        model: str,
        client: Client,
        tools: Optional[List[Callable]] = None,
        token_tracker: Optional[MASTokenTracker] = None,
    ):
        self.name = name
        self.role = role
        self.guidelines = guidelines
        self.model = model
        self.client = client
        self.tools = tools or []
        self.token_tracker = token_tracker
        self.call_logger: Optional[MASAgentCallLogger] = None
        self.original_model = model
        self.team_agents: List["MASAgent"] = []

    def set_team_agents(self, team_agents: List["MASAgent"]) -> None:
        self.team_agents = team_agents

    def run_with_retry(self, parts, tools, max_retries):
        attempts = 0

        alternative_models = ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"]
        # Models that are in the list can be alternated/substituted with each other (other models e.g. image/audio generation can't be substituted)
        model_index = alternative_models.index(self.model) if self.model in alternative_models else -1

        response = None

        while attempts < max_retries + 1:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(tools=tools),
                )
                if self.token_tracker is not None:
                    self.token_tracker.record(
                        source=f"agent:{self.name}",
                        response=response,
                        model=self.model,
                    )
                if self.call_logger is not None:
                    self.call_logger.record_response(
                        agent_name=self.name,
                        model=self.model,
                        response=response,
                    )
                break  # success
            except (ClientError, ServerError) as e:
                msg = str(e)
                attempts += 1
                retrying = attempts < (max_retries + 1)
                if self.call_logger is not None:
                    self.call_logger.record_error(
                        agent_name=self.name,
                        model=self.model,
                        error=e,
                        stage="client_error",
                        attempt=attempts,
                        max_retries=max_retries,
                        retrying=retrying,
                    )

                if attempts >= max_retries + 1:
                    raise
                elif "RESOURCE_EXHAUSTED" in msg or  "UNAVAILABLE" in msg or "INTERNAL" in msg:
                    if model_index != -1 and attempts > 1:
                        model_index = (model_index + 1) % len(alternative_models)
                        self.model = alternative_models[model_index]
                    
                    delay = 15
                    print(f"[{self.name}] Retry with {self.model}; {attempts}/{max_retries} after {delay}s due to: {e}")
                    time.sleep(delay)
                else:
                    raise 
            except KeyError as e:
                attempts += 1
                retrying = attempts <= max_retries
                if self.call_logger is not None:
                    self.call_logger.record_error(
                        agent_name=self.name,
                        model=self.model,
                        error=e,
                        stage="tool_keyerror",
                        attempt=attempts,
                        max_retries=max_retries,
                        retrying=retrying,
                    )

                if attempts > max_retries:
                    raise

                delay = 10
                print(f"[{self.name}] Tool-calling KeyError {e!r} – likely bad function name from model. "
                    f"Retrying {attempts}/{max_retries} in {delay}s...")
                time.sleep(delay)
                continue
            except Exception as e:
                # Transport/network errors (e.g. httpx.ConnectError, DNS failures) are often transient.
                msg = str(e).lower()
                is_retryable_transport = (
                    "connecterror" in e.__class__.__name__.lower()
                    or "nodename nor servname provided" in msg
                    or "name or service not known" in msg
                    or "temporary failure in name resolution" in msg
                    or "connection reset by peer" in msg
                    or "timed out" in msg
                )

                if not is_retryable_transport:
                    if self.call_logger is not None:
                        self.call_logger.record_error(
                            agent_name=self.name,
                            model=self.model,
                            error=e,
                            stage="unexpected_error",
                            attempt=attempts + 1,
                            max_retries=max_retries,
                            retrying=False,
                        )
                    raise

                attempts += 1
                retrying = attempts <= max_retries
                if self.call_logger is not None:
                    self.call_logger.record_error(
                        agent_name=self.name,
                        model=self.model,
                        error=e,
                        stage="transport_error",
                        attempt=attempts,
                        max_retries=max_retries,
                        retrying=retrying,
                    )
                if attempts > max_retries:
                    raise

                delay = min(30, 5 * attempts)
                print(
                    f"[{self.name}] Network/transport error {e.__class__.__name__}: {e}. "
                    f"Retrying {attempts}/{max_retries} in {delay}s..."
                )
                time.sleep(delay)
                continue
        
        return response

    def eval_dump(self, parts, tools, response):
        # Lightweight hook for debugging/inspection.
        return None


@dataclass
class MASRunConfig:
    max_turns: int = GLOBAL_MAX_TURNS
    max_retries: int = MAX_RETRIES
    coder_fix_attempts: int = DEFAULT_TURN_CODER_FIX_ATTEMPTS
    final_render_fix_attempts: int = DEFAULT_FINAL_RENDER_FIX_ATTEMPTS
    coder_regenerate_tries: int = DEFAULT_CODER_REGENERATE_TRIES
    max_code_token_length: int = 10000
    worker_model: str = "gemini-3-flash-preview"
    orchestrator_model: str = "gemini-3-flash-preview"
    section_parallel_workers: int = 4
    clear_logs: bool = False
    case_index: Optional[int] = None
    enable_storyboard_asset_enhancement: bool = True
    storyboard_asset_enhancement_turn: int = 2
    iconfinder_api_key: str = ""


def generate_outline_with_code2video_stage1(
    knowledge_point: str,
    duration_minutes: int = OUTLINE_DURATION_MINUTES,
    max_regenerate_tries: int = OUTLINE_MAX_REGENERATE_TRIES,
    model: str = OUTLINE_MODEL,
    request_client: Optional[Client] = None,
    token_tracker: Optional[MASTokenTracker] = None,
    usage_source: str = "stage1:outline",
) -> TeachingOutline:
    """
    Generate an outline using the original Code2Video Stage-1 prompt style.
    """
    if not knowledge_point or not knowledge_point.strip():
        raise ValueError("knowledge_point must be a non-empty string.")

    prompt = get_prompt1_outline(knowledge_point=knowledge_point.strip(), duration=duration_minutes)
    req_client = request_client or client

    last_error: Optional[Exception] = None
    for attempt in range(1, max_regenerate_tries + 1):
        try:
            response = req_client.models.generate_content(
                model=model,
                contents=[prompt],
            )
            if token_tracker is not None:
                token_tracker.record(
                    source=usage_source,
                    response=response,
                    model=model,
                )
            content = _extract_text_from_gemini_response(response)
            if not content:
                content = str(response)

            outline_data = json.loads(extract_json_from_markdown(content))
            return TeachingOutline(
                topic=outline_data["topic"],
                target_audience=outline_data["target_audience"],
                sections=outline_data["sections"],
            )
        except Exception as e:
            last_error = e
            if attempt < max_regenerate_tries:
                continue

    raise ValueError(
        f"Failed to generate outline for topic '{knowledge_point}' "
        f"after {max_regenerate_tries} attempts. Last error: {last_error}"
    )


def build_video_state_from_outline(
    outline: TeachingOutline,
    storyboard_sections: Optional[List[Section]] = None,
    code_by_section_id: Optional[Dict[str, str]] = None,
) -> VideoMASState:
    """
    Build a centralized VideoMASState directly from Code2Video stage outputs.
    """
    storyboard_by_id = {section.id: deepcopy(section) for section in (storyboard_sections or [])}
    storyboard: List[Section] = []
    code: List[str] = []
    issues: List[Issue] = []
    coder_assignments: Dict[str, str] = {}
    issue_id = 1

    for idx, outline_section in enumerate(outline.sections, start=1):
        section_id = outline_section["id"]
        storyboard_section = storyboard_by_id.get(
            section_id,
            Section(
                id=section_id,
                title=outline_section["title"],
                lecture_lines=[],
                animations=[],
            ),
        )
        storyboard.append(storyboard_section)

        section_code = (code_by_section_id or {}).get(section_id, "")
        code.append(section_code)

        coder_name = f"Coder{idx}"
        coder_assignments[section_id] = coder_name

        if not storyboard_section.lecture_lines:
            issues.append(
                Issue(
                    id=issue_id,
                    fromAgent=ORCHESTRATOR,
                    toAgent=SCRIPT_WRITER,
                    description=(
                        f"[{section_id}] Lecture lines are missing. Generate concise lecture lines aligned "
                        "with this section's high-level content."
                    ),
                    isActive=True,
                    section_id=section_id,
                )
            )
            issue_id += 1

        if not storyboard_section.animations:
            issues.append(
                Issue(
                    id=issue_id,
                    fromAgent=ORCHESTRATOR,
                    toAgent=ANIMATION_PLANNER,
                    description=(
                        f"[{section_id}] Animation steps are missing. Generate codable animation steps aligned "
                        "one-to-one with lecture lines."
                    ),
                    isActive=True,
                    section_id=section_id,
                )
            )
            issue_id += 1

        issues.append(
            Issue(
                id=issue_id,
                fromAgent=ORCHESTRATOR,
                toAgent=coder_name,
                description=f"[{section_id}] Code is missing or outdated. Generate/repair runnable Manim code for this section.",
                isActive=bool(section_code.strip()),
                section_id=section_id,
            )
        )
        issue_id += 1

    return VideoMASState(
        topic=outline.topic,
        target_audience=outline.target_audience,
        code=code,
        outline=deepcopy(outline),
        storyboard=storyboard,
        issues=issues,
        coder_assignments=coder_assignments,
    )


class VideoTeamBaseAgent(MASAgent):
    def __init__(
        self,
        name: str,
        role: str,
        guidelines: str,
        video_state: VideoMASState,
        model: str,
        client: Client,
        managed_sections: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        token_tracker: Optional[MASTokenTracker] = None,
    ):
        super().__init__(
            name=name,
            role=role,
            guidelines=guidelines,
            model=model,
            client=client,
            tools=tools,
            token_tracker=token_tracker,
        )
        self.video_state = video_state
        self.managed_sections = managed_sections or self.video_state.section_ids()

    def _validate_section_access(self, section_id: str) -> int:
        if section_id not in self.video_state.section_id_to_index:
            raise ValueError(f"Unknown section_id '{section_id}'. Valid ids: {self.video_state.section_ids()}")
        if section_id not in self.managed_sections:
            raise ValueError(
                f"Agent '{self.name}' is not assigned to section '{section_id}'. "
                f"Managed sections: {self.managed_sections}"
            )
        return self.video_state.section_index(section_id)

    def _next_issue_id(self) -> int:
        return (max((x.id for x in self.video_state.issues), default=0) + 1)

    def _find_issue(self, issue_id: int) -> Issue:
        for issue in self.video_state.issues:
            if issue.id == issue_id:
                return issue
        raise ValueError(f"Issue with id={issue_id} does not exist.")

    def _issues_for_me(self, active_only: bool = True) -> List[Issue]:
        issues = [x for x in self.video_state.issues if x.toAgent == self.name and not x.resolved]
        if active_only:
            issues = [x for x in issues if x.isActive]
        return issues

    def add_issue(self, section_id: str, toAgent: str, description: str) -> int:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Issue description must be a non-empty string.")

        self._validate_section_access(section_id)

        valid_agent_names = {agent.name for agent in self.team_agents}
        if toAgent not in valid_agent_names:
            raise ValueError(f"Agent '{toAgent}' does not exist. Valid agents: {sorted(valid_agent_names)}")

        issue = Issue(
            id=self._next_issue_id(),
            fromAgent=self.name,
            toAgent=toAgent,
            description=description.strip(),
            isActive=True,
            section_id=section_id,
        )
        self.video_state.issues.append(issue)
        return issue.id

    def update_issue(
        self,
        issue_id: int,
        under_review: Optional[bool] = None,
        resolution_note: Optional[str] = None,
    ) -> None:
        if under_review is None and resolution_note is None:
            raise ValueError("At least one field must be updated: under_review or resolution_note.")
        if under_review is not None and not isinstance(under_review, bool):
            raise ValueError("under_review must be a boolean when provided.")
        if resolution_note is not None and (not isinstance(resolution_note, str) or not resolution_note.strip()):
            raise ValueError("resolution_note must be a non-empty string when provided.")

        target_issue = self._find_issue(issue_id)
        if target_issue.section_id:
            self._validate_section_access(target_issue.section_id)

        if under_review is not None:
            target_issue.under_review = under_review
        if resolution_note is not None:
            target_issue.resolution_note = resolution_note.strip()

    def replace_lecture_lines(self, section_id: str, lecture_lines: List[str]) -> Dict[str, object]:
        section_idx = self._validate_section_access(section_id)
        if not isinstance(lecture_lines, list):
            raise ValueError("lecture_lines must be a list of strings.")

        cleaned_lines: List[str] = []
        for line in lecture_lines:
            if not isinstance(line, str):
                raise ValueError("Every lecture line must be a string.")
            normalized = line.strip()
            if not normalized:
                raise ValueError("Lecture lines cannot contain empty strings.")
            cleaned_lines.append(normalized)

        self.video_state.storyboard[section_idx].lecture_lines = cleaned_lines
        return {
            "status": "ok",
            "section_id": section_id,
            "lecture_lines": cleaned_lines,
            "count": len(cleaned_lines),
        }

    def replace_animations(self, section_id: str, animations: List[str]) -> Dict[str, object]:
        section_idx = self._validate_section_access(section_id)
        if not isinstance(animations, list):
            raise ValueError("animations must be a list of strings.")

        cleaned_animations: List[str] = []
        for animation in animations:
            if not isinstance(animation, str):
                raise ValueError("Every animation must be a string.")
            normalized = animation.strip()
            if not normalized:
                raise ValueError("Animations cannot contain empty strings.")
            cleaned_animations.append(normalized)

        self.video_state.storyboard[section_idx].animations = cleaned_animations
        return {
            "status": "ok",
            "section_id": section_id,
            "animations": cleaned_animations,
            "count": len(cleaned_animations),
        }

    def replace_code_for_section(self, section_id: str, code: str) -> Dict[str, object]:
        section_idx = self._validate_section_access(section_id)
        if not isinstance(code, str):
            raise ValueError("code must be a string.")
        normalized = code.strip()
        if not normalized:
            raise ValueError("code cannot be empty.")

        section = self.video_state.storyboard[section_idx]
        normalized = normalize_code_to_code2video(
            code=normalized,
            section_id=section_id,
            section_title=section.title,
            lecture_lines=section.lecture_lines,
        )
        self.video_state.code[section_idx] = normalized
        return {
            "status": "ok",
            "section_id": section_id,
            "char_count": len(normalized),
            "line_count": len(normalized.splitlines()),
        }

    def replace_code(self, code: str) -> Dict[str, object]:
        if len(self.managed_sections) != 1:
            raise ValueError(
                "replace_code(code) is only available for agents assigned to exactly one section. "
                "Use replace_code_for_section(section_id, code) otherwise."
            )
        return self.replace_code_for_section(self.managed_sections[0], code)

    def _managed_section_payload(self) -> Dict[str, Dict[str, object]]:
        payload: Dict[str, Dict[str, object]] = {}
        for section_id in self.managed_sections:
            section_idx = self.video_state.section_index(section_id)
            payload[section_id] = {
                "highLevel": self.video_state.section_outline(section_id),
                "section": asdict(self.video_state.storyboard[section_idx]),
                "code": self.video_state.code[section_idx],
                "render_status": self.video_state.render_status[section_idx],
                "render_error": self.video_state.render_error[section_idx],
                "rendered_video_path": self.video_state.rendered_video_path[section_idx],
            }
        return payload

    def _build_coder_generation_prompt(
        self,
        *,
        agent_lines: str,
        my_active_issues: List[Issue],
        codegen_attempt: int,
        max_codegen_attempts: int,
        failure_context: str,
    ) -> str:
        section_id = self.managed_sections[0]
        section_idx = self.video_state.section_index(section_id)
        section = self.video_state.storyboard[section_idx]
        section_snapshot = self._managed_section_payload()[section_id]
        current_code = self.video_state.code[section_idx] or ""
        regenerate_note = ""
        if codegen_attempt > 1:
            regenerate_note = get_regenerate_note(
                codegen_attempt,
                MAX_REGENERATE_TRIES=max_codegen_attempts,
            )

        stage3_prompt = get_prompt3_code(
            regenerate_note=regenerate_note,
            section=section,
            base_class=TEACHING_SCENE_BASE_CLASS,
        )
        failure_block = failure_context.strip() or "None"
        current_code_block = current_code if current_code.strip() else "# No existing code yet."

        return f"""You are {self.name} in a single shared MAS team building one multi-section video.

Topic: {self.video_state.topic}
Target audience: {self.video_state.target_audience}

Team members:
{agent_lines}

Coder assignments by section:
{self.video_state.coder_assignments}

You are assigned to exactly one section:
{section_id}

Current section snapshot:
{json.dumps(section_snapshot, ensure_ascii=False, indent=2)}

Current code for this section:
```python
{current_code_block}
```

Active issues assigned to you:
{my_active_issues}

Latest render/debug failure context:
{failure_block}

MAS-specific instructions:
1. Resolve your assigned active issues first.
2. Use replace_code(code) to update your assigned section.
3. Mark each attempted issue with update_issue(issue_id, under_review=True, resolution_note=...).
4. Keep changes section-specific and avoid editing sections outside your assignment.
5. Preserve any existing [Asset: ...] references in animations and use them in code.
6. If blocked, add targeted cross-agent issues via add_issue(section_id, toAgent, description).
7. Any issue you create for another agent must be self-contained: include the concrete problem, relevant local context, and the action you want that agent to take.
8. If you answer in plain text instead of a tool call, return only the complete Python code for this section.

Original Code2Video Stage-3 prompt (follow this fully):
{stage3_prompt}
"""

    def _apply_text_response_code_fallback(self, response: object, previous_code: str) -> bool:
        if self.name in (SCRIPT_WRITER, ANIMATION_PLANNER) or len(self.managed_sections) != 1:
            return False

        section_id = self.managed_sections[0]
        section_idx = self.video_state.section_index(section_id)
        current_code = self.video_state.code[section_idx] or ""
        if current_code.strip() and current_code != (previous_code or ""):
            return False

        fallback_code = _extract_python_code_from_response(response)
        if not fallback_code:
            return False

        self.replace_code(fallback_code)
        updated_code = self.video_state.code[section_idx] or ""
        if not updated_code.strip():
            return False

        print(f"[{self.name}] Applied text-response code fallback for {section_id}.")
        return True

    def run(
        self,
        max_retries: int = 3,
        *,
        codegen_attempt: int = 1,
        max_codegen_attempts: int = 1,
        failure_context: str = "",
    ) -> str:
        tools = [self.add_issue, self.update_issue]
        tools.extend(self.tools)
        previous_code = ""
        if self.name not in (SCRIPT_WRITER, ANIMATION_PLANNER) and len(self.managed_sections) == 1:
            section_idx = self.video_state.section_index(self.managed_sections[0])
            previous_code = self.video_state.code[section_idx] or ""

        agent_lines = "\n".join(f"- {agent.name}: {agent.role}" for agent in self.team_agents)
        my_active_issues = self._issues_for_me(active_only=True)

        if self.name == SCRIPT_WRITER:
            edit_instruction = (
                "Use replace_lecture_lines(section_id, lecture_lines) to update narration for any section. "
                "Do not edit or remove existing [Asset: ...] references in animations."
            )
            base_class_context = ""
        elif self.name == ANIMATION_PLANNER:
            edit_instruction = (
                "Use replace_animations(section_id, animations) to update animation steps for any section. "
                "Preserve any existing [Asset: ...] references unless an issue explicitly tells you to replace them."
            )
            base_class_context = ""
        else:
            prompt = self._build_coder_generation_prompt(
                agent_lines=agent_lines,
                my_active_issues=my_active_issues,
                codegen_attempt=codegen_attempt,
                max_codegen_attempts=max_codegen_attempts,
                failure_context=failure_context,
            )
            response = self.run_with_retry([prompt], tools, max_retries)
            if response is not None:
                self._apply_text_response_code_fallback(response, previous_code)
                self.eval_dump([prompt], tools, response)
                return [response.usage_metadata, self.model, self.name]
            return None

        prompt = f"""You are {self.name} in a single shared MAS team building one multi-section video.

Topic: {self.video_state.topic}
Target audience: {self.video_state.target_audience}

Team members:
{agent_lines}

Coder assignments by section:
{self.video_state.coder_assignments}

Sections you are allowed to edit:
{self.managed_sections}

Section snapshots for your scope:
{self._managed_section_payload()}

{base_class_context}

Active issues assigned to you:
{my_active_issues}

Instructions:
1. Resolve your assigned active issues first.
2. Mark each attempted issue with update_issue(issue_id, under_review=True, resolution_note=...).
3. Keep changes section-specific and avoid editing sections outside your assignment.
4. If your section snapshot or issues contain existing [Asset: ...] references, preserve them. Coders must use them in code, and non-coder agents must not remove them unless explicitly instructed.
5. {edit_instruction}
6. You do NOT see the global issue list; only your assigned issues are visible.
7. If blocked, add targeted cross-agent issues via add_issue(section_id, toAgent, description).
8. Any issue you create for another agent must be self-contained: include the concrete problem, relevant local context, and the action you want that agent to take.

Guidelines:
{self.guidelines}
"""

        response = self.run_with_retry([prompt], tools, max_retries)
        if response is not None:
            self._apply_text_response_code_fallback(response, previous_code)
            self.eval_dump([prompt], tools, response)
            return [response.usage_metadata, self.model, self.name]
        return None


class VideoOrchestratorTeamAgent(VideoTeamBaseAgent):
    def __init__(
        self,
        name: str,
        role: str,
        guidelines: str,
        video_state: VideoMASState,
        model: str,
        client: Client,
        managed_sections: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        token_tracker: Optional[MASTokenTracker] = None,
    ):
        super().__init__(
            name=name,
            role=role,
            guidelines=guidelines,
            video_state=video_state,
            model=model,
            client=client,
            managed_sections=managed_sections,
            tools=tools,
            token_tracker=token_tracker,
        )
        self.extractor = GridPositionExtractor()
        self.grid_img_path = Path(__file__).resolve().parent.parent / "assets" / "reference" / "GRID.png"

    @staticmethod
    def _coerce_optional_str(value: object) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    def _build_video_feedback_issue_description(self, section_id: str, improvement: Dict[str, object]) -> str:
        problem = self._coerce_optional_str(improvement.get("problem"))
        solution = self._coerce_optional_str(improvement.get("solution"))
        object_affected = self._coerce_optional_str(improvement.get("object_affected"))
        line_number_raw = improvement.get("line_number")

        if not problem and not solution:
            return ""

        metadata_parts: List[str] = []
        if line_number_raw is not None and str(line_number_raw).strip():
            metadata_parts.append(f"line={line_number_raw}")
        if object_affected:
            metadata_parts.append(f"object={object_affected}")

        description = (
            f"[VideoCritic][{section_id}] "
            f"Problem: {problem or 'Layout issue detected.'} "
            f"Fix: {solution or 'Adjust layout, spacing, and grid placement to remove conflict.'}"
        )
        if metadata_parts:
            description += f" ({', '.join(metadata_parts)})"
        return " ".join(description.split())

    def _sync_video_feedback_issues(self, section_id: str, analysis: Dict[str, object]) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {
            "created_issue_ids": [],
            "reopened_issue_ids": [],
            "existing_issue_ids": [],
        }
        layout = analysis.get("layout")
        if not isinstance(layout, dict):
            return result
        if layout.get("has_issues") is False:
            return result

        improvements = layout.get("improvements")

        coder_name = self.video_state.coder_assignments.get(section_id)
        if not coder_name:
            return result

        def _upsert_issue(description: str) -> None:
            matched_issue: Optional[Issue] = None
            for issue in self.video_state.issues:
                if (
                    issue.section_id == section_id
                    and issue.fromAgent == self.name
                    and issue.toAgent == coder_name
                    and issue.description == description
                ):
                    matched_issue = issue
                    break

            if matched_issue is None:
                issue_id = self.add_issue(section_id=section_id, toAgent=coder_name, description=description)
                result["created_issue_ids"].append(issue_id)
                return

            if matched_issue.resolved:
                matched_issue.resolved = False
                result["reopened_issue_ids"].append(matched_issue.id)
            else:
                result["existing_issue_ids"].append(matched_issue.id)
            matched_issue.isActive = True
            matched_issue.under_review = False

        if not isinstance(improvements, list) or not improvements:
            _upsert_issue(
                f"[VideoCritic][{section_id}] Critic flagged layout issues. "
                "Re-check layout for overlap, obstruction, off-screen placement, and grid misuse."
            )
            return result

        for improvement in improvements:
            if not isinstance(improvement, dict):
                continue
            description = self._build_video_feedback_issue_description(section_id, improvement)
            if not description:
                continue
            _upsert_issue(description)

        return result

    def _auto_review_rendered_sections(self) -> List[Dict[str, object]]:
        review_results: List[Dict[str, object]] = []
        for section_id in self.video_state.section_ids():
            section_idx = self.video_state.section_index(section_id)
            if not self.video_state.rendered_video_path[section_idx]:
                continue
            try:
                review_results.append(self.review_rendered_video(section_id))
            except Exception as e:
                review_results.append(
                    {
                        "status": "failed",
                        "section_id": section_id,
                        "reason": f"review_rendered_video failed: {e}",
                    }
                )
        return review_results

    def mark_task_complete(self, issue_id: int) -> Dict[str, object]:
        if not isinstance(issue_id, int):
            raise ValueError("issue_id must be an integer.")
        target_issue = self._find_issue(issue_id)
        target_issue.resolved = True
        target_issue.isActive = False
        target_issue.under_review = False
        return {
            "status": "ok",
            "issue_id": target_issue.id,
            "resolved": target_issue.resolved,
            "isActive": target_issue.isActive,
        }

    def update_issue(
        self,
        issue_id: int,
        under_review: Optional[bool] = None,
        resolution_note: Optional[str] = None,
        isActive: Optional[bool] = None,
    ) -> None:
        if under_review is None and resolution_note is None and isActive is None:
            raise ValueError("At least one field must be updated: under_review, resolution_note, or isActive.")
        if under_review is not None and not isinstance(under_review, bool):
            raise ValueError("under_review must be a boolean when provided.")
        if resolution_note is not None and (not isinstance(resolution_note, str) or not resolution_note.strip()):
            raise ValueError("resolution_note must be a non-empty string when provided.")
        if isActive is not None and not isinstance(isActive, bool):
            raise ValueError("isActive must be a boolean when provided.")

        target_issue = self._find_issue(issue_id)
        if under_review is not None:
            target_issue.under_review = under_review
        if resolution_note is not None:
            target_issue.resolution_note = resolution_note.strip()
        if isActive is not None:
            target_issue.isActive = isActive

    def review_rendered_video(self, section_id: str) -> Dict[str, object]:
        section_idx = self._validate_section_access(section_id)
        video_path = self.video_state.rendered_video_path[section_idx]
        if not video_path:
            return {"status": "skipped", "section_id": section_id, "reason": "No rendered video path available."}
        if not Path(video_path).exists():
            return {"status": "failed", "section_id": section_id, "reason": f"Rendered video not found at {video_path}"}
        if not self.grid_img_path.exists():
            return {"status": "failed", "section_id": section_id, "reason": f"Grid reference image missing at {self.grid_img_path}"}

        position_table = self.extractor.generate_position_table(
            self.extractor.extract_grid_positions(self.video_state.code[section_idx] or "")
        )
        prompt = get_prompt4_layout_feedback(
            section=self.video_state.storyboard[section_idx],
            position_table=position_table,
        )

        response, usage = request_gemini_video_img_token(
            prompt=prompt,
            video_path=video_path,
            image_path=str(self.grid_img_path),
        )
        if self.token_tracker is not None:
            self.token_tracker.record(
                source=f"video_review:{section_id}",
                response=response,
                usage=usage,
            )
        raw = extract_answer_from_response(response)
        parsed: Dict[str, object]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw_response": raw}

        result = {
            "status": "ok",
            "section_id": section_id,
            "video_path": video_path,
            "analysis": parsed,
        }
        if isinstance(parsed, dict):
            result.update(self._sync_video_feedback_issues(section_id, parsed))
        self.video_state.video_review[section_idx] = result
        return result

    def run(self, max_retries: int = 3) -> str:
        tools = [self.mark_task_complete, self.add_issue, self.update_issue, self.review_rendered_video]
        tools.extend(self.tools)

        agent_lines = "\n".join(f"- {agent.name}: {agent.role}" for agent in self.team_agents)
        under_review = [x for x in self.video_state.issues if x.under_review and not x.resolved]
        auto_review_results = self._auto_review_rendered_sections()

        prompt = f"""You are {self.name}, the orchestrator for one shared MAS team producing all sections of a video.

Topic: {self.video_state.topic}
Target audience: {self.video_state.target_audience}

Team roster:
{agent_lines}

Coder assignments:
{self.video_state.coder_assignments}

Section summaries:
{self.video_state.section_summaries()}

Issues currently under review:
{under_review}

Automatic video review results this turn:
{auto_review_results}

All active issues:
{self.video_state.active_issues()}

Instructions:
1. Review issue resolutions under review and either:
   - mark_task_complete(issue_id), or
   - update_issue(issue_id, under_review=False, resolution_note=..., isActive=True/False).
2. Automatic video review has already run for sections with rendered videos, and corresponding coder issues may already be created.
3. Call review_rendered_video(section_id) only when you need an additional re-check (for example after major changes in code/render).
4. Workers do NOT see the global issue list; they only see issues assigned to them.
5. Therefore every issue you create must be self-contained. Include the concrete problem, the relevant section context, any dependency or upstream reason, and the exact action expected from that recipient. Do not assume they can infer missing context from other issues.
6. If an existing worker-facing issue is too vague to stand alone, create a new richer replacement issue and deactivate the stale or redundant one.
7. Add only high-impact new issues with add_issue(section_id, toAgent, description).
8. Route coding issues to the correct dedicated coder (Coder1/Coder2/... based on coder assignments).
9. Deactivate duplicates/out-of-scope issues with update_issue(..., isActive=False).

Guidelines:
{self.guidelines}
"""

        response = self.run_with_retry([prompt], tools, max_retries)
        if response is not None:
            self.eval_dump([prompt], tools, response)
            return [response.usage_metadata, self.model, self.name]
        return None


class MASVideoRunner:
    def __init__(
        self,
        video_state: VideoMASState,
        logs_dir: Path,
        cfg: MASRunConfig,
        client_override: Optional[Client] = None,
        token_tracker: Optional[MASTokenTracker] = None,
    ):
        self.video_state = video_state
        self.logs_dir = logs_dir
        self.cfg = cfg
        self.client = client_override or client
        self.token_tracker = token_tracker or MASTokenTracker(initial_summary=video_state.token_usage)

        self.case_dir = self.logs_dir
        self.state_logs_dir = self.case_dir / "mas_state"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.state_logs_dir.mkdir(parents=True, exist_ok=True)
        self.call_logger = MASAgentCallLogger(self.case_dir / "agent_function_calls.jsonl")
        self.assets_dir = Path(__file__).resolve().parent.parent / "assets" / "icon"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.iconfinder_api_key = self.cfg.iconfinder_api_key or _load_iconfinder_api_key()

        self._sync_token_usage()

        self._sync_case_documents()
        self._sync_section_code_files()

        self.coder_runtimes: Dict[str, CoderRuntime] = {}
        for section_id in self.video_state.section_ids():
            self.coder_runtimes[section_id] = CoderRuntime(
                runtime_dir=self.case_dir,
                request_fn=lambda prompt, max_tokens=8000, section_id=section_id: _scope_refine_request_with_client(
                    self.client,
                    prompt,
                    max_tokens=max_tokens,
                    token_tracker=self.token_tracker,
                    usage_source=f"scope_refine:{section_id}",
                ),
                max_code_token_length=self.cfg.max_code_token_length,
            )

        self.script_writer_agent = VideoTeamBaseAgent(
            name=SCRIPT_WRITER,
            role=SCRIPT_WRITER_ROLE,
            guidelines=SCRIPT_WRITER_GUIDELINES,
            video_state=self.video_state,
            model=self.cfg.worker_model,
            client=self.client,
            managed_sections=self.video_state.section_ids(),
            tools=[],
            token_tracker=self.token_tracker,
        )
        self.script_writer_agent.tools.append(self.script_writer_agent.replace_lecture_lines)

        self.animation_planner_agent = VideoTeamBaseAgent(
            name=ANIMATION_PLANNER,
            role=ANIMATION_PLANNER_ROLE,
            guidelines=ANIMATION_PLANNER_GUIDELINES,
            video_state=self.video_state,
            model=self.cfg.worker_model,
            client=self.client,
            managed_sections=self.video_state.section_ids(),
            tools=[],
            token_tracker=self.token_tracker,
        )
        self.animation_planner_agent.tools.append(self.animation_planner_agent.replace_animations)

        self.coder_agents: List[VideoTeamBaseAgent] = []
        self.coder_agents_by_section: Dict[str, VideoTeamBaseAgent] = {}
        for section_id in self.video_state.section_ids():
            coder_name = self.video_state.coder_assignments[section_id]
            coder_agent = VideoTeamBaseAgent(
                name=coder_name,
                role=f"{CODER_ROLE} Assigned section: {section_id}",
                guidelines=CODER_GUIDELINES,
                video_state=self.video_state,
                model=self.cfg.worker_model,
                client=self.client,
                managed_sections=[section_id],
                tools=[],
                token_tracker=self.token_tracker,
            )
            coder_agent.tools.append(coder_agent.replace_code)
            self.coder_agents.append(coder_agent)
            self.coder_agents_by_section[section_id] = coder_agent

        self.team_agents: List[VideoTeamBaseAgent] = [
            self.script_writer_agent,
            self.animation_planner_agent,
            *self.coder_agents,
        ]

        self.orchestrator_agent = VideoOrchestratorTeamAgent(
            name="OrchestratorAgent",
            role=ORCHESTRATOR,
            guidelines="",
            video_state=self.video_state,
            model=self.cfg.orchestrator_model,
            client=self.client,
            managed_sections=self.video_state.section_ids(),
            tools=[],
            token_tracker=self.token_tracker,
        )

        for agent in self.team_agents + [self.orchestrator_agent]:
            agent.set_team_agents(self.team_agents)
            agent.call_logger = self.call_logger

    def _sync_token_usage(self) -> None:
        self.video_state.token_usage = self.token_tracker.snapshot()
        token_usage_path = self.case_dir / "token_usage_summary.json"
        with token_usage_path.open("w", encoding="utf-8") as f:
            json.dump(self.video_state.token_usage, f, ensure_ascii=False, indent=2)

    def _storyboard_payload(self) -> Dict[str, object]:
        return {
            "topic": self.video_state.topic,
            "target_audience": self.video_state.target_audience,
            "sections": [asdict(section) for section in self.video_state.storyboard],
        }

    def _storyboard_contains_assets(self) -> bool:
        return any(
            isinstance(animation, str) and "[Asset:" in animation
            for section in self.video_state.storyboard
            for animation in section.animations
        )

    def _request_asset_enhancement(self, prompt: str, max_tokens: int = 10000):
        return _scope_refine_request_with_client(
            self.client,
            prompt,
            max_tokens=max_tokens,
            token_tracker=self.token_tracker,
            usage_source="assets",
        )

    def _ensure_asset_integration_issue(
        self,
        section_id: str,
        asset_animations: List[str],
        turn_idx: int,
    ) -> None:
        coder_name = self.video_state.coder_assignments.get(section_id)
        if not coder_name:
            return

        issue_marker = "[AUTO-ASSET-INTEGRATION]"
        for issue in self.video_state.issues:
            if (
                issue.section_id == section_id
                and issue.toAgent == coder_name
                and not issue.resolved
                and issue_marker in issue.description
            ):
                issue.isActive = True
                issue.under_review = False
                return

        issue_description = (
            f"{issue_marker} [{section_id}] The storyboard was enhanced with asset references at the end of turn "
            f"{turn_idx}. Update this section's Manim code so it uses every [Asset: ...] reference now present in "
            "the animation descriptions. Load and place the referenced files in the scene where the storyboard "
            "calls for them, while preserving the existing lecture-line alignment and layout constraints. "
            f"Asset-tagged animations: {asset_animations}"
        )
        self.video_state.issues.append(
            Issue(
                id=max((x.id for x in self.video_state.issues), default=0) + 1,
                fromAgent=ORCHESTRATOR,
                toAgent=coder_name,
                description=issue_description,
                isActive=True,
                section_id=section_id,
            )
        )

    def _enhance_storyboard_with_assets(self, turn_idx: int) -> None:
        if self._storyboard_contains_assets():
            print("[VideoMAS] Storyboard already contains asset tags; skipping asset enhancement.")
            return

        storyboard_payload = self._storyboard_payload()
        original_animations_by_section = {
            section.id: list(section.animations)
            for section in self.video_state.storyboard
        }

        try:
            enhanced_storyboard = process_storyboard_with_assets(
                storyboard=storyboard_payload,
                api_function=self._request_asset_enhancement,
                assets_dir=str(self.assets_dir),
                iconfinder_api_key=self.iconfinder_api_key,
            )
        except Exception as e:
            print(f"[VideoMAS] Storyboard asset enhancement failed: {e}")
            return

        enhanced_sections = enhanced_storyboard.get("sections", []) if isinstance(enhanced_storyboard, dict) else []
        if not isinstance(enhanced_sections, list):
            print("[VideoMAS] Storyboard asset enhancement returned an invalid payload; skipping.")
            return

        sections_by_id = {
            str(section_data.get("id", "")): section_data
            for section_data in enhanced_sections
            if isinstance(section_data, dict)
        }
        sections_with_new_assets: List[Tuple[str, List[str]]] = []

        for section in self.video_state.storyboard:
            section_data = sections_by_id.get(section.id)
            if section_data is None:
                continue

            enhanced_animations = section_data.get("animations", [])
            if not isinstance(enhanced_animations, list):
                continue

            cleaned_animations: List[str] = []
            for animation in enhanced_animations:
                if not isinstance(animation, str):
                    continue
                normalized = animation.strip()
                if normalized:
                    cleaned_animations.append(normalized)

            if not cleaned_animations:
                continue

            previous_animations = original_animations_by_section.get(section.id, [])
            section.animations = cleaned_animations
            asset_animations = [animation for animation in cleaned_animations if "[Asset:" in animation]
            if asset_animations and cleaned_animations != previous_animations:
                sections_with_new_assets.append((section.id, asset_animations))

        if not sections_with_new_assets:
            print("[VideoMAS] Storyboard asset enhancement completed, but no new asset-tagged animations were added.")
            return

        for section_id, asset_animations in sections_with_new_assets:
            self._ensure_asset_integration_issue(section_id, asset_animations, turn_idx)

        print(
            "[VideoMAS] Storyboard enhanced with assets for sections: "
            f"{[section_id for section_id, _ in sections_with_new_assets]}"
        )

    def _sync_case_documents(self) -> None:
        outline_path = self.case_dir / "outline.json"
        with outline_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.video_state.outline), f, ensure_ascii=False, indent=2)

        storyboard_payload = self._storyboard_payload()
        storyboard_path = self.case_dir / "storyboard.json"
        with storyboard_path.open("w", encoding="utf-8") as f:
            json.dump(storyboard_payload, f, ensure_ascii=False, indent=2)

        storyboard_assets_path = self.case_dir / "storyboard_with_assets.json"
        with storyboard_assets_path.open("w", encoding="utf-8") as f:
            json.dump(storyboard_payload, f, ensure_ascii=False, indent=2)

    def _sync_section_code_files(self) -> None:
        for section_id in self.video_state.section_ids():
            section_idx = self.video_state.section_index(section_id)
            code_text = self.video_state.code[section_idx] or ""
            code_path = self.case_dir / f"{section_id}.py"
            code_path.write_text(code_text, encoding="utf-8")

    def _merge_optimized_videos(self) -> Optional[str]:
        video_list_file = self.case_dir / "video_list.txt"
        if not video_list_file.exists() or not video_list_file.read_text(encoding="utf-8").strip():
            return None

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            try:
                import imageio_ffmpeg

                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                return None

        safe_name = topic_to_safe_name(self.video_state.topic) or "final_video"
        output_path = self.case_dir / f"{safe_name}.mp4"
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(video_list_file), "-c", "copy", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return str(output_path)

    def _finalize_case_outputs(self) -> None:
        self._sync_case_documents()
        self._sync_section_code_files()

        optimized_dir = self.case_dir / "optimized_videos"
        optimized_dir.mkdir(parents=True, exist_ok=True)

        concat_entries: List[str] = []
        for section_id in self.video_state.section_ids():
            section_idx = self.video_state.section_index(section_id)
            rendered_path = self.video_state.rendered_video_path[section_idx]
            if not rendered_path:
                continue

            source_path = Path(rendered_path)
            if not source_path.exists():
                continue

            optimized_path = optimized_dir / f"{section_id}_optimized.mp4"
            shutil.copy2(source_path, optimized_path)
            rel_path = optimized_path.relative_to(self.case_dir)
            concat_entries.append(f"file '{rel_path.as_posix()}'")

        video_list_path = self.case_dir / "video_list.txt"
        if concat_entries:
            video_list_path.write_text("\n".join(concat_entries) + "\n", encoding="utf-8")
        else:
            video_list_path.write_text("", encoding="utf-8")

        self._merge_optimized_videos()

    def _activate_ready_coder_issues(self) -> None:
        for section_id, coder_name in self.video_state.coder_assignments.items():
            section_idx = self.video_state.section_index(section_id)
            section = self.video_state.storyboard[section_idx]
            has_content = bool(section.lecture_lines) and bool(section.animations)
            for issue in self.video_state.issues:
                if issue.section_id != section_id or issue.toAgent != coder_name or issue.resolved:
                    continue
                if has_content:
                    issue.isActive = True
                elif issue.fromAgent == ORCHESTRATOR:
                    issue.isActive = False

    def _select_next_agents(self) -> List[VideoTeamBaseAgent]:
        by_name = {agent.name: agent for agent in self.team_agents}
        selected: Dict[str, VideoTeamBaseAgent] = {}
        for issue in self.video_state.active_issues():
            agent = by_name.get(issue.toAgent)
            if agent is not None:
                selected[agent.name] = agent
        return list(selected.values())

    def _run_workers_parallel(self) -> List[str]:
        next_agents = self._select_next_agents()
        if not next_agents:
            print("[VideoMAS] No active issues mapped to workers; skipping worker execution for this turn.")
            return []

        completed_agent_names: List[str] = []
        print(f"[VideoMAS] Running workers: {[agent.name for agent in next_agents]}")
        with ThreadPoolExecutor(max_workers=len(next_agents)) as executor:
            futures = {
                executor.submit(
                    agent.run,
                    max_retries=self.cfg.max_retries,
                    max_codegen_attempts=self.cfg.coder_regenerate_tries,
                ): agent
                for agent in next_agents
            }
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    future.result()
                    completed_agent_names.append(agent.name)
                    print(f"[VideoMAS][{agent.name}] completed")
                except Exception as e:
                    print(f"[VideoMAS][{agent.name}] failed: {e}")
        return completed_agent_names

    def _apply_coder_runtime_result(self, section_id: str, result: Dict[str, object]) -> None:
        section_idx = self.video_state.section_index(section_id)
        section = self.video_state.storyboard[section_idx]
        self.video_state.code[section_idx] = normalize_code_to_code2video(
            code=result.get("code", self.video_state.code[section_idx]),
            section_id=section_id,
            section_title=section.title,
            lecture_lines=section.lecture_lines,
        )
        self.video_state.render_status[section_idx] = result.get("status")
        self.video_state.render_error[section_idx] = result.get("error", "")
        if result.get("video_path"):
            self.video_state.rendered_video_path[section_idx] = result["video_path"]
        else:
            self.video_state.rendered_video_path[section_idx] = None
            self.video_state.video_review[section_idx] = None

    def _run_coder_runtime_for_section(
        self,
        section_id: str,
        max_fix_attempts: int,
        phase_label: str,
    ) -> Tuple[str, Dict[str, object]]:
        runtime = self.coder_runtimes[section_id]
        section_idx = self.video_state.section_index(section_id)
        section = self.video_state.storyboard[section_idx]
        current_code = self.video_state.code[section_idx] or ""
        if not current_code.strip():
            return section_id, {
                "success": False,
                "status": "skipped",
                "error": "No code available for section",
            }

        print(f"[{section_id}][CoderRuntime][{phase_label}] Debugging updated code.")
        self.video_state.code[section_idx] = normalize_code_to_code2video(
            code=current_code,
            section_id=section_id,
            section_title=section.title,
            lecture_lines=section.lecture_lines,
        )
        result = runtime.debug_and_fix(
            section_id=section_id,
            code=self.video_state.code[section_idx],
            max_fix_attempts=max_fix_attempts,
            section_title=section.title,
            lecture_lines=section.lecture_lines,
        )
        self._apply_coder_runtime_result(section_id, result)
        return section_id, result

    def _sections_with_updated_code(
        self,
        prior_code_by_section: Dict[str, str],
        completed_agent_names: List[str],
    ) -> List[str]:
        completed_names = set(completed_agent_names)
        updated_sections: List[str] = []

        for section_id, coder_name in self.video_state.coder_assignments.items():
            if coder_name not in completed_names:
                continue

            section_idx = self.video_state.section_index(section_id)
            previous_code = prior_code_by_section.get(section_id, "") or ""
            current_code = self.video_state.code[section_idx] or ""
            if current_code.strip() and current_code != previous_code:
                updated_sections.append(section_id)

        return updated_sections

    def _run_coder_runtimes_parallel(
        self,
        max_fix_attempts: int,
        phase_label: str,
        sections_to_run: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        if sections_to_run is None:
            runnable_sections = [
                section_id
                for section_id in self.video_state.section_ids()
                if bool((self.video_state.code[self.video_state.section_index(section_id)] or "").strip())
            ]
        else:
            runnable_sections = list(sections_to_run)

        if not runnable_sections:
            print(f"[VideoMAS] No sections require coder runtime execution for {phase_label}.")
            return results

        max_workers = min(max(1, len(runnable_sections)), max(1, self.cfg.section_parallel_workers))
        tasks = []
        for section_id in runnable_sections:
            section_idx = self.video_state.section_index(section_id)
            section = self.video_state.storyboard[section_idx]
            current_code = self.video_state.code[section_idx] or ""
            if not current_code.strip():
                results[section_id] = {
                    "success": False,
                    "status": "skipped",
                    "error": "No code available for section",
                }
                continue

            print(f"[{section_id}][CoderRuntime][{phase_label}] Debugging updated code.")
            tasks.append(
                {
                    "runtime_dir": str(self.case_dir),
                    "topic": self.video_state.topic,
                    "section_id": section_id,
                    "code": current_code,
                    "max_fix_attempts": max_fix_attempts,
                    "max_code_token_length": self.cfg.max_code_token_length,
                    "section_title": section.title,
                    "lecture_lines": list(section.lecture_lines),
                }
            )

        if not tasks:
            print(f"[VideoMAS] No sections require coder runtime execution for {phase_label}.")
            return results

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_coder_runtime_worker, **task): task["section_id"]
                for task in tasks
            }
            for future in as_completed(futures):
                section_id = futures[future]
                try:
                    _, result, token_usage_summary = future.result()
                    self.token_tracker.merge_summary(token_usage_summary)
                    self._apply_coder_runtime_result(section_id, result)
                    results[section_id] = result
                except Exception as e:
                    print(f"[{section_id}][CoderRuntime][{phase_label}] failed: {e}")
                    failure_result = {
                        "success": False,
                        "status": "failed",
                        "error": str(e),
                        "code": self.video_state.code[self.video_state.section_index(section_id)] or "",
                    }
                    self._apply_coder_runtime_result(section_id, failure_result)
                    results[section_id] = failure_result

        return results

    def _collect_failed_coder_runtime_sections(
        self,
        runtime_results: Dict[str, Dict[str, object]],
        attempted_sections: List[str],
    ) -> Dict[str, str]:
        failed_sections: Dict[str, str] = {}
        for section_id in attempted_sections:
            result = runtime_results.get(section_id)
            if not isinstance(result, dict):
                failed_sections[section_id] = "No coder runtime result recorded."
                continue
            if result.get("success"):
                continue
            failed_sections[section_id] = str(
                result.get("error")
                or result.get("reason")
                or result.get("status")
                or "Coder runtime failed."
            )
        return failed_sections

    def _regenerate_failed_sections(
        self,
        *,
        failed_sections: Dict[str, str],
        phase_label: str,
        max_fix_attempts: int,
    ) -> None:
        if not failed_sections:
            return

        for codegen_attempt in range(2, self.cfg.coder_regenerate_tries + 1):
            if not failed_sections:
                return

            print(
                f"[VideoMAS] Regenerating failed coder sections for {phase_label}, "
                f"attempt {codegen_attempt}/{self.cfg.coder_regenerate_tries}: {sorted(failed_sections)}"
            )
            sections_to_rerun: List[str] = []
            pending_failures: Dict[str, str] = {}

            for section_id, failure_reason in failed_sections.items():
                coder_agent = self.coder_agents_by_section.get(section_id)
                if coder_agent is None:
                    pending_failures[section_id] = failure_reason
                    continue

                section_idx = self.video_state.section_index(section_id)
                previous_code = self.video_state.code[section_idx] or ""
                try:
                    coder_agent.run(
                        max_retries=self.cfg.max_retries,
                        codegen_attempt=codegen_attempt,
                        max_codegen_attempts=self.cfg.coder_regenerate_tries,
                        failure_context=failure_reason,
                    )
                except Exception as e:
                    pending_failures[section_id] = f"Coder regeneration attempt {codegen_attempt} failed: {e}"
                    continue

                current_code = self.video_state.code[section_idx] or ""
                if current_code.strip() and current_code != previous_code:
                    sections_to_rerun.append(section_id)
                elif not current_code.strip():
                    pending_failures[section_id] = (
                        f"Coder regeneration attempt {codegen_attempt} produced no code. "
                        f"Last failure: {failure_reason}"
                    )
                else:
                    pending_failures[section_id] = (
                        f"Coder regeneration attempt {codegen_attempt} did not change the code. "
                        f"Last failure: {failure_reason}"
                    )

            if sections_to_rerun:
                rerun_results = self._run_coder_runtimes_parallel(
                    max_fix_attempts=max_fix_attempts,
                    phase_label=f"{phase_label}_regen_{codegen_attempt:02d}",
                    sections_to_run=sections_to_rerun,
                )
                pending_failures.update(
                    self._collect_failed_coder_runtime_sections(rerun_results, sections_to_rerun)
                )

            failed_sections = pending_failures

        if failed_sections:
            print(f"[VideoMAS] Sections still failing after regeneration for {phase_label}: {failed_sections}")

    def run(self) -> VideoMASState:
        while self.video_state.unresolved_issues() and self.video_state.turns_run < self.cfg.max_turns:
            turn_idx = self.video_state.turns_run + 1
            print(f"[VideoMAS] === Turn {turn_idx} ===")

            prior_code_by_section = {
                section_id: self.video_state.code[self.video_state.section_index(section_id)] or ""
                for section_id in self.video_state.section_ids()
            }
            self._activate_ready_coder_issues()
            completed_agent_names = self._run_workers_parallel()
            if (
                self.cfg.enable_storyboard_asset_enhancement
                and turn_idx == self.cfg.storyboard_asset_enhancement_turn
            ):
                self._enhance_storyboard_with_assets(turn_idx)
            self._activate_ready_coder_issues()
            sections_to_debug = self._sections_with_updated_code(prior_code_by_section, completed_agent_names)
            runtime_results = self._run_coder_runtimes_parallel(
                max_fix_attempts=self.cfg.coder_fix_attempts,
                phase_label=f"turn_{turn_idx:02d}",
                sections_to_run=sections_to_debug,
            )
            self._regenerate_failed_sections(
                failed_sections=self._collect_failed_coder_runtime_sections(runtime_results, sections_to_debug),
                phase_label=f"turn_{turn_idx:02d}",
                max_fix_attempts=self.cfg.coder_fix_attempts,
            )

            self._sync_case_documents()
            self._sync_section_code_files()
            self._sync_token_usage()
            _save_video_state_json(self.video_state, self.state_logs_dir, f"turn_{turn_idx:02d}_video_state.json")

            self.video_state.turns_run += 1
            if self.video_state.turns_run >= self.cfg.max_turns:
                break

            self.orchestrator_agent.run(max_retries=self.cfg.max_retries)
            self._sync_case_documents()
            self._sync_token_usage()
            _save_video_state_json(
                self.video_state,
                self.state_logs_dir,
                f"turn_{turn_idx:02d}_orchestrator_video_state.json",
            )

        # Final render-readiness pass: align with original Code2Video "final render" bug-fix budget.
        final_render_sections = [
            section_id
            for section_id in self.video_state.section_ids()
            if bool((self.video_state.code[self.video_state.section_index(section_id)] or "").strip())
        ]
        final_render_results = self._run_coder_runtimes_parallel(
            max_fix_attempts=self.cfg.final_render_fix_attempts,
            phase_label="final_render",
            sections_to_run=final_render_sections,
        )
        self._regenerate_failed_sections(
            failed_sections=self._collect_failed_coder_runtime_sections(
                final_render_results,
                final_render_sections,
            ),
            phase_label="final_render",
            max_fix_attempts=self.cfg.final_render_fix_attempts,
        )
        self._sync_case_documents()
        self._sync_section_code_files()
        self._sync_token_usage()
        _save_video_state_json(self.video_state, self.state_logs_dir, "final_render_pass_video_state.json")
        self._finalize_case_outputs()

        self._sync_token_usage()
        _save_video_state_json(self.video_state, self.state_logs_dir, "final_video_state.json")
        return self.video_state


def run_mas_for_video_state(
    video_state: VideoMASState,
    logs_root: Optional[Path] = None,
    cfg: Optional[MASRunConfig] = None,
    client_override: Optional[Client] = None,
    token_tracker: Optional[MASTokenTracker] = None,
) -> VideoMASState:
    """
    Primary centralized runner: all team agents read/write one VideoMASState object.
    """
    cfg = cfg or MASRunConfig()

    logs_root = logs_root or (Path(__file__).resolve().parent.parent / "mas_logs")
    if cfg.clear_logs and logs_root.exists():
        shutil.rmtree(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)

    safe_topic = topic_to_safe_name(video_state.topic) or "untitled_topic"
    if cfg.case_index is None:
        max_existing_idx = -1
        for child in logs_root.iterdir():
            if not child.is_dir():
                continue
            match = re.match(r"^(\d+)-", child.name)
            if not match:
                continue
            max_existing_idx = max(max_existing_idx, int(match.group(1)))
        resolved_case_index = max_existing_idx + 1
    else:
        resolved_case_index = cfg.case_index

    case_dir = logs_root / f"{resolved_case_index}-{safe_topic}"
    if case_dir.exists():
        # Preserve previous runs even when a fixed case_index is reused.
        suffix = time.strftime("%Y%m%d_%H%M%S")
        case_dir = logs_root / f"{resolved_case_index}-{safe_topic}_{suffix}"

    case_dir.mkdir(parents=True, exist_ok=True)
    video_state.run_output_dir = str(case_dir)
    if token_tracker is not None:
        video_state.token_usage = token_tracker.snapshot()

    runner = MASVideoRunner(
        video_state=video_state,
        logs_dir=case_dir,
        cfg=cfg,
        client_override=client_override,
        token_tracker=token_tracker,
    )
    return runner.run()


if __name__ == "__main__":
    start_time = time.time()
    token_tracker = MASTokenTracker()
    generated_outline = generate_outline_with_code2video_stage1(
        knowledge_point=TOPIC,
        duration_minutes=OUTLINE_DURATION_MINUTES,
        max_regenerate_tries=OUTLINE_MAX_REGENERATE_TRIES,
        model=OUTLINE_MODEL,
        token_tracker=token_tracker,
    )

    video_state = build_video_state_from_outline(generated_outline)
    logs_root = Path(__file__).resolve().parent.parent / "mas_logs"

    cfg = MASRunConfig(
        max_turns=GLOBAL_MAX_TURNS,
        max_retries=MAX_RETRIES,
        section_parallel_workers=len(video_state.section_ids()),
        clear_logs=False,
    )

    final_video_state = run_mas_for_video_state(
        video_state=video_state,
        logs_root=logs_root,
        cfg=cfg,
        token_tracker=token_tracker,
    )

    duration_minutes = (time.time() - start_time) / 60
    total_tokens = _total_tokens_from_summary(final_video_state.token_usage)
    final_video_path = _final_video_output_path(final_video_state)
    was_successful = bool(final_video_path and final_video_path.exists())

    print(f"Generated outline topic: {generated_outline.topic}")
    print(f"Finished MAS run for {len(final_video_state.section_ids())} sections.")
    print(f"Run outputs written to: {final_video_state.run_output_dir}")
    status_icon = "✅" if was_successful else "❌"
    print(
        f"{status_icon} Knowledge topic '{generated_outline.topic}' processed. "
        f"Cost Time: {duration_minutes:.2f} minutes, Tokens used: {total_tokens}"
    )
    if was_successful:
        print("\n" + "=" * 50)
        print("   Total knowledge points: 1")
        print("   Successfully processed: 1 (100.0%)")
        print(f"   Average duration [min]: {duration_minutes:.2f} minutes/knowledge point")
        print(f"   Average token consumption: {total_tokens:,.0f} tokens/knowledge point")
        print("=" * 50)
    else:
        print("\nAll knowledge points failed, cannot calculate average.")
    print(_format_token_usage_summary(final_video_state.token_usage))
