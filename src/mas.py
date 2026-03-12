from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable, Dict, List, Optional, Tuple

from type_utils import *
from gpt_request import request_gemini_video_img
from scope_refine import ScopeRefineFixer, GridPositionExtractor
from utils import extract_answer_from_response
from utils import extract_json_from_markdown
from utils import replace_base_class
from utils import topic_to_safe_name

try:
    from prompts import get_prompt1_outline, get_prompt4_layout_feedback
except ModuleNotFoundError:
    # Allow running `python src/mas.py` without manually setting PYTHONPATH.
    import sys

    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from prompts import get_prompt1_outline, get_prompt4_layout_feedback

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
- Do not involve any external elements (such as SVGs or other assets that require downloading or dependencies).
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


API_KEY = _load_api_key()
client = Client(api_key=API_KEY)


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


def _normalize_scene_class_name(code: str, scene_name: str) -> str:
    class_pattern = re.compile(r"^(\s*class\s+)([A-Za-z_]\w*)(\s*\(([^)]*)\)\s*:)", re.MULTILINE)

    for match in class_pattern.finditer(code):
        class_name = match.group(2)
        bases = match.group(4) or ""

        if class_name == "TeachingScene":
            continue

        if "TeachingScene" in bases or "Scene" in bases:
            if class_name == scene_name:
                return code
            return f"{code[:match.start(2)]}{scene_name}{code[match.end(2):]}"

    return code


def _ensure_scene_inherits_teaching_scene(code: str, scene_name: str) -> str:
    # class X(Base): -> class X(TeachingScene):
    class_with_bases = re.compile(
        rf"^(\s*class\s+{re.escape(scene_name)}\s*)\(([^)]*)\)(\s*:)",
        re.MULTILINE,
    )
    match = class_with_bases.search(code)
    if match:
        if "TeachingScene" in (match.group(2) or ""):
            return code
        return f"{code[:match.start(2)]}TeachingScene{code[match.end(2):]}"

    # class X: -> class X(TeachingScene):
    class_without_bases = re.compile(
        rf"^(\s*class\s+{re.escape(scene_name)}\s*)(:)",
        re.MULTILINE,
    )
    match = class_without_bases.search(code)
    if match:
        return f"{code[:match.start(2)]}(TeachingScene){code[match.start(2):]}"

    return code


def _remove_non_stage3_helper_stubs(code: str) -> str:
    stub_line = re.compile(
        r"^\s*self\.(add_line|highlight_line|place_at_grid|place_in_area)\s*=\s*lambda\b.*$",
        re.MULTILINE,
    )
    return stub_line.sub("", code)


def _ensure_setup_layout_call(code: str, scene_name: str, section_title: str, lecture_lines: List[str]) -> str:
    lines = code.splitlines()

    class_idx = None
    class_indent = 0
    class_pattern = re.compile(
        rf"^(\s*)class\s+{re.escape(scene_name)}\s*(?:\([^)]*\))?\s*:",
    )
    for idx, line in enumerate(lines):
        class_match = class_pattern.match(line)
        if class_match:
            class_idx = idx
            class_indent = len(class_match.group(1))
            break

    if class_idx is None:
        fallback = [
            "",
            f"class {scene_name}(TeachingScene):",
            "    def construct(self):",
            f"        self.setup_layout({json.dumps(section_title)}, {json.dumps(lecture_lines, ensure_ascii=False)})",
            "        self.wait(1)",
        ]
        return (code.rstrip() + "\n" + "\n".join(fallback)).strip() + "\n"

    def_idx = None
    def_indent = 0
    for idx in range(class_idx + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip(" "))
        if stripped and current_indent <= class_indent:
            break
        if re.match(r"^\s*def\s+construct\s*\(self[^\)]*\)\s*:", line):
            def_idx = idx
            def_indent = current_indent
            break

    setup_call = f"self.setup_layout({json.dumps(section_title)}, {json.dumps(lecture_lines, ensure_ascii=False)})"
    if def_idx is None:
        # Add construct() to the scene class when missing.
        insert_idx = class_idx + 1
        while insert_idx < len(lines):
            line = lines[insert_idx]
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip(" "))
            if stripped and current_indent <= class_indent:
                break
            insert_idx += 1

        method_block = [
            " " * (class_indent + 4) + "def construct(self):",
            " " * (class_indent + 8) + setup_call,
            " " * (class_indent + 8) + "self.wait(1)",
        ]
        lines[insert_idx:insert_idx] = method_block
        return "\n".join(lines).rstrip() + "\n"

    has_setup_layout = False
    for idx in range(def_idx + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip(" "))
        if stripped and current_indent <= def_indent:
            break
        if "self.setup_layout(" in line:
            has_setup_layout = True
            break

    if has_setup_layout:
        return "\n".join(lines).rstrip() + "\n"

    insert_idx = def_idx + 1
    while insert_idx < len(lines) and lines[insert_idx].strip() == "":
        insert_idx += 1

    lines.insert(insert_idx, " " * (def_indent + 4) + setup_call)
    return "\n".join(lines).rstrip() + "\n"


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

    if not re.search(r"^\s*from\s+manim\s+import\s+\*", normalized, re.MULTILINE):
        normalized = f"from manim import *\n\n{normalized}"

    scene_name = _scene_name_from_section_id(section_id)
    normalized = _remove_non_stage3_helper_stubs(normalized)
    normalized = _normalize_scene_class_name(normalized, scene_name)
    normalized = _ensure_scene_inherits_teaching_scene(normalized, scene_name)
    normalized = replace_base_class(normalized, TEACHING_SCENE_BASE_CLASS)
    normalized = _ensure_setup_layout_call(normalized, scene_name, section_title, lecture_lines)
    return normalized.strip() + "\n"


class _FakeChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeChoiceMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


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


def _scope_refine_request_with_client(
    request_client: Client,
    prompt: str,
    max_tokens: int = 8000,
):
    """
    Client-bound adapter variant for parallel section runners.
    """
    generate_config = None
    if DEBUG_FIX_THINKING_BUDGET is not None or DEBUG_FIX_INCLUDE_THOUGHTS is not None:
        thinking_kwargs: Dict[str, object] = {}
        if DEBUG_FIX_THINKING_BUDGET is not None:
            thinking_kwargs["thinking_budget"] = DEBUG_FIX_THINKING_BUDGET
        if DEBUG_FIX_INCLUDE_THOUGHTS is not None:
            thinking_kwargs["include_thoughts"] = DEBUG_FIX_INCLUDE_THOUGHTS
        generate_config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(**thinking_kwargs))

    request_kwargs = {
        "model": DEBUG_FIX_MODEL,
        "contents": [prompt],
    }
    if generate_config is not None:
        request_kwargs["config"] = generate_config

    response = request_client.models.generate_content(**request_kwargs)
    content = _extract_text_from_gemini_response(response)
    fake_completion = _FakeCompletion(content)

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_md = getattr(response, "usage_metadata", None)
    if usage_md is not None:
        usage["prompt_tokens"] = getattr(usage_md, "prompt_token_count", 0) or 0
        usage["completion_tokens"] = getattr(usage_md, "candidates_token_count", 0) or 0
        usage["total_tokens"] = getattr(usage_md, "total_token_count", 0) or 0

    return fake_completion, usage


class CoderRuntime:
    """
    Minimal coder runtime:
    1) persist current section code to disk
    2) run Manim
    3) apply ScopeRefine smart fixes on failure
    4) return fixed code + render metadata
    """

    def __init__(
        self,
        runtime_dir: Path,
        request_fn: Callable,
        max_code_token_length: int = 10000,
    ):
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.scope_refine_fixer = ScopeRefineFixer(request_fn, max_code_token_length)

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

    def _normalize_scene_class_name(self, code: str, scene_name: str) -> str:
        """
        Ensure the runnable scene class name matches the expected `{section_id}Scene`.
        This prevents ScopeRefine dry-run import failures when LLM renames the class.
        """
        return _normalize_scene_class_name(code, scene_name)

    def debug_and_fix(
        self,
        section_id: str,
        code: str,
        max_fix_attempts: int = 3,
        section_title: str = "",
        lecture_lines: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        if not code or not code.strip():
            return {"success": False, "status": "skipped", "reason": "No code available for section"}

        scene_name = self._scene_name(section_id)
        code_file = self.runtime_dir / f"{section_id}.py"
        current_code = normalize_code_to_code2video(
            code=code,
            section_id=section_id,
            section_title=section_title,
            lecture_lines=lecture_lines or [],
        )
        current_code = self._normalize_scene_class_name(current_code, scene_name)
        code_file.write_text(current_code, encoding="utf-8")

        last_error = ""
        for fix_attempt in range(1, max_fix_attempts + 1):
            print(f"[CoderRuntime] Debugging {section_id} ({fix_attempt}/{max_fix_attempts})")
            try:
                result = subprocess.run(
                    ["manim", "-ql", code_file.name, scene_name],
                    capture_output=True,
                    text=True,
                    cwd=self.runtime_dir,
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "status": "failed",
                    "attempts": fix_attempt,
                    "error": "Manim execution timed out",
                }
            except Exception as e:
                return {
                    "success": False,
                    "status": "failed",
                    "attempts": fix_attempt,
                    "error": str(e),
                }

            if result.returncode == 0:
                video_path = self._find_video_path(section_id, scene_name)
                return {
                    "success": True,
                    "status": "ok",
                    "attempts": fix_attempt,
                    "code": current_code,
                    "video_path": str(video_path) if video_path else None,
                }

            last_error = (result.stderr or result.stdout or "Unknown runtime error").strip()
            fixed_code = self.scope_refine_fixer.fix_code_smart(
                section_id=section_id,
                code=current_code,
                error_msg=last_error,
                output_dir=self.runtime_dir,
            )
            if not fixed_code:
                break

            current_code = normalize_code_to_code2video(
                code=fixed_code,
                section_id=section_id,
                section_title=section_title,
                lecture_lines=lecture_lines or [],
            )
            current_code = self._normalize_scene_class_name(current_code, scene_name)
            code_file.write_text(current_code, encoding="utf-8")

        return {
            "success": False,
            "status": "failed",
            "attempts": max_fix_attempts,
            "code": current_code,
            "error": last_error,
        }


class MASAgent:
    def __init__(
        self,
        name: str,
        role: str,
        guidelines: str,
        model: str,
        client: Client,
        tools: Optional[List[Callable]] = None,
    ):
        self.name = name
        self.role = role
        self.guidelines = guidelines
        self.model = model
        self.client = client
        self.tools = tools or []
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
                break  # success
            except (ClientError, ServerError) as e:
                msg = str(e)
                attempts += 1

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
                    raise

                attempts += 1
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
    max_code_token_length: int = 10000
    worker_model: str = "gemini-3-flash-preview"
    orchestrator_model: str = "gemini-3-flash-preview"
    section_parallel_workers: int = 4
    clear_logs: bool = False
    case_index: Optional[int] = None


def generate_outline_with_code2video_stage1(
    knowledge_point: str,
    duration_minutes: int = OUTLINE_DURATION_MINUTES,
    max_regenerate_tries: int = OUTLINE_MAX_REGENERATE_TRIES,
    model: str = OUTLINE_MODEL,
    request_client: Optional[Client] = None,
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
    ):
        super().__init__(
            name=name,
            role=role,
            guidelines=guidelines,
            model=model,
            client=client,
            tools=tools,
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

    def run(self, max_retries: int = 3) -> str:
        tools = [self.add_issue, self.update_issue]
        tools.extend(self.tools)

        agent_lines = "\n".join(f"- {agent.name}: {agent.role}" for agent in self.team_agents)
        my_active_issues = self._issues_for_me(active_only=True)
        global_active_issues = self.video_state.active_issues()

        if self.name == SCRIPT_WRITER:
            edit_instruction = (
                "Use replace_lecture_lines(section_id, lecture_lines) to update narration for any section."
            )
            base_class_context = ""
        elif self.name == ANIMATION_PLANNER:
            edit_instruction = (
                "Use replace_animations(section_id, animations) to update animation steps for any section."
            )
            base_class_context = ""
        else:
            section_id = self.managed_sections[0]
            section_idx = self.video_state.section_index(section_id)
            section = self.video_state.storyboard[section_idx]
            scene_name = _scene_name_from_section_id(section_id)
            required_setup_layout = (
                f"self.setup_layout({json.dumps(section.title)}, "
                f"{json.dumps(section.lecture_lines, ensure_ascii=False)})"
            )
            edit_instruction = (
                "You are a dedicated coder for one section. Use replace_code(code) for your assigned section. "
                "Return only Python code. "
                f"Mandatory structure: class {scene_name}(TeachingScene) and a construct() containing "
                f"`{required_setup_layout}` before section animations."
            )
            base_class_context = (
                "Provided TeachingScene base class (use as-is; do not assume extra helper methods):\n"
                "```python\n"
                f"{TEACHING_SCENE_BASE_CLASS}\n"
                "```"
            )

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

All active issues (video-wide):
{global_active_issues}

Instructions:
1. Resolve your assigned active issues first.
2. Mark each attempted issue with update_issue(issue_id, under_review=True, resolution_note=...).
3. Keep changes section-specific and avoid editing sections outside your assignment.
4. {edit_instruction}
5. If blocked, add targeted cross-agent issues via add_issue(section_id, toAgent, description).

Guidelines:
{self.guidelines}
"""

        response = self.run_with_retry([prompt], tools, max_retries)
        if response is not None:
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

        response = request_gemini_video_img(
            prompt=prompt,
            video_path=video_path,
            image_path=str(self.grid_img_path),
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
4. Add only high-impact new issues with add_issue(section_id, toAgent, description).
5. Route coding issues to the correct dedicated coder (Coder1/Coder2/... based on coder assignments).
6. Deactivate duplicates/out-of-scope issues with update_issue(..., isActive=False).

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
    ):
        self.video_state = video_state
        self.logs_dir = logs_dir
        self.cfg = cfg
        self.client = client_override or client

        self.case_dir = self.logs_dir
        self.state_logs_dir = self.case_dir / "mas_state"
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.state_logs_dir.mkdir(parents=True, exist_ok=True)

        self._sync_case_documents()
        self._sync_section_code_files()

        self.coder_runtimes: Dict[str, CoderRuntime] = {}
        for section_id in self.video_state.section_ids():
            self.coder_runtimes[section_id] = CoderRuntime(
                runtime_dir=self.case_dir,
                request_fn=lambda prompt, max_tokens=8000: _scope_refine_request_with_client(
                    self.client, prompt, max_tokens=max_tokens
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
        )
        self.animation_planner_agent.tools.append(self.animation_planner_agent.replace_animations)

        self.coder_agents: List[VideoTeamBaseAgent] = []
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
            )
            coder_agent.tools.append(coder_agent.replace_code)
            self.coder_agents.append(coder_agent)

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
        )

        for agent in self.team_agents + [self.orchestrator_agent]:
            agent.set_team_agents(self.team_agents)

    def _storyboard_payload(self) -> Dict[str, object]:
        return {
            "topic": self.video_state.topic,
            "target_audience": self.video_state.target_audience,
            "sections": [asdict(section) for section in self.video_state.storyboard],
        }

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

    def _run_workers_parallel(self) -> None:
        next_agents = self._select_next_agents()
        if not next_agents:
            print("[VideoMAS] No active issues mapped to workers; skipping worker execution for this turn.")
            return

        print(f"[VideoMAS] Running workers: {[agent.name for agent in next_agents]}")
        with ThreadPoolExecutor(max_workers=len(next_agents)) as executor:
            futures = {executor.submit(agent.run, max_retries=self.cfg.max_retries): agent for agent in next_agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    future.result()
                    print(f"[VideoMAS][{agent.name}] completed")
                except Exception as e:
                    print(f"[VideoMAS][{agent.name}] failed: {e}")

    def _run_coder_runtime_for_section(
        self,
        section_id: str,
        max_fix_attempts: int,
    ) -> Tuple[str, Dict[str, object]]:
        runtime = self.coder_runtimes[section_id]
        section_idx = self.video_state.section_index(section_id)
        section = self.video_state.storyboard[section_idx]
        self.video_state.code[section_idx] = normalize_code_to_code2video(
            code=self.video_state.code[section_idx],
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
        return section_id, result

    def _run_coder_runtimes_parallel(self, max_fix_attempts: int, phase_label: str) -> None:
        runnable_sections = [
            section_id
            for section_id in self.video_state.section_ids()
            if bool((self.video_state.code[self.video_state.section_index(section_id)] or "").strip())
        ]
        if not runnable_sections:
            print("[VideoMAS] No section has code yet; skipping coder runtime execution.")
            return

        max_workers = min(max(1, len(runnable_sections)), max(1, self.cfg.section_parallel_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_coder_runtime_for_section, section_id, max_fix_attempts): section_id
                for section_id in runnable_sections
            }
            for future in as_completed(futures):
                section_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{section_id}][CoderRuntime][{phase_label}] failed: {e}")

    def run(self) -> VideoMASState:
        while self.video_state.unresolved_issues() and self.video_state.turns_run < self.cfg.max_turns:
            turn_idx = self.video_state.turns_run + 1
            print(f"[VideoMAS] === Turn {turn_idx} ===")

            self._activate_ready_coder_issues()
            self._run_workers_parallel()
            self._activate_ready_coder_issues()
            self._run_coder_runtimes_parallel(
                max_fix_attempts=self.cfg.coder_fix_attempts,
                phase_label=f"turn_{turn_idx:02d}",
            )

            self._sync_case_documents()
            self._sync_section_code_files()
            _save_video_state_json(self.video_state, self.state_logs_dir, f"turn_{turn_idx:02d}_video_state.json")

            self.video_state.turns_run += 1
            if self.video_state.turns_run >= self.cfg.max_turns:
                break

            self.orchestrator_agent.run(max_retries=self.cfg.max_retries)
            self._sync_case_documents()
            _save_video_state_json(
                self.video_state,
                self.state_logs_dir,
                f"turn_{turn_idx:02d}_orchestrator_video_state.json",
            )

        # Final render-readiness pass: align with original Code2Video "final render" bug-fix budget.
        self._run_coder_runtimes_parallel(
            max_fix_attempts=self.cfg.final_render_fix_attempts,
            phase_label="final_render",
        )
        self._sync_case_documents()
        self._sync_section_code_files()
        _save_video_state_json(self.video_state, self.state_logs_dir, "final_render_pass_video_state.json")
        self._finalize_case_outputs()

        _save_video_state_json(self.video_state, self.state_logs_dir, "final_video_state.json")
        return self.video_state


def run_mas_for_video_state(
    video_state: VideoMASState,
    logs_root: Optional[Path] = None,
    cfg: Optional[MASRunConfig] = None,
    client_override: Optional[Client] = None,
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

    runner = MASVideoRunner(
        video_state=video_state,
        logs_dir=case_dir,
        cfg=cfg,
        client_override=client_override,
    )
    return runner.run()


if __name__ == "__main__":
    generated_outline = generate_outline_with_code2video_stage1(
        knowledge_point=TOPIC,
        duration_minutes=OUTLINE_DURATION_MINUTES,
        max_regenerate_tries=OUTLINE_MAX_REGENERATE_TRIES,
        model=OUTLINE_MODEL,
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
    )

    print(f"Generated outline topic: {generated_outline.topic}")
    print(f"Finished MAS run for {len(final_video_state.section_ids())} sections.")
    print(f"Run outputs written to: {final_video_state.run_output_dir}")
