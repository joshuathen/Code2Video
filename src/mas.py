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

# Role Guidelines
CODER_GUIDELINES = f"""
Objective:
- Convert the current section state into executable Manim code and keep it codable.

Core Responsibilities:
- Validate each animation step is technically feasible in Manim CE v0.19.0.
- Implement or revise code for your assigned section via `replace_code(...)` (or `replace_code_for_section(...)`).
- Flag non-codable animation requirements as issues to AnimationPlanner.

Code Constraints (from stage3):
- Use TeachingScene without modification.
- Use only `place_at_grid` / `place_in_area` for positioning.
- Never use `.to_edge()`, `.move_to()`, or manual coordinate placement.
- Keep lecture line animations limited to color changes only.
- Keep labels close to their target object (within 1 grid unit).
- Avoid 3D methods, complex panel systems, and fragile effects.
- Use light, distinguishable HEX colors.
- If an animation includes `[Asset: ...]`, incorporate it in code.

Issue Workflow:
- Resolve all active issues assigned to you.
- After each attempted resolution, call `update_issue(..., under_review=True, resolution_note=...)`.
- If blocked, create precise issues for AnimationPlanner or ScriptWriter via `add_issue(...)`.

Turn Behavior:
- Keep changes minimal and stable in early turns.
- On turn {GLOBAL_MAX_TURNS}, prioritise finalisation: code should be runnable and aligned with current section design.
""".strip()

ANIMATION_PLANNER_GUIDELINES = f"""
Objective:
- Own the `animations` list and ensure spatially coherent, codable visual stages.

Core Responsibilities:
- Replace/improve animation stages using `replace_animations(...)`.
- Ensure one-to-one alignment between lecture lines and animation steps.
- Keep animation logic clear, incremental, and easy to implement in Manim.

Storyboard Constraints (from stage2):
- No 3D, no panel-heavy layouts, avoid axes unless necessary.
- Prefer simple, high-signal visual actions (appear, move, color shift, fade, scale).
- Keep descriptions concrete enough for coding (objects, transitions, spatial intent).
- Respect visual clarity: avoid occlusion and overcrowding.
- Use assets only when explicitly needed and already available in the section context.

Cross-Agent Collaboration:
- If a line is too long/unclear for animation, assign ScriptWriter a rewrite issue.
- If a planned stage is hard to encode reliably, coordinate with Coder and revise.
- Review Coder feedback and quickly reshape non-feasible animation steps.

Turn Behavior:
- Prioritise fixing structural animation issues in early turns.
- On turn {GLOBAL_MAX_TURNS}, finalise a coherent, codable animation list with no unresolved blocking dependencies.
""".strip()

SCRIPT_WRITER_GUIDELINES = f"""
Objective:
- Own the `lecture_lines` list and ensure concise, teachable narration aligned to visuals.

Core Responsibilities:
- Replace/improve lecture lines using `replace_lecture_lines(...)`.
- Keep each line short and direct (target <= 10 words where possible, per stage2).
- Maintain strict one-to-one alignment with animation stages.

Quality Constraints (from stage2/stage3):
- Lecture text should describe the concept, not animation mechanics.
- Keep terminology consistent with section title and high-level outline.
- Preserve pacing and progression across lines (no jumps or redundancy).
- Ensure lines are compatible with color-highlight-only lecture rendering.

Cross-Agent Collaboration:
- If animations imply missing narrative context, raise an issue to AnimationPlanner.
- If code constraints require wording changes (brevity/clarity), incorporate Coder feedback.
- Avoid introducing new conceptual scope that the current section cannot visualise.

Turn Behavior:
- Improve clarity and alignment in early turns.
- On turn {GLOBAL_MAX_TURNS}, finalise concise lines that cleanly pair with the final animation list.
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


def _save_section_state_json(section_state: Dict[str, object], logs_dir: Path, filename: str) -> Path:
    output_path = logs_dir / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(section_state, f, ensure_ascii=False, indent=2)
    return output_path


def _save_video_state_json(video_state: VideoMASState, logs_dir: Path, filename: str) -> Path:
    output_path = logs_dir / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(video_state), f, ensure_ascii=False, indent=2)
    return output_path


def _build_section_snapshot(video_state: VideoMASState, section_id: str) -> Dict[str, object]:
    idx = video_state.section_index(section_id)
    section = video_state.storyboard[idx]
    outline_section = video_state.section_outline(section_id)
    section_issues = [asdict(issue) for issue in video_state.issues if issue.section_id == section_id]
    return {
        "topic": video_state.topic,
        "target_audience": video_state.target_audience,
        "section": asdict(section),
        "highLevel": outline_section,
        "code": video_state.code[idx],
        "rendered_video_path": video_state.rendered_video_path[idx],
        "render_status": video_state.render_status[idx],
        "render_error": video_state.render_error[idx],
        "video_review": video_state.video_review[idx],
        "issues": section_issues,
        "turns_run": video_state.turns_run,
        "finalised": video_state.finalised,
    }


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
    response = request_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt],
    )
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
        return f"{section_id.title().replace('_', '')}Scene"

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

    def debug_and_fix(self, section_id: str, code: str, max_fix_attempts: int = 3) -> Dict[str, object]:
        if not code or not code.strip():
            return {"success": False, "status": "skipped", "reason": "No code available for section"}

        scene_name = self._scene_name(section_id)
        code_file = self.runtime_dir / f"{section_id}.py"
        current_code = self._normalize_scene_class_name(code, scene_name)
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

            current_code = self._normalize_scene_class_name(fixed_code, scene_name)
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
    clear_logs: bool = True


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
        elif self.name == ANIMATION_PLANNER:
            edit_instruction = (
                "Use replace_animations(section_id, animations) to update animation steps for any section."
            )
        else:
            edit_instruction = (
                "You are a dedicated coder for one section. Use replace_code(code) for your assigned section."
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
        prompt = f"""
Analyze this rendered educational Manim video for layout/spatial quality.
Return JSON only with this shape:
{{
  "layout": {{
    "has_issues": true,
    "improvements": [
      {{"problem": "...", "solution": "..."}}
    ]
  }}
}}

Section title: {self.video_state.storyboard[section_idx].title}
Lecture lines: {'; '.join(self.video_state.storyboard[section_idx].lecture_lines)}
Current grid positions:
{position_table}
""".strip()

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
        self.video_state.video_review[section_idx] = result
        return result

    def run(self, max_retries: int = 3) -> str:
        tools = [self.mark_task_complete, self.add_issue, self.update_issue, self.review_rendered_video]
        tools.extend(self.tools)

        agent_lines = "\n".join(f"- {agent.name}: {agent.role}" for agent in self.team_agents)
        under_review = [x for x in self.video_state.issues if x.under_review and not x.resolved]

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

All active issues:
{self.video_state.active_issues()}

Instructions:
1. Review issue resolutions under review and either:
   - mark_task_complete(issue_id), or
   - update_issue(issue_id, under_review=False, resolution_note=..., isActive=True/False).
2. If rendered video exists for a section, call review_rendered_video(section_id) before deciding closure.
3. Add only high-impact new issues with add_issue(section_id, toAgent, description).
4. Route coding issues to the correct dedicated coder (Coder1/Coder2/... based on coder assignments).
5. Deactivate duplicates/out-of-scope issues with update_issue(..., isActive=False).

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

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.coder_runtimes: Dict[str, CoderRuntime] = {}
        for section_id in self.video_state.section_ids():
            section_log_dir = self.logs_dir / section_id
            section_log_dir.mkdir(parents=True, exist_ok=True)
            self.coder_runtimes[section_id] = CoderRuntime(
                runtime_dir=section_log_dir / "coder_runtime",
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

    def _save_all_sections(self, filename: str) -> None:
        for section_id in self.video_state.section_ids():
            section_dir = self.logs_dir / section_id
            section_dir.mkdir(parents=True, exist_ok=True)
            _save_section_state_json(
                section_state=_build_section_snapshot(self.video_state, section_id),
                logs_dir=section_dir,
                filename=filename,
            )

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
                    usage = future.result()
                    print(f"[VideoMAS][{agent.name}] completed: {usage}")
                except Exception as e:
                    print(f"[VideoMAS][{agent.name}] failed: {e}")

    def _run_coder_runtime_for_section(
        self,
        section_id: str,
        max_fix_attempts: int,
    ) -> Tuple[str, Dict[str, object]]:
        runtime = self.coder_runtimes[section_id]
        section_idx = self.video_state.section_index(section_id)
        result = runtime.debug_and_fix(
            section_id=section_id,
            code=self.video_state.code[section_idx],
            max_fix_attempts=max_fix_attempts,
        )
        self.video_state.code[section_idx] = result.get("code", self.video_state.code[section_idx])
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

            self._save_all_sections(filename=f"turn_{turn_idx:02d}.json")
            _save_video_state_json(self.video_state, self.logs_dir, f"turn_{turn_idx:02d}_video_state.json")

            self.video_state.turns_run += 1
            if self.video_state.turns_run >= self.cfg.max_turns:
                break

            self.orchestrator_agent.run(max_retries=self.cfg.max_retries)
            self._save_all_sections(filename=f"turn_{turn_idx:02d}_orchestrator.json")
            _save_video_state_json(
                self.video_state,
                self.logs_dir,
                f"turn_{turn_idx:02d}_orchestrator_video_state.json",
            )

        # Final render-readiness pass: align with original Code2Video "final render" bug-fix budget.
        self._run_coder_runtimes_parallel(
            max_fix_attempts=self.cfg.final_render_fix_attempts,
            phase_label="final_render",
        )
        self._save_all_sections(filename="final_render_pass.json")
        _save_video_state_json(self.video_state, self.logs_dir, "final_render_pass_video_state.json")

        self._save_all_sections(filename="final_section_state.json")
        _save_video_state_json(self.video_state, self.logs_dir, "final_video_state.json")
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
    if cfg.clear_logs:
        if logs_root.exists():
            shutil.rmtree(logs_root)
        logs_root.mkdir(parents=True, exist_ok=True)
    else:
        logs_root.mkdir(parents=True, exist_ok=True)

    runner = MASVideoRunner(
        video_state=video_state,
        logs_dir=logs_root,
        cfg=cfg,
        client_override=client_override,
    )
    return runner.run()


if __name__ == "__main__":
    placeholder_outline = TeachingOutline(
        topic="Linear Transformations and Matrices: The Geometry of Space",
        target_audience="Undergraduate students in STEM or advanced high school students studying linear algebra.",
        sections=[
            {
                "id": "section_1",
                "title": "The Canvas: Prerequisites of Vector Space",
                "content": (
                    "Establish the 2D Cartesian plane as our playground. Introduce the unit basis vectors: "
                    "'i-hat' (1,0) and 'j-hat' (0,1). Explain that any point in space is just a combination of "
                    "these two building blocks."
                ),
                "example": (
                    "Imagine a character named 'Pixel the Robot' standing at coordinates (3, 2). Pixel is "
                    "essentially standing on a spot reached by walking 3 units of i-hat and 2 units of j-hat."
                ),
            },
            {
                "id": "section_2",
                "title": "The Morphing Rule: Defining Linear Transformations",
                "content": (
                    "Define a 'transformation' as a function that takes a vector and spits out a new one. "
                    "A transformation is 'linear' only if the origin remains fixed at (0,0) and all grid "
                    "lines remain parallel and evenly spaced after the move."
                ),
                "example": (
                    "Visualize the entire grid stretching or tilting like a sheet of rubber. If the grid lines "
                    "curve or the center moves, Pixel the Robot knows it's not a linear transformation."
                ),
            },
            {
                "id": "section_3",
                "title": "The Secret of the Basis Vectors",
                "content": (
                    "Introduce the breakthrough concept: To track where the entire infinite plane goes, you only "
                    "need to track where the two basis vectors (i-hat and j-hat) land. Everything else follows "
                    "their lead."
                ),
                "example": (
                    "If i-hat moves from (1,0) to (1, -2) and j-hat moves from (0,1) to (3, 0), the entire grid "
                    "follows this 'recipe' to reshape itself."
                ),
            },
            {
                "id": "section_4",
                "title": "From Motion to Matrix: Recording the Change",
                "content": (
                    "Explain that a matrix is simply a numerical 'container' for these new coordinates. "
                    "The first column of a 2x2 matrix is where i-hat landed; the second column is where "
                    "j-hat landed."
                ),
                "example": (
                    "Using the previous movement, we write the matrix M = [[1, 3], [-2, 0]]. "
                    "This 2x2 grid of numbers is actually a compact map of the entire transformation."
                ),
            },
            {
                "id": "section_5",
                "title": "Applying the Transformation: Moving Pixel",
                "content": (
                    "Show the mathematical operation (Matrix-Vector multiplication) as a way to find Pixel's new "
                    "location. The new vector is '3 times the new i-hat' + '2 times the new j-hat'."
                ),
                "example": (
                    "By multiplying our matrix M by Pixel's original position (3, 2), we see Pixel instantly "
                    "'teleport' to his new transformed position on the tilted grid."
                ),
            },
            {
                "id": "section_6",
                "title": "Visual Summary: Matrices as Functions",
                "content": (
                    "Recap the core idea: Matrices are not just boxes of numbers; they are geometric actions "
                    "(rotations, shears, scales) expressed through the language of basis vectors."
                ),
                "example": (
                    "Show an animation of a square 'cat' face being squashed by a matrix [[0.5, 0], [0, 2]], "
                    "demonstrating a vertical stretch and horizontal compression."
                ),
            },
        ],
    )

    video_state = build_video_state_from_outline(placeholder_outline)
    logs_dir = Path(__file__).resolve().parent.parent / "mas_logs"

    cfg = MASRunConfig(
        max_turns=GLOBAL_MAX_TURNS,
        max_retries=MAX_RETRIES,
        section_parallel_workers=len(video_state.section_ids()),
        clear_logs=True,
    )

    final_video_state = run_mas_for_video_state(
        video_state=video_state,
        logs_root=logs_dir,
        cfg=cfg,
    )

    print(f"Finished MAS run for {len(final_video_state.section_ids())} sections.")
    print(f"Section state logs written to: {logs_dir}")
