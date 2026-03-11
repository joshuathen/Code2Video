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
from scope_refine import ScopeRefineFixer

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
RETRY_BASE_DELAY_SECONDS = 1.0

# Role Guidelines
CODER_GUIDELINES = f"""
Objective:
- Convert the current section state into executable Manim code and keep it codable.

Core Responsibilities:
- Validate each animation step is technically feasible in Manim CE v0.19.0.
- Implement or revise the section code in `section_object.code`.
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
    resolved: bool = False
    under_review: bool = False
    resolution_note: str = ""


@dataclass
class SectionMASState:
    section: Section
    highLevel: SectionOutline  # High-level outline of the section
    topic: str
    target_audience: str
    issues: List[Issue] = field(default_factory=list)
    code: Optional[str] = None
    finalised: bool = False
    turns_run: int = 0

    def unresolved_issues(self) -> List[Issue]:
        return [x for x in self.issues if not x.resolved]

    def active_issues(self) -> List[Issue]:
        return [x for x in self.issues if x.isActive and not x.resolved]


def _reset_mas_logs_dir() -> Path:
    logs_dir = Path(__file__).resolve().parent.parent / "mas_logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _save_section_state_json(section_state: SectionMASState, logs_dir: Path, filename: str) -> Path:
    output_path = logs_dir / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(section_state), f, ensure_ascii=False, indent=2)
    return output_path


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


def _scope_refine_request(prompt: str, max_tokens: int = 8000):
    """
    Adapter for ScopeRefineFixer.
    Returns OpenAI-like `(completion, usage)` shape expected by scope_refine.py.
    """
    response = client.models.generate_content(
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
    4) sync fixed code back to section state
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

    def debug_and_fix(self, section_state: SectionMASState, max_fix_attempts: int = 3) -> Dict[str, object]:
        if not section_state.code or not section_state.code.strip():
            return {"success": False, "status": "skipped", "reason": "No code available on section state"}

        section_id = section_state.section.id
        scene_name = self._scene_name(section_id)
        code_file = self.runtime_dir / f"{section_id}.py"
        current_code = self._normalize_scene_class_name(section_state.code, scene_name)
        section_state.code = current_code
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
                section_state.code = current_code
                video_path = self._find_video_path(section_id, scene_name)
                return {
                    "success": True,
                    "status": "ok",
                    "attempts": fix_attempt,
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
            section_state.code = current_code
            code_file.write_text(current_code, encoding="utf-8")

        return {
            "success": False,
            "status": "failed",
            "attempts": max_fix_attempts,
            "error": last_error,
        }


agents = [] # List of agents


class MASAgent:
    def __init__(self, name: str, role: str, guidelines: str, section_object: SectionMASState, model: str, client: Client, section: Section, tools: Optional[List[Callable]] = None):
        self.name = name
        self.role = role
        self.guidelines = guidelines
        self.section_object = section_object
        self.model = model
        self.client = client
        self.section = section
        self.tools = tools or []
        self.original_model = model
    
    def add_issue(self, toAgent: str, description: str) -> int:
        """
        Adds a new issue to the current section state and assigns it to a target agent.

        Args:
            toAgent (str): The agent responsible for resolving the issue.
            description (str): A clear description of what needs to be resolved.

        Returns:
            int: The ID of the newly created issue.

        Raises:
            ValueError: If the target agent does not exist in the global agents list.
            ValueError: If the issue description is empty.
        """
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Issue description must be a non-empty string.")

        valid_agent_names = {agent.name for agent in agents}
        if toAgent not in valid_agent_names:
            raise ValueError(f"Agent '{toAgent}' does not exist. Valid agents: {sorted(valid_agent_names)}")

        issue = Issue(
            id=len(self.section_object.issues) + 1,
            fromAgent=self.name,
            toAgent=toAgent,
            description=description,
            isActive=True
        )
        self.section_object.issues.append(issue)
        return issue.id
    
    def update_issue(
        self,
        issue_id: int,
        under_review: Optional[bool] = None,
        resolution_note: Optional[str] = None,
    ) -> None:
        """
        Updates an existing issue's review status and/or resolution note.

        Args:
            issue_id (int): The unique identifier of the issue to update.
            under_review (Optional[bool]): Whether the issue is currently under orchestrator review.
            resolution_note (Optional[str]): Notes describing what was attempted or resolved.

        Returns:
            None

        Raises:
            ValueError: If the issue ID does not exist.
            ValueError: If neither `under_review` nor `resolution_note` is provided.
            ValueError: If `under_review` is provided but is not a boolean.
            ValueError: If `resolution_note` is provided but is empty.
        """
        if under_review is None and resolution_note is None:
            raise ValueError("At least one field must be updated: under_review or resolution_note.")
        if under_review is not None and not isinstance(under_review, bool):
            raise ValueError("under_review must be a boolean when provided.")
        if resolution_note is not None and (not isinstance(resolution_note, str) or not resolution_note.strip()):
            raise ValueError("resolution_note must be a non-empty string when provided.")

        target_issue = None
        for issue in self.section_object.issues:
            if issue.id == issue_id:
                target_issue = issue
                break
        if target_issue is None:
            raise ValueError(f"Issue with id={issue_id} does not exist.")

        if under_review is not None:
            target_issue.under_review = under_review
        if resolution_note is not None:
            target_issue.resolution_note = resolution_note

    def replace_lecture_lines(self, lecture_lines: List[str]) -> Dict[str, object]:
        """
        Replaces the entire lecture_lines list on the section object.

        Args:
            lecture_lines (List[str]): New lecture lines to fully replace the existing list.

        Returns:
            Dict[str, object]: Summary of the update, including updated lines and count.

        Raises:
            ValueError: If `lecture_lines` is not a list of non-empty strings.
        """
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

        # Keep both references in sync.
        self.section_object.section.lecture_lines = cleaned_lines
        self.section.lecture_lines = cleaned_lines

        return {
            "status": "ok",
            "lecture_lines": cleaned_lines,
            "count": len(cleaned_lines),
        }

    def replace_animations(self, animations: List[str]) -> Dict[str, object]:
        """
        Replaces the entire animations list on the section object.

        Args:
            animations (List[str]): New animations to fully replace the existing list.

        Returns:
            Dict[str, object]: Summary of the update, including updated animations and count.

        Raises:
            ValueError: If `animations` is not a list of non-empty strings.
        """
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

        # Keep both references in sync.
        self.section_object.section.animations = cleaned_animations
        self.section.animations = cleaned_animations

        return {
            "status": "ok",
            "animations": cleaned_animations,
            "count": len(cleaned_animations),
        }

    def replace_code(self, code: str) -> Dict[str, object]:
        """
        Replaces the entire code string on the section state object.

        Args:
            code (str): New code to fully replace the existing code.

        Returns:
            Dict[str, object]: Summary of the update, including character and line counts.

        Raises:
            ValueError: If `code` is not a non-empty string.
        """
        if not isinstance(code, str):
            raise ValueError("code must be a string.")

        normalized = code.strip()
        if not normalized:
            raise ValueError("code cannot be empty.")

        self.section_object.code = normalized

        return {
            "status": "ok",
            "char_count": len(normalized),
            "line_count": len(normalized.splitlines()),
        }

    def run_with_retry(self, parts, tools, max_retries):
        attempts = 0

        alternative_models = ["gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"]
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
                print(f"===== {self.name} =====")
                response_text = (response.text or "").rstrip()
                print(response_text)
                print("\n\n")
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
    
    def run(self, max_retries: int = 3) -> str:
        tools = [self.add_issue, self.update_issue]
        tools.extend(self.tools)  # Add any custom agent-specific tools provided at initialisation


        agent_lines = "\n".join(f"- {agent.name}: {agent.role}" for agent in agents)

        prompt = f"""You are {self.name} - an agent in a Manim Animation Code AI team that is creating a section of a larger video (using Manim code) on {self.section_object.topic} for {self.section_object.target_audience}. Your role is specifically {self.role}.

        The following object represents the high-level outline for the section you are working on. This includes the content that is to be presented in the section as well as relevant examples. You should use this as a guide to inform your work on the section, but you may need to deviate from it if there are issues that need to be resolved or if you identify improvements that can be made.
        {self.section_object.highLevel}
        
        The following object represents the current state of the design - this includes the section title, lecture lines and animations that have been generated so far. This is the current state of the section that you are working on.
        {self.section_object.section}

        Here are the steps you should follow:
        1. Review the current state of the section and identify any issues assigned to you that need to be resolved
            - Review any issues assigned to you in the list below:
            {self.section_object.active_issues()}
            - Identify the how the issue is relevant to the current state of the section and use this to inform your approach to resolving the issue
        2. Address the issues assigned to you and try to resolve them; update their status accordingly using the update_issue() function
            - Use the functions provided to you to update the design state as required to resolve the issue
            - Once the issue is resolved, use the update_issue() function to set under_review to True and provide a detailed resolution note describing what was done to resolve the issue
            - If the issue cannot be resolved, provide detailed reasoning in the resolution note and set under_review to True using the update_issue() function
            - Try and resolve ALL issues assigned to you
        3. Identify any improvements you see (where possible) and add issues (the most important) to the list of issues using the add_issue() function
            - Minimise the number of issues being assigned - try and assign only the most important (or a couple) issues
            - Provide detailed description of the issue in the add_issue() function
            - Make sure to only add issues that are directly addressable by the other agents (make sure to assign the issue to the most appropriate agent based on the agent's role and the nature of the issue - i.e. assign coding related issues to the coder, animation related issues to the animation planner etc.). The list of agents are:
            {agent_lines}
        
        Guidelines:
        {self.guidelines}
        """

        parts = [prompt]

        response = self.run_with_retry(parts, tools, max_retries)

        if response != None:
            self.eval_dump(parts, tools, response)
            return [response.usage_metadata, self.model, self.name]

    def eval_dump(self, parts, tools, response):
        # Lightweight hook for debugging/inspection.
        return None


class OrchestratorAgent(MASAgent):
    def __init__(
        self,
        name: str,
        section_object: SectionMASState,
        model: str,
        client: Client,
        section: Section,
        prompt: str = "",
        guidelines: str = "",
        tools: Optional[List[Callable]] = None,
    ):
        super().__init__(
            name=name,
            role=ORCHESTRATOR,
            guidelines=guidelines,
            section_object=section_object,
            model=model,
            client=client,
            section=section,
            tools=tools,
        )

    def mark_task_complete(self, issue_id: int) -> Dict[str, object]:
        """
        Marks an issue/task as completed by id.

        Args:
            issue_id (int): The id of the issue/task to complete.

        Returns:
            Dict[str, object]: Completion status and updated issue metadata.
        """
        if not isinstance(issue_id, int):
            raise ValueError("issue_id must be an integer.")

        target_issue = None
        for issue in self.section_object.issues:
            if issue.id == issue_id:
                target_issue = issue
                break

        if target_issue is None:
            raise ValueError(f"Issue with id={issue_id} does not exist.")

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
        """
        Orchestrator-only issue updater with support for activity toggling.

        Args:
            issue_id (int): The unique identifier of the issue to update.
            under_review (Optional[bool]): Whether the issue is under orchestrator review.
            resolution_note (Optional[str]): Notes describing what was attempted or decided.
            isActive (Optional[bool]): Whether the issue should remain active.

        Returns:
            None
        """
        if under_review is None and resolution_note is None and isActive is None:
            raise ValueError("At least one field must be updated: under_review, resolution_note, or isActive.")
        if under_review is not None and not isinstance(under_review, bool):
            raise ValueError("under_review must be a boolean when provided.")
        if resolution_note is not None and (not isinstance(resolution_note, str) or not resolution_note.strip()):
            raise ValueError("resolution_note must be a non-empty string when provided.")
        if isActive is not None and not isinstance(isActive, bool):
            raise ValueError("isActive must be a boolean when provided.")

        target_issue = None
        for issue in self.section_object.issues:
            if issue.id == issue_id:
                target_issue = issue
                break
        if target_issue is None:
            raise ValueError(f"Issue with id={issue_id} does not exist.")

        if under_review is not None:
            target_issue.under_review = under_review
        if resolution_note is not None:
            target_issue.resolution_note = resolution_note
        if isActive is not None:
            target_issue.isActive = isActive

    def run(self, max_retries: int = 3) -> str:
        tools = [self.mark_task_complete, self.add_issue, self.update_issue]
        tools.extend(self.tools)

        agent_lines = "\n".join(f"- {agent.name}: {agent.role}; GUIDELINES: {agent.guidelines}" for agent in agents)

        prompt = f"""You are {self.name} - an agent in a Manim Animation Code AI team that is in charge of leading and directing the team of agents to help generate a section of a larger video (using Manim code) on {self.section_object.topic} for {self.section_object.target_audience}. The list of agents in the team is provided below along with their roles and guidelines:
        {agent_lines}

        The following object represents the high-level outline for the section you are working on. This includes the content that is to be presented in the section as well as relevant examples. You should use this as a high-level guide to provide some information on the overall purpose of the scene:
        {self.section_object.highLevel}
        
        The following object represents the current state of the design - this includes the section title, lecture lines and animations that have been generated so far. This is the current state of the section that you will be evaluating. Use this object to identify issues and also get a feel for the progress that has been made on the section so far:
        {self.section_object.section}

        Here are the steps you should follow:
        1. Review the current state of the section thoroughly
        2. Review the following issues which agents attempted to resolve in the previous turn and identify whether they have been resolved or not based on the current state of the section. The list may be empty:
            {[x for x in self.section_object.issues if x.under_review]}
            - If the issue has been resolved, use the mark_task_complete() function to mark the issue as completed and resolved
            - If the issue has not been resolved, provide detailed feedback in the resolution note and description using the update_issue() function and set under_review to False to indicate that the issue needs to be re-addressed by the assigned agent in the next turn
        3. Identify any other additional issues that need to be resolved based on the current state of the section and add them to the list of issues using the add_issue() function
            - Minimise the number of issues being assigned - try and assign only the most important (or a couple) issues
            - Provide detailed description of the issue in the add_issue() function
            - Make sure to only add issues that are directly addressable by the other agents (make sure to assign the issue to the most appropriate agent based on the agent's role and the nature of the issue - i.e. assign coding related issues to the coder, animation related issues to the animation planner etc.). Use the previously provided list of agents and their roles to help you determine which agent to assign the issue to.
        4. Deactivate any issues that can't be resolved in the next turn (due to dependencies on other issues being resolved first, or if they are out of scope for the current section) by setting isActive to False using the update_issue() function and providing detailed reasoning in the resolution note.

        Guidelines:
        {self.guidelines}
        """

        parts = [prompt]

        response = self.run_with_retry(parts, tools, max_retries)

        if response != None:
            self.eval_dump(parts, tools, response)
            return [response.usage_metadata, self.model, self.name]
        
        return None


if __name__ == "__main__":
    logs_dir = _reset_mas_logs_dir()
    coder_runtime_dir = logs_dir / "coder_runtime"
    coder_runtime = CoderRuntime(
        runtime_dir=coder_runtime_dir,
        request_fn=_scope_refine_request,
        max_code_token_length=10000,
    )

    # Test - using 
    placeholder_section = Section(
        id="section_1",
        title="The Canvas: Prerequisites of Vector Space",
        lecture_lines=[],
        animations=[],
    )

    placeholder_high_level = SectionOutline(
        id="section_1",
        title="The Canvas: Prerequisites of Vector Space",
        content="Establish the 2D Cartesian plane as our playground. Introduce the unit basis vectors: 'i-hat' (1,0) and 'j-hat' (0,1). Explain that any point in space is just a combination of these two building blocks.",
        example="Imagine a character named 'Pixel the Robot' standing at coordinates (3, 2). Pixel is essentially standing on a spot reached by walking 3 units of i-hat and 2 units of j-hat.",
    )

    section_state = SectionMASState(
        section=placeholder_section,
        highLevel=placeholder_high_level,
        topic="Linear Transformations and Matrices: The Geometry of Space",
        target_audience="Undergraduate students in STEM or advanced high school students studying linear algebra.",
        issues=[
            Issue(id=1, fromAgent=ORCHESTRATOR, toAgent=SCRIPT_WRITER, description="The lecture lines are currently empty. Please generate concise and clear lecture lines that align with the section content and example provided in the high-level outline.", isActive=True),
            Issue(id=2, fromAgent=ORCHESTRATOR, toAgent=ANIMATION_PLANNER, description="The animations list is currently empty. Please generate a list of animations that visually represent the concepts described in the section content and example provided in the high-level outline.", isActive=True),
            Issue(id=3, fromAgent=ORCHESTRATOR, toAgent=CODER, description="The code list is currently empty. Please generate the necessary code to represent the animations.", isActive=False)
        ],
    )

    script_writer_agent = MASAgent(
        name=SCRIPT_WRITER,
        role=SCRIPT_WRITER_ROLE,
        guidelines=SCRIPT_WRITER_GUIDELINES,
        section_object=section_state,
        model="gemini-3-flash-preview",
        client=client,
        section=placeholder_section,
        tools=[],
    )
    script_writer_agent.tools.append(script_writer_agent.replace_lecture_lines)

    animation_planner_agent = MASAgent(
        name=ANIMATION_PLANNER,
        role=ANIMATION_PLANNER_ROLE,
        guidelines=ANIMATION_PLANNER_GUIDELINES,
        section_object=section_state,
        model="gemini-3-flash-preview",
        client=client,
        section=placeholder_section,
        tools=[],
    )
    animation_planner_agent.tools.append(animation_planner_agent.replace_animations)

    coder_agent = MASAgent(
        name=CODER,
        role=CODER_ROLE,
        guidelines=CODER_GUIDELINES,
        section_object=section_state,
        model="gemini-3-flash-preview",
        client=client,
        section=placeholder_section,
        tools=[],
    )
    coder_agent.tools.append(coder_agent.replace_code)

    orchestrator_agent = OrchestratorAgent(
        name="OrchestratorAgent",
        guidelines="",
        section_object=section_state,
        model="gemini-3.1-pro-preview",
        client=client,
        section=placeholder_section,
        tools=[],
    )

    agents.clear()
    agents.extend([script_writer_agent, animation_planner_agent, coder_agent])

    while section_state.unresolved_issues() and section_state.turns_run < GLOBAL_MAX_TURNS:
        print(f"=== Turn {section_state.turns_run + 1} ===")
        
        # Identify the needed agents for the next turn based on active issues
        next_agent = set()
        for issue in section_state.active_issues():
            # identify the agent responsible for the issue and add to next_agent set
            # only checks for agents in valid list (does not contain orchestrator)
            print(issue.toAgent)
            for agent in agents:
                if agent.name == issue.toAgent:
                    next_agent.add(agent)
                    break
        
        if len(next_agent) == 0:
            print("No active issues found, but there are unresolved issues. Skipping agent operation and asking producer to activate issues.")
        else:
            print(f"Running agents: {[agent.name for agent in next_agent]}")

            # Run all selected agents in parallel for this turn.
            with ThreadPoolExecutor(max_workers=len(next_agent)) as executor:
                futures = {
                    executor.submit(agent.run, max_retries=MAX_RETRIES): agent for agent in next_agent
                }
                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        usage = future.result()
                        print(f"[{agent.name}] completed: {usage}")
                    except Exception as e:
                        print(f"[{agent.name}] failed: {e}")

        runtime_result = coder_runtime.debug_and_fix(
            section_state=section_state,
            max_fix_attempts=3,
        )
        print(f"[CoderRuntime] result: {runtime_result}")
        _save_section_state_json(
            section_state=section_state,
            logs_dir=logs_dir,
            filename=f"turn_{section_state.turns_run + 1:02d}.json",
        )
        
        # CODE TO RUN THE ORCHESTRATOR AGENT TO REVIEW ISSUES AND UPDATE THEIR STATUS
        orchestrator_agent.run(max_retries=MAX_RETRIES)
        _save_section_state_json(
            section_state=section_state,
            logs_dir=logs_dir,
            filename=f"turn_{section_state.turns_run + 1:02d}_orchestrator.json",
        )
        
        section_state.turns_run += 1

    # try:
    #     result = script_writer_agent.run(
    #         max_retries=MAX_RETRIES,
    #     )
    #     print("=== Smoke Test Output ===")
    #     print(result)
    #     _save_section_state_json(
    #         section_state=section_state,
    #         logs_dir=logs_dir,
    #         filename="final_section_state.json",
    #     )
    #     print(f"Section state logs written to: {logs_dir}")
    # except Exception as e:
    #     print(f"Smoke test failed: {e}")
