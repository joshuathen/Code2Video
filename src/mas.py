from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

from type_utils import *

from google.genai import types, Client
from google.genai.errors import ClientError, ServerError

# MAS Team
ORCHESTRATOR = "Orchestrator"
CODER = "Coder"
ANIMATION_PLANNER = "AnimationPlanner"
SCRIPT_WRITER = "ScriptWriter"


# Global Settings
GLOBAL_MAX_TURNS = 3
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0


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


if __name__ == "__main__":
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
    )

    test_agent = MASAgent(
        name="Issuer",
        role="Script Writer",
        guidelines="IMPORTANT: use replace_code() to write python code that prints the topic.",
        section_object=section_state,
        model="gemini-3-flash-preview",
        client=client,
        section=placeholder_section,
        tools=[],
    )
    # test_agent.tools.append(test_agent.replace_lecture_lines)
    # test_agent.tools.append(test_agent.replace_animations)
    test_agent.tools.append(test_agent.replace_code)
    agents.clear()
    agents.append(test_agent)

    try:
        result = test_agent.run(
            max_retries=MAX_RETRIES,
        )
        print("=== Smoke Test Output ===")
        print(result)
        print("=== Issues After Run ===")
        print(section_state)
    except Exception as e:
        print(f"Smoke test failed: {e}")
