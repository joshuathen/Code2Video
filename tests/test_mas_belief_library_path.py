import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mas import (  # noqa: E402
    _belief_injection_enabled_for_role,
    _resolve_belief_library_path,
)


class BeliefLibraryPathResolutionTests(unittest.TestCase):
    def test_ordinary_agent_prompt_injection_is_disabled_by_default(self):
        self.assertFalse(_belief_injection_enabled_for_role("Coder"))
        self.assertFalse(_belief_injection_enabled_for_role("Coder6"))
        self.assertFalse(_belief_injection_enabled_for_role("Orchestrator"))
        self.assertFalse(_belief_injection_enabled_for_role("ScriptWriter"))
        self.assertFalse(_belief_injection_enabled_for_role("AnimationPlanner"))

    def test_historical_broad_injection_can_be_explicitly_reproduced(self):
        with patch.dict("os.environ", {"ENABLE_BROAD_BELIEF_INJECTION": "1"}):
            self.assertTrue(_belief_injection_enabled_for_role("Coder"))
            self.assertTrue(_belief_injection_enabled_for_role("Orchestrator"))
            self.assertTrue(_belief_injection_enabled_for_role("ScriptWriter"))
            self.assertTrue(_belief_injection_enabled_for_role("AnimationPlanner"))

    def test_coder_wide_injection_only_enables_ordinary_coder_prompts(self):
        with patch.dict(
            "os.environ",
            {
                "ENABLE_BROAD_BELIEF_INJECTION": "0",
                "ENABLE_CODER_WIDE_BELIEF_INJECTION": "1",
            },
        ):
            self.assertTrue(_belief_injection_enabled_for_role("Coder"))
            self.assertTrue(_belief_injection_enabled_for_role("Coder6"))
            self.assertFalse(_belief_injection_enabled_for_role("Orchestrator"))
            self.assertFalse(_belief_injection_enabled_for_role("ScriptWriter"))
            self.assertFalse(_belief_injection_enabled_for_role("AnimationPlanner"))

    def test_omitted_path_does_not_auto_discover_beliefs(self):
        self.assertIsNone(_resolve_belief_library_path(None))
        self.assertIsNone(_resolve_belief_library_path(""))

    def test_explicit_library_path_is_resolved(self):
        expected = Path(__file__).resolve()
        self.assertEqual(
            _resolve_belief_library_path(str(expected)),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
