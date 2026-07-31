import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_normalization import normalize_code_to_code2video  # noqa: E402
from mas_belief_reflection import _build_turn_code_trajectories  # noqa: E402


class BeliefTurnCodeTrajectoryTests(unittest.TestCase):
    def test_reconstructs_turn_edits_and_render_feedback_without_full_source(self):
        raw_original = "from manim import *\nclass Section1Scene(Scene):\n    pass\n"
        raw_revised = "from manim import *\nclass Section1Scene(Scene):\n    def construct(self):\n        self.add(Dot())\n"
        original = normalize_code_to_code2video(raw_original, "section_1", "Title", [])
        revised = normalize_code_to_code2video(raw_revised, "section_1", "Title", [])

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state_dir = run_dir / "mas_state"
            attempt_dir = run_dir / "coder_debugger" / "section_1" / "attempt_01"
            state_dir.mkdir(parents=True)
            attempt_dir.mkdir(parents=True)

            state_base = {"storyboard": [{"id": "section_1"}]}
            (state_dir / "turn_01_orchestrator_video_state.json").write_text(
                json.dumps({**state_base, "code": [original]}), encoding="utf-8"
            )
            (state_dir / "turn_02_video_state.json").write_text(
                json.dumps({**state_base, "code": [revised]}), encoding="utf-8"
            )
            (run_dir / "agent_function_calls.jsonl").write_text(
                json.dumps(
                    {
                        "event_type": "tool_call",
                        "agent": "Coder1",
                        "turn_number": 2,
                        "tool_name": "replace_code",
                        "input": {"code": raw_revised},
                        "output": {"section_id": "section_1"},
                        "status": "success",
                        "timestamp": "2026-07-30T00:00:00Z",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (attempt_dir / "attempt.json").write_text(
                json.dumps(
                    {
                        "attempt_number": 1,
                        "phase": "turn_02",
                        "code_before": revised,
                        "render_error": "",
                        "elapsed_seconds": 1.25,
                        "repair_strategy": "none",
                        "render_outcome": {
                            "success": True,
                            "returncode": 0,
                            "timed_out": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            trajectories = _build_turn_code_trajectories(run_dir)

        trajectory = trajectories["2"]["section_1"]
        self.assertEqual(trajectory["start_code"]["char_count"], len(original))
        self.assertEqual(trajectory["start_code"]["full_source"], original)
        self.assertEqual(trajectory["end_code"]["char_count"], len(revised))
        self.assertEqual(len(trajectory["modifications"]), 1)
        self.assertEqual(trajectory["modifications"][0]["change_type"], "agent_edit")
        self.assertTrue(trajectory["modifications"][0]["normalization_changed_code"])
        self.assertIn("+        self.add(Dot())", trajectory["modifications"][0]["diff"])
        self.assertEqual(
            trajectory["render_attempts"][0]["code_sha1"],
            hashlib.sha1(revised.encode("utf-8")).hexdigest(),
        )
        self.assertTrue(trajectory["render_attempts"][0]["outcome"]["success"])
        self.assertNotIn("code", trajectory["modifications"][0]["after"])
        self.assertNotIn("full_source", trajectory["modifications"][0]["after"])

    def test_initial_generation_includes_full_source_once(self):
        raw_code = "from manim import *\nclass Section1Scene(Scene):\n    pass\n"
        normalized = normalize_code_to_code2video(raw_code, "section_1", "Title", [])
        raw_revised = raw_code.replace("pass", "def construct(self):\n        self.add(Dot())")
        normalized_revised = normalize_code_to_code2video(
            raw_revised, "section_1", "Title", []
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            state_dir = run_dir / "mas_state"
            state_dir.mkdir(parents=True)
            state = {"storyboard": [{"id": "section_1", "title": "Title"}], "code": [normalized]}
            (state_dir / "turn_02_video_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            (state_dir / "turn_02_orchestrator_video_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            (state_dir / "turn_03_video_state.json").write_text(
                json.dumps({**state, "code": [normalized_revised]}), encoding="utf-8"
            )
            (state_dir / "final_video_state.json").write_text(
                json.dumps({**state, "code": [normalized_revised]}), encoding="utf-8"
            )
            (run_dir / "agent_function_calls.jsonl").write_text(
                json.dumps(
                    {
                        "agent": "Coder1",
                        "turn_number": 2,
                        "tool_name": "replace_code",
                        "input": {"code": raw_code},
                        "output": {"section_id": "section_1"},
                        "status": "success",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "agent": "Coder1",
                        "turn_number": 3,
                        "tool_name": "replace_code",
                        "input": {"code": raw_revised},
                        "output": {"section_id": "section_1"},
                        "status": "success",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            trajectories = _build_turn_code_trajectories(run_dir)
            trajectory = trajectories["2"]["section_1"]

        self.assertIsNone(trajectory["start_code"]["sha1"])
        self.assertEqual(
            trajectory["modifications"][0]["after"]["full_source"],
            normalized,
        )
        self.assertNotIn("full_source", trajectory["end_code"])
        self.assertNotIn("full_source", trajectories["3"]["section_1"]["start_code"])
        self.assertNotIn(
            "full_source",
            trajectories["3"]["section_1"]["modifications"][0]["after"],
        )


if __name__ == "__main__":
    unittest.main()
