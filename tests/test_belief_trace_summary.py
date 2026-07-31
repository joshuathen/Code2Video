import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mas_belief_reflection import _summarize_function_trace  # noqa: E402


class BeliefTraceSummaryTests(unittest.TestCase):
    def test_empty_events_are_removed_but_calls_and_errors_are_retained(self):
        rows = [
            {
                "timestamp": "t1",
                "event_type": "tool_call",
                "agent": "Coder1",
                "text_parts": [],
                "function_calls": [],
                "function_responses": [],
            },
            {
                "timestamp": "t2",
                "event_type": "agent_response",
                "agent": "Coder1",
                "text_parts": ["Applied the repair."],
                "function_calls": [
                    {"name": "replace_code", "arguments": {"section_id": "s1"}}
                ],
                "function_responses": [],
            },
            {
                "timestamp": "t3",
                "event_type": "transport_error",
                "agent": "Coder1",
                "error": "connection reset",
                "retrying": True,
                "text_parts": [],
                "function_calls": [],
                "function_responses": [],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_function_calls.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = _summarize_function_trace(Path(directory))

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["function_calls"][0]["name"], "replace_code")
        self.assertEqual(summary[1]["error"], "connection reset")
        self.assertTrue(summary[1]["retrying"])


if __name__ == "__main__":
    unittest.main()
