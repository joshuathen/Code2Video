import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scope_refine import (  # noqa: E402
    ManimCodeErrorAnalyzer,
    ScopeRefineFixer,
    build_compact_error_diagnostic,
    get_completion_only,
)


class ScopeRefineInteractionsTests(unittest.TestCase):
    def test_compacts_python_traceback_to_generated_frame_and_source(self):
        code = "\n".join(
            [
                "class Section3Scene:",
                "    def construct(self):",
                "        value = 1",
                "        self.play(None)",
                "        return value",
            ]
        )
        error = "\n".join(
            [
                'File "/tmp/section_3.py", line 4, in construct',
                "    self.play(None)",
                'File "/venv/manim/scene.py", line 100, in play',
                "TypeError: Unexpected argument None",
            ]
        )

        diagnostic = build_compact_error_diagnostic(error, code, "section_3")

        self.assertEqual(diagnostic["category"], "runtime_type")
        self.assertEqual(diagnostic["generated_frames"][0]["line"], 4)
        self.assertTrue(any("self.play(None)" in line for line in diagnostic["source_context"]))
        self.assertEqual(diagnostic["final_exception"]["type"], "TypeError")

    def test_latex_diagnostic_is_bounded(self):
        error = "\n".join(
            ["startup noise"] * 100
            + ["! Undefined control sequence.", "l.8 \\invalidcommand{x}"]
            + ["latex error converting to dvi"]
            + ["trailing noise"] * 100
        )

        diagnostic = build_compact_error_diagnostic(error, "", "section_1")

        self.assertEqual(diagnostic["category"], "latex")
        self.assertLessEqual(len(diagnostic["primary_diagnostic"]), 20)
        self.assertTrue(
            any("Undefined control sequence" in line for line in diagnostic["primary_diagnostic"])
        )

    def test_unknown_diagnostic_has_bounded_fallback(self):
        error = "\n".join(f"unknown diagnostic line {index} " + "x" * 100 for index in range(200))

        diagnostic = build_compact_error_diagnostic(error, "", "section_1")

        self.assertEqual(diagnostic["category"], "unknown")
        self.assertLessEqual(len("\n".join(diagnostic["primary_diagnostic"])), 6000)

    def test_missing_error_line_uses_complete_source(self):
        analyzer = ManimCodeErrorAnalyzer()
        code = "def construct(self):\n    self.wait(1)"

        result = analyzer.analyze_error(code, "TypeError: invalid argument")

        self.assertIsNone(result["line_number"])
        self.assertEqual(result["fix_scope"], "function")
        self.assertEqual(result["relevant_code_block"], code)

    def test_extracts_tuple_wrapped_interactions_output_text(self):
        response = SimpleNamespace(
            output_text="```python\nfrom manim import *\n```",
            text="fallback",
        )
        self.assertEqual(
            get_completion_only((response, {"total_tokens": 10})),
            "```python\nfrom manim import *\n```",
        )

    def test_preserves_plain_string_response(self):
        self.assertEqual(get_completion_only("x = 1"), "x = 1")

    def test_clean_code_format_removes_fence_and_explanation(self):
        cleaner = ScopeRefineFixer.__new__(ScopeRefineFixer)
        cleaned = cleaner._clean_code_format(
            "Here is the fix:\n```python\nfrom manim import *\n\nx = 1\n```\nDone."
        )
        self.assertEqual(cleaned, "from manim import *\n\nx = 1")

    def test_clean_code_format_accepts_unclosed_fence(self):
        cleaner = ScopeRefineFixer.__new__(ScopeRefineFixer)
        cleaned = cleaner._clean_code_format("```python\nx = 1")
        self.assertEqual(cleaned, "x = 1")

    def test_invalid_response_does_not_replace_valid_retry_baseline(self):
        fixer = ScopeRefineFixer.__new__(ScopeRefineFixer)
        fixer.MAX_CODE_TOKEN_LENGTH = 100
        prompt_sources = []
        responses = iter(
            [
                (SimpleNamespace(output_text="this is invalid"), {}),
                (SimpleNamespace(output_text="x = 2"), {}),
            ]
        )
        fixer.generate_fix_prompt = (
            lambda section_id, current_code, error_msg, attempt: (
                prompt_sources.append(current_code) or "prompt"
            )
        )
        fixer.request_gpt = lambda prompt, max_tokens: next(responses)
        fixer.validate_code_syntax = lambda code: (
            (True, None)
            if code == "x = 2"
            else (False, "SyntaxError")
        )
        fixer.dry_run_test = lambda code, section_id, output_dir: (True, None)

        fixed = fixer.fix_code_with_multi_stage_validation(
            "section_1",
            "x = 1",
            "original error",
            Path("."),
            max_attempts=2,
        )
        self.assertEqual(fixed, "x = 2")
        self.assertEqual(prompt_sources, ["x = 1", "x = 1"])


if __name__ == "__main__":
    unittest.main()
