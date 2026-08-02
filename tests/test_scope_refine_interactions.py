import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scope_refine import ScopeRefineFixer, get_completion_only  # noqa: E402


class ScopeRefineInteractionsTests(unittest.TestCase):
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
