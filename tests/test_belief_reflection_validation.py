import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mas_belief_reflection import (  # noqa: E402
    ReflectionAssessmentPayload,
    _evidence_categories_to_values,
    _extract_structured_json,
)


def valid_assessment():
    return {
        "belief_id": None,
        "instruction": "Create persistent text objects outside per-frame updater functions.",
        "scope": {
            "roles": ["Coder"],
            "stages": ["coding", "debugging"],
            "problem_description": "Per-frame text construction causes excessive render time.",
            "context_conditions": ["render_timeout", "dynamic_text_creation"],
        },
        "applicable": True,
        "compliance": "followed",
        "outcome": "positive",
        "action": "ADD",
        "merge_belief_ids": [],
        "evidence_confidence": 0.9,
        "strategy_application": "full",
        "attribution_strength": "strong",
        "evidence_reliability": "direct",
        "outcome_improvement": "resolved",
        "impact": "high",
        "belief_type": "confirmed",
        "timing": "both",
        "agent": "Coder",
        "section_id": "section_1",
        "reason": "The simplified version rendered successfully after the earlier timeout.",
        "evidence": "The matching code hash rendered in 8 seconds after the previous version timed out.",
    }


class BeliefReflectionValidationTests(unittest.TestCase):
    def test_scope_fields_cannot_be_empty(self):
        payload = valid_assessment()
        payload["scope"]["problem_description"] = ""
        payload["scope"]["context_conditions"] = []
        with self.assertRaises(ValidationError):
            ReflectionAssessmentPayload.model_validate(payload)

    def test_numeric_probability_is_not_accepted_as_an_option(self):
        payload = valid_assessment()
        payload["strategy_application"] = 0.95
        with self.assertRaises(ValidationError):
            ReflectionAssessmentPayload.model_validate(payload)

    def test_valid_consistent_assessment_is_accepted(self):
        parsed = ReflectionAssessmentPayload.model_validate(valid_assessment())
        self.assertEqual(parsed.scope.roles, ["Coder"])
        self.assertEqual(parsed.outcome_improvement, "resolved")

    def test_categories_map_to_fixed_bbn_values(self):
        values = _evidence_categories_to_values(
            "full", "strong", "direct", "resolved"
        )
        self.assertEqual(
            values,
            {
                "strategy_applied_probability": 0.95,
                "attribution_probability": 0.90,
                "reliability_probability": 1.0,
                "improvement": 1.0,
            },
        )

    def test_markdown_fenced_json_is_unwrapped(self):
        self.assertEqual(
            _extract_structured_json('```json\n{"assessments":[]}\n```'),
            '{"assessments":[]}',
        )


if __name__ == "__main__":
    unittest.main()
