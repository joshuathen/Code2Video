import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mas_belief_three_stage import (  # noqa: E402
    CandidateDecision,
    ConsolidatedBeliefPayload,
    ConsolidationResponse,
    EvidenceObservation,
    RetrospectiveResponse,
    _apply_discovery_decisions,
    _consolidation_quality_concerns,
    _consolidation_prompt,
    _derive_update_direction,
    _discovery_prompt,
    _observation_matrix_payload,
    _validate_consolidation,
    _validate_evaluation,
)


def decision(action, candidate_id=None, instruction=None):
    return CandidateDecision.model_validate(
        {
            "action": action,
            "candidate_id": candidate_id,
            "instruction": instruction,
            "scope": (
                {
                    "roles": ["Coder"],
                    "stages": ["coding"],
                    "problem_description": "Repeated construction causes render slowdown.",
                    "context_conditions": ["per_frame_construction"],
                }
                if instruction is not None
                else None
            ),
            "impact": "high",
            "belief_type": "confirmed",
            "timing": "both",
            "compliance": "followed",
            "outcome": "positive",
            "strategy_application": "full",
            "attribution_strength": "strong",
            "evidence_reliability": "direct",
            "outcome_improvement": "resolved",
            "evidence": "The exact revised code rendered after the prior timeout.",
            "reason": "The changed construction strategy preceded the successful render.",
            "agent": "Coder",
            "section_id": "section_1",
        }
    )


class ThreeStageBeliefTests(unittest.TestCase):
    def test_discovery_can_reject_non_belief_observation(self):
        candidates = []
        rejections = []
        rejected = decision("REJECT")
        rejected.reason = "This merely restates an existing pipeline instruction."

        counts = _apply_discovery_decisions(
            candidates,
            [rejected],
            run_id="run-1",
            topic="Topic 1",
            rejections=rejections,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(rejections[0]["run_id"], "run-1")
        self.assertIn("restates", rejections[0]["reason"])

    def test_prompts_include_contrastive_examples_and_rejection(self):
        discovery = _discovery_prompt({}, [])
        consolidation = _consolidation_prompt([])

        self.assertIn("BUNDLED CLAIM -> SPLIT/REJECT", discovery)
        self.assertIn("REJECT: the observation does not justify", discovery)
        self.assertIn("FINAL ACCEPTANCE GATE", consolidation)
        self.assertIn("Existing pipeline constraints", consolidation)

    def test_revision_is_deferred_without_mutating_candidate(self):
        candidates = []
        _apply_discovery_decisions(
            candidates,
            [decision("ADD", instruction="Construct persistent objects outside frame updaters.")],
            run_id="run-1",
            topic="Topic 1",
        )
        original = candidates[0]["instruction"]
        _apply_discovery_decisions(
            candidates,
            [
                decision(
                    "PROPOSE_REVISION",
                    candidate_id="C001",
                    instruction="Precompute persistent objects outside every frame updater.",
                )
            ],
            run_id="run-2",
            topic="Topic 2",
        )
        self.assertEqual(candidates[0]["instruction"], original)
        self.assertEqual(len(candidates[0]["revision_proposals"]), 1)

    def test_consolidation_requires_exact_candidate_coverage(self):
        response = ConsolidationResponse(
            beliefs=[
                ConsolidatedBeliefPayload(
                    instruction="Construct persistent objects outside frame updaters.",
                    scope={
                        "roles": ["Coder"],
                        "stages": ["coding"],
                        "problem_description": "Per-frame construction slows rendering.",
                        "context_conditions": ["per_frame_construction"],
                    },
                    impact="high",
                    belief_type="confirmed",
                    timing="both",
                    source_candidate_ids=["C001"],
                    consolidation_reason="Both records describe the same construction mechanism.",
                )
            ],
            excluded_candidates=[],
        )
        with self.assertRaises(ValueError):
            _validate_consolidation(response, ["C001", "C002"])

    def test_missing_frozen_belief_is_reconciled_as_insufficient(self):
        response = RetrospectiveResponse(
            observations=[],
            applicable_belief_ids=[],
            not_applicable_belief_ids=["B001"],
            insufficient_belief_ids=[],
        )
        _validate_evaluation(response, ["B001", "B002"])
        self.assertEqual(response.not_applicable_belief_ids, ["B001"])
        self.assertEqual(response.insufficient_belief_ids, ["B002"])

    def test_unknown_belief_id_is_rejected(self):
        response = RetrospectiveResponse(
            observations=[],
            applicable_belief_ids=[],
            not_applicable_belief_ids=["B999"],
            insufficient_belief_ids=[],
        )
        with self.assertRaises(ValueError):
            _validate_evaluation(response, ["B001"])

    def test_one_belief_can_have_positive_and_negative_effectiveness_evidence(self):
        common = {
            "compliance": "followed",
            "outcome": "mixed",
            "strategy_application": "full",
            "attribution_strength": "strong",
            "evidence_reliability": "direct",
            "reason": "The transition isolates the strategy and the target outcome.",
            "agent": "Coder",
            "section_id": "section_1",
        }
        response = RetrospectiveResponse(
            observations=[
                EvidenceObservation(
                    belief_id="B001",
                    **common,
                    outcome_improvement="resolved",
                    evidence="The first change removed the directly observed error.",
                ),
                EvidenceObservation(
                    belief_id="B001",
                    **common,
                    outcome_improvement="worsened",
                    evidence="A later isolated use caused the target error to return.",
                ),
            ],
            applicable_belief_ids=["B001"],
            not_applicable_belief_ids=[],
            insufficient_belief_ids=["B002"],
        )
        _validate_evaluation(response, ["B001", "B002"])
        self.assertEqual(len(response.observations), 2)
        self.assertEqual(
            _derive_update_direction(response.observations[0]), ("support", True)
        )
        self.assertEqual(
            _derive_update_direction(response.observations[1]), ("contradict", True)
        )

    def test_noncompliance_with_predicted_problem_is_neutral(self):
        observation = EvidenceObservation(
            belief_id="B001",
            compliance="violated",
            outcome="negative",
            strategy_application="none",
            attribution_strength="strong",
            evidence_reliability="direct",
            outcome_improvement="worsened",
            evidence="The strategy was absent and the predicted overlap was observed.",
            reason="This establishes relevance but cannot establish the counterfactual fix.",
            agent="Coder",
            section_id="section_2",
        )
        self.assertEqual(_derive_update_direction(observation), ("neutral", False))
        payload = _observation_matrix_payload(observation)
        self.assertEqual(payload["update_direction"], "neutral")
        self.assertFalse(payload["update_eligible"])

    def test_applicable_without_observation_becomes_insufficient(self):
        response = RetrospectiveResponse(
            observations=[],
            applicable_belief_ids=["B001"],
            not_applicable_belief_ids=[],
            insufficient_belief_ids=[],
        )
        _validate_evaluation(response, ["B001"])
        self.assertEqual(response.applicable_belief_ids, [])
        self.assertEqual(response.insufficient_belief_ids, ["B001"])

    def test_consolidated_instruction_can_exceed_candidate_limit(self):
        payload = {
            "instruction": "A" * 600,
            "scope": {
                "roles": ["Coder"],
                "stages": ["coding"],
                "problem_description": "Several related mechanisms require conditional handling.",
                "context_conditions": ["compound_condition"],
            },
            "impact": "medium",
            "belief_type": "confirmed",
            "timing": "both",
            "source_candidate_ids": ["C001"],
            "consolidation_reason": "The source candidates share one conditional mechanism.",
        }
        parsed = ConsolidatedBeliefPayload.model_validate(payload)
        self.assertEqual(len(parsed.instruction), 600)

    def test_consolidated_belief_cannot_bundle_more_than_four_candidates(self):
        payload = {
            "instruction": "Use one coherent strategy for one reusable mechanism.",
            "scope": {
                "roles": ["Coder"],
                "stages": ["coding"],
                "problem_description": "One mechanism needs one testable strategy.",
                "context_conditions": ["atomic_belief"],
            },
            "impact": "medium",
            "belief_type": "confirmed",
            "timing": "both",
            "source_candidate_ids": ["C001", "C002", "C003", "C004", "C005"],
            "consolidation_reason": "These candidates were incorrectly bundled.",
        }
        with self.assertRaises(ValidationError):
            ConsolidatedBeliefPayload.model_validate(payload)

    def test_consolidation_can_reject_more_than_quarter_of_candidates(self):
        candidate_ids = [f"C{index:03d}" for index in range(1, 9)]
        response = ConsolidationResponse(
            beliefs=[],
            excluded_candidates=[
                {
                    "candidate_id": candidate_id,
                    "reason": "This candidate only repeats an existing prompt constraint.",
                }
                for candidate_id in candidate_ids
            ],
        )
        _validate_consolidation(response, candidate_ids)

    def test_unsupported_confirmed_belief_is_reported_without_blocking(self):
        response = ConsolidationResponse(
            beliefs=[
                ConsolidatedBeliefPayload(
                    instruction="Use persistent geometry instead of rebuilding it every frame.",
                    scope={
                        "roles": ["Coder"],
                        "stages": ["coding"],
                        "problem_description": "Per-frame reconstruction increases render cost.",
                        "context_conditions": ["per_frame_reconstruction"],
                    },
                    impact="high",
                    belief_type="confirmed",
                    timing="preventative",
                    source_candidate_ids=["C001"],
                    consolidation_reason="The candidate describes one reusable mechanism.",
                )
            ],
            excluded_candidates=[],
        )
        candidates = [
            {
                "candidate_id": "C001",
                "origins": [
                    {
                        "strategy_application": "unclear",
                        "attribution_strength": "weak",
                        "evidence_reliability": "inferred",
                        "outcome_improvement": "unclear",
                    }
                ],
            }
        ]
        _validate_consolidation(response, ["C001"], candidates)
        concerns = _consolidation_quality_concerns(response, candidates)
        self.assertEqual(
            [item["category"] for item in concerns],
            ["unsupported_confirmed_classification"],
        )

    def test_existing_constraint_restatement_is_reported_without_blocking(self):
        response = ConsolidationResponse(
            beliefs=[
                ConsolidatedBeliefPayload(
                    instruction="Keep lecture lines under 10 words in every section.",
                    scope={
                        "roles": ["ScriptWriter"],
                        "stages": ["planning"],
                        "problem_description": "Lecture lines may be too long.",
                        "context_conditions": ["lecture_line_generation"],
                    },
                    impact="low",
                    belief_type="quality",
                    timing="preventative",
                    source_candidate_ids=["C001"],
                    consolidation_reason="This repeats a useful formatting requirement.",
                )
            ],
            excluded_candidates=[],
        )
        candidates = [{"candidate_id": "C001", "origins": []}]
        _validate_consolidation(response, ["C001"], candidates)
        concerns = _consolidation_quality_concerns(response, candidates)
        self.assertEqual(
            [item["category"] for item in concerns],
            ["existing_constraint_restatement"],
        )

    def test_context_condition_punctuation_is_normalized_without_retry(self):
        response = ConsolidationResponse(
            beliefs=[
                ConsolidatedBeliefPayload(
                    instruction="Increase connector clearance when a scaled object overlaps its line tip.",
                    scope={
                        "roles": ["Coder"],
                        "stages": ["coding"],
                        "problem_description": "Scaled objects can overlap connector endpoints.",
                        "context_conditions": ["mobject_scale_greater_than_1.0"],
                    },
                    impact="medium",
                    belief_type="quality",
                    timing="preventative",
                    source_candidate_ids=["C001"],
                    consolidation_reason="The evidence identifies one visual-clearance mechanism.",
                )
            ],
            excluded_candidates=[],
        )

        _validate_consolidation(
            response,
            ["C001"],
            [{"candidate_id": "C001", "origins": []}],
        )

        self.assertEqual(
            response.beliefs[0].scope.context_conditions,
            ["mobject_scale_greater_than_1_0"],
        )


if __name__ == "__main__":
    unittest.main()
