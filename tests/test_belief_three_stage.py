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

    def test_evaluation_requires_every_frozen_belief(self):
        response = RetrospectiveResponse(
            observations=[],
            applicable_belief_ids=[],
            not_applicable_belief_ids=["B001"],
            insufficient_belief_ids=[],
        )
        with self.assertRaises(ValueError):
            _validate_evaluation(response, ["B001", "B002"])

    def test_one_belief_can_have_support_and_contradiction_in_one_topic(self):
        common = {
            "compliance": "followed",
            "outcome": "mixed",
            "strategy_application": "full",
            "attribution_strength": "strong",
            "evidence_reliability": "direct",
            "evidence_confidence": 0.9,
            "reason": "The transition isolates the strategy and the target outcome.",
            "agent": "Coder",
            "section_id": "section_1",
        }
        response = RetrospectiveResponse(
            observations=[
                EvidenceObservation(
                    belief_id="B001",
                    **common,
                    direction="support",
                    outcome_improvement="resolved",
                    evidence="The first change removed the directly observed error.",
                ),
                EvidenceObservation(
                    belief_id="B001",
                    **common,
                    direction="contradict",
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


if __name__ == "__main__":
    unittest.main()
