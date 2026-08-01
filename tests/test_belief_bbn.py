import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from belief_bbn import (  # noqa: E402
    ApplicabilityEvidence,
    BBNParameters,
    BeliefEmbeddingIndex,
    BeliefSelector,
    BeliefSituation,
    TransitionEvidence,
    fit_bbn_parameters,
    normalize_stage,
    update_beta_posterior,
)


class BeliefBBNTests(unittest.TestCase):
    def test_matching_evidence_increases_applicability(self):
        model = BBNParameters()
        poor = model.probability(ApplicabilityEvidence(0, 0, 0, 0))
        strong = model.probability(ApplicabilityEvidence(1, 1, 1, 1))
        self.assertLess(poor, strong)
        self.assertGreater(strong, 0.5)

    def test_fractional_beta_update(self):
        update = update_beta_posterior(
            2.0,
            2.0,
            TransitionEvidence(
                p_applicable=0.9,
                p_strategy_applied=0.8,
                p_attributable=0.75,
                p_reliable=1.0,
                improvement=1.0,
            ),
        )
        self.assertAlmostEqual(update["evidence_weight"], 0.54)
        self.assertAlmostEqual(update["alpha"], 2.54)
        self.assertAlmostEqual(update["beta"], 2.0)

    def test_topic_weight_cap(self):
        update = update_beta_posterior(
            2.0,
            2.0,
            TransitionEvidence(1.0, 1.0, 1.0, 1.0, 1.0),
            remaining_topic_weight=0.2,
        )
        self.assertAlmostEqual(update["evidence_weight"], 0.2)
        self.assertAlmostEqual(update["alpha"], 2.2)

    def test_selector_filters_ranks_thresholds_and_logs(self):
        beliefs = [
            {
                "belief_id": "B001",
                "instruction": "Precompute dynamic geometry after render timeouts.",
                "scope": {
                    "roles": ["Coder"],
                    "stages": ["debugging"],
                    "problem_description": "dynamic geometry render timeout",
                    "context_conditions": ["render_timeout"],
                },
                "alpha": 8,
                "beta": 2,
                "status": "active",
                "timing": "reactive",
            },
            {
                "belief_id": "B002",
                "instruction": "Shorten narration.",
                "scope": {
                    "roles": ["ScriptWriter"],
                    "stages": ["revision"],
                    "problem_description": "long narration",
                },
                "alpha": 9,
                "beta": 1,
                "status": "active",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "selections.jsonl"
            selector = BeliefSelector(beliefs, log_path=log_path)
            selected = selector.select(
                BeliefSituation(
                    topic="Brachistochrone",
                    agent_role="Coder",
                    pipeline_stage="debugging",
                    problem_text="dynamic geometry render timeout",
                    context_tags=["render_timeout"],
                    timing="reactive",
                ),
                threshold=0.5,
                top_k=2,
            )
            self.assertEqual([item["belief_id"] for item in selected], ["B001"])
            row = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(row["selected_belief_ids"], ["B001"])
            candidate = row["candidates"][0]
            self.assertEqual(candidate["role_match"], 1.0)
            self.assertEqual(candidate["stage_match"], 1.0)
            self.assertEqual(candidate["context_match"], 1.0)
            mismatched = next(
                item for item in row["candidates"] if item["belief_id"] == "B002"
            )
            self.assertEqual(mismatched["role_match"], 0.0)
            self.assertEqual(mismatched["stage_match"], 0.0)
            self.assertFalse(mismatched["selected"])

    def test_runtime_stage_aliases_match_frozen_stage_vocabulary(self):
        self.assertEqual(normalize_stage("debugging"), "coding")
        self.assertEqual(normalize_stage("writing"), "planning")
        self.assertEqual(normalize_stage("revision"), "planning")

    def test_parameter_fit_learns_from_labelled_cases(self):
        cases = [
            {
                "role_match": 1,
                "stage_match": 1,
                "problem_match": 1,
                "context_match": 1,
                "applicable": 1,
            }
            for _ in range(10)
        ] + [
            {
                "role_match": 0,
                "stage_match": 0,
                "problem_match": 0,
                "context_match": 0,
                "applicable": 0,
            }
            for _ in range(10)
        ]
        initial = BBNParameters()
        fitted = fit_bbn_parameters(cases, initial=initial, epochs=100)
        self.assertEqual(fitted.version, initial.version + 1)
        self.assertGreater(
            fitted.probability(ApplicabilityEvidence(1, 1, 1, 1)),
            fitted.probability(ApplicabilityEvidence(0, 0, 0, 0)),
        )

    def test_embedding_cache_detects_changed_belief_text(self):
        belief = {
            "belief_id": "B001",
            "instruction": "Fallback instruction",
            "scope": {"problem_description": "render timeout"},
        }
        import hashlib

        index = BeliefEmbeddingIndex(
            model=None,
            model_name="test-model",
            belief_ids=["B001"],
            embeddings=[],
            text_hashes={
                "B001": hashlib.sha256(
                    b"render timeout"
                ).hexdigest()
            },
        )
        self.assertEqual(index.validate_beliefs([belief]), [])
        belief["scope"]["problem_description"] = "different problem"
        self.assertEqual(index.validate_beliefs([belief]), ["B001"])

    def test_selector_uses_stored_similarity(self):
        class FakeIndex:
            def similarities(self, query):
                return {"B001": 1.0}

        belief = {
            "belief_id": "B001",
            "instruction": "Stored vector belief.",
            "scope": {"roles": ["Coder"], "problem_description": "unrelated words"},
            "alpha": 9,
            "beta": 1,
            "status": "active",
        }
        selector = BeliefSelector(
            [belief],
            embedding_index=FakeIndex(),
            similarity_fn=lambda first, second: self.fail(
                "lexical fallback should not run for a stored belief"
            ),
        )
        selected = selector.select(
            BeliefSituation(
                topic="T",
                agent_role="Coder",
                pipeline_stage="debugging",
                problem_text="query",
            ),
            threshold=0.5,
        )
        self.assertEqual([item["belief_id"] for item in selected], ["B001"])

    def test_selector_uses_semantic_context_as_bbn_evidence(self):
        class FakeIndex:
            def similarities_with_context(self, query, context_texts):
                return {"B001": 0.9}, {"B001": 0.8}

        belief = {
            "belief_id": "B001",
            "instruction": "Use area placement for wide labels.",
            "scope": {
                "roles": ["Coder"],
                "stages": ["coding"],
                "problem_description": "wide label alignment",
                "context_conditions": ["grid layout with wide labels"],
            },
            "alpha": 9,
            "beta": 1,
            "status": "active",
            "timing": "preventative",
        }
        selector = BeliefSelector([belief], embedding_index=FakeIndex())
        selected = selector.select(
            BeliefSituation(
                topic="T",
                agent_role="Coder",
                pipeline_stage="coding",
                problem_text="A long title must be centered across a six-column grid.",
                timing="preventative",
            ),
            threshold=0.5,
        )
        self.assertEqual([item["belief_id"] for item in selected], ["B001"])
        self.assertEqual(selected[0]["context_match_semantic"], 0.8)

    def test_probationary_belief_can_collect_selection_evidence(self):
        belief = {
            "belief_id": "B058",
            "instruction": "Qualify rate functions.",
            "scope": {
                "roles": ["Coder"],
                "stages": ["coding"],
                "problem_description": "Render failure due to NameError.",
            },
            "alpha": 3,
            "beta": 1,
            "status": "probation",
            "timing": "reactive",
        }
        selector = BeliefSelector(
            [belief],
            similarity_fn=lambda first, second: 1.0,
        )
        selected = selector.select(
            BeliefSituation(
                topic="T",
                agent_role="Coder",
                pipeline_stage="coding",
                problem_text="NameError: smooth is not defined",
                context_tags=["missing_import", "render_error"],
                timing="reactive",
            ),
            threshold=0.0,
        )
        self.assertEqual([item["belief_id"] for item in selected], ["B058"])
        self.assertEqual(selected[0]["status"], "probation")

        belief["timing"] = "preventative"
        preventative = selector.select(
            BeliefSituation(
                topic="T",
                agent_role="Coder",
                pipeline_stage="coding",
                problem_text="NameError while preparing a section",
                context_tags=["missing_import"],
                timing="preventative",
            ),
            threshold=0.0,
        )
        self.assertEqual(
            [item["belief_id"] for item in preventative],
            ["B058"],
        )


if __name__ == "__main__":
    unittest.main()
