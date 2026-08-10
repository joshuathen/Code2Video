#!/usr/bin/env python3
"""Order-aware three-stage belief generation for completed MAS pipelines.

Stages:
1. Sequential candidate discovery with deferred revision proposals.
2. One global consolidation pass that freezes stable belief definitions.
3. Retrospective evaluation of the same frozen bank against every topic,
   followed by deterministic Bayesian updating.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from google.genai import Client
from pydantic import BaseModel, Field, ValidationError, model_validator

from belief_bbn import BBNParameters, BeliefEmbeddingIndex
from mas_interactions import create_interaction, response_usage_dict
from mas_belief_reflection import (
    ACTION_CONTRADICT,
    ACTION_OBSERVE,
    ACTION_SUPPORT,
    DEFAULT_REFLECTION_MODEL,
    BeliefAssessment,
    BeliefRecord,
    BeliefScope,
    ReflectionScopePayload,
    _build_reflection_context,
    _evidence_categories_to_values,
    _extract_structured_json,
    _find_final_state_json,
    _iter_run_dirs_from_pipeline,
    _safe_mean,
    _state_for_run,
    _write_json,
    apply_assessments,
    cfg,
    save_evidence,
    summarise_run,
)


EXISTING_PIPELINE_CONSTRAINTS = [
    "Lecture lines contain no more than ten words.",
    "Use the established left-text/right-animation layout and supplied TeachingScene base class.",
    "Use hexadecimal colours and the existing 6x6 placement grid.",
    "Render each section within the configured 180-second timeout.",
    "Preserve and use storyboard asset references supplied by the asset-enhancement stage.",
]

CONTRASTIVE_BELIEF_EXAMPLES = r"""
CONTRASTIVE EXAMPLES (learn the reasoning rule, not the domain wording):

1. BUNDLED CLAIM -> SPLIT/REJECT
BAD: "Prevent timeouts and LaTeX failures by simplifying formulas, changing
configuration, and removing updaters."
WHY BAD: Two failure mechanisms and several independently falsifiable interventions.
BETTER A: "Avoid rebuilding large mobject collections inside frame updaters; update
persistent geometry instead." Use only when before/after evidence isolates that mechanism.
BETTER B: "After a confirmed LaTeX compilation failure, simplify the failing MathTex
expression while preserving complete command boundaries."

2. CO-OCCURRENCE -> HYPOTHESIS, NOT CONFIRMED
OBSERVATION: A timed-out scene contained an updater.
BAD: "Updaters cause render timeouts."
BETTER: Classify as hypothesis unless a comparable successful transition removes or
simplifies per-frame reconstruction without material confounding changes.

3. EXISTING PROMPT RULE -> REJECT
BAD: "Keep lecture lines under ten words."
WHY BAD: It merely repeats an existing constraint and is not a learned finding.
ACCEPTABLE ONLY IF EVIDENCE ADDS A BOUNDARY: "The ten-word limit did not prevent
overflow when five lines were displayed; also constrain the rendered lecture block height."

4. TOPIC-SPECIFIC FACT -> REUSABLE MECHANISM
BAD: "In the Moser section, move the purple point away from C4."
BETTER: "When a geometric construction assumes general position, avoid coincident
intersections by selecting non-degenerate parameters and verify the resulting count."

5. IMPACT/STAGE CALIBRATION
BAD: Mark "use NumPy for matrices" as critical and leave its workflow scope vague.
BETTER: A recoverable incorrect displayed value is medium impact. List every workflow
stage where the exact strategy applies, such as generation, fix, or refine.

6. STRONG BEFORE/AFTER EVIDENCE -> SUPPORTED/CONFIRMED
Attempt A rebuilds hundreds of objects per frame and times out. Attempt B changes only
that construction to persistent geometry and renders successfully. Direct matching code
hashes and no material confounders can support a confirmed, narrowly worded belief.
""".strip()


class EvidenceClassification(BaseModel):
    compliance: Literal["followed", "violated", "mixed", "unclear", "observed_pattern"]
    outcome: Literal["positive", "negative", "mixed", "unclear"]
    strategy_application: Literal["full", "partial", "unclear", "none"]
    attribution_strength: Literal["strong", "moderate", "weak", "unclear", "none"]
    evidence_reliability: Literal[
        "direct", "corroborated", "indirect", "inferred", "unverifiable"
    ]
    outcome_improvement: Literal[
        "resolved", "improved", "unchanged", "worsened", "unclear"
    ]
    evidence: str = Field(min_length=10, max_length=500)
    reason: str = Field(min_length=10, max_length=500)
    agent: Optional[
        Literal["Orchestrator", "Coder", "AnimationPlanner", "ScriptWriter"]
    ] = None
    section_id: Optional[str] = None


class CandidateDecision(EvidenceClassification):
    action: Literal["ADD", "MATCH", "PROPOSE_REVISION", "REJECT"]
    candidate_id: Optional[str] = None
    instruction: Optional[str] = Field(default=None, max_length=500)
    scope: Optional[ReflectionScopePayload] = None
    impact: Literal["low", "medium", "high", "critical"]
    belief_type: Literal["confirmed", "precaution", "hypothesis", "quality"]

    @model_validator(mode="after")
    def validate_target(self) -> "CandidateDecision":
        if self.action == "ADD":
            if self.candidate_id is not None or self.instruction is None or self.scope is None:
                raise ValueError("ADD requires instruction and scope and candidate_id=null")
        elif self.action in {"MATCH", "PROPOSE_REVISION"}:
            if not self.candidate_id:
                raise ValueError(f"{self.action} requires candidate_id")
        elif self.candidate_id is not None:
            raise ValueError("REJECT records a rejected observation and requires candidate_id=null")
        if self.action == "PROPOSE_REVISION" and (
            self.instruction is None or self.scope is None
        ):
            raise ValueError("PROPOSE_REVISION requires replacement instruction and scope")
        return self


class DiscoveryResponse(BaseModel):
    decisions: List[CandidateDecision]


class ConsolidatedBeliefPayload(BaseModel):
    # Consolidation sometimes needs a compound conditional instruction to
    # replace several over-specific candidates. Keep a bound, but allow more
    # room than the per-topic candidate schema.
    instruction: str = Field(min_length=10, max_length=750)
    scope: ReflectionScopePayload
    impact: Literal["low", "medium", "high", "critical"]
    belief_type: Literal["confirmed", "precaution", "hypothesis", "quality"]
    # Candidate discovery already uses MATCH for duplicate evidence. A final
    # belief should therefore need only a small number of genuinely overlapping
    # candidate definitions. This bound prevents broad thematic bundles.
    source_candidate_ids: List[str] = Field(min_length=1, max_length=4)
    consolidation_reason: str = Field(min_length=10, max_length=500)


class ExcludedCandidatePayload(BaseModel):
    candidate_id: str
    reason: str = Field(min_length=10, max_length=500)


class ConsolidationResponse(BaseModel):
    beliefs: List[ConsolidatedBeliefPayload]
    excluded_candidates: List[ExcludedCandidatePayload] = Field(default_factory=list)


class EvidenceObservation(EvidenceClassification):
    belief_id: str


class RetrospectiveResponse(BaseModel):
    observations: List[EvidenceObservation]
    applicable_belief_ids: List[str]
    not_applicable_belief_ids: List[str]
    insufficient_belief_ids: List[str]


def _atomic_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _structured_call(
    client: Client,
    *,
    prompt: str,
    schema: Type[BaseModel],
    label: str,
    raw_output_dir: Path,
    max_output_tokens: Optional[int],
    semantic_validator: Optional[Callable[[BaseModel], None]] = None,
    max_attempts: int = 2,
) -> tuple[BaseModel, Dict[str, int], int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if attempt > 1:
            error_feedback = (
                str(last_error)[:2000] if last_error is not None else "unknown validation error"
            )
            attempt_prompt += (
                "\n\nRETRY: Return one complete JSON object conforming exactly to the "
                "provided schema. Use only the permitted categorical values and no commentary."
                "\nThe previous response failed local validation. Here is the precise "
                "validation error:\n"
                + error_feedback
                + "\nCorrect every issue named in that error while preserving all otherwise "
                "valid distinct content. If the error reports an invalid evidence partition, "
                "ensure every expected evidence ID appears exactly once across the partition: "
                "add all missing IDs, remove duplicate occurrences, and omit unknown IDs. "
                "Verify the completed partition against the full input evidence list before "
                "returning the corrected JSON."
            )
        response = create_interaction(
            client,
            model=DEFAULT_REFLECTION_MODEL,
            input_value=attempt_prompt,
            response_schema=schema,
            max_output_tokens=max_output_tokens,
        )
        call_usage = response_usage_dict(response)
        for key in usage:
            usage[key] += int(call_usage.get(key, 0) or 0)
        try:
            parsed = schema.model_validate_json(
                _extract_structured_json(response.output_text)
            )
            if semantic_validator is not None:
                semantic_validator(parsed)
            return parsed, usage, attempt
        except (ValidationError, ValueError) as exc:
            last_error = exc
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            (raw_output_dir / f"{label}_invalid_response_{attempt}.txt").write_text(
                response.output_text,
                encoding="utf-8",
            )
    raise ValueError(
        f"Gemini returned invalid {label} JSON {max_attempts} times"
    ) from last_error


def _compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "instruction": candidate["instruction"],
        "scope": candidate["scope"],
        "impact": candidate["impact"],
        "belief_type": candidate["belief_type"],
        "origin_count": len(candidate.get("origins", [])),
        "match_count": len(candidate.get("matches", [])),
        "revision_proposal_count": len(candidate.get("revision_proposals", [])),
    }


def _discovery_prompt(
    run_context: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> str:
    payload = {
        "run_context": run_context,
        "candidate_bank": [_compact_candidate(item) for item in candidates],
    }
    return f"""
You are performing Stage 1: candidate belief discovery for a completed MAS topic.

Inspect every material event, but retain only distinct, reusable, evidence-grounded
learnings that could improve a future MAS run. It is valid for a topic to yield no ADD
decisions. The candidate bank exists only to reduce duplicates.

For each discovered mechanism choose exactly one action:
- ADD: no candidate covers the same mechanism. Provide a complete instruction and scope.
- MATCH: an existing candidate already covers it. Do not rewrite that candidate.
- PROPOSE_REVISION: evidence suggests an existing candidate should be narrowed,
  broadened, or corrected. Provide the complete proposed replacement and scope.
- REJECT: the observation does not justify a reusable belief. Set candidate_id,
  instruction, and scope to null and explain the rejection in reason.

Revision proposals are recorded but are NOT applied during discovery. Do not perform
Bayesian updating, assign posterior confidence, or evaluate unrelated candidates.
Do not let an existing candidate prevent discovery of a genuinely different mechanism.
Use role names, not agent instances. Generalize only as far as direct evidence permits.
Classify evidence using the permitted categorical options; never invent probabilities.
Use critical only for unrecoverable termination, corruption, data loss, or uncontrolled
external effects; ordinary render timeouts are high.

EVIDENCE AND FORMULATION RULES:
- One ADD or PROPOSE_REVISION must contain one target mechanism, one coherent
  intervention, and one primary outcome. If clauses could be independently supported
  or contradicted, return separate decisions.
- Presence of a construct near a failure is co-occurrence, not causation. Use hypothesis
  unless an observed intervention and improved outcome support stronger wording.
- A later successful render does not establish which of several simultaneous changes
  caused success. Reflect confounding in attribution_strength and belief_type.
- Reject generic good practice, administrative advice, topic facts, and restatements of
  existing pipeline constraints unless evidence supports a measurable refinement,
  contradiction, or boundary condition.
- problem_description states WHAT reusable problem the belief addresses.
  context_conditions state WHEN it activates and must use concise observable
  snake_case conditions derivable from logs, code, role, or stage.
- confirmed requires direct/corroborated evidence of strategy application followed by
  resolved/improved outcome with strong/moderate attribution. Otherwise use hypothesis,
  precaution, or quality as appropriate.

Existing pipeline constraints (not new learnings):
{json.dumps(EXISTING_PIPELINE_CONSTRAINTS, ensure_ascii=False)}

{CONTRASTIVE_BELIEF_EXAMPLES}

Return JSON conforming to the supplied schema. For MATCH, instruction and scope may be
null. For ADD and REJECT set candidate_id=null. Evidence must identify the concrete
observation; REJECT reason must state why it cannot support a belief.

Input:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _apply_discovery_decisions(
    candidates: List[Dict[str, Any]],
    decisions: List[CandidateDecision],
    *,
    run_id: str,
    topic: str,
    rejections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    by_id = {item["candidate_id"]: item for item in candidates}
    counts = {"added": 0, "matched": 0, "revision_proposed": 0, "rejected": 0}
    for decision in decisions:
        evidence = {
            "run_id": run_id,
            "topic": topic,
            **decision.model_dump(
                exclude={"action", "candidate_id", "instruction", "scope"}
            ),
        }
        if decision.action == "REJECT":
            if rejections is not None:
                rejections.append(evidence)
            counts["rejected"] += 1
            continue
        if decision.action == "ADD":
            candidate_id = f"C{len(candidates) + 1:03d}"
            candidate = {
                "candidate_id": candidate_id,
                "instruction": decision.instruction,
                "scope": decision.scope.model_dump() if decision.scope else {},
                "impact": decision.impact,
                "belief_type": decision.belief_type,
                "origins": [evidence],
                "matches": [],
                "revision_proposals": [],
            }
            candidates.append(candidate)
            by_id[candidate_id] = candidate
            counts["added"] += 1
            continue
        candidate = by_id.get(str(decision.candidate_id))
        if candidate is None:
            raise ValueError(f"Unknown candidate_id: {decision.candidate_id}")
        if decision.action == "MATCH":
            candidate["matches"].append(evidence)
            counts["matched"] += 1
        else:
            candidate["revision_proposals"].append(
                {
                    **evidence,
                    "proposed_instruction": decision.instruction,
                    "proposed_scope": decision.scope.model_dump() if decision.scope else {},
                }
            )
            counts["revision_proposed"] += 1
    return counts


def _consolidation_prompt(candidates: List[Dict[str, Any]]) -> str:
    return f"""
You are performing Stage 2: global consolidation of a candidate belief bank.

You can see every candidate, origin, match, and deferred revision proposal together.
Produce stable, reusable, ATOMIC beliefs:
- merge candidates only when they describe the same underlying causal mechanism,
  the same problem, and a compatible mitigation;
- resolve revision proposals using the complete evidence, avoiding topic-order drift;
- do not treat origin/match count as confidence or prefer early candidates merely
  because they had more opportunities to be matched; one direct late candidate may
  remain a frozen probationary belief for retrospective evaluation;
- replace fixed topic-specific coordinates or thresholds with a conditional principle
  only when the combined evidence demonstrates that principle;
- preserve distinct causes, roles, workflow stages, or mitigations as separate beliefs;
- exclude candidates that are demonstrably incorrect, redundant without contributing
  distinct evidence, lack any concrete origin evidence, merely restate an existing
  constraint, make an unsupported causal/API claim, or are not reusable, and explain why;
- use critical only for unrecoverable termination, corruption, data loss, or uncontrolled
  external effects; render timeouts are high.

ATOMICITY REQUIREMENTS:
- One final belief must express one coherent strategy whose effectiveness can be
  represented by one alpha/beta posterior.
- A belief must be split whenever evidence could support one clause while contradicting
  another clause.
- Do not merge candidates merely because they concern the same role, stage, visual
  theme, grid system, API library, or broad objective.
- Do not create grab-bag beliefs listing unrelated Manim APIs, unrelated layout rules,
  unrelated numerical scale factors, or multiple pedagogical techniques.
- A shared word such as "layout", "quality", "rendering", "MathTex", "SVG", or
  "animation" is not by itself a shared mechanism.
- Keep generation guidance separate from fix-stage recovery when the actions or
  triggering conditions differ; use multiple scope.stages only when the exact same
  instruction genuinely applies in each listed stage.
- Keep runtime correctness, render performance, visual composition, mathematical
  accuracy, and pedagogy as separate mechanisms.
- Prefer retaining two narrowly testable beliefs over producing one broad instruction.
- Each final belief may contain at most four source candidate IDs. This is an upper
  bound, not a target; most beliefs should contain one or two.
- Do not use exclusions merely to avoid producing appropriately separate beliefs, but
  treat REJECT as a valid and desirable outcome when acceptance criteria are not met.
- "Niche", "low frequency", "single occurrence", or "low impact" alone are not valid
  exclusion reasons. A single occurrence may remain a hypothesis/precaution when it is
  reusable and evidence-grounded; reject it when it is only generic advice or a topic fact.
- If a candidate is a narrower example of the exact same mechanism and strategy, include
  its ID in that belief's source_candidate_ids rather than excluding it as redundant.
- Exclusion is reserved for a demonstrably incorrect claim, absence of concrete origin
  evidence, a purely topic-specific fact with no reusable strategy, or a non-belief
  administrative/meta-process statement.

Before returning each belief, perform these tests:
1. Do these candidates have the same problem?
2. Do these candidates have the same causal mechanism?
3. Do these candidates have the same strategy?
4. Do these candidates have one interpretable effectiveness variable?
Only merge when all four answers are YES. Otherwise return separate beliefs.

FINAL ACCEPTANCE GATE FOR EACH BELIEF:
1. One independently falsifiable mechanism and one coherent intervention.
2. Traceable concrete origin evidence; causal wording no stronger than attribution.
3. Reusable problem description, not a topic/section-specific instruction.
4. Observable snake_case context conditions, not subjective states such as "complex".
5. Not merely an existing prompt constraint.
6. Exact Manim API claims are supported by direct evidence; otherwise use cautious
   precaution/hypothesis wording rather than asserting compatibility.
7. confirmed is permitted only with direct/corroborated full/partial strategy evidence,
   resolved/improved outcome, and strong/moderate attribution.
8. critical is permitted only for termination, corruption, data loss, or uncontrolled
   external effect documented in evidence.

Existing pipeline constraints (reject mere restatements):
{json.dumps(EXISTING_PIPELINE_CONSTRAINTS, ensure_ascii=False)}

{CONTRASTIVE_BELIEF_EXAMPLES}

Use these questions as internal merge guidance. Do not add merge-check fields to the
output object. Explain the shared mechanism concisely in consolidation_reason.
When separate atomic candidate IDs exist, preserve them as separate beliefs whenever
equivalence is uncertain. If one source candidate is itself irreducibly bundled and its
clauses cannot be traced to separate origin evidence, exclude it as non-atomic rather
than inventing unsupported split claims.

Examples of merges that are NOT allowed:
- combining invalid Manim class names, colour constants, vector normalization, and SVG
  positioning into one "API correctness" belief;
- combining title clearance, label adjacency, scaling, coordinate-system padding, and
  screen balance into one "layout" belief;
- combining metaphors, worked examples, variable colours, prerequisite coverage, and
  answer verification into one "pedagogy" belief.

Every candidate ID must appear exactly once, either in one belief's
source_candidate_ids or in excluded_candidates. Do not calculate confidence or perform
Bayesian updates. The resulting definitions will be frozen before evaluation.
Before submission, construct a checklist from the supplied candidate IDs and verify that
none is missing, duplicated, or invented.

Candidate bank:
{json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _validate_consolidation(
    response: ConsolidationResponse,
    candidate_ids: List[str],
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> None:
    assigned = [
        candidate_id
        for belief in response.beliefs
        for candidate_id in belief.source_candidate_ids
    ] + [item.candidate_id for item in response.excluded_candidates]
    if sorted(assigned) != sorted(candidate_ids):
        missing = sorted(set(candidate_ids) - set(assigned))
        duplicated = sorted(
            candidate_id for candidate_id in set(assigned) if assigned.count(candidate_id) > 1
        )
        unknown = sorted(set(assigned) - set(candidate_ids))
        raise ValueError(
            f"Invalid consolidation coverage: missing={missing}, "
            f"duplicated={duplicated}, unknown={unknown}"
        )
    for belief in response.beliefs:
        # Context tags are identifiers, so punctuation differences are
        # cosmetic rather than a reason to discard an otherwise valid global
        # consolidation. Normalize deterministically (for example, 1.0 ->
        # 1_0) instead of spending API retries asking the model to reformat the
        # same semantic output.
        conditions = []
        for raw_condition in belief.scope.context_conditions:
            normalized_condition = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(raw_condition).strip().lower(),
            ).strip("_")
            if normalized_condition and normalized_condition not in conditions:
                conditions.append(normalized_condition)
        belief.scope.context_conditions = conditions


def _consolidation_quality_concerns(
    response: ConsolidationResponse,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Report academic-quality concerns without blocking the pipeline."""
    candidate_by_id = {
        str(item.get("candidate_id")): item for item in candidates
    }
    existing_constraint_patterns = [
        re.compile(r"lecture lines?.*(?:ten|10) words?", re.I),
        re.compile(r"two[- ]column.*(?:left|lecture)", re.I),
        re.compile(r"(?:hexadecimal|hex).*(?:colou?r)", re.I),
        re.compile(r"TeachingScene", re.I),
        re.compile(r"6x6.*grid", re.I),
    ]
    refinement_markers = re.compile(
        r"refin|boundary|contradict|insufficient|did not|measured|evidence shows",
        re.I,
    )
    concerns: List[Dict[str, Any]] = []

    for belief_index, belief in enumerate(response.beliefs, start=1):
        belief_ref = f"provisional_B{belief_index:03d}"
        if any(pattern.search(belief.instruction) for pattern in existing_constraint_patterns):
            if not refinement_markers.search(belief.consolidation_reason):
                concerns.append(
                    {
                        "belief_ref": belief_ref,
                        "category": "existing_constraint_restatement",
                        "instruction": belief.instruction,
                    }
                )

        source_candidates = [
            candidate_by_id[candidate_id]
            for candidate_id in belief.source_candidate_ids
            if candidate_id in candidate_by_id
        ]
        origins = [
            origin
            for candidate in source_candidates
            for origin in candidate.get("origins", [])
        ]
        if belief.belief_type == "confirmed":
            has_confirming_transition = any(
                origin.get("strategy_application") in {"full", "partial"}
                and origin.get("attribution_strength") in {"strong", "moderate"}
                and origin.get("evidence_reliability") in {"direct", "corroborated"}
                and origin.get("outcome_improvement") in {"resolved", "improved"}
                for origin in origins
            )
            if not has_confirming_transition:
                concerns.append(
                    {
                        "belief_ref": belief_ref,
                        "category": "unsupported_confirmed_classification",
                        "instruction": belief.instruction,
                    }
                )

        if belief.impact == "critical":
            critical_evidence = " ".join(
                str(origin.get("evidence") or "") + " " + str(origin.get("reason") or "")
                for origin in origins
            )
            if not re.search(
                r"unrecoverable|terminat|corrupt|data loss|lost data|uncontrolled external",
                critical_evidence,
                re.I,
            ):
                concerns.append(
                    {
                        "belief_ref": belief_ref,
                        "category": "unsupported_critical_impact",
                        "instruction": belief.instruction,
                    }
                )
    return concerns


def _frozen_beliefs(response: ConsolidationResponse) -> List[Dict[str, Any]]:
    return [
        {
            "belief_id": f"B{index:03d}",
            **belief.model_dump(),
            "definition_frozen": True,
        }
        for index, belief in enumerate(response.beliefs, start=1)
    ]


def _evaluation_prompt(
    run_context: Dict[str, Any],
    frozen_beliefs: List[Dict[str, Any]],
) -> str:
    compact_beliefs = [
        {
            "belief_id": item["belief_id"],
            "instruction": item["instruction"],
            "scope": item["scope"],
        }
        for item in frozen_beliefs
    ]
    return f"""
You are performing Stage 3: retrospective evidence evaluation.

Evaluate EVERY frozen belief against this completed topic. Do not add, rewrite, merge,
or reinterpret belief definitions.

Partition every belief ID exactly once:
- applicable_belief_ids: the topic contains a concrete opportunity and one or more
  distinct observations that inform the exact frozen belief;
- not_applicable_belief_ids: the belief's problem/role/context did not occur;
- insufficient_belief_ids: potentially relevant, but evidence cannot establish the
  strategy, outcome, reliability, or attribution.

This corpus was generated without injecting the frozen beliefs. You are evaluating
observational evidence, not a controlled belief-provided versus belief-withheld
experiment. Be conservative and do not turn co-occurrence into causal effectiveness.

MULTIPLE OBSERVATIONS WITHIN ONE TOPIC:
- Return each belief ID only once across applicable_belief_ids,
  not_applicable_belief_ids, and insufficient_belief_ids.
- Put every distinct evidential event for applicable beliefs in the top-level
  observations list. The same belief_id may occur multiple times in observations.
  Separate observations may refer to different sections, turns, errors, fixes, or
  outcomes.
- NEVER put a belief in applicable_belief_ids unless you also return at least one
  observation for it. If its context seems relevant but you cannot describe a
  concrete observation, put it in insufficient_belief_ids.
- A single belief may have multiple distinct observations in the same topic. Preserve
  them separately; do not average them into one judgement.
- Do not repeat the same event using different wording. Each observation must identify
  a distinct transition or outcome.
- Do not output support, contradiction, update direction, update eligibility, or a
  numeric confidence. Local deterministic code derives Bayesian update eligibility
  from your four categorical judgements.
- The local updater caps the combined Bayesian evidence weight contributed by all
  update-eligible observations for one belief in one topic, so report distinct evidence
  honestly rather than trying to compress it.

SEPARATE THESE JUDGEMENTS:
1. Contextual applicability: did the matching problem and opportunity occur?
2. Strategy application: was the exact strategy in the frozen instruction followed?
3. Source reliability: how directly is the underlying event recorded?
4. Causal attribution: how well does the evidence isolate this strategy as the cause?
5. Outcome improvement: is there an observed before/action/after change in the target
   problem?

The presence of a technique in successful final code proves only that the technique was
used. It does NOT by itself prove that the technique caused success or improvement.
Likewise, a successful render, high AES/TQ score, positive reviewer comment, or later
pipeline completion must not be attributed to a particular strategy unless the evidence
isolates that relationship.

YOUR TASK IS EVIDENCE CLASSIFICATION, NOT THE FINAL EFFECTIVENESS DECISION:
For every concrete observation, classify these four inputs independently:
1. strategy_application;
2. outcome_improvement;
3. attribution_strength; and
4. evidence_reliability.
Do not try to make these categories collectively imply support. Report what the
artifacts establish for each category. Local code—not you—will apply a fixed,
auditable rule to decide whether the observation updates effectiveness.

PARTITION GUIDANCE:
- Use not_applicable only when the belief's scoped problem, role, stage, or opportunity
  genuinely did not occur.
- Use insufficient when the context may match but no concrete event establishes whether
  the exact strategy was applied, violated, or encountered.
- Use applicable when a concrete strategy application, non-application, or matching
  problem event is directly observable, even when effectiveness cannot be established.
- Do not classify a belief as applicable merely because its API, keyword, visual theme,
  or broad best practice appears somewhere in the topic.

STRATEGY APPLICATION:
- full: the exact strategy was clearly implemented;
- partial: only a material subset was implemented;
- none: a matching opportunity occurred but the strategy was not implemented;
- unclear: the artifacts cannot establish implementation.
Do not infer full application from a final successful state when the relevant transition
or action is absent.

EVIDENCE RELIABILITY DESCRIBES THE SOURCE, NOT CAUSALITY:
- direct: an exact traceback, issue, code diff, matching render result, or measured
  before/after outcome records the claimed event;
- corroborated: multiple independent artifacts record the same event;
- indirect: a related artifact supports but does not directly record the event;
- inferred: the conclusion primarily comes from semantic interpretation;
- unverifiable: the supplied artifacts cannot verify it.
A direct code occurrence may still have weak attribution and unclear improvement.
Set evidence_reliability="direct" only for what the source directly records. Direct
evidence that code exists is not direct evidence that the code caused an improvement.
Reviewer interpretation of visual quality is indirect or inferred unless accompanied by
a concrete before/after observation.

ATTRIBUTION STRENGTH:
- strong: the strategy was the only material targeted change, or evidence otherwise
  isolates it from simultaneous changes;
- moderate: the transition supports the strategy but a small number of plausible
  alternatives remain;
- weak: the strategy and outcome co-occur, but several changes or explanations remain;
- none: there is no evidence connecting the strategy to the outcome;
- unclear: the artifacts do not permit an attribution judgement.
Chronological order alone is not strong attribution. A later success after several edits
is weak or unclear unless the target change can be isolated.

OUTCOME IMPROVEMENT:
- resolved: a directly observed target problem existed before the strategy and was absent
  after it;
- improved: a directly observed target problem measurably decreased after the strategy;
- unchanged: a comparable before/after observation shows no material change;
- worsened: a comparable before/after observation shows deterioration;
- unclear: no comparable before/after observation exists.
Never use improved or resolved merely because the final render succeeded, the strategy
appears in code, or an aggregate evaluation score was positive.

CAUSAL EDGE CASES:
- strategy full/partial + isolated improvement should be classified literally as
  full/partial, improved/resolved, strong/moderate, and direct/corroborated;
- strategy full/partial + isolated unchanged/worsened outcome should be classified
  literally using those corresponding categories;
- strategy full/partial + no before/after outcome means improvement is unclear;
- strategy none + predicted problem occurred shows relevance/noncompliance, but does not
  directly demonstrate that providing the strategy would have fixed it;
- strategy none + successful outcome still has strategy_application="none"; do not
  reinterpret non-application as application;
- aggregate AES/TQ scores cannot establish attribution to one coding, layout, planning,
  or pedagogical strategy without specific supporting evidence.

AUDIT LOCATION:
- Populate agent whenever the responsible agent can be identified.
- Populate section_id whenever the evidence belongs to a particular section.
- Keep either null only when the event is genuinely topic-wide or the source does not
  permit identification.

Before returning an observation, ask:
1. What exact event was directly observed?
2. Where is the before/action/after transition?
3. Which simultaneous changes offer alternative explanations?
4. Does the evidence support the complete frozen instruction or only a clause/example?
5. Would outcome_improvement="unclear" or weaker attribution be more honest?

Prefer weak attribution or unclear improvement over an unsupported causal claim. Use
insufficient when even the concrete application or problem event cannot be established.
Do not aim for balanced categories; classify only what the supplied artifacts establish.

Frozen beliefs:
{json.dumps(compact_beliefs, ensure_ascii=False, separators=(",", ":"))}

Topic evidence:
{json.dumps(run_context, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _validate_evaluation(
    response: RetrospectiveResponse,
    belief_ids: List[str],
) -> None:
    # Reconcile bookkeeping locally. Concrete observations establish the
    # applicable partition. Missing, duplicated, or unsupported known IDs are
    # conservatively insufficient; unknown IDs still indicate a corrupt model
    # response and must fail.
    expected = set(belief_ids)
    observed_ids = {item.belief_id for item in response.observations}
    supplied_ids = (
        set(response.applicable_belief_ids)
        | set(response.not_applicable_belief_ids)
        | set(response.insufficient_belief_ids)
        | observed_ids
    )
    unknown = sorted(supplied_ids - expected)
    if unknown:
        raise ValueError(f"Invalid evidence response: unknown belief IDs={unknown}")

    original_not_applicable = set(response.not_applicable_belief_ids)
    original_insufficient = set(response.insufficient_belief_ids)
    response.applicable_belief_ids = [
        belief_id for belief_id in belief_ids if belief_id in observed_ids
    ]
    response.not_applicable_belief_ids = [
        belief_id
        for belief_id in belief_ids
        if belief_id not in observed_ids
        and belief_id in original_not_applicable
        and belief_id not in original_insufficient
    ]
    not_applicable = set(response.not_applicable_belief_ids)
    response.insufficient_belief_ids = [
        belief_id
        for belief_id in belief_ids
        if belief_id not in observed_ids and belief_id not in not_applicable
    ]

    assigned = (
        response.applicable_belief_ids
        + response.not_applicable_belief_ids
        + response.insufficient_belief_ids
    )
    if sorted(assigned) != sorted(belief_ids):
        raise AssertionError("Evidence partition reconciliation lost a frozen belief")


def _records_from_frozen(frozen: List[Dict[str, Any]]) -> Dict[str, BeliefRecord]:
    records: Dict[str, BeliefRecord] = {}
    for item in frozen:
        scope = item["scope"]
        record = BeliefRecord(
            belief_id=item["belief_id"],
            instruction=item["instruction"],
            scope=BeliefScope(
                roles=list(scope["roles"]),
                stages=list(scope["stages"]),
                problem_description=scope["problem_description"],
                context_conditions=list(scope["context_conditions"]),
            ),
            status="probation",
            impact=item["impact"],
            belief_type=item["belief_type"],
        )
        record.update_confidence()
        record.status = "probation"
        records[record.belief_id] = record
    return records


def _record_payload(record: BeliefRecord) -> Dict[str, Any]:
    """Serialize a three-stage belief without legacy timing metadata."""
    payload = asdict(record)
    payload.pop("timing", None)
    for evidence in payload.get("evidence", []):
        if isinstance(evidence, dict):
            evidence.pop("timing", None)
    return payload


def _assessment_from_observation(
    item: EvidenceObservation,
    belief: BeliefRecord,
    *,
    run_id: str,
    topic: str,
) -> BeliefAssessment:
    values = _evidence_categories_to_values(
        item.strategy_application,
        item.attribution_strength,
        item.evidence_reliability,
        item.outcome_improvement,
    )
    update_direction, update_eligible = _derive_update_direction(item)
    action = {
        "support": ACTION_SUPPORT,
        "contradict": ACTION_CONTRADICT,
        "neutral": ACTION_OBSERVE,
    }[update_direction]
    # An attributable unchanged/worsened outcome is evidence against the
    # strategy's effectiveness, so its Beta outcome is deterministically zero.
    if update_direction == "contradict":
        values["improvement"] = 0.0
    derived_confidence = values["reliability_probability"]
    return BeliefAssessment(
        run_id=run_id,
        belief_id=belief.belief_id,
        instruction=belief.instruction,
        applicable=True,
        compliance=item.compliance,
        outcome=item.outcome,
        action=action,
        weight=derived_confidence,
        scope=belief.scope,
        evidence_confidence=derived_confidence,
        impact=belief.impact,
        belief_type=belief.belief_type,
        agent=item.agent,
        section_id=item.section_id,
        reason=item.reason,
        evidence={
            "summary": item.evidence,
            "update_direction": update_direction,
            "update_eligible": update_eligible,
            "confidence_source": "derived_from_evidence_reliability",
        },
        topic=topic,
        strategy_application=item.strategy_application,
        attribution_strength=item.attribution_strength,
        evidence_reliability=item.evidence_reliability,
        outcome_improvement=item.outcome_improvement,
        **values,
        reflection_call="retrospective_evidence",
    )


def _derive_update_direction(
    item: EvidenceObservation,
) -> tuple[Literal["support", "contradict", "neutral"], bool]:
    """Derive the effectiveness update from auditable categorical evidence."""
    strategy_applied = item.strategy_application in {"full", "partial"}
    attributable = item.attribution_strength in {"strong", "moderate"}
    reliable = item.evidence_reliability in {"direct", "corroborated"}
    if (
        strategy_applied
        and attributable
        and reliable
        and item.outcome_improvement in {"resolved", "improved"}
    ):
        return "support", True
    if (
        strategy_applied
        and attributable
        and reliable
        and item.outcome_improvement in {"unchanged", "worsened"}
    ):
        return "contradict", True
    return "neutral", False


def _observation_matrix_payload(item: EvidenceObservation) -> Dict[str, Any]:
    direction, eligible = _derive_update_direction(item)
    return {
        **item.model_dump(),
        "update_direction": direction,
        "update_eligible": eligible,
    }


def _cap_topic_observation_weights(
    assessments: List[BeliefAssessment],
) -> List[BeliefAssessment]:
    """Share the unit topic cap proportionally across repeated belief evidence."""
    updating_actions = {ACTION_SUPPORT, ACTION_CONTRADICT}
    by_belief: Dict[str, List[BeliefAssessment]] = {}
    for assessment in assessments:
        if (
            assessment.belief_id
            and assessment.action in updating_actions
            and assessment.outcome_improvement != "unclear"
        ):
            by_belief.setdefault(assessment.belief_id, []).append(assessment)

    for grouped in by_belief.values():
        raw_weights = [
            assessment.strategy_applied_probability
            * assessment.attribution_probability
            * assessment.reliability_probability
            for assessment in grouped
        ]
        total = sum(raw_weights)
        scale = min(1.0, 1.0 / total) if total > 0.0 else 1.0
        for assessment, raw_weight in zip(grouped, raw_weights):
            assessment.reliability_probability *= scale
            assessment.evidence["uncapped_evidence_weight"] = raw_weight
            assessment.evidence["topic_weight_scale"] = scale
    return assessments


def _evidence_quality_summary(
    matrix_rows: List[Dict[str, Any]],
    records: Dict[str, BeliefRecord],
) -> Dict[str, Any]:
    observations = [
        observation
        for row in matrix_rows
        for observation in row.get("observations", [])
    ]
    evaluation_counts = Counter(row["evaluation"] for row in matrix_rows)
    direction_counts = Counter(
        observation["update_direction"] for observation in observations
    )
    derived_reliability_weights = {
        "direct": 1.0,
        "corroborated": 0.85,
        "indirect": 0.65,
        "inferred": 0.4,
        "unverifiable": 0.0,
    }
    category_fields = (
        "strategy_application",
        "attribution_strength",
        "evidence_reliability",
        "outcome_improvement",
        "compliance",
        "outcome",
    )
    per_topic: Dict[str, Dict[str, Any]] = {}
    for row in matrix_rows:
        topic_summary = per_topic.setdefault(
            row["topic"],
            {
                "evaluation_counts": Counter(),
                "observation_count": 0,
                "update_direction_counts": Counter(),
            },
        )
        topic_summary["evaluation_counts"][row["evaluation"]] += 1
        topic_observations = row.get("observations", [])
        topic_summary["observation_count"] += len(topic_observations)
        topic_summary["update_direction_counts"].update(
            observation["update_direction"] for observation in topic_observations
        )
    topic_weights = {
        topic: round(
            sum(record.topic_evidence_weights.get(topic, 0.0) for record in records.values()),
            6,
        )
        for topic in per_topic
    }
    for topic, summary in per_topic.items():
        summary["evaluation_counts"] = dict(summary["evaluation_counts"])
        summary["update_direction_counts"] = dict(
            summary["update_direction_counts"]
        )
        summary["bayesian_evidence_weight"] = topic_weights[topic]

    return {
        "evaluation_counts": dict(evaluation_counts),
        "observation_count": len(observations),
        "update_eligible_count": sum(
            bool(observation["update_eligible"]) for observation in observations
        ),
        "update_direction_counts": dict(direction_counts),
        "category_distributions": {
            field: dict(Counter(observation.get(field) for observation in observations))
            for field in category_fields
        },
        "derived_reliability_weight_distribution": dict(
            Counter(
                str(derived_reliability_weights[observation["evidence_reliability"]])
                for observation in observations
            )
        ),
        "audit_location_coverage": {
            "agent_identified": sum(
                observation.get("agent") is not None for observation in observations
            ),
            "section_identified": sum(
                observation.get("section_id") is not None for observation in observations
            ),
            "total_observations": len(observations),
        },
        "rows_with_multiple_observations": sum(
            len(row.get("observations", [])) > 1 for row in matrix_rows
        ),
        "bayesian_evidence_weight_total": round(sum(topic_weights.values()), 6),
        "per_topic": per_topic,
    }


def _assign_operational_status(record: BeliefRecord) -> None:
    # A semantic mention with zero/negligible Bayesian weight is not an
    # independent effectiveness observation.
    topics = sum(
        1 for weight in record.topic_evidence_weights.values() if float(weight) >= 0.10
    )
    if record.contradiction_count > 0 and record.weighted_contradiction >= record.weighted_support:
        record.status = "contested"
    elif topics >= 3 and record.confidence >= 0.65:
        record.status = "active"
    else:
        record.status = "probation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--belief-embedding-model", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--start-stage",
        choices=["discovery", "consolidation", "evidence"],
        default="discovery",
        help=(
            "Start from discovery, or load an existing completed "
            "belief_candidates.json for consolidation, or load frozen_beliefs.json "
            "and begin retrospective evidence evaluation."
        ),
    )
    args = parser.parse_args()

    pipeline_dir = Path(args.pipeline_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else pipeline_dir / "belief_three_stage"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": output_dir / "belief_candidates.json",
        "consolidation": output_dir / "belief_consolidation.json",
        "quality_concerns": output_dir / "belief_quality_concerns.json",
        "frozen": output_dir / "frozen_beliefs.json",
        "matrix": output_dir / "belief_evidence_matrix.jsonl",
        "library": output_dir / "belief_library.json",
        "evidence": output_dir / "belief_evidence.jsonl",
        "analysis": output_dir / "belief_pipeline_analysis.json",
        "progress": output_dir / "belief_pipeline_progress.json",
        "bbn": output_dir / "bbn_parameters.json",
        "embeddings": output_dir / "belief_embeddings.npz",
        "embedding_metadata": output_dir / "belief_embedding_metadata.json",
    }
    if args.fresh and args.start_stage != "discovery":
        raise ValueError(
            "--fresh cannot be combined with a resumed start stage because it "
            "would delete the checkpoint being resumed"
        )
    if args.fresh:
        for path in paths.values():
            if path.exists():
                path.unlink()

    run_dirs = [
        path
        for path in _iter_run_dirs_from_pipeline(pipeline_dir)
        if _find_final_state_json(path) is not None
    ]
    provisional = [summarise_run(path, None) for path in run_dirs]
    baseline = _safe_mean(run.combined_score for run in provisional)
    runs = [summarise_run(path, baseline) for path in run_dirs]
    api_key = cfg("gemini", "api_key")
    if not api_key:
        raise ValueError("Missing Gemini API key")
    client = Client(api_key=api_key)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    call_log: List[Dict[str, Any]] = []

    # Stage 1, or resume from its completed checkpoint.
    candidates: List[Dict[str, Any]]
    discovery_rejections: List[Dict[str, Any]] = []
    if args.start_stage == "consolidation":
        if not paths["candidates"].exists():
            raise FileNotFoundError(
                f"Cannot resume; candidate checkpoint not found: "
                f"{paths['candidates']}"
            )
        candidate_payload = json.loads(paths["candidates"].read_text(encoding="utf-8"))
        if not candidate_payload.get("complete"):
            raise ValueError(
                "Cannot start at consolidation because belief_candidates.json is "
                "not marked complete"
            )
        candidates = list(candidate_payload.get("candidates") or [])
        discovery_rejections = list(candidate_payload.get("rejected_observations") or [])
        if not candidates:
            raise ValueError("Candidate checkpoint contains no candidates")
        print(
            f"[belief-3stage][resume] Loaded {len(candidates)} completed candidates "
            f"from {paths['candidates']}",
            flush=True,
        )
    elif args.start_stage == "evidence":
        # Candidates are not an input to Stage 3. Load them only for final
        # provenance counts when the checkpoint is still available.
        if paths["candidates"].exists():
            candidate_payload = json.loads(
                paths["candidates"].read_text(encoding="utf-8")
            )
            candidates = list(candidate_payload.get("candidates") or [])
            discovery_rejections = list(
                candidate_payload.get("rejected_observations") or []
            )
        else:
            candidates = []
    else:
        candidates = []
        print(f"[belief-3stage][discovery] Starting {len(runs)} topics", flush=True)
        for index, run in enumerate(runs, start=1):
            started = time.perf_counter()
            context = _build_reflection_context(run, _state_for_run(Path(run.run_dir)), {})
            parsed, usage, attempts = _structured_call(
                client,
                prompt=_discovery_prompt(context, candidates),
                schema=DiscoveryResponse,
                label=f"discovery_{index:02d}",
                raw_output_dir=output_dir,
                max_output_tokens=12000,
            )
            counts = _apply_discovery_decisions(
                candidates,
                parsed.decisions,
                run_id=run.run_id,
                topic=run.topic,
                rejections=discovery_rejections,
            )
            for key in usage_total:
                usage_total[key] += usage[key]
            call_log.append(
                {
                    "stage": "discovery",
                    "run_id": run.run_id,
                    "usage": usage,
                    "attempts": attempts,
                }
            )
            _write_json(
                paths["candidates"],
                {
                    "stage": "discovery",
                    "complete": index == len(runs),
                    "candidates": candidates,
                    "rejected_observations": discovery_rejections,
                },
            )
            _write_json(
                paths["progress"],
                {
                    "status": "running",
                    "stage": "discovery",
                    "completed": index,
                    "total": len(runs),
                    "candidate_count": len(candidates),
                },
            )
            print(
                f"[belief-3stage][discovery][{index}/{len(runs)}] {run.topic} | "
                f"decisions={len(parsed.decisions)} candidates={len(candidates)} "
                f"updates={counts} attempts={attempts} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    # Stage 2, or resume directly from its frozen output.
    if args.start_stage == "evidence":
        if not paths["frozen"].exists():
            raise FileNotFoundError(
                f"Cannot start at evidence; frozen belief checkpoint not found: "
                f"{paths['frozen']}"
            )
        frozen_payload = json.loads(paths["frozen"].read_text(encoding="utf-8"))
        if not frozen_payload.get("definition_frozen"):
            raise ValueError(
                "Cannot start at evidence because frozen_beliefs.json is not "
                "marked definition_frozen=true"
            )
        frozen = list(frozen_payload.get("beliefs") or [])
        if not frozen:
            raise ValueError("Frozen belief checkpoint contains no beliefs")
        frozen_ids = [str(item.get("belief_id") or "") for item in frozen]
        if any(not belief_id for belief_id in frozen_ids) or len(frozen_ids) != len(
            set(frozen_ids)
        ):
            raise ValueError("Frozen belief checkpoint has missing or duplicate belief IDs")
        print(
            f"[belief-3stage][resume] Loaded {len(frozen)} frozen beliefs from "
            f"{paths['frozen']}; restarting Stage 3 from topic 1",
            flush=True,
        )
    else:
        print(
            f"[belief-3stage][consolidation] Consolidating {len(candidates)} candidates",
            flush=True,
        )
        candidate_ids = [item["candidate_id"] for item in candidates]
        # Do not recover responses from an earlier consolidation policy: a
        # policy change must always trigger a fresh model judgement.
        consolidated, usage, attempts = _structured_call(
            client,
            prompt=_consolidation_prompt(candidates),
            schema=ConsolidationResponse,
            label="consolidation_atomic_v5_prompt_guidance",
            raw_output_dir=output_dir,
            # Consolidation must account for every candidate and may need to emit
            # many complete frozen belief definitions. Do not impose an additional
            # application-level output cap on this one global call.
            max_output_tokens=None,
            semantic_validator=lambda response: _validate_consolidation(
                response,
                candidate_ids,
                candidates,
            ),
            max_attempts=3,
        )
        for key in usage_total:
            usage_total[key] += usage[key]
        call_log.append(
            {"stage": "consolidation", "usage": usage, "attempts": attempts}
        )
        quality_concerns = _consolidation_quality_concerns(
            consolidated,
            candidates,
        )
        _write_json(
            paths["quality_concerns"],
            {
                "blocking": False,
                "concern_count": len(quality_concerns),
                "concerns": quality_concerns,
            },
        )
        if quality_concerns:
            print(
                f"[belief-3stage][consolidation] Non-blocking academic quality "
                f"concerns={len(quality_concerns)}; see "
                f"{paths['quality_concerns']}",
                flush=True,
            )
        frozen = _frozen_beliefs(consolidated)
        _write_json(paths["consolidation"], consolidated.model_dump())
        _write_json(
            paths["frozen"],
            {
                "definition_frozen": True,
                "belief_count": len(frozen),
                "beliefs": frozen,
            },
        )
        print(
            f"[belief-3stage][consolidation] Frozen beliefs={len(frozen)} "
            f"excluded={len(consolidated.excluded_candidates)} attempts={attempts}",
            flush=True,
        )

    # Stage 3
    records = _records_from_frozen(frozen)
    matrix_rows: List[Dict[str, Any]] = []
    belief_ids = list(records)
    print(
        f"[belief-3stage][evidence] Evaluating {len(belief_ids)} frozen beliefs "
        f"against {len(runs)} topics",
        flush=True,
    )
    for index, run in enumerate(runs, start=1):
        started = time.perf_counter()
        context = _build_reflection_context(run, _state_for_run(Path(run.run_dir)), {})
        evaluated, usage, attempts = _structured_call(
            client,
            prompt=_evaluation_prompt(context, frozen),
            schema=RetrospectiveResponse,
            label=f"evidence_{index:02d}",
            raw_output_dir=output_dir,
            # One belief may now contain several distinct observations. Do not
            # impose a client-side cap that can truncate an otherwise valid
            # structured topic response.
            max_output_tokens=None,
            semantic_validator=lambda response: _validate_evaluation(
                response, belief_ids
            ),
        )
        assessments = _cap_topic_observation_weights(
            [
                _assessment_from_observation(
                    observation,
                    records[observation.belief_id],
                    run_id=run.run_id,
                    topic=run.topic,
                )
                for observation in evaluated.observations
            ]
        )
        update_summary = apply_assessments(records, assessments)
        observations_by_id: Dict[str, List[EvidenceObservation]] = {}
        for observation in evaluated.observations:
            observations_by_id.setdefault(observation.belief_id, []).append(observation)
        for belief_id in belief_ids:
            if belief_id in evaluated.applicable_belief_ids:
                row = {
                    "run_id": run.run_id,
                    "topic": run.topic,
                    "belief_id": belief_id,
                    "evaluation": "applicable",
                    "observations": [
                        _observation_matrix_payload(observation)
                        for observation in observations_by_id[belief_id]
                    ],
                }
            elif belief_id in evaluated.not_applicable_belief_ids:
                row = {
                    "run_id": run.run_id,
                    "topic": run.topic,
                    "belief_id": belief_id,
                    "evaluation": "not_applicable",
                }
            else:
                row = {
                    "run_id": run.run_id,
                    "topic": run.topic,
                    "belief_id": belief_id,
                    "evaluation": "insufficient",
                }
            matrix_rows.append(row)
        _atomic_jsonl(paths["matrix"], matrix_rows)
        save_library(
            paths["library"],
            records,
            metadata={
                "method": "three_stage",
                "definitions_frozen": True,
                "checkpoint": True,
                "evidence_topics_completed": index,
                "evidence_topics_total": len(runs),
            },
        )
        save_evidence(paths["evidence"], records)
        for key in usage_total:
            usage_total[key] += usage[key]
        call_log.append(
            {"stage": "evidence", "run_id": run.run_id, "usage": usage, "attempts": attempts}
        )
        _write_json(
            paths["progress"],
            {
                "status": "running",
                "stage": "evidence",
                "completed": index,
                "total": len(runs),
                "frozen_belief_count": len(records),
            },
        )
        print(
            f"[belief-3stage][evidence][{index}/{len(runs)}] {run.topic} | "
            f"applicable={len(evaluated.applicable_belief_ids)} "
            f"observations={len(assessments)} "
            f"not_applicable={len(evaluated.not_applicable_belief_ids)} "
            f"insufficient={len(evaluated.insufficient_belief_ids)} "
            f"updates={update_summary} attempts={attempts} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    for record in records.values():
        _assign_operational_status(record)
        for evidence in record.evidence:
            if isinstance(evidence, dict):
                evidence.pop("timing", None)
    _write_json(
        paths["library"],
        {
            "metadata": {
            "method": "three_stage",
            "definitions_frozen": True,
            "source_pipeline": str(pipeline_dir),
            "candidate_count": len(candidates),
            "belief_count": len(records),
            "topic_count": len(runs),
            },
            "beliefs": [
                _record_payload(record)
                for record in sorted(records.values(), key=lambda item: item.belief_id)
            ],
        },
    )
    save_evidence(paths["evidence"], records)
    _write_json(paths["bbn"], BBNParameters().to_payload())
    if args.belief_embedding_model:
        BeliefEmbeddingIndex.build(
            [_record_payload(record) for record in records.values()],
            model_name_or_path=args.belief_embedding_model,
            embeddings_path=paths["embeddings"],
            metadata_path=paths["embedding_metadata"],
        )
    _write_json(
        paths["analysis"],
        {
            "method": "three_stage",
            "pipeline_dir": str(pipeline_dir),
            "topic_count": len(runs),
            "candidate_count": len(candidates),
            "frozen_belief_count": len(records),
            "evidence_matrix_rows": len(matrix_rows),
            "usage": usage_total,
            "calls": call_log,
            "evidence_quality": _evidence_quality_summary(matrix_rows, records),
            "status_counts": {
                status: sum(1 for record in records.values() if record.status == status)
                for status in ("active", "probation", "contested", "deprecated")
            },
        },
    )
    _write_json(
        paths["progress"],
        {
            "status": "complete",
            "stage": "complete",
            "topics": len(runs),
            "candidates": len(candidates),
            "beliefs": len(records),
            "evidence_matrix_rows": len(matrix_rows),
        },
    )
    print(
        f"[belief-3stage] Complete | candidates={len(candidates)} "
        f"beliefs={len(records)} evidence_rows={len(matrix_rows)}",
        flush=True,
    )
    print(f"Final library: {paths['library']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
