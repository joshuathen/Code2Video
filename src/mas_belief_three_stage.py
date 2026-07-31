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
import time
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
    save_library,
    summarise_run,
)


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
    action: Literal["ADD", "MATCH", "PROPOSE_REVISION"]
    candidate_id: Optional[str] = None
    instruction: Optional[str] = Field(default=None, max_length=500)
    scope: Optional[ReflectionScopePayload] = None
    impact: Literal["low", "medium", "high", "critical"]
    belief_type: Literal["confirmed", "precaution", "hypothesis", "quality"]
    timing: Literal["preventative", "reactive", "both"]

    @model_validator(mode="after")
    def validate_target(self) -> "CandidateDecision":
        if self.action == "ADD":
            if self.candidate_id is not None or self.instruction is None or self.scope is None:
                raise ValueError("ADD requires instruction and scope and candidate_id=null")
        else:
            if not self.candidate_id:
                raise ValueError(f"{self.action} requires candidate_id")
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
    timing: Literal["preventative", "reactive", "both"]
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
    direction: Literal["support", "contradict", "neutral"]
    evidence_confidence: float = Field(ge=0.0, le=1.0)


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
                "\nThe previous response failed local validation:\n"
                + error_feedback
                + "\nCorrect this error while preserving all otherwise valid distinct content."
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
        "timing": candidate["timing"],
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

Discover every distinct, reusable, evidence-grounded learning that could improve a
future MAS run. The candidate bank exists only to reduce duplicates.

For each discovered mechanism choose exactly one action:
- ADD: no candidate covers the same mechanism. Provide a complete instruction and scope.
- MATCH: an existing candidate already covers it. Do not rewrite that candidate.
- PROPOSE_REVISION: evidence suggests an existing candidate should be narrowed,
  broadened, or corrected. Provide the complete proposed replacement and scope.

Revision proposals are recorded but are NOT applied during discovery. Do not perform
Bayesian updating, assign posterior confidence, or evaluate unrelated candidates.
Do not let an existing candidate prevent discovery of a genuinely different mechanism.
Use role names, not agent instances. Generalize only as far as direct evidence permits.
Classify evidence using the permitted categorical options; never invent probabilities.
Use critical only for unrecoverable termination, corruption, data loss, or uncontrolled
external effects; ordinary render timeouts are high.

Return JSON conforming to the supplied schema. For MATCH, instruction and scope may be
null. For ADD set candidate_id=null. Evidence must identify the concrete observation.

Input:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _apply_discovery_decisions(
    candidates: List[Dict[str, Any]],
    decisions: List[CandidateDecision],
    *,
    run_id: str,
    topic: str,
) -> Dict[str, int]:
    by_id = {item["candidate_id"]: item for item in candidates}
    counts = {"added": 0, "matched": 0, "revision_proposed": 0}
    for decision in decisions:
        evidence = {
            "run_id": run_id,
            "topic": topic,
            **decision.model_dump(
                exclude={"action", "candidate_id", "instruction", "scope"}
            ),
        }
        if decision.action == "ADD":
            candidate_id = f"C{len(candidates) + 1:03d}"
            candidate = {
                "candidate_id": candidate_id,
                "instruction": decision.instruction,
                "scope": decision.scope.model_dump() if decision.scope else {},
                "impact": decision.impact,
                "belief_type": decision.belief_type,
                "timing": decision.timing,
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
- preserve distinct causes, roles, timing, or mitigations as separate beliefs;
- exclude candidates that are demonstrably incorrect, redundant without contributing
  distinct evidence, lack any concrete origin evidence, or are not reusable, and explain why;
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
- Keep preventative guidance separate from reactive recovery when the actions or
  triggering conditions differ.
- Keep runtime correctness, render performance, visual composition, mathematical
  accuracy, and pedagogy as separate mechanisms.
- Prefer retaining two narrowly testable beliefs over producing one broad instruction.
- Each final belief may contain at most four source candidate IDs. This is an upper
  bound, not a target; most beliefs should contain one or two.
- Do not use exclusions to avoid producing appropriately separate beliefs.
- "Niche", "low frequency", "single occurrence", "low impact", "tool-specific",
  "basic library knowledge", or "only one topic" are NOT valid exclusion reasons.
  Preserve such candidates as standalone probationary beliefs when they express a reusable,
  evidence-grounded mechanism.
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

Use these questions as internal merge guidance. Do not add merge-check fields to the
output object. Explain the shared mechanism concisely in consolidation_reason.
Splitting candidates is always acceptable and preferred when equivalence is uncertain.

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
    maximum_exclusions = max(3, len(candidate_ids) // 4)
    if len(response.excluded_candidates) > maximum_exclusions:
        raise ValueError(
            f"Consolidation excluded {len(response.excluded_candidates)} of "
            f"{len(candidate_ids)} candidates; maximum permitted is "
            f"{maximum_exclusions}. Preserve uncertain reusable candidates as "
            "separate probationary beliefs instead of mass-excluding them."
        )


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
            "timing": item["timing"],
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
- A single belief may have both supporting and contradicting observations in the same
  topic. Preserve both; do not average them into one judgement.
- Do not repeat the same event using different wording. Each observation must identify
  a distinct transition or outcome.
- direction="support" means that this observation provides positive evidence that
  applying the exact frozen instruction caused or materially contributed to a better
  target outcome. Compliance alone is not support.
- direction="contradict" means that this observation provides negative evidence about
  the effectiveness of the exact frozen instruction. Violation alone is not
  contradiction.
- direction="neutral" records an informative application, non-application, relevance,
  or outcome event that does not justify a positive or negative effectiveness update.
- Absence of the strategy, mere final-code presence, or an unclear before/after outcome
  is normally neutral—not automatically support or contradiction.
- The local updater caps the combined Bayesian evidence weight contributed by all
  observations for one belief in one topic, so report distinct evidence honestly
  rather than trying to compress it.

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

DIRECTION IS AN EFFECTIVENESS JUDGEMENT, NOT A COMPLIANCE JUDGEMENT:
Choose direction only AFTER completing strategy_application, attribution_strength,
evidence_reliability, and outcome_improvement.

Use direction="support" only when ALL of these hold:
1. the exact strategy was applied fully or partially;
2. the target problem is observed before application and resolved or improved after it;
3. the transition has strong or moderate attribution to that strategy; and
4. the source directly or corroboratively records the relevant transition.

Otherwise, do not label the observation support. In particular:
- strategy present in final code + successful render => neutral;
- strategy followed + positive AES/TQ score => neutral unless a specific target
  before/after change is isolated;
- strategy absent + predicted problem occurred => neutral evidence of relevance and
  noncompliance, not evidence that the strategy would have fixed it;
- a reviewer saying that a technique is good, clear, consistent, or helpful without a
  comparable transition => neutral;
- an instruction was followed and no target outcome is available => neutral.

Use direction="contradict" only when the evidence tests effectiveness negatively:
- the exact strategy was applied, the target outcome was unchanged or worsened, and
  attribution is strong or moderate; or
- a genuinely comparable alternative succeeds without the strategy AND the frozen
  instruction specifically claims that its strategy is necessary or uniquely required.
Do not treat ordinary noncompliance, an alternative implementation, or success without
a merely recommended strategy as contradiction.

Use direction="neutral" for every concrete applicability/compliance observation that
does not pass the support or contradiction tests. Neutral is a useful result: it records
role, context, application, and outcome information without changing effectiveness.
When uncertain between support and neutral, choose neutral. When uncertain between
contradiction and neutral, choose neutral.

PARTITION GUIDANCE:
- Use not_applicable only when the belief's scoped problem, role, stage, or opportunity
  genuinely did not occur.
- Use insufficient when the context may match but no concrete event establishes whether
  the exact strategy was applied, violated, or encountered.
- Use applicable with direction="neutral" when a concrete strategy application,
  non-application, or matching problem event is directly observable but effectiveness
  cannot be established.
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
- strategy full/partial + isolated improvement may support effectiveness;
- strategy full/partial + isolated worsening may contradict effectiveness;
- strategy full/partial + no before/after outcome means improvement is unclear;
- strategy none + predicted problem occurred shows relevance/noncompliance, but does not
  directly demonstrate that providing the strategy would have fixed it;
- strategy none + successful outcome may suggest the strategy is unnecessary, but only
  contradicts it when the opportunity and alternative explanation are comparable;
- aggregate AES/TQ scores cannot establish attribution to one coding, layout, planning,
  or pedagogical strategy without specific supporting evidence.

EVIDENCE CONFIDENCE:
- Reserve 1.0 for an explicit, unambiguous and well-isolated recorded transition.
- Use 0.8-0.9 when the event is clear but interpretation or attribution has some
  uncertainty.
- Use 0.5-0.7 for indirect, incomplete, or weakly attributable observations.
- Confidence describes confidence in the complete observation, not confidence that a
  code token, issue, or final state exists.

AUDIT LOCATION:
- Populate agent whenever the responsible agent can be identified.
- Populate section_id whenever the evidence belongs to a particular section.
- Keep either null only when the event is genuinely topic-wide or the source does not
  permit identification.

Before returning a support or contradiction direction, ask:
1. What exact event was directly observed?
2. Where is the before/action/after transition?
3. Which simultaneous changes offer alternative explanations?
4. Does the evidence support the complete frozen instruction or only a clause/example?
5. Would direction="neutral" or outcome_improvement="unclear" be more honest?

Prefer neutral direction, weak attribution, or unclear improvement over an unsupported
causal claim. Use insufficient when even the concrete application or problem event
cannot be established. Do not aim for balanced numbers, but actively search for
contradictory, unchanged, and noncompliant evidence as well as supportive evidence.

Frozen beliefs:
{json.dumps(compact_beliefs, ensure_ascii=False, separators=(",", ":"))}

Topic evidence:
{json.dumps(run_context, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _validate_evaluation(
    response: RetrospectiveResponse,
    belief_ids: List[str],
) -> None:
    # Gemini sometimes uses "applicable" to mean merely contextually relevant
    # while returning observations only for evidentially useful beliefs. The
    # latter is our operational definition. Reclassify unsupported applicable
    # IDs as insufficient locally rather than inventing evidence or paying for
    # a full retry.
    observed_ids = {item.belief_id for item in response.observations}
    unsupported_applicable = [
        belief_id
        for belief_id in response.applicable_belief_ids
        if belief_id not in observed_ids
    ]
    if unsupported_applicable:
        unsupported_set = set(unsupported_applicable)
        response.applicable_belief_ids = [
            belief_id
            for belief_id in response.applicable_belief_ids
            if belief_id not in unsupported_set
        ]
        response.insufficient_belief_ids.extend(
            belief_id
            for belief_id in unsupported_applicable
            if belief_id not in response.insufficient_belief_ids
        )

    assigned = (
        response.applicable_belief_ids
        + response.not_applicable_belief_ids
        + response.insufficient_belief_ids
    )
    if sorted(assigned) != sorted(belief_ids):
        missing = sorted(set(belief_ids) - set(assigned))
        duplicated = sorted(
            belief_id for belief_id in set(assigned) if assigned.count(belief_id) > 1
        )
        unknown = sorted(set(assigned) - set(belief_ids))
        raise ValueError(
            f"Invalid evidence partition: missing={missing}, "
            f"duplicated={duplicated}, unknown={unknown}"
        )
    applicable = set(response.applicable_belief_ids)
    observation_ids = [item.belief_id for item in response.observations]
    missing_observations = sorted(applicable - set(observation_ids))
    misplaced_observations = sorted(set(observation_ids) - applicable)
    if missing_observations or misplaced_observations:
        raise ValueError(
            "Invalid evidence observations: "
            f"applicable_without_observations={missing_observations}, "
            f"observations_for_non_applicable={misplaced_observations}"
        )


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
            timing=item["timing"],
        )
        record.update_confidence()
        record.status = "probation"
        records[record.belief_id] = record
    return records


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
    action = {
        "support": ACTION_SUPPORT,
        "contradict": ACTION_CONTRADICT,
        "neutral": ACTION_OBSERVE,
    }[item.direction]
    # Direction is the explicit effectiveness judgement. Force the numeric
    # outcome to the corresponding side of the Beta update; neutral records are
    # retained but ACTION_OBSERVE is deliberately excluded from updating.
    if item.direction == "support":
        values["improvement"] = max(0.5, values["improvement"])
    elif item.direction == "contradict":
        values["improvement"] = min(0.499999, values["improvement"])
    return BeliefAssessment(
        run_id=run_id,
        belief_id=belief.belief_id,
        instruction=belief.instruction,
        applicable=True,
        compliance=item.compliance,
        outcome=item.outcome,
        action=action,
        weight=item.evidence_confidence,
        scope=belief.scope,
        evidence_confidence=item.evidence_confidence,
        impact=belief.impact,
        belief_type=belief.belief_type,
        timing=belief.timing,
        agent=item.agent,
        section_id=item.section_id,
        reason=item.reason,
        evidence={"summary": item.evidence, "direction": item.direction},
        topic=topic,
        strategy_application=item.strategy_application,
        attribution_strength=item.attribution_strength,
        evidence_reliability=item.evidence_reliability,
        outcome_improvement=item.outcome_improvement,
        **values,
        reflection_call="retrospective_evidence",
    )


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
            label="consolidation_atomic_v4_prompt_guidance",
            raw_output_dir=output_dir,
            # Consolidation must account for every candidate and may need to emit
            # many complete frozen belief definitions. Do not impose an additional
            # application-level output cap on this one global call.
            max_output_tokens=None,
            semantic_validator=lambda response: _validate_consolidation(
                response,
                candidate_ids,
            ),
            max_attempts=3,
        )
        for key in usage_total:
            usage_total[key] += usage[key]
        call_log.append(
            {"stage": "consolidation", "usage": usage, "attempts": attempts}
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
                        observation.model_dump()
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
    save_library(
        paths["library"],
        records,
        metadata={
            "method": "three_stage",
            "definitions_frozen": True,
            "source_pipeline": str(pipeline_dir),
            "candidate_count": len(candidates),
            "belief_count": len(records),
            "topic_count": len(runs),
        },
    )
    save_evidence(paths["evidence"], records)
    _write_json(paths["bbn"], BBNParameters().to_payload())
    if args.belief_embedding_model:
        BeliefEmbeddingIndex.build(
            [asdict(record) for record in records.values()],
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
