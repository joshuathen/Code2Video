"""Bayesian belief applicability, retrieval, and posterior updating.

The LLM-facing reflection pipeline produces structured observations.  This
module owns the deterministic probability calculations so that a model never
directly invents posterior parameters.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def clamp_probability(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def posterior_mean(alpha: float, beta: float) -> float:
    denominator = float(alpha) + float(beta)
    return float(alpha) / denominator if denominator > 0 else 0.5


@dataclass(frozen=True)
class ApplicabilityEvidence:
    role_match: float
    stage_match: float
    problem_match: float
    context_match: float
    exact_error_match: float = 0.0


@dataclass
class BBNParameters:
    """Logistic CPD for the latent Applicability node.

    This is a compact conditional probability distribution for:
    P(Applicability | RoleMatch, StageMatch, ProblemMatch, ContextMatch).
    """

    intercept: float = -4.0
    role_match: float = 1.5
    stage_match: float = 1.5
    problem_match: float = 3.0
    context_match: float = 2.0
    exact_error_match: float = 4.0
    version: int = 2

    @classmethod
    def from_path(cls, path: Optional[Path]) -> "BBNParameters":
        if path is None or not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("applicability_model", payload)
        allowed = {
            "intercept",
            "role_match",
            "stage_match",
            "problem_match",
            "context_match",
            "exact_error_match",
            "version",
        }
        return cls(**{key: value for key, value in model.items() if key in allowed})

    def probability(self, evidence: ApplicabilityEvidence) -> float:
        log_odds = (
            self.intercept
            + self.role_match * clamp_probability(evidence.role_match)
            + self.stage_match * clamp_probability(evidence.stage_match)
            + self.problem_match * clamp_probability(evidence.problem_match)
            + self.context_match * clamp_probability(evidence.context_match)
            + self.exact_error_match
            * clamp_probability(evidence.exact_error_match, default=0.0)
        )
        return 1.0 / (1.0 + math.exp(-log_odds))

    def to_payload(self) -> Dict[str, Any]:
        return {"applicability_model": asdict(self)}


def fit_bbn_parameters(
    cases: Sequence[Dict[str, Any]],
    *,
    initial: Optional[BBNParameters] = None,
    learning_rate: float = 0.05,
    epochs: int = 500,
    prior_strength: float = 0.1,
) -> BBNParameters:
    """Fit the applicability CPD from labelled belief–situation pairs.

    L2 shrinkage toward the documented initial parameters acts as the Bayesian
    prior and prevents a small validation set from producing extreme weights.
    """

    prior = initial or BBNParameters()
    names = [
        "intercept",
        "role_match",
        "stage_match",
        "problem_match",
        "context_match",
        "exact_error_match",
    ]
    weights = [getattr(prior, name) for name in names]
    prior_weights = list(weights)
    usable = [case for case in cases if "applicable" in case]
    if not usable:
        return prior

    for _ in range(max(0, epochs)):
        gradient = [0.0] * len(weights)
        for case in usable:
            features = [
                1.0,
                clamp_probability(case.get("role_match")),
                clamp_probability(case.get("stage_match")),
                clamp_probability(case.get("problem_match")),
                clamp_probability(case.get("context_match")),
                clamp_probability(case.get("exact_error_match"), default=0.0),
            ]
            predicted = 1.0 / (
                1.0 + math.exp(-sum(w * x for w, x in zip(weights, features)))
            )
            label = clamp_probability(case["applicable"])
            case_weight = max(0.0, float(case.get("weight", 1.0)))
            for index, feature in enumerate(features):
                gradient[index] += case_weight * (label - predicted) * feature

        scale = max(1.0, sum(max(0.0, float(c.get("weight", 1.0))) for c in usable))
        for index in range(len(weights)):
            shrinkage = prior_strength * (weights[index] - prior_weights[index])
            weights[index] += learning_rate * (gradient[index] / scale - shrinkage)

    return BBNParameters(
        intercept=weights[0],
        role_match=weights[1],
        stage_match=weights[2],
        problem_match=weights[3],
        context_match=weights[4],
        exact_error_match=weights[5],
        version=prior.version + 1,
    )


@dataclass(frozen=True)
class BeliefSituation:
    topic: str
    agent_role: str
    pipeline_stage: str
    problem_text: str
    context_tags: Sequence[str] = field(default_factory=tuple)
    section_ids: Sequence[str] = field(default_factory=tuple)
    timing: str = "both"


@dataclass(frozen=True)
class TransitionEvidence:
    p_applicable: float
    p_strategy_applied: float
    p_attributable: float
    p_reliable: float
    improvement: float

    @property
    def weight(self) -> float:
        return (
            clamp_probability(self.p_applicable)
            * clamp_probability(self.p_strategy_applied)
            * clamp_probability(self.p_attributable)
            * clamp_probability(self.p_reliable)
        )


def update_beta_posterior(
    alpha: float,
    beta: float,
    evidence: TransitionEvidence,
    *,
    remaining_topic_weight: float = 1.0,
) -> Dict[str, float]:
    """Return a fractional Beta-Binomial update.

    A topic cap prevents many dependent repair attempts in one topic from
    masquerading as independent replications.
    """

    weight = min(evidence.weight, max(0.0, remaining_topic_weight))
    improvement = clamp_probability(evidence.improvement)
    alpha_increment = weight * improvement
    beta_increment = weight * (1.0 - improvement)
    new_alpha = float(alpha) + alpha_increment
    new_beta = float(beta) + beta_increment
    return {
        "alpha": new_alpha,
        "beta": new_beta,
        "posterior_mean": posterior_mean(new_alpha, new_beta),
        "evidence_weight": weight,
        "alpha_increment": alpha_increment,
        "beta_increment": beta_increment,
    }


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def lexical_similarity(first: str, second: str) -> float:
    """Cheap fallback when BGE/sentence-transformers is unavailable."""

    left = set(_TOKEN_RE.findall((first or "").lower()))
    right = set(_TOKEN_RE.findall((second or "").lower()))
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def build_bge_similarity(
    model_name_or_path: Optional[str],
) -> Optional[Callable[[str, str], float]]:
    """Build a local BGE cosine scorer when sentence-transformers is installed."""

    if not model_name_or_path:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE selection requested but sentence-transformers is not installed"
        ) from exc

    model = SentenceTransformer(model_name_or_path)

    def _similarity(first: str, second: str) -> float:
        vectors = model.encode(
            [first or "", second or ""],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Negative semantic similarity is not useful evidence of a match. Keep
        # the positive cosine on its natural scale rather than shifting every
        # mediocre match upward with ``(cosine + 1) / 2``.
        cosine = float(vectors[0] @ vectors[1])
        return clamp_probability(cosine, default=0.0)

    return _similarity


def _embedding_text(belief: Dict[str, Any]) -> str:
    scope = belief.get("scope") if isinstance(belief.get("scope"), dict) else {}
    return str(
        scope.get("problem_description")
        or belief.get("problem_description")
        or belief.get("instruction")
        or ""
    ).strip()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BeliefEmbeddingIndex:
    """Persisted normalized belief vectors plus a query encoder."""

    def __init__(
        self,
        *,
        model: Any,
        model_name: str,
        belief_ids: Sequence[str],
        embeddings: Any,
        text_hashes: Dict[str, str],
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.belief_ids = list(belief_ids)
        self.embeddings = embeddings
        self.text_hashes = dict(text_hashes)
        self._embedding_lock = Lock()
        self._text_vector_cache: Dict[str, Any] = {}

    @classmethod
    def build(
        cls,
        beliefs: Sequence[Dict[str, Any]],
        *,
        model_name_or_path: str,
        embeddings_path: Path,
        metadata_path: Path,
    ) -> "BeliefEmbeddingIndex":
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Stored BGE embeddings require numpy and sentence-transformers"
            ) from exc

        rows = []
        for belief in beliefs:
            belief_id = str(belief.get("belief_id", belief.get("lesson_id")) or "")
            text = _embedding_text(belief)
            if belief_id and text:
                rows.append((belief_id, text))

        model = SentenceTransformer(model_name_or_path)
        texts = [text for _, text in rows]
        if texts:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            dimension = int(model.get_sentence_embedding_dimension())
            embeddings = np.empty((0, dimension), dtype=np.float32)

        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(embeddings_path, embeddings=embeddings)
        metadata = {
            "version": 1,
            "model": model_name_or_path,
            "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "normalized": True,
            "belief_ids": [belief_id for belief_id, _ in rows],
            "text_hashes": {
                belief_id: _text_sha256(text) for belief_id, text in rows
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cls(
            model=model,
            model_name=model_name_or_path,
            belief_ids=metadata["belief_ids"],
            embeddings=embeddings,
            text_hashes=metadata["text_hashes"],
        )

    @classmethod
    def load(
        cls,
        *,
        embeddings_path: Path,
        metadata_path: Path,
        model_name_or_path: Optional[str] = None,
    ) -> "BeliefEmbeddingIndex":
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Stored BGE embeddings require numpy and sentence-transformers"
            ) from exc

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_model = str(metadata["model"])
        requested_model = model_name_or_path or stored_model
        if requested_model != stored_model:
            raise ValueError(
                f"Embedding model mismatch: cache={stored_model!r}, "
                f"requested={requested_model!r}"
            )
        archive = np.load(embeddings_path, allow_pickle=False)
        embeddings = archive["embeddings"]
        belief_ids = [str(item) for item in metadata.get("belief_ids", [])]
        if len(belief_ids) != len(embeddings):
            raise ValueError("Embedding metadata and matrix row counts do not match")
        return cls(
            model=SentenceTransformer(requested_model),
            model_name=requested_model,
            belief_ids=belief_ids,
            embeddings=embeddings,
            text_hashes={
                str(key): str(value)
                for key, value in (metadata.get("text_hashes") or {}).items()
            },
        )

    def validate_beliefs(self, beliefs: Sequence[Dict[str, Any]]) -> List[str]:
        """Return belief IDs missing from or stale in the persisted cache."""

        stale = []
        for belief in beliefs:
            belief_id = str(belief.get("belief_id", belief.get("lesson_id")) or "")
            text = _embedding_text(belief)
            if belief_id and self.text_hashes.get(belief_id) != _text_sha256(text):
                stale.append(belief_id)
        return stale

    def similarities(self, query: str) -> Dict[str, float]:
        if not self.belief_ids:
            return {}
        query_vector = self.model.encode(
            [query or ""],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        cosine_scores = self.embeddings @ query_vector
        return {
            belief_id: clamp_probability(float(score), default=0.0)
            for belief_id, score in zip(self.belief_ids, cosine_scores)
        }

    def similarities_with_context(
        self,
        query: str,
        context_texts: Dict[str, str],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Score problem descriptions and context conditions with one BGE query."""
        if not self.belief_ids:
            return {}, {}
        with self._embedding_lock:
            query_vector = self.model.encode(
                [query or ""],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            missing_texts = [
                text
                for text in dict.fromkeys(context_texts.values())
                if text and text not in self._text_vector_cache
            ]
            if missing_texts:
                encoded = self.model.encode(
                    missing_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                self._text_vector_cache.update(zip(missing_texts, encoded))

        problem_scores = self.embeddings @ query_vector
        problem_similarities = {
            belief_id: clamp_probability(float(score), default=0.0)
            for belief_id, score in zip(self.belief_ids, problem_scores)
        }
        context_similarities = {}
        for belief_id, text in context_texts.items():
            if not text:
                context_similarities[belief_id] = 1.0
                continue
            vector = self._text_vector_cache.get(text)
            if vector is not None:
                context_similarities[belief_id] = clamp_probability(
                    float(vector @ query_vector), default=0.0
                )
        return problem_similarities, context_similarities


def context_match(required: Iterable[str], observed: Iterable[str]) -> float:
    required_set = {str(item).strip().lower() for item in required if str(item).strip()}
    observed_set = {str(item).strip().lower() for item in observed if str(item).strip()}
    if not required_set:
        return 1.0
    return len(required_set & observed_set) / len(required_set)


def normalize_role(value: Any) -> str:
    role = re.sub(r"\d+$", "", str(value or "").strip())
    return "Orchestrator" if role == "OrchestratorAgent" else role


def normalize_stage(value: Any) -> str:
    stage = str(value or "").strip().lower()
    aliases = {
        "writing": "planning",
        "revision": "planning",
        "debugging": "fix",
        "coding": "generation",
        "evaluation": "coordination",
    }
    return aliases.get(stage, stage)


def select_pipeline_stage(
    agent_role: Any,
    invocation_phase: str,
    *,
    has_runtime_failure: bool = False,
) -> str:
    """Choose the workflow stage from the work being performed."""
    if normalize_role(agent_role) == "Coder" and has_runtime_failure:
        return "fix"
    return "generation" if invocation_phase == "preventative" else "refine"


def embedding_query_for_situation(situation: BeliefSituation) -> str:
    """Return a compact semantic query suited to the current workflow stage.

    Runtime-repair situations often include the topic, audience, storyboard, and
    active-issue state before the actual exception.  That lesson context is
    useful to agents, but it dilutes retrieval of a technical repair belief.
    Exact identifier matching continues to inspect the complete ``problem_text``;
    only the embedding query is narrowed here.
    """
    problem_text = str(situation.problem_text or "").strip()
    if normalize_stage(situation.pipeline_stage) != "fix":
        return problem_text

    marker = "Current exception or failure:"
    marker_index = problem_text.lower().find(marker.lower())
    if marker_index >= 0:
        diagnostic = problem_text[marker_index + len(marker) :].strip()
    else:
        diagnostic = problem_text

    # Error payloads can contain very long tracebacks or renderer output.  The
    # normalized exception and implicated identifiers occur at the start of the
    # payload produced by the MAS error extractor, so retain that focused prefix.
    diagnostic = diagnostic[:2000].strip()
    tags = ", ".join(
        str(tag).strip() for tag in situation.context_tags if str(tag).strip()
    )
    parts = [diagnostic]
    if tags:
        parts.append("Error categories: " + tags)
    return "\n".join(part for part in parts if part).strip()


def exact_error_match(problem_text: str, belief_text: str) -> float:
    """Match concrete runtime identifiers before broad semantic similarity.

    A matching missing name, key, attribute, class, or API token is decisive.
    Matching only an exception family is weak evidence because many unrelated
    repair beliefs mention the same broad exception type.
    """
    problem = str(problem_text or "")
    belief = str(belief_text or "")
    problem_lower = problem.lower()
    belief_lower = belief.lower()

    # Generalize unavailable Manim colour constants beyond the examples stored
    # in a belief (for example CYAN -> MAGENTA or STEELBLUE).  Requiring a
    # recognizable colour-family token avoids treating every uppercase missing
    # name as a colour error.
    missing_name_match = re.search(
        r"name\s+['\"]([^'\"]+)['\"]\s+is\s+not\s+defined",
        problem,
        flags=re.IGNORECASE,
    )
    if missing_name_match:
        missing_name = missing_name_match.group(1).strip().upper()
        colour_tokens = (
            "BLACK",
            "BLUE",
            "BROWN",
            "CYAN",
            "GOLD",
            "GRAY",
            "GREY",
            "GREEN",
            "MAGENTA",
            "MAROON",
            "ORANGE",
            "PINK",
            "PURPLE",
            "RED",
            "TEAL",
            "WHITE",
            "YELLOW",
        )
        colour_belief = any(
            phrase in belief_lower
            for phrase in ("color specification", "colour specification", "hexadecimal")
        )
        if colour_belief and any(token in missing_name for token in colour_tokens):
            return 1.0

    # Custom helper signature errors identify the callable even when the
    # unexpected keyword itself has never appeared in the belief library.
    unexpected_keyword = re.search(
        r"([A-Za-z_][A-Za-z0-9_.]*)\(\)\s+got an unexpected keyword argument\s+['\"]([^'\"]+)['\"]",
        problem,
        flags=re.IGNORECASE,
    )
    if unexpected_keyword:
        callable_name = unexpected_keyword.group(1).split(".")[-1].lower()
        if (
            callable_name
            and re.search(
                r"(?<![A-Za-z0-9_])"
                + re.escape(callable_name)
                + r"(?![A-Za-z0-9_])",
                belief_lower,
            )
            and "typeerror" in belief_lower
        ):
            return 1.0

    if (
        "manimcolor" in problem_lower
        and "get_color" in problem_lower
        and any(
            phrase in belief_lower
            for phrase in ("color specification", "colour specification", "hexadecimal")
        )
    ):
        return 1.0

    identifier_patterns = (
        r"name\s+['\"]([^'\"]+)['\"]\s+is\s+not\s+defined",
        r"keyerror\s*:\s*['\"]([^'\"]+)['\"]",
        r"has\s+no\s+attribute\s+['\"]([^'\"]+)['\"]",
    )
    identifiers = {
        match.group(1).strip().lower()
        for pattern in identifier_patterns
        for match in re.finditer(pattern, problem, flags=re.IGNORECASE)
        if match.group(1).strip()
    }
    # Backtick-quoted API/class names are also intentionally exact.
    identifiers.update(
        match.group(1).strip().lower()
        for match in re.finditer(r"`([^`]+)`", problem)
        if match.group(1).strip()
    )
    if identifiers:
        identifier_match = any(
            re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])",
                belief_lower,
            )
            for identifier in identifiers
        )
        if identifier_match:
            exception_types = {
                item.lower()
                for item in re.findall(
                    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b", problem
                )
            }
            # Full priority requires both the concrete identifier and its
            # exception family. Identifier-only overlap (for example the word
            # "Plane" in a layout belief) is useful but not decisive.
            if exception_types and any(item in belief_lower for item in exception_types):
                return 1.0
            return 0.5
        # Once the runtime supplies a concrete missing name/key/attribute, a
        # belief about a different NameError is not an exact match merely
        # because both texts contain the same exception family.
        return 0.0

    exception_types = {
        item.lower()
        for item in re.findall(
            r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b", problem
        )
    }
    if exception_types and any(item in belief_lower for item in exception_types):
        return 0.25
    return 0.0


class BeliefSelector:
    """Evaluate belief records against one current MAS situation."""

    def __init__(
        self,
        beliefs: Sequence[Dict[str, Any]],
        *,
        parameters: Optional[BBNParameters] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        embedding_index: Optional[BeliefEmbeddingIndex] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.beliefs = list(beliefs)
        self.parameters = parameters or BBNParameters()
        self.similarity_fn = similarity_fn or lexical_similarity
        self.embedding_index = embedding_index
        self.log_path = log_path
        self._log_lock = Lock()

    def select(
        self,
        situation: BeliefSituation,
        *,
        top_k: int = 3,
        threshold: float = 0.0,
        minimum_effectiveness: float = 0.50,
        candidate_limit: int = 20,
    ) -> List[Dict[str, Any]]:
        role = normalize_role(situation.agent_role)
        stage = normalize_stage(situation.pipeline_stage)
        evaluated: List[Dict[str, Any]] = []
        embedding_query = embedding_query_for_situation(situation)
        context_query = embedding_query
        context_texts = {}
        for belief in self.beliefs:
            belief_id = str(
                belief.get("belief_id", belief.get("lesson_id")) or ""
            )
            scope = belief.get("scope") if isinstance(belief.get("scope"), dict) else {}
            conditions = [
                str(item).strip()
                for item in scope.get("context_conditions", [])
                if str(item).strip()
            ]
            context_texts[belief_id] = "\n".join(conditions)

        if (
            self.embedding_index is not None
            and hasattr(self.embedding_index, "similarities_with_context")
        ):
            stored_similarities, semantic_context_similarities = (
                self.embedding_index.similarities_with_context(
                    context_query,
                    context_texts,
                )
            )
        else:
            stored_similarities = (
                self.embedding_index.similarities(embedding_query)
                if self.embedding_index is not None
                else {}
            )
            semantic_context_similarities = {}

        for belief in self.beliefs:
            status = str(belief.get("status") or "active")
            # Probation represents uncertainty, not inapplicability. Let its
            # alpha-beta posterior influence ranking so probationary beliefs can
            # gather prospective evidence in any selection context.
            if status not in {"active", "probation"}:
                continue
            if str(belief.get("belief_type") or "confirmed") == "hypothesis":
                continue

            scope = belief.get("scope") if isinstance(belief.get("scope"), dict) else {}
            roles = {normalize_role(item) for item in scope.get("roles", [])}
            stages = {normalize_stage(item) for item in scope.get("stages", [])}
            # Role scope is an eligibility constraint, not merely weak
            # evidence of applicability. A Coder belief must never enter a
            # ScriptWriter, AnimationPlanner, or Orchestrator prompt. Beliefs
            # without an explicit role remain eligible for legacy libraries.
            if roles and role not in roles:
                continue
            problem_description = str(
                scope.get("problem_description")
                or belief.get("problem_description")
                or belief.get("instruction")
                or ""
            )
            exact_error = exact_error_match(
                situation.problem_text,
                "\n".join(
                    [
                        problem_description,
                        str(belief.get("instruction") or ""),
                        " ".join(str(item) for item in scope.get("context_conditions", [])),
                    ]
                ),
            )
            belief_id = str(
                belief.get("belief_id", belief.get("lesson_id")) or ""
            )
            if belief_id in stored_similarities:
                raw_similarity = stored_similarities[belief_id]
            else:
                raw_similarity = self.similarity_fn(
                    embedding_query, problem_description
                )
            similarity = clamp_probability(raw_similarity, default=0.0)
            exact_context = context_match(
                scope.get("context_conditions", []),
                situation.context_tags,
            )
            semantic_context = semantic_context_similarities.get(belief_id)
            if semantic_context is None:
                context_text = context_texts.get(belief_id, "")
                semantic_context = (
                    lexical_similarity(context_query, context_text)
                    if context_text
                    else 1.0
                )
            context_score = max(
                clamp_probability(exact_context, default=0.0),
                clamp_probability(semantic_context, default=0.0),
            )
            evaluated.append(
                {
                    "belief": belief,
                    "status": status,
                    "problem_similarity": similarity,
                    "role_match": 1.0,
                    "stage_match": 1.0 if not stages or stage in stages else 0.0,
                    "context_match": context_score,
                    "context_match_exact": exact_context,
                    "context_match_semantic": semantic_context,
                    "exact_error_match": exact_error,
                }
            )

        evaluated.sort(
            key=lambda item: (
                -(1 if item["exact_error_match"] >= 1.0 else 0),
                -item["stage_match"],
                -item["exact_error_match"],
                -item["problem_similarity"],
            )
        )
        evaluated = evaluated[: max(0, candidate_limit)]

        results: List[Dict[str, Any]] = []
        for item in evaluated:
            belief = item["belief"]
            applicability = self.parameters.probability(
                ApplicabilityEvidence(
                    role_match=item["role_match"],
                    stage_match=item["stage_match"],
                    problem_match=item["problem_similarity"],
                    context_match=item["context_match"],
                    exact_error_match=item["exact_error_match"],
                )
            )
            effectiveness = posterior_mean(
                float(belief.get("alpha", 2.0)),
                float(belief.get("beta", 2.0)),
            )
            # Historical effectiveness governs library eligibility only.  Once
            # a belief clears the inclusive neutral-prior floor, selection is
            # based entirely on how applicable it is to the current situation.
            # This prevents a common, well-supported layout belief from
            # outranking a less-observed belief that directly matches an error.
            effectiveness_eligible = effectiveness >= minimum_effectiveness
            selection_score = applicability
            result = {
                "belief_id": belief.get("belief_id", belief.get("lesson_id")),
                "instruction": str(belief.get("instruction") or ""),
                "status": item["status"],
                "p_applicable": round(applicability, 6),
                "p_effective": round(effectiveness, 6),
                # Keep the legacy field for downstream compatibility.  It now
                # records the applicability-only selection score.
                "usefulness": round(selection_score, 6),
                "selection_score": round(selection_score, 6),
                "effectiveness_eligible": effectiveness_eligible,
                "problem_similarity": round(item["problem_similarity"], 6),
                "role_match": round(item["role_match"], 6),
                "stage_match": round(item["stage_match"], 6),
                "context_match": round(item["context_match"], 6),
                "context_match_exact": round(item["context_match_exact"], 6),
                "context_match_semantic": round(
                    item["context_match_semantic"], 6
                ),
                "exact_error_match": round(item["exact_error_match"], 6),
                "selected": effectiveness_eligible and selection_score >= threshold,
            }
            results.append(result)

        results.sort(key=lambda item: (-item["usefulness"], str(item["belief_id"])))
        eligible = [item for item in results if item["selected"]]
        exact = [item for item in eligible if item["exact_error_match"] >= 1.0]
        stage_matched = [
            item
            for item in eligible
            if item["exact_error_match"] < 1.0 and item["stage_match"] >= 1.0
        ]
        cross_stage = [
            item
            for item in eligible
            if item["exact_error_match"] < 1.0 and item["stage_match"] < 1.0
        ]
        # Exact signature matches come first, followed by current-stage
        # beliefs. Cross-stage beliefs only fill otherwise unused slots.
        # Historical ``timing`` metadata is deliberately ignored.
        if stage == "fix":
            # Runtime embeddings are useful for candidate discovery but were
            # not calibrated well enough to distinguish technical repairs in
            # the completed 30-topic replay: irrelevant candidates routinely
            # had high combined applicability.  Inject one decisive structured
            # match.  Otherwise permit one exceptionally close semantic match
            # only when its raw problem similarity is >= 0.90 and clearly
            # separated from the next current-stage candidate.  In the replay,
            # the highest unmatched/irrelevant raw similarity was 0.823.
            if exact:
                selected = exact[:1]
            else:
                semantic_fix = sorted(
                    stage_matched,
                    key=lambda item: (
                        -item["problem_similarity"],
                        str(item["belief_id"]),
                    ),
                )
                best = semantic_fix[0] if semantic_fix else None
                runner_up = semantic_fix[1] if len(semantic_fix) > 1 else None
                margin = (
                    best["problem_similarity"] - runner_up["problem_similarity"]
                    if best is not None and runner_up is not None
                    else 1.0
                )
                exact_context_fix = [
                    item
                    for item in semantic_fix
                    if item["context_match_exact"] >= 1.0
                    and item["problem_similarity"] >= 0.70
                ]
                if exact_context_fix:
                    selected = exact_context_fix[:1]
                else:
                    selected = (
                        [best]
                        if best is not None
                        and best["problem_similarity"] >= 0.90
                        and margin >= 0.05
                        else []
                    )
        else:
            selected = (exact + stage_matched + cross_stage)[: max(0, top_k)]
        selected_ids = {item["belief_id"] for item in selected}
        for item in results:
            item["selected"] = item["belief_id"] in selected_ids
        self._log_selection(
            situation,
            results,
            top_k=top_k,
            threshold=threshold,
            minimum_effectiveness=minimum_effectiveness,
            candidate_limit=candidate_limit,
        )
        return selected

    def _log_selection(
        self,
        situation: BeliefSituation,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: int,
        threshold: float,
        minimum_effectiveness: float,
        candidate_limit: int,
    ) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "situation": asdict(situation),
            "bbn_parameters": asdict(self.parameters),
            "decision_policy": {
                "top_k": top_k,
                "threshold": threshold,
                "minimum_effectiveness": minimum_effectiveness,
                "ranking": "applicability_only",
                "fix_selection": (
                    "top_1_structured_else_exact_context_else_"
                    "semantic_0.90_with_0.05_margin"
                ),
                "candidate_limit": candidate_limit,
            },
            "candidates": list(candidates),
            "selected_belief_ids": [
                item["belief_id"] for item in candidates if item.get("selected")
            ],
        }
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_selected_beliefs(selected: Sequence[Dict[str, Any]]) -> str:
    if not selected:
        return ""
    lines = [
        "Contextually selected reusable beliefs:",
        "Apply only when the current shared state confirms their stated conditions.",
    ]
    for index, item in enumerate(selected, start=1):
        lines.append(
            f"{index}. [{item['belief_id']} | useful={item['usefulness']:.3f} | "
            f"applicable={item['p_applicable']:.3f} | effective={item['p_effective']:.3f}] "
            f"{item['instruction']}"
        )
    return "\n".join(lines)
