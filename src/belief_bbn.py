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


@dataclass
class BBNParameters:
    """Logistic CPD for the latent Applicability node.

    This is a compact conditional probability distribution for:
    P(Applicability | RoleMatch, StageMatch, ProblemMatch, ContextMatch).
    """

    intercept: float = -4.0
    role_match: float = 1.5
    stage_match: float = 1.2
    problem_match: float = 2.0
    context_match: float = 1.5
    version: int = 1

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
        # Cosine for normalized vectors lies in [-1, 1]. Map it to [0, 1]
        # pending project-specific calibration against labelled pairs.
        cosine = float(vectors[0] @ vectors[1])
        return (cosine + 1.0) / 2.0

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
            belief_id: clamp_probability((float(score) + 1.0) / 2.0)
            for belief_id, score in zip(self.belief_ids, cosine_scores)
        }


def context_match(required: Iterable[str], observed: Iterable[str]) -> float:
    required_set = {str(item).strip().lower() for item in required if str(item).strip()}
    observed_set = {str(item).strip().lower() for item in observed if str(item).strip()}
    if not required_set:
        return 1.0
    return len(required_set & observed_set) / len(required_set)


def normalize_role(value: Any) -> str:
    role = re.sub(r"\d+$", "", str(value or "").strip())
    return "Orchestrator" if role == "OrchestratorAgent" else role


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
        threshold: float = 0.60,
        candidate_limit: int = 20,
    ) -> List[Dict[str, Any]]:
        role = normalize_role(situation.agent_role)
        stage = situation.pipeline_stage.strip().lower()
        timing = situation.timing.strip().lower()
        evaluated: List[Dict[str, Any]] = []
        stored_similarities = (
            self.embedding_index.similarities(situation.problem_text)
            if self.embedding_index is not None
            else {}
        )

        for belief in self.beliefs:
            if str(belief.get("status") or "active") != "active":
                continue
            if str(belief.get("belief_type") or "confirmed") == "hypothesis":
                continue

            scope = belief.get("scope") if isinstance(belief.get("scope"), dict) else {}
            roles = {normalize_role(item) for item in scope.get("roles", [])}
            stages = {str(item).strip().lower() for item in scope.get("stages", [])}
            belief_timing = str(belief.get("timing") or "both").lower()
            if roles and role not in roles:
                continue
            if stages and stage not in stages:
                continue
            if timing != "both" and belief_timing not in {timing, "both"}:
                continue

            problem_description = str(
                scope.get("problem_description")
                or belief.get("problem_description")
                or belief.get("instruction")
                or ""
            )
            belief_id = str(
                belief.get("belief_id", belief.get("lesson_id")) or ""
            )
            if belief_id in stored_similarities:
                raw_similarity = stored_similarities[belief_id]
            else:
                raw_similarity = self.similarity_fn(
                    situation.problem_text, problem_description
                )
            similarity = clamp_probability(raw_similarity, default=0.0)
            evaluated.append(
                {
                    "belief": belief,
                    "problem_similarity": similarity,
                    "role_match": 1.0 if not roles or role in roles else 0.0,
                    "stage_match": 1.0 if not stages or stage in stages else 0.0,
                    "context_match": context_match(
                        scope.get("context_conditions", []),
                        situation.context_tags,
                    ),
                }
            )

        evaluated.sort(key=lambda item: item["problem_similarity"], reverse=True)
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
                )
            )
            effectiveness = posterior_mean(
                float(belief.get("alpha", 2.0)),
                float(belief.get("beta", 2.0)),
            )
            usefulness = applicability * effectiveness
            result = {
                "belief_id": belief.get("belief_id", belief.get("lesson_id")),
                "instruction": str(belief.get("instruction") or ""),
                "p_applicable": round(applicability, 6),
                "p_effective": round(effectiveness, 6),
                "usefulness": round(usefulness, 6),
                "problem_similarity": round(item["problem_similarity"], 6),
                "context_match": round(item["context_match"], 6),
                "selected": usefulness >= threshold,
            }
            results.append(result)

        results.sort(key=lambda item: (-item["usefulness"], str(item["belief_id"])))
        selected = [item for item in results if item["selected"]][: max(0, top_k)]
        selected_ids = {item["belief_id"] for item in selected}
        for item in results:
            item["selected"] = item["belief_id"] in selected_ids
        self._log_selection(situation, results)
        return selected

    def _log_selection(
        self,
        situation: BeliefSituation,
        candidates: Sequence[Dict[str, Any]],
    ) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "situation": asdict(situation),
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
