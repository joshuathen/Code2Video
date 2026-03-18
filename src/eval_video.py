import argparse
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval_AES import VideoEvaluator
from eval_TQ import format_evaluation_report, load_questions_from_json, run_one_concept
from gpt_request import request_gemini_with_video


DEFAULT_QUESTIONS_JSON = Path(__file__).resolve().parent.parent / "json_files" / "questions_by_topic_10.json"


def _normalize_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def resolve_tq_concept(topic: str, available_concepts: Iterable[str]) -> str:
    concepts = list(available_concepts)
    if topic in concepts:
        return topic

    lowered_lookup = {concept.lower(): concept for concept in concepts}
    if topic.lower() in lowered_lookup:
        return lowered_lookup[topic.lower()]

    normalized_topic = _normalize_topic(topic)
    normalized_lookup: Dict[str, List[str]] = {}
    for concept in concepts:
        normalized_lookup.setdefault(_normalize_topic(concept), []).append(concept)

    normalized_exact = normalized_lookup.get(normalized_topic, [])
    if len(normalized_exact) == 1:
        return normalized_exact[0]
    if len(normalized_exact) > 1:
        raise ValueError(
            "Topic matches multiple TQ concepts after normalization: "
            + ", ".join(sorted(normalized_exact))
            + ". Pass --tq-concept explicitly."
        )

    containment_matches = [
        concept
        for concept in concepts
        if normalized_topic in _normalize_topic(concept) or _normalize_topic(concept) in normalized_topic
    ]
    if len(containment_matches) == 1:
        return containment_matches[0]
    if len(containment_matches) > 1:
        raise ValueError(
            "Topic is ambiguous across multiple TQ concepts: "
            + ", ".join(sorted(containment_matches))
            + ". Pass --tq-concept explicitly."
        )

    close_matches = get_close_matches(topic, concepts, n=5, cutoff=0.45)
    if not close_matches:
        close_matches = get_close_matches(normalized_topic, list(normalized_lookup.keys()), n=5, cutoff=0.45)
        close_matches = [normalized_lookup[match][0] for match in close_matches]

    suggestion_text = ", ".join(close_matches[:5]) if close_matches else "no close matches found"
    raise ValueError(
        f"Could not resolve topic '{topic}' to a TQ concept. Closest matches: {suggestion_text}. "
        "Pass --tq-concept explicitly."
    )


def evaluate_video(
    video_path: Path,
    topic: str,
    questions_json: Path,
    tq_concept: str | None = None,
    per_question_workers: int = 5,
) -> Dict[str, Any]:
    concept_questions = load_questions_from_json(str(questions_json))
    resolved_tq_concept = tq_concept or resolve_tq_concept(topic, concept_questions.keys())

    result: Dict[str, Any] = {
        "video_path": str(video_path.resolve()),
        "topic": topic,
        "questions_json": str(questions_json.resolve()),
        "tq_concept": resolved_tq_concept,
        "aes": {"ok": False, "result": None, "error": None},
        "tq": {"ok": False, "result": None, "report": None, "error": None},
    }

    evaluator = VideoEvaluator(request_gemini_with_video)
    try:
        aes_result = evaluator.evaluate_video(video_path=str(video_path), knowledge_point=topic)
        result["aes"]["ok"] = True
        result["aes"]["result"] = _to_jsonable(aes_result)
    except Exception as exc:
        result["aes"]["error"] = str(exc)

    try:
        questions = concept_questions[resolved_tq_concept]
        tq_result = run_one_concept(
            concept=resolved_tq_concept,
            questions=questions,
            video_path=str(video_path),
            per_question_workers=per_question_workers,
        )
        result["tq"]["ok"] = True
        result["tq"]["result"] = _to_jsonable(tq_result)
        result["tq"]["report"] = format_evaluation_report([tq_result])
    except Exception as exc:
        result["tq"]["error"] = str(exc)

    result["success"] = bool(result["aes"]["ok"] and result["tq"]["ok"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run both AES and TQ evaluation on a single generated video.")
    parser.add_argument("--video-path", required=True, help="Path to the generated video file.")
    parser.add_argument("--topic", required=True, help="Knowledge topic for AES and TQ concept resolution.")
    parser.add_argument(
        "--questions-json",
        default=str(DEFAULT_QUESTIONS_JSON),
        help="Path to questions_by_topic_10.json. Defaults to the repo copy.",
    )
    parser.add_argument(
        "--tq-concept",
        default=None,
        help="Optional explicit TQ concept key. Use this if topic auto-resolution is ambiguous.",
    )
    parser.add_argument(
        "--per-question-workers",
        type=int,
        default=5,
        help="Parallel workers used inside each TQ evaluation stage.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for combined JSON output. Defaults to <video_stem>_eval.json next to the video.",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    questions_json = Path(args.questions_json).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else video_path.with_name(f"{video_path.stem}_eval.json")
    )

    if not video_path.exists():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        return 1
    if not questions_json.exists():
        print(f"Questions JSON not found: {questions_json}", file=sys.stderr)
        return 1

    try:
        combined_result = evaluate_video(
            video_path=video_path,
            topic=args.topic,
            questions_json=questions_json,
            tq_concept=args.tq_concept,
            per_question_workers=args.per_question_workers,
        )
    except Exception as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(json.dumps(_to_jsonable(combined_result), indent=2), encoding="utf-8")

    print(f"Saved combined evaluation: {output_path}")
    print(f"TQ concept: {combined_result['tq_concept']}")

    aes_result = combined_result["aes"]["result"]
    if aes_result:
        print(f"AES overall score: {aes_result['overall_score']}")
    else:
        print(f"AES failed: {combined_result['aes']['error']}")

    tq_result = combined_result["tq"]["result"]
    if tq_result:
        print(f"TQ learning gain: {tq_result['learning_gain']}")
        print(f"TQ post-video score: {tq_result['post_video_score']}")
    else:
        print(f"TQ failed: {combined_result['tq']['error']}")

    return 0 if combined_result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
