import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Keep both the repo root and src on sys.path so this launcher works the same
# whether it is invoked from the repo root or directly via an absolute path.
for candidate in (SRC_DIR, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


AGENT_API_CHOICES = ["gpt-41", "claude", "gpt-5", "gpt-4o", "gpt-o4mini", "Gemini"]

# Shared launcher defaults, kept explicit here so the single entrypoint stays
# readable and the agent/MAS runners can inherit the same baseline behavior.
DEFAULT_API = "Gemini"
DEFAULT_USE_FEEDBACK = False
DEFAULT_USE_ASSETS = False
DEFAULT_MAX_CODE_TOKEN_LENGTH = 10000
DEFAULT_MAX_FIX_BUG_TRIES = 10
DEFAULT_MAX_REGENERATE_TRIES = 10
DEFAULT_MAX_FEEDBACK_GEN_CODE_TRIES = 3
DEFAULT_MAX_MLLM_FIX_BUGS_TRIES = 3
DEFAULT_FEEDBACK_ROUNDS = 2

# MAS-only defaults that do not have a direct single-run analogue in agent.py.
DEFAULT_OUTLINE_DURATION_MINUTES = 5
DEFAULT_OUTLINE_MODEL = "gemini-3-flash-preview"
DEFAULT_MAS_MAX_TURNS = 3
DEFAULT_MAS_MAX_RETRIES = 3
DEFAULT_KNOWLEDGE_FILE = "long_video_topics_list.json"
DEFAULT_PER_QUESTION_WORKERS = 5


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    parser.add_argument(f"--{name}", action="store_true", default=default)
    parser.add_argument(f"--no_{name}", action="store_false", dest=name)


def _load_iconfinder_api_key() -> str:
    cfg_path = SRC_DIR / "api_config.json"
    if not cfg_path.exists():
        return ""

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return str(cfg.get("iconfinder", {}).get("api_key", "") or "")


def _resolve_with_fallback(value: int, fallback: int) -> int:
    return fallback if value is None else value


def _resolve_repo_path(path_or_name: Union[str, Path], default_dir: Optional[Path] = None) -> Path:
    candidate = Path(path_or_name).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if default_dir is not None:
        default_candidate = default_dir / candidate
        repo_candidate = PROJECT_ROOT / candidate
        if default_candidate.exists() or not repo_candidate.exists():
            return default_candidate.resolve()
        return repo_candidate.resolve()

    return (PROJECT_ROOT / candidate).resolve()


def load_knowledge_points(
    knowledge_file: Union[str, Path] = DEFAULT_KNOWLEDGE_FILE,
    max_topics: Optional[int] = None,
) -> List[str]:
    knowledge_path = _resolve_repo_path(knowledge_file, PROJECT_ROOT / "json_files")
    with knowledge_path.open("r", encoding="utf-8") as f:
        knowledge_points = json.load(f)

    if not isinstance(knowledge_points, list):
        raise ValueError(f"Knowledge file must contain a JSON list: {knowledge_path}")

    normalized_points = [str(item) for item in knowledge_points]
    if max_topics is not None and max_topics >= 0:
        normalized_points = normalized_points[:max_topics]
    return normalized_points


def _resolve_questions_json_path(questions_json: Optional[Union[str, Path]]) -> Path:
    if questions_json is None:
        from eval_video import DEFAULT_QUESTIONS_JSON

        resolved = DEFAULT_QUESTIONS_JSON.resolve()
    else:
        resolved = _resolve_repo_path(questions_json, PROJECT_ROOT / "json_files")

    if not resolved.exists():
        raise FileNotFoundError(f"Questions JSON not found: {resolved}")
    return resolved


def _ensure_agent_iconfinder_api_key(args: argparse.Namespace) -> str:
    cached_key = getattr(args, "_iconfinder_api_key", None)
    if cached_key is None:
        cached_key = _load_iconfinder_api_key()
        setattr(args, "_iconfinder_api_key", cached_key)
    return cached_key


def _report_agent_asset_config(args: argparse.Namespace) -> None:
    if not args.use_assets:
        return

    iconfinder_api_key = _ensure_agent_iconfinder_api_key(args)
    if iconfinder_api_key:
        print("Iconfinder API key loaded from config.")
    else:
        print("WARNING: Iconfinder API key not found in config file. Asset enhancement may be limited.")


def _print_single_run_summary(
    *,
    topic: str,
    duration_minutes: float,
    total_tokens: int,
    was_successful: bool,
) -> None:
    status_icon = "✅" if was_successful else "❌"
    print(
        f"{status_icon} Knowledge topic '{topic}' processed. "
        f"Cost Time: {duration_minutes:.2f} minutes, Tokens used: {total_tokens}"
    )
    if was_successful:
        print("\n" + "=" * 50)
        print("   Total knowledge points: 1")
        print("   Successfully processed: 1 (100.0%)")
        print(f"   Average duration [min]: {duration_minutes:.2f} minutes/knowledge point")
        print(f"   Average token consumption: {total_tokens:,.0f} tokens/knowledge point")
        print("=" * 50)
    else:
        print("\nAll knowledge points failed, cannot calculate average.")


def _resolve_versioned_directory(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    suffix = time.strftime("%Y%m%d_%H%M%S")
    candidate = base_dir.parent / f"{base_dir.name}_{suffix}"
    attempt = 1
    while candidate.exists():
        candidate = base_dir.parent / f"{base_dir.name}_{suffix}_{attempt}"
        attempt += 1
    return candidate


def _resolve_agent_output_root(args: argparse.Namespace) -> Path:
    cached_root = getattr(args, "_resolved_agent_output_root", None)
    if cached_root is not None:
        return Path(cached_root)

    from agent import get_api_and_output

    _, folder_name = get_api_and_output(args.API)
    requested_root = PROJECT_ROOT / "CASES" / f"{args.folder_prefix}_{folder_name}"
    resolved_root = _resolve_versioned_directory(requested_root)
    if resolved_root != requested_root:
        print(f"Agent output root exists, using versioned folder: {resolved_root}")

    setattr(args, "_resolved_agent_output_root", resolved_root)
    return resolved_root


def _resolve_mas_pipeline_output_root(args: argparse.Namespace) -> Path:
    cached_root = getattr(args, "_resolved_mas_pipeline_output_root", None)
    if cached_root is not None:
        return Path(cached_root)

    logs_root = PROJECT_ROOT / "mas_logs"
    suffix = time.strftime("%Y%m%d_%H%M%S")
    requested_root = logs_root / f"pipeline_{suffix}"
    resolved_root = requested_root
    attempt = 1
    while resolved_root.exists():
        resolved_root = logs_root / f"pipeline_{suffix}_{attempt}"
        attempt += 1

    print(f"MAS pipeline output root: {resolved_root}")
    setattr(args, "_resolved_mas_pipeline_output_root", resolved_root)
    return resolved_root


def _print_batch_pipeline_summary(runner_name: str, results: List[Dict[str, Any]]) -> None:
    total_requested = len(results)
    generated_runs = [item for item in results if (item.get("generation") or {}).get("success")]
    successful_evaluations = [item for item in results if (item.get("evaluation") or {}).get("success")]
    successful_pipelines = [item for item in results if item.get("success")]

    print("\n" + "=" * 50)
    print(f"   Runner: {runner_name}")
    print(f"   Total knowledge points: {total_requested}")
    print(f"   Successfully generated: {len(generated_runs)} ({len(generated_runs) / total_requested * 100:.1f}%)")
    print(
        f"   Successfully evaluated: {len(successful_evaluations)} "
        f"({len(successful_evaluations) / total_requested * 100:.1f}%)"
    )
    print(
        f"   End-to-end success: {len(successful_pipelines)} "
        f"({len(successful_pipelines) / total_requested * 100:.1f}%)"
    )

    if generated_runs:
        total_duration = sum(float(item["generation"]["duration_minutes"]) for item in generated_runs)
        total_tokens = sum(int(item["generation"]["total_tokens"]) for item in generated_runs)
        print(f"   Average duration [min]: {total_duration / len(generated_runs):.2f} minutes/knowledge point")
        print(f"   Average token consumption: {total_tokens / len(generated_runs):,.0f} tokens/knowledge point")
    print("=" * 50)


def _build_paper_style_row(results: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    completed_runs = []
    for item in results:
        generation = item.get("generation") or {}
        evaluation = item.get("evaluation") or {}
        combined_result = evaluation.get("result") or {}
        aes_result = ((combined_result.get("aes") or {}).get("result") or {})
        tq_result = ((combined_result.get("tq") or {}).get("result") or {})

        if not (
            generation.get("success")
            and evaluation.get("success")
            and aes_result
            and tq_result
        ):
            continue

        completed_runs.append(
            {
                "duration_minutes": float(generation["duration_minutes"]),
                "total_tokens": float(generation["total_tokens"]),
                "element_layout": float(aes_result["element_layout"]) / 20.0 * 100.0,
                "attractiveness": float(aes_result["attractiveness"]) / 20.0 * 100.0,
                "logic_flow": float(aes_result["logic_flow"]) / 20.0 * 100.0,
                "visual_consistency": float(aes_result["visual_consistency"]) / 20.0 * 100.0,
                "accuracy_depth": float(aes_result["accuracy_depth"]) / 20.0 * 100.0,
                "avg": float(aes_result["overall_score"]),
                "quiz": float(tq_result["learning_gain"]) * 100.0,
            }
        )

    if not completed_runs:
        return None

    sample_size = len(completed_runs)

    def _mean(key: str) -> float:
        return sum(item[key] for item in completed_runs) / sample_size

    return {
        "sample_size": float(sample_size),
        "time_minutes": _mean("duration_minutes"),
        "token_k": _mean("total_tokens") / 1000.0,
        "EL": _mean("element_layout"),
        "AT": _mean("attractiveness"),
        "LF": _mean("logic_flow"),
        "VC": _mean("visual_consistency"),
        "AD": _mean("accuracy_depth"),
        "Avg": _mean("avg"),
        "Quiz": _mean("quiz"),
    }


def _pipeline_output_root(results: List[Dict[str, Any]]) -> Optional[Path]:
    generated_dirs = [
        Path(item["generation"]["output_dir"]).resolve()
        for item in results
        if (item.get("generation") or {}).get("output_dir")
    ]
    if not generated_dirs:
        return None
    return generated_dirs[0].parent


def _write_batch_pipeline_summary(runner_name: str, results: List[Dict[str, Any]]) -> None:
    paper_row = _build_paper_style_row(results)
    if paper_row is None:
        print("No complete AES+TQ evaluation results available for a paper-style aggregate row.")
        return

    output_root = _pipeline_output_root(results)
    method_label = output_root.name if output_root is not None else runner_name

    print("\nPaper-style aggregate row")
    print("   Time [min] | Token [K] | EL | AT | LF | VC | AD | Avg | Quiz")
    print(
        "   "
        f"{paper_row['time_minutes']:.1f} | "
        f"{paper_row['token_k']:.1f} | "
        f"{paper_row['EL']:.1f} | "
        f"{paper_row['AT']:.1f} | "
        f"{paper_row['LF']:.1f} | "
        f"{paper_row['VC']:.1f} | "
        f"{paper_row['AD']:.1f} | "
        f"{paper_row['Avg']:.1f} | "
        f"{paper_row['Quiz']:.1f}"
    )

    if output_root is None:
        return

    markdown_row = (
        f"| {method_label} | "
        f"{paper_row['time_minutes']:.1f} | "
        f"{paper_row['token_k']:.1f} | "
        f"{paper_row['EL']:.1f} | "
        f"{paper_row['AT']:.1f} | "
        f"{paper_row['LF']:.1f} | "
        f"{paper_row['VC']:.1f} | "
        f"{paper_row['AD']:.1f} | "
        f"{paper_row['Avg']:.1f} | "
        f"{paper_row['Quiz']:.1f} |"
    )
    summary_payload = {
        "runner": runner_name,
        "method_label": method_label,
        "sample_size": int(paper_row["sample_size"]),
        "paper_row": {
            key: round(value, 1) if key != "sample_size" else int(value)
            for key, value in paper_row.items()
        },
        "markdown_row": markdown_row,
    }

    summary_json_path = output_root / "pipeline_summary.json"
    summary_md_path = output_root / "pipeline_summary.md"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        "\n".join(
            [
                "# Pipeline Summary",
                "",
                f"- Runner: {runner_name}",
                f"- Method label: {method_label}",
                f"- Samples: {int(paper_row['sample_size'])}",
                "",
                "| Method | Time | Token (K) | EL | AT | LF | VC | AD | Avg | Quiz |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                markdown_row,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved pipeline summary: {summary_json_path}")
    print(f"Saved pipeline summary table: {summary_md_path}")


def _build_agent_generation_result(args: argparse.Namespace, idx: int) -> Dict[str, Any]:
    from agent import RunConfig, TeachingVideoAgent, get_api_and_output

    api, _ = get_api_and_output(args.API)
    folder = _resolve_agent_output_root(args)

    cfg = RunConfig(
        api=api,
        iconfinder_api_key=_ensure_agent_iconfinder_api_key(args),
        use_feedback=args.use_feedback,
        use_assets=args.use_assets,
        max_code_token_length=args.max_code_token_length,
        max_fix_bug_tries=args.max_fix_bug_tries,
        max_regenerate_tries=args.max_regenerate_tries,
        max_feedback_gen_code_tries=args.max_feedback_gen_code_tries,
        max_mllm_fix_bugs_tries=args.max_mllm_fix_bugs_tries,
        feedback_rounds=args.feedback_rounds,
    )

    print(f"Running agent flow for knowledge topic: {args.knowledge_point}")
    start_time = time.time()
    agent = TeachingVideoAgent(
        idx=idx,
        knowledge_point=args.knowledge_point,
        folder=folder,
        cfg=cfg,
    )
    final_video_path = agent.GENERATE_VIDEO()
    duration_minutes = (time.time() - start_time) / 60
    total_tokens = int(agent.token_usage.get("total_tokens", 0) or 0)

    return {
        "runner": "agent",
        "topic": args.knowledge_point,
        "generated_topic": args.knowledge_point,
        "output_dir": str(agent.output_dir.resolve()),
        "final_video_path": str(Path(final_video_path).resolve()) if final_video_path else None,
        "duration_minutes": duration_minutes,
        "total_tokens": total_tokens,
        "success": bool(final_video_path),
        "token_usage_summary": agent.token_usage_summary,
    }


def _build_mas_generation_result(args: argparse.Namespace, idx: int) -> Dict[str, Any]:
    from mas import (
        MASTokenTracker,
        MASRunConfig,
        _final_video_output_path,
        _format_token_usage_summary,
        _total_tokens_from_summary,
        build_video_state_from_outline,
        generate_outline_with_code2video_stage1,
        run_mas_for_video_state,
    )

    print(f"Running MAS flow for knowledge topic: {args.knowledge_point}")
    start_time = time.time()
    token_tracker = MASTokenTracker()
    outline_max_regenerate_tries = _resolve_with_fallback(
        args.outline_max_regenerate_tries,
        args.max_regenerate_tries,
    )
    coder_fix_attempts = _resolve_with_fallback(
        args.coder_fix_attempts,
        args.max_fix_bug_tries,
    )
    final_render_fix_attempts = _resolve_with_fallback(
        args.final_render_fix_attempts,
        args.max_fix_bug_tries,
    )
    coder_regenerate_tries = _resolve_with_fallback(
        args.coder_regenerate_tries,
        args.max_regenerate_tries,
    )

    generated_outline = generate_outline_with_code2video_stage1(
        knowledge_point=args.knowledge_point,
        duration_minutes=args.outline_duration_minutes,
        max_regenerate_tries=outline_max_regenerate_tries,
        model=args.outline_model,
        token_tracker=token_tracker,
    )

    video_state = build_video_state_from_outline(generated_outline)
    section_parallel_workers = args.section_parallel_workers or len(video_state.section_ids())
    logs_root = getattr(args, "_resolved_mas_pipeline_output_root", None)
    if logs_root is None:
        logs_root = PROJECT_ROOT / "mas_logs"
    else:
        logs_root = Path(logs_root)

    cfg = MASRunConfig(
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        coder_fix_attempts=coder_fix_attempts,
        final_render_fix_attempts=final_render_fix_attempts,
        coder_regenerate_tries=coder_regenerate_tries,
        max_code_token_length=args.max_code_token_length,
        worker_model=args.worker_model,
        orchestrator_model=args.orchestrator_model,
        section_parallel_workers=section_parallel_workers,
        clear_logs=args.clear_logs,
        case_index=args.case_index,
        iconfinder_api_key=_load_iconfinder_api_key(),
    )

    final_video_state = run_mas_for_video_state(
        video_state=video_state,
        logs_root=logs_root,
        cfg=cfg,
        token_tracker=token_tracker,
    )

    final_video_path = _final_video_output_path(final_video_state)
    duration_minutes = (time.time() - start_time) / 60
    total_tokens = _total_tokens_from_summary(final_video_state.token_usage)

    return {
        "runner": "mas",
        "topic": args.knowledge_point,
        "generated_topic": generated_outline.topic,
        "output_dir": str(Path(final_video_state.run_output_dir).resolve()),
        "final_video_path": str(final_video_path.resolve()) if final_video_path else None,
        "duration_minutes": duration_minutes,
        "total_tokens": total_tokens,
        "success": bool(final_video_path and final_video_path.exists()),
        "token_usage_summary": final_video_state.token_usage,
        "token_usage_summary_text": _format_token_usage_summary(final_video_state.token_usage),
        "section_count": len(final_video_state.section_ids()),
    }


def _evaluate_generated_video(
    *,
    video_path: Union[str, Path],
    topic: str,
    questions_json: Path,
    per_question_workers: int,
) -> Dict[str, Any]:
    from eval_video import evaluate_video

    resolved_video_path = Path(video_path).expanduser().resolve()
    combined_result = evaluate_video(
        video_path=resolved_video_path,
        topic=topic,
        questions_json=questions_json,
        per_question_workers=per_question_workers,
    )
    output_path = resolved_video_path.with_name(f"{resolved_video_path.stem}_eval.json")
    output_path.write_text(json.dumps(combined_result, indent=2), encoding="utf-8")

    return {
        "success": bool(combined_result.get("success")),
        "output_path": str(output_path),
        "result": combined_result,
    }


def _clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def _resolve_mas_case_index(base_case_index: Optional[int], idx: int) -> int:
    if base_case_index is None:
        return idx
    return base_case_index + idx


def _run_generation_and_evaluation_pipeline(
    *,
    runner_name: str,
    knowledge_points: List[str],
    base_args: argparse.Namespace,
    generation_fn: Callable[[argparse.Namespace, int], Dict[str, Any]],
    questions_json: Path,
    per_question_workers: int,
) -> List[Dict[str, Any]]:
    if not knowledge_points:
        print(f"No knowledge points selected for the {runner_name} pipeline.")
        return []

    results: List[Dict[str, Any]] = []
    total_topics = len(knowledge_points)

    for idx, knowledge_point in enumerate(knowledge_points):
        print("\n" + "=" * 80)
        print(f"[{idx + 1}/{total_topics}] Running {runner_name} pipeline for: {knowledge_point}")
        print("=" * 80)

        topic_args = _clone_args(base_args)
        topic_args.knowledge_point = knowledge_point
        if runner_name == "mas":
            topic_args.case_index = _resolve_mas_case_index(getattr(base_args, "case_index", None), idx)

        try:
            generation_result = generation_fn(topic_args, idx)
            print(f"Run outputs written to: {generation_result['output_dir']}")
            if generation_result.get("final_video_path"):
                print(f"Final video: {generation_result['final_video_path']}")

            evaluation_result = None
            if generation_result["success"] and generation_result.get("final_video_path"):
                evaluation_result = _evaluate_generated_video(
                    video_path=generation_result["final_video_path"],
                    topic=knowledge_point,
                    questions_json=questions_json,
                    per_question_workers=per_question_workers,
                )
                print(f"Saved evaluation: {evaluation_result['output_path']}")
            else:
                print("Skipping evaluation because no final video was produced.")

            pipeline_result = {
                "runner": runner_name,
                "topic": knowledge_point,
                "generation": generation_result,
                "evaluation": evaluation_result,
                "success": bool(generation_result["success"] and evaluation_result and evaluation_result["success"]),
            }
            results.append(pipeline_result)
        except Exception as exc:
            print(f"❌ {runner_name} pipeline failed for '{knowledge_point}': {exc}")
            results.append(
                {
                    "runner": runner_name,
                    "topic": knowledge_point,
                    "generation": None,
                    "evaluation": None,
                    "success": False,
                    "error": str(exc),
                }
            )

    _print_batch_pipeline_summary(runner_name, results)
    _write_batch_pipeline_summary(runner_name, results)
    return results


def run_agent_generation_and_evaluation_pipeline(
    *,
    knowledge_file: Union[str, Path] = DEFAULT_KNOWLEDGE_FILE,
    max_topics: Optional[int] = None,
    questions_json: Optional[Union[str, Path]] = None,
    per_question_workers: int = DEFAULT_PER_QUESTION_WORKERS,
    folder_prefix: str = "TEST",
    api_name: str = DEFAULT_API,
    use_feedback: bool = DEFAULT_USE_FEEDBACK,
    use_assets: bool = DEFAULT_USE_ASSETS,
    max_code_token_length: int = DEFAULT_MAX_CODE_TOKEN_LENGTH,
    max_fix_bug_tries: int = DEFAULT_MAX_FIX_BUG_TRIES,
    max_regenerate_tries: int = DEFAULT_MAX_REGENERATE_TRIES,
    max_feedback_gen_code_tries: int = DEFAULT_MAX_FEEDBACK_GEN_CODE_TRIES,
    max_mllm_fix_bugs_tries: int = DEFAULT_MAX_MLLM_FIX_BUGS_TRIES,
    feedback_rounds: int = DEFAULT_FEEDBACK_ROUNDS,
) -> List[Dict[str, Any]]:
    """Run generation plus evaluation for the first N topics, or all topics when max_topics is None/-1."""
    knowledge_points = load_knowledge_points(knowledge_file=knowledge_file, max_topics=max_topics)
    questions_json_path = _resolve_questions_json_path(questions_json)

    base_args = argparse.Namespace(
        API=api_name,
        folder_prefix=folder_prefix,
        max_code_token_length=max_code_token_length,
        use_feedback=use_feedback,
        use_assets=use_assets,
        max_fix_bug_tries=max_fix_bug_tries,
        max_regenerate_tries=max_regenerate_tries,
        max_feedback_gen_code_tries=max_feedback_gen_code_tries,
        max_mllm_fix_bugs_tries=max_mllm_fix_bugs_tries,
        feedback_rounds=feedback_rounds,
        knowledge_point="",
    )
    _resolve_agent_output_root(base_args)
    _report_agent_asset_config(base_args)

    return _run_generation_and_evaluation_pipeline(
        runner_name="agent",
        knowledge_points=knowledge_points,
        base_args=base_args,
        generation_fn=_build_agent_generation_result,
        questions_json=questions_json_path,
        per_question_workers=per_question_workers,
    )


def run_mas_generation_and_evaluation_pipeline(
    *,
    knowledge_file: Union[str, Path] = DEFAULT_KNOWLEDGE_FILE,
    max_topics: Optional[int] = None,
    questions_json: Optional[Union[str, Path]] = None,
    per_question_workers: int = DEFAULT_PER_QUESTION_WORKERS,
    max_code_token_length: int = DEFAULT_MAX_CODE_TOKEN_LENGTH,
    max_fix_bug_tries: int = DEFAULT_MAX_FIX_BUG_TRIES,
    max_regenerate_tries: int = DEFAULT_MAX_REGENERATE_TRIES,
    outline_duration_minutes: int = DEFAULT_OUTLINE_DURATION_MINUTES,
    outline_max_regenerate_tries: Optional[int] = None,
    outline_model: str = DEFAULT_OUTLINE_MODEL,
    max_turns: int = DEFAULT_MAS_MAX_TURNS,
    max_retries: int = DEFAULT_MAS_MAX_RETRIES,
    coder_fix_attempts: Optional[int] = None,
    final_render_fix_attempts: Optional[int] = None,
    coder_regenerate_tries: Optional[int] = None,
    section_parallel_workers: Optional[int] = None,
    worker_model: str = DEFAULT_OUTLINE_MODEL,
    orchestrator_model: str = DEFAULT_OUTLINE_MODEL,
    clear_logs: bool = False,
    case_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run MAS generation plus evaluation for the first N topics, or all topics when max_topics is None/-1."""
    knowledge_points = load_knowledge_points(knowledge_file=knowledge_file, max_topics=max_topics)
    questions_json_path = _resolve_questions_json_path(questions_json)

    base_args = argparse.Namespace(
        knowledge_point="",
        max_code_token_length=max_code_token_length,
        max_fix_bug_tries=max_fix_bug_tries,
        max_regenerate_tries=max_regenerate_tries,
        outline_duration_minutes=outline_duration_minutes,
        outline_max_regenerate_tries=outline_max_regenerate_tries,
        outline_model=outline_model,
        max_turns=max_turns,
        max_retries=max_retries,
        coder_fix_attempts=coder_fix_attempts,
        final_render_fix_attempts=final_render_fix_attempts,
        coder_regenerate_tries=coder_regenerate_tries,
        section_parallel_workers=section_parallel_workers,
        worker_model=worker_model,
        orchestrator_model=orchestrator_model,
        clear_logs=clear_logs,
        case_index=case_index,
    )
    _resolve_mas_pipeline_output_root(base_args)

    return _run_generation_and_evaluation_pipeline(
        runner_name="mas",
        knowledge_points=knowledge_points,
        base_args=base_args,
        generation_fn=_build_mas_generation_result,
        questions_json=questions_json_path,
        per_question_workers=per_question_workers,
    )


def _run_pipeline_from_cli(args: argparse.Namespace) -> int:
    if args.runner == "agent":
        results = run_agent_generation_and_evaluation_pipeline(
            knowledge_file=args.knowledge_file,
            max_topics=args.max_topics,
            questions_json=args.questions_json,
            per_question_workers=args.per_question_workers,
            folder_prefix=args.folder_prefix,
            api_name=args.API,
            use_feedback=args.use_feedback,
            use_assets=args.use_assets,
            max_code_token_length=args.max_code_token_length,
            max_fix_bug_tries=args.max_fix_bug_tries,
            max_regenerate_tries=args.max_regenerate_tries,
            max_feedback_gen_code_tries=args.max_feedback_gen_code_tries,
            max_mllm_fix_bugs_tries=args.max_mllm_fix_bugs_tries,
            feedback_rounds=args.feedback_rounds,
        )
    else:
        results = run_mas_generation_and_evaluation_pipeline(
            knowledge_file=args.knowledge_file,
            max_topics=args.max_topics,
            questions_json=args.questions_json,
            per_question_workers=args.per_question_workers,
            max_code_token_length=args.max_code_token_length,
            max_fix_bug_tries=args.max_fix_bug_tries,
            max_regenerate_tries=args.max_regenerate_tries,
            outline_duration_minutes=args.outline_duration_minutes,
            outline_max_regenerate_tries=args.outline_max_regenerate_tries,
            outline_model=args.outline_model,
            max_turns=args.max_turns,
            max_retries=args.max_retries,
            coder_fix_attempts=args.coder_fix_attempts,
            final_render_fix_attempts=args.final_render_fix_attempts,
            coder_regenerate_tries=args.coder_regenerate_tries,
            section_parallel_workers=args.section_parallel_workers,
            worker_model=args.worker_model,
            orchestrator_model=args.orchestrator_model,
            clear_logs=args.clear_logs,
            case_index=args.case_index,
        )

    if not results:
        return 0
    return 0 if all(item.get("success") for item in results) else 1


def build_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single Code2Video topic, or run the generation+evaluation pipeline "
            "across a topic list, with either the classic agent flow or the MAS flow."
        ),
    )
    parser.add_argument("--runner", choices=["agent", "mas"], default="agent")
    parser.add_argument(
        "--knowledge_point",
        type=str,
        default=None,
        help="Single topic to generate. Required unless --run_pipeline is set.",
    )
    parser.add_argument(
        "--run_pipeline",
        action="store_true",
        default=False,
        help="Run generation and evaluation for the first N topics, or all topics from --knowledge_file.",
    )
    parser.add_argument(
        "--knowledge_file",
        type=str,
        default=DEFAULT_KNOWLEDGE_FILE,
        help="Topic-list JSON used by --run_pipeline. Relative paths resolve from json_files/ first.",
    )
    parser.add_argument(
        "--max_topics",
        type=int,
        default=None,
        help="Limit --run_pipeline to the first N topics. Use -1 or omit to run all topics.",
    )
    parser.add_argument(
        "--questions_json",
        type=str,
        default=None,
        help="Optional questions_by_topic JSON override for pipeline evaluation.",
    )
    parser.add_argument(
        "--per_question_workers",
        type=int,
        default=DEFAULT_PER_QUESTION_WORKERS,
        help="Parallel workers used within each TQ evaluation during --run_pipeline.",
    )
    parser.add_argument(
        "--folder_prefix",
        type=str,
        default="TEST",
        help="Prefix for agent outputs under CASES. MAS logs always use the normal mas_logs directory.",
    )
    parser.add_argument("--max_code_token_length", type=int, default=DEFAULT_MAX_CODE_TOKEN_LENGTH)

    # Agent runner options.
    parser.add_argument("--API", type=str, choices=AGENT_API_CHOICES, default=DEFAULT_API)
    _add_bool_flag(parser, "use_feedback", DEFAULT_USE_FEEDBACK)
    _add_bool_flag(parser, "use_assets", DEFAULT_USE_ASSETS)
    parser.add_argument("--max_fix_bug_tries", type=int, default=DEFAULT_MAX_FIX_BUG_TRIES)
    parser.add_argument("--max_regenerate_tries", type=int, default=DEFAULT_MAX_REGENERATE_TRIES)
    parser.add_argument("--max_feedback_gen_code_tries", type=int, default=DEFAULT_MAX_FEEDBACK_GEN_CODE_TRIES)
    parser.add_argument("--max_mllm_fix_bugs_tries", type=int, default=DEFAULT_MAX_MLLM_FIX_BUGS_TRIES)
    parser.add_argument("--feedback_rounds", type=int, default=DEFAULT_FEEDBACK_ROUNDS)

    # MAS runner options.
    parser.add_argument("--outline_duration_minutes", type=int, default=DEFAULT_OUTLINE_DURATION_MINUTES)
    parser.add_argument(
        "--outline_max_regenerate_tries",
        type=int,
        default=None,
        help="Defaults to --max_regenerate_tries for closer agent/MAS alignment.",
    )
    parser.add_argument("--outline_model", type=str, default=DEFAULT_OUTLINE_MODEL)
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAS_MAX_TURNS)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAS_MAX_RETRIES)
    parser.add_argument(
        "--coder_fix_attempts",
        type=int,
        default=None,
        help="Defaults to --max_fix_bug_tries for closer agent/MAS alignment.",
    )
    parser.add_argument(
        "--final_render_fix_attempts",
        type=int,
        default=None,
        help="Defaults to --max_fix_bug_tries for closer agent/MAS alignment.",
    )
    parser.add_argument(
        "--coder_regenerate_tries",
        type=int,
        default=None,
        help="Defaults to --max_regenerate_tries for closer agent/MAS alignment.",
    )
    parser.add_argument("--section_parallel_workers", type=int, default=None)
    parser.add_argument("--worker_model", type=str, default=DEFAULT_OUTLINE_MODEL)
    parser.add_argument("--orchestrator_model", type=str, default=DEFAULT_OUTLINE_MODEL)
    parser.add_argument("--clear_logs", action="store_true", default=False)
    parser.add_argument("--case_index", type=int, default=None)

    return parser.parse_args()


def run_single_agent(args: argparse.Namespace) -> int:
    _report_agent_asset_config(args)
    result = _build_agent_generation_result(args, idx=0)

    print(f"Runner: agent")
    print(f"Run outputs written to: {result['output_dir']}")
    if result["final_video_path"]:
        print(f"Final video: {result['final_video_path']}")
    _print_single_run_summary(
        topic=result["topic"],
        duration_minutes=result["duration_minutes"],
        total_tokens=result["total_tokens"],
        was_successful=result["success"],
    )
    return 0 if result["success"] else 1


def run_single_mas(args: argparse.Namespace) -> int:
    result = _build_mas_generation_result(args, idx=0)

    print(f"Runner: mas")
    print(f"Generated outline topic: {result['generated_topic']}")
    print(f"Finished MAS run for {result.get('section_count', 0)} sections.")
    print(f"Run outputs written to: {result['output_dir']}")
    if result["final_video_path"]:
        print(f"Final video: {result['final_video_path']}")
    _print_single_run_summary(
        topic=result["generated_topic"],
        duration_minutes=result["duration_minutes"],
        total_tokens=result["total_tokens"],
        was_successful=result["success"],
    )
    print(result["token_usage_summary_text"])
    return 0 if result["success"] else 1


def main() -> int:
    args = build_and_parse_args()
    if args.run_pipeline:
        return _run_pipeline_from_cli(args)

    if not args.knowledge_point:
        raise ValueError("--knowledge_point is required unless --run_pipeline is set.")

    if args.runner == "agent":
        return run_single_agent(args)
    return run_single_mas(args)


if __name__ == "__main__":
    raise SystemExit(main())


# Example usage:
# python src/main.py --runner agent --knowledge_point "Implicit Differentiation"
# python src/main.py --runner mas --knowledge_point "Implicit Differentiation" --outline_duration_minutes 10 --max_turns 5 --max_retries 5
# python src/main.py --runner agent --run_pipeline --max_topics 3
# python src/main.py --runner mas --run_pipeline --max_topics -1
# from main import run_agent_generation_and_evaluation_pipeline
# from main import run_mas_generation_and_evaluation_pipeline
