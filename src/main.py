import argparse
import json
import sys
import time
from pathlib import Path


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


def build_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Code2Video topic with either the classic agent flow or the MAS flow.",
    )
    parser.add_argument("--runner", choices=["agent", "mas"], default="agent")
    parser.add_argument("--knowledge_point", type=str, required=True)
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
    parser.add_argument("--section_parallel_workers", type=int, default=None)
    parser.add_argument("--worker_model", type=str, default=DEFAULT_OUTLINE_MODEL)
    parser.add_argument("--orchestrator_model", type=str, default=DEFAULT_OUTLINE_MODEL)
    parser.add_argument("--clear_logs", action="store_true", default=False)
    parser.add_argument("--case_index", type=int, default=None)

    return parser.parse_args()


def run_single_agent(args: argparse.Namespace) -> int:
    from agent import RunConfig, TeachingVideoAgent, get_api_and_output

    api, folder_name = get_api_and_output(args.API)
    folder = PROJECT_ROOT / "CASES" / f"{args.folder_prefix}_{folder_name}"
    iconfinder_api_key = _load_iconfinder_api_key()

    if args.use_assets:
        if iconfinder_api_key:
            print("Iconfinder API key loaded from config.")
        else:
            print("WARNING: Iconfinder API key not found in config file. Asset enhancement may be limited.")

    cfg = RunConfig(
        api=api,
        iconfinder_api_key=iconfinder_api_key,
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
        idx=0,
        knowledge_point=args.knowledge_point,
        folder=folder,
        cfg=cfg,
    )
    final_video_path = agent.GENERATE_VIDEO()
    duration_minutes = (time.time() - start_time) / 60
    total_tokens = int(agent.token_usage.get("total_tokens", 0) or 0)
    was_successful = bool(final_video_path)

    print(f"Runner: agent")
    print(f"Run outputs written to: {agent.output_dir}")
    if final_video_path:
        print(f"Final video: {final_video_path}")
    _print_single_run_summary(
        topic=args.knowledge_point,
        duration_minutes=duration_minutes,
        total_tokens=total_tokens,
        was_successful=was_successful,
    )
    return 0 if was_successful else 1


def run_single_mas(args: argparse.Namespace) -> int:
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

    generated_outline = generate_outline_with_code2video_stage1(
        knowledge_point=args.knowledge_point,
        duration_minutes=args.outline_duration_minutes,
        max_regenerate_tries=outline_max_regenerate_tries,
        model=args.outline_model,
        token_tracker=token_tracker,
    )

    video_state = build_video_state_from_outline(generated_outline)
    section_parallel_workers = args.section_parallel_workers or len(video_state.section_ids())

    logs_root = PROJECT_ROOT / "mas_logs"

    cfg = MASRunConfig(
        max_turns=args.max_turns,
        max_retries=args.max_retries,
        coder_fix_attempts=coder_fix_attempts,
        final_render_fix_attempts=final_render_fix_attempts,
        max_code_token_length=args.max_code_token_length,
        worker_model=args.worker_model,
        orchestrator_model=args.orchestrator_model,
        section_parallel_workers=section_parallel_workers,
        clear_logs=args.clear_logs,
        case_index=args.case_index,
    )

    final_video_state = run_mas_for_video_state(
        video_state=video_state,
        logs_root=logs_root,
        cfg=cfg,
        token_tracker=token_tracker,
    )

    duration_minutes = (time.time() - start_time) / 60
    total_tokens = _total_tokens_from_summary(final_video_state.token_usage)
    final_video_path = _final_video_output_path(final_video_state)
    was_successful = bool(final_video_path and final_video_path.exists())

    print(f"Runner: mas")
    print(f"Generated outline topic: {generated_outline.topic}")
    print(f"Finished MAS run for {len(final_video_state.section_ids())} sections.")
    print(f"Run outputs written to: {final_video_state.run_output_dir}")
    if final_video_path:
        print(f"Final video: {final_video_path}")
    _print_single_run_summary(
        topic=generated_outline.topic,
        duration_minutes=duration_minutes,
        total_tokens=total_tokens,
        was_successful=was_successful,
    )
    print(_format_token_usage_summary(final_video_state.token_usage))
    return 0 if was_successful else 1


def main() -> int:
    args = build_and_parse_args()
    if args.runner == "agent":
        return run_single_agent(args)
    return run_single_mas(args)


if __name__ == "__main__":
    raise SystemExit(main())


# Example usage:
# python src/main.py --runner agent --knowledge_point "Implicit Differentiation"
# python src/main.py --runner mas --knowledge_point "Implicit Differentiation" --outline_duration_minutes 10 --max_turns 5 --max_retries 5
