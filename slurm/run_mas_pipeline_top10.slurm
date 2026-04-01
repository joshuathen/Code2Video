#!/usr/bin/env bash
#SBATCH --job-name=code2video-mas-top10
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Uncomment these if Kaya requires them for your jobs.
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
  REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
fi

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/jthen/.conda/envs/py31020/bin/python3.10}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# MAS pipeline defaults. Override any of these via exported env vars before sbatch.
KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-long_video_topics_list.json}"
MAX_TOPICS="${MAX_TOPICS:-10}"
RANDOM_SEED="${RANDOM_SEED:-}"
PER_QUESTION_WORKERS="${PER_QUESTION_WORKERS:-5}"

MAX_CODE_TOKEN_LENGTH="${MAX_CODE_TOKEN_LENGTH:-10000}"
MAX_FIX_BUG_TRIES="${MAX_FIX_BUG_TRIES:-10}"
MAX_REGENERATE_TRIES="${MAX_REGENERATE_TRIES:-10}"
OUTLINE_DURATION_MINUTES="${OUTLINE_DURATION_MINUTES:-5}"
OUTLINE_MAX_REGENERATE_TRIES="${OUTLINE_MAX_REGENERATE_TRIES:-}"
OUTLINE_MODEL="${OUTLINE_MODEL:-gemini-3-flash-preview}"
MAX_TURNS="${MAX_TURNS:-3}"
MAX_RETRIES="${MAX_RETRIES:-3}"
CODER_FIX_ATTEMPTS="${CODER_FIX_ATTEMPTS:-}"
FINAL_RENDER_FIX_ATTEMPTS="${FINAL_RENDER_FIX_ATTEMPTS:-}"
SECTION_PARALLEL_WORKERS="${SECTION_PARALLEL_WORKERS:-}"
WORKER_MODEL="${WORKER_MODEL:-gemini-3-flash-preview}"
ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-gemini-3-flash-preview}"

RESOLVED_KNOWLEDGE_FILE="$KNOWLEDGE_FILE"
if [[ "$KNOWLEDGE_FILE" != /* ]]; then
  if [[ -f "$REPO_ROOT/json_files/$KNOWLEDGE_FILE" ]]; then
    RESOLVED_KNOWLEDGE_FILE="$REPO_ROOT/json_files/$KNOWLEDGE_FILE"
  elif [[ -f "$REPO_ROOT/$KNOWLEDGE_FILE" ]]; then
    RESOLVED_KNOWLEDGE_FILE="$REPO_ROOT/$KNOWLEDGE_FILE"
  fi
fi

SAMPLED_KNOWLEDGE_FILE=$(mktemp "${TMPDIR:-/tmp}/code2video-mas-top10.XXXXXX.json")
cleanup() {
  rm -f "$SAMPLED_KNOWLEDGE_FILE"
}
trap cleanup EXIT

if [[ -n "$RANDOM_SEED" ]]; then
  SAMPLE_ARGS=("$RESOLVED_KNOWLEDGE_FILE" "$SAMPLED_KNOWLEDGE_FILE" "$MAX_TOPICS" "$RANDOM_SEED")
else
  SAMPLE_ARGS=("$RESOLVED_KNOWLEDGE_FILE" "$SAMPLED_KNOWLEDGE_FILE" "$MAX_TOPICS")
fi

"$PYTHON_BIN" -c '
import json
import random
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
max_topics = int(sys.argv[3])
seed = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "" else None

with source_path.open("r", encoding="utf-8") as f:
    topics = json.load(f)

if not isinstance(topics, list):
    raise ValueError(f"Knowledge file must contain a JSON list: {source_path}")

normalized_topics = [str(item) for item in topics]
rng = random.Random(seed)
sample_size = len(normalized_topics) if max_topics < 0 else min(max_topics, len(normalized_topics))
sampled_topics = rng.sample(normalized_topics, sample_size) if sample_size < len(normalized_topics) else normalized_topics[:]
if sample_size == len(normalized_topics):
    rng.shuffle(sampled_topics)

with target_path.open("w", encoding="utf-8") as f:
    json.dump(sampled_topics, f, ensure_ascii=False, indent=2)

print(f"Randomly selected {len(sampled_topics)} topics from {source_path}")
if seed is not None:
    print(f"Random seed: {seed}")
for idx, topic in enumerate(sampled_topics, start=1):
    print(f"{idx}. {topic}")
' "${SAMPLE_ARGS[@]}"

CMD=(
  "$PYTHON_BIN" src/main.py
  --runner mas
  --run_pipeline
  --knowledge_file "$SAMPLED_KNOWLEDGE_FILE"
  --max_topics "$MAX_TOPICS"
  --per_question_workers "$PER_QUESTION_WORKERS"
  --max_code_token_length "$MAX_CODE_TOKEN_LENGTH"
  --max_fix_bug_tries "$MAX_FIX_BUG_TRIES"
  --max_regenerate_tries "$MAX_REGENERATE_TRIES"
  --outline_duration_minutes "$OUTLINE_DURATION_MINUTES"
  --outline_model "$OUTLINE_MODEL"
  --max_turns "$MAX_TURNS"
  --max_retries "$MAX_RETRIES"
  --worker_model "$WORKER_MODEL"
  --orchestrator_model "$ORCHESTRATOR_MODEL"
)

if [ -n "${QUESTIONS_JSON:-}" ]; then
  CMD+=(--questions_json "$QUESTIONS_JSON")
fi

if [ -n "$OUTLINE_MAX_REGENERATE_TRIES" ]; then
  CMD+=(--outline_max_regenerate_tries "$OUTLINE_MAX_REGENERATE_TRIES")
fi

if [ -n "$CODER_FIX_ATTEMPTS" ]; then
  CMD+=(--coder_fix_attempts "$CODER_FIX_ATTEMPTS")
fi

if [ -n "$FINAL_RENDER_FIX_ATTEMPTS" ]; then
  CMD+=(--final_render_fix_attempts "$FINAL_RENDER_FIX_ATTEMPTS")
fi

if [ -n "$SECTION_PARALLEL_WORKERS" ]; then
  CMD+=(--section_parallel_workers "$SECTION_PARALLEL_WORKERS")
fi

if [ "${CLEAR_LOGS:-0}" = "1" ]; then
  CMD+=(--clear_logs)
fi

if [ -n "${CASE_INDEX:-}" ]; then
  CMD+=(--case_index "$CASE_INDEX")
fi

echo "Starting MAS generation+evaluation pipeline at $(date)"
echo "Repository: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Topics file: $KNOWLEDGE_FILE"
echo "Sampled topics file: $SAMPLED_KNOWLEDGE_FILE"
echo "Max topics: $MAX_TOPICS"
if [ -n "$RANDOM_SEED" ]; then
  echo "Random seed: $RANDOM_SEED"
fi
echo "Command: ${CMD[*]}"

"${CMD[@]}"

echo "Finished at $(date)"
