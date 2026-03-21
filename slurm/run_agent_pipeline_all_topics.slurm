#!/usr/bin/env bash
#SBATCH --job-name=code2video-agent-all
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

# Agent pipeline defaults. Override any of these via exported env vars before sbatch.
API_NAME="${API_NAME:-Gemini}"
FOLDER_PREFIX="${FOLDER_PREFIX:-TEST_Gemini}"
KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-long_video_topics_list.json}"
MAX_TOPICS="${MAX_TOPICS:--1}"
PER_QUESTION_WORKERS="${PER_QUESTION_WORKERS:-5}"

MAX_CODE_TOKEN_LENGTH="${MAX_CODE_TOKEN_LENGTH:-10000}"
MAX_FIX_BUG_TRIES="${MAX_FIX_BUG_TRIES:-10}"
MAX_REGENERATE_TRIES="${MAX_REGENERATE_TRIES:-10}"
MAX_FEEDBACK_GEN_CODE_TRIES="${MAX_FEEDBACK_GEN_CODE_TRIES:-3}"
MAX_MLLM_FIX_BUGS_TRIES="${MAX_MLLM_FIX_BUGS_TRIES:-3}"
FEEDBACK_ROUNDS="${FEEDBACK_ROUNDS:-2}"

CMD=(
  "$PYTHON_BIN" src/main.py
  --runner agent
  --run_pipeline
  --API "$API_NAME"
  --folder_prefix "$FOLDER_PREFIX"
  --knowledge_file "$KNOWLEDGE_FILE"
  --max_topics "$MAX_TOPICS"
  --per_question_workers "$PER_QUESTION_WORKERS"
  --use_feedback
  --use_assets
  --max_code_token_length "$MAX_CODE_TOKEN_LENGTH"
  --max_fix_bug_tries "$MAX_FIX_BUG_TRIES"
  --max_regenerate_tries "$MAX_REGENERATE_TRIES"
  --max_feedback_gen_code_tries "$MAX_FEEDBACK_GEN_CODE_TRIES"
  --max_mllm_fix_bugs_tries "$MAX_MLLM_FIX_BUGS_TRIES"
  --feedback_rounds "$FEEDBACK_ROUNDS"
)

if [ -n "${QUESTIONS_JSON:-}" ]; then
  CMD+=(--questions_json "$QUESTIONS_JSON")
fi

echo "Starting agent generation+evaluation pipeline at $(date)"
echo "Repository: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Topics file: $KNOWLEDGE_FILE"
echo "Max topics: $MAX_TOPICS"
echo "Output prefix: $FOLDER_PREFIX"
echo "Command: ${CMD[*]}"

"${CMD[@]}"

echo "Finished at $(date)"
