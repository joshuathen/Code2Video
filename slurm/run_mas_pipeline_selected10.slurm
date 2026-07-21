#!/usr/bin/env bash
#SBATCH --job-name=code2video-mas-selected10
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

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/software/projects/pawsey1357/jthen/containers/manim-0.19.0.sif}"
PYTHON_BIN="${PYTHON_BIN:-/software/projects/pawsey1357/jthen/venvs/code2video-manim/bin/python}"

if [ -n "$CONTAINER_IMAGE" ]; then
  if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "Container image not found: $CONTAINER_IMAGE" >&2
    exit 1
  fi
  module load singularity/4.1.0-nompi
  SINGULARITY_JOB_TMP="${TMPDIR:-/tmp}/${USER}-code2video-${SLURM_JOB_ID:-$$}"
  export SINGULARITY_CACHEDIR="$SINGULARITY_JOB_TMP/cache"
  export SINGULARITY_TMPDIR="$SINGULARITY_JOB_TMP/tmp"
  mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
  PYTHON_CMD=(singularity exec "$CONTAINER_IMAGE" "$PYTHON_BIN")
else
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
  fi
  PYTHON_CMD=("$PYTHON_BIN")
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_PATH="$(dirname "$PYTHON_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Curated 10-topic MAS pipeline defaults. Override via exported env vars
# before sbatch if you want to tweak the run.
RUN_LABEL="${RUN_LABEL:-selected-10}"
KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-json_files/agent_selected_topics_10.json}"
MAX_TOPICS="${MAX_TOPICS:--1}"
PER_QUESTION_WORKERS="${PER_QUESTION_WORKERS:-5}"

MAX_CODE_TOKEN_LENGTH="${MAX_CODE_TOKEN_LENGTH:-10000}"
MAX_FIX_BUG_TRIES="${MAX_FIX_BUG_TRIES:-10}"
MAX_REGENERATE_TRIES="${MAX_REGENERATE_TRIES:-10}"
OUTLINE_DURATION_MINUTES="${OUTLINE_DURATION_MINUTES:-5}"
MAS_MODEL="${MAS_MODEL:-gemini-3-flash-preview}"
CODE_MODEL="${CODE_MODEL:-gemini-3-flash-preview}"
REPAIR_MODEL="${REPAIR_MODEL:-gemini-3-flash-preview}"
EVAL_MODEL="${EVAL_MODEL:-gemini-2.5-pro}"
export MAS_MODEL CODE_MODEL REPAIR_MODEL EVAL_MODEL
OUTLINE_MAX_REGENERATE_TRIES="${OUTLINE_MAX_REGENERATE_TRIES:-}"
OUTLINE_MODEL="$MAS_MODEL"
MAX_TURNS="${MAX_TURNS:-3}"
MAX_RETRIES="${MAX_RETRIES:-3}"
CODER_FIX_ATTEMPTS="${CODER_FIX_ATTEMPTS:-}"
FINAL_RENDER_FIX_ATTEMPTS="${FINAL_RENDER_FIX_ATTEMPTS:-}"
CODER_REGENERATE_TRIES="${CODER_REGENERATE_TRIES:-}"
SECTION_PARALLEL_WORKERS="${SECTION_PARALLEL_WORKERS:-}"
WORKER_MODEL="$MAS_MODEL"
ORCHESTRATOR_MODEL="$MAS_MODEL"
TOPIC_PARALLEL_WORKERS="${TOPIC_PARALLEL_WORKERS:-1}"

CMD=(
  "${PYTHON_CMD[@]}" src/main.py
  --runner mas
  --run_pipeline
  --knowledge_file "$KNOWLEDGE_FILE"
  --max_topics "$MAX_TOPICS"
  --per_question_workers "$PER_QUESTION_WORKERS"
  --topic_parallel_workers "$TOPIC_PARALLEL_WORKERS"
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

if [ -n "$CODER_REGENERATE_TRIES" ]; then
  CMD+=(--coder_regenerate_tries "$CODER_REGENERATE_TRIES")
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

echo "Starting $RUN_LABEL MAS generation+evaluation pipeline at $(date)"
echo "Repository: $REPO_ROOT"
echo "Python: $PYTHON_BIN"
echo "Container: ${CONTAINER_IMAGE:-<none>}"
echo "Topics file: $KNOWLEDGE_FILE"
echo "MAS model: $MAS_MODEL"
echo "Code model: $CODE_MODEL"
echo "Repair model: $REPAIR_MODEL"
echo "Evaluation model: $EVAL_MODEL"
echo "Max topics: $MAX_TOPICS"
echo "Topic parallel workers: $TOPIC_PARALLEL_WORKERS"
echo "Command: ${CMD[*]}"

"${CMD[@]}"

echo "Finished at $(date)"
