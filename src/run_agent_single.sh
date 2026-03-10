#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PY=python3
ENTRY="$SCRIPT_DIR/agent.py"

# Ensure sibling packages (e.g. prompts/) are importable regardless of cwd.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 1) Default values and constants
# -------------------------------------------------------------

# Common defaults (if not overridden by command line)
API="gpt-41"
FOLDER_PREFIX="TEST-single"

# Hyperparameters
MAX_CODE_TOKEN_LENGTH=10000
MAX_FIX_BUG_TRIES=10
MAX_REGENERATE_TRIES=10
MAX_FEEDBACK_GEN_CODE_TRIES=3
MAX_MLLM_FIX_BUGS_TRIES=3
FEEDBACK_ROUNDS=2

# 2) KNOWLEDGE_POINT
# -------------------------------------------------------------

DEFAULT_KNOWLEDGE_POINT="Linear transformations and matrices"
HAS_KNOWLEDGE_POINT=0
for arg in "$@"; do
  if [ "$arg" = "--knowledge_point" ]; then
    HAS_KNOWLEDGE_POINT=1
    break
  fi
done
if [ "$HAS_KNOWLEDGE_POINT" -eq 0 ]; then
  echo "INFO: Using default knowledge point: $DEFAULT_KNOWLEDGE_POINT"
  exec "$PY" "$ENTRY" \
    --API "$API" \
    --folder_prefix "$FOLDER_PREFIX" \
    --use_feedback \
    --use_assets \
    --max_code_token_length "$MAX_CODE_TOKEN_LENGTH" \
    --max_fix_bug_tries "$MAX_FIX_BUG_TRIES" \
    --max_regenerate_tries "$MAX_REGENERATE_TRIES" \
    --max_feedback_gen_code_tries "$MAX_FEEDBACK_GEN_CODE_TRIES" \
    --max_mllm_fix_bugs_tries "$MAX_MLLM_FIX_BUGS_TRIES" \
    --feedback_rounds "$FEEDBACK_ROUNDS" \
    --parallel \
    --knowledge_point "$DEFAULT_KNOWLEDGE_POINT" \
    "$@"
fi

# 3) execute
# -------------------------------------------------------------

exec "$PY" "$ENTRY" \
  --API "$API" \
  --folder_prefix "$FOLDER_PREFIX" \
  --use_feedback \
  --use_assets \
  --max_code_token_length "$MAX_CODE_TOKEN_LENGTH" \
  --max_fix_bug_tries "$MAX_FIX_BUG_TRIES" \
  --max_regenerate_tries "$MAX_REGENERATE_TRIES" \
  --max_feedback_gen_code_tries "$MAX_FEEDBACK_GEN_CODE_TRIES" \
  --max_mllm_fix_bugs_tries "$MAX_MLLM_FIX_BUGS_TRIES" \
  --feedback_rounds "$FEEDBACK_ROUNDS" \
  --parallel \
  "$@"
