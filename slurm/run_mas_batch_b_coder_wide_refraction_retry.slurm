#!/usr/bin/env bash
#SBATCH --job-name=mas-b-coder-retry
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/pawsey1357/jthen/Code2Video"
BELIEF_ASSET_DIR="mas_logs/pipeline_20260814_075635/belief_three_stage"

for required in \
  "$REPO_ROOT/slurm/run_mas_pipeline_random30_with_beliefs.slurm" \
  "$REPO_ROOT/json_files/mas_batch_b_refraction_retry.json" \
  "$REPO_ROOT/json_files/questions_by_topic_10.json" \
  "$REPO_ROOT/$BELIEF_ASSET_DIR/belief_library.json" \
  "$REPO_ROOT/$BELIEF_ASSET_DIR/bbn_parameters.json"; do
  if [ ! -e "$required" ]; then
    echo "Required input not found: $required" >&2
    exit 1
  fi
done

export REPO_ROOT BELIEF_ASSET_DIR
export KNOWLEDGE_FILE="json_files/mas_batch_b_refraction_retry.json"
export QUESTIONS_JSON="json_files/questions_by_topic_10.json"
export RUN_LABEL="batch-b-coder-wide-refraction-retry"
export CASE_INDEX=3
export MAS_MODEL="gemini-3.1-flash-lite"
export CODE_MODEL="gemini-3.1-flash-lite"
export REPAIR_MODEL="gemini-3.1-flash-lite"
export EVAL_MODEL="gemini-2.5-pro"
export MAX_TOPICS=-1
export TOPIC_PARALLEL_WORKERS=1

export INJECT_BELIEFS_INTO_PROMPTS=1
export CONTEXTUAL_BELIEF_SELECTION=1
export ENABLE_BROAD_BELIEF_INJECTION=0
export ENABLE_CODER_WIDE_BELIEF_INJECTION=1
export TOP_GENERAL_BELIEFS=0
export TOP_SPECIALIZED_BELIEFS=5
export BELIEF_USEFULNESS_THRESHOLD=0.0
export BELIEF_CANDIDATE_LIMIT=82
export BELIEF_LIBRARY_PATH="$BELIEF_ASSET_DIR/belief_library.json"
export BBN_PARAMETERS_PATH="$BELIEF_ASSET_DIR/bbn_parameters.json"

exec bash "$REPO_ROOT/slurm/run_mas_pipeline_random30_with_beliefs.slurm"
