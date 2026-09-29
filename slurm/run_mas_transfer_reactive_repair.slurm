#!/usr/bin/env bash
#SBATCH --job-name=mas-belief-transfer
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/pawsey1357/jthen/Code2Video"
if [ ! -f "$REPO_ROOT/slurm/run_mas_pipeline_random30_with_beliefs.slurm" ]; then
  echo "Transfer launcher could not find the Pawsey Code2Video repository: $REPO_ROOT" >&2
  exit 1
fi
export REPO_ROOT
if [ -z "${BELIEF_ASSET_DIR:-}" ]; then
  echo "Set BELIEF_ASSET_DIR to the source library's belief_three_stage directory." >&2
  exit 1
fi
if [ ! -f "$REPO_ROOT/$BELIEF_ASSET_DIR/belief_library.json" ]; then
  echo "Belief library not found: $BELIEF_ASSET_DIR/belief_library.json" >&2
  exit 1
fi
if [ ! -f "$REPO_ROOT/$BELIEF_ASSET_DIR/bbn_parameters.json" ]; then
  echo "BBN parameters not found: $BELIEF_ASSET_DIR/bbn_parameters.json" >&2
  exit 1
fi

export KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-json_files/mas_random_topics_30_seed_20260904.json}"
export RUN_LABEL="${RUN_LABEL:-belief-transfer-reactive-repair}"
export MAS_MODEL="gemini-3.1-flash-lite"
export CODE_MODEL="gemini-3.1-flash-lite"
export REPAIR_MODEL="gemini-3.1-flash-lite"
export EVAL_MODEL="gemini-2.5-pro"
export MAX_TOPICS=-1
export TOPIC_PARALLEL_WORKERS=3

# Match the completed A/B transferability runs: retrieve up to five
# context-selected beliefs only after a concrete Coder runtime failure.
export INJECT_BELIEFS_INTO_PROMPTS=1
export CONTEXTUAL_BELIEF_SELECTION=1
export ENABLE_BROAD_BELIEF_INJECTION=0
export ENABLE_CODER_WIDE_BELIEF_INJECTION=0
export TOP_GENERAL_BELIEFS=0
export TOP_SPECIALIZED_BELIEFS=5
export BELIEF_USEFULNESS_THRESHOLD=0.0
export BELIEF_CANDIDATE_LIMIT="${BELIEF_CANDIDATE_LIMIT:-82}"
export BELIEF_LIBRARY_PATH="$BELIEF_ASSET_DIR/belief_library.json"
export BBN_PARAMETERS_PATH="$BELIEF_ASSET_DIR/bbn_parameters.json"

exec bash "$REPO_ROOT/slurm/run_mas_pipeline_random30_with_beliefs.slurm"
