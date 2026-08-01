#!/usr/bin/env bash
#SBATCH --job-name=code2video-mas-random30-beliefs
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/slurm/run_mas_pipeline_selected10_with_beliefs.slurm" ]; then
  echo "Submit from the Code2Video repository root or export REPO_ROOT." >&2
  exit 1
fi

# Use the identical fixed topic sample as the completed belief-free baseline.
export KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-json_files/mas_random_topics_30_seed_20260728.json}"
export MAX_TOPICS="${MAX_TOPICS:--1}"
export RUN_LABEL="${RUN_LABEL:-random-30-with-beliefs}"
export TOPIC_PARALLEL_WORKERS="${TOPIC_PARALLEL_WORKERS:-3}"

# Explicit smoke mode processes only the first fixed random-30 topic and uses
# one topic worker. CASE_INDEX only affects output-directory numbering; it does
# not select or limit input topics.
if [ "${SMOKE_ONE_TOPIC:-0}" = "1" ]; then
  export MAX_TOPICS=1
  export TOPIC_PARALLEL_WORKERS=1
  export RUN_LABEL="${SMOKE_RUN_LABEL:-random-30-with-beliefs-smoke-1}"
fi

# Contextual BBN retrieval defaults inherited by the maintained launcher:
# top_k=5 with no usefulness threshold, all 82 frozen candidates considered, stored BGE
# embeddings, posterior effectiveness, and the versioned BBN parameter file.
exec bash "$REPO_ROOT/slurm/run_mas_pipeline_selected10_with_beliefs.slurm"
