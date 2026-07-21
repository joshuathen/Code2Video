#!/usr/bin/env bash
#SBATCH --job-name=code2video-mas-random15
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Slurm executes a copied script from /var/spool/slurmd, so $0 cannot be used
# to locate files in the repository. sbatch records the real submission
# directory in SLURM_SUBMIT_DIR.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/slurm/run_mas_pipeline_selected10.slurm" ]; then
  echo "Could not locate the Code2Video repository from SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}" >&2
  echo "Submit this job from the repository root, or export REPO_ROOT=/path/to/Code2Video." >&2
  exit 1
fi

# Fixed, reproducible sample of 15 topics drawn from the 117-topic list after
# excluding the curated selected-ten topics (and equivalent duplicate titles).
export KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-json_files/mas_random_topics_15_excluding_selected10.json}"
export MAX_TOPICS="${MAX_TOPICS:--1}"
export RUN_LABEL="random-15"

# Run three independent topics concurrently. Section-level parallelism is left
# unset so each topic retains the MAS pipeline's original behavior of rendering
# its sections concurrently.
export TOPIC_PARALLEL_WORKERS="${TOPIC_PARALLEL_WORKERS:-3}"

# Reuse the maintained MAS/container launcher. Its #SBATCH directives are
# ignored here because this wrapper already owns the active allocation.
exec bash "$REPO_ROOT/slurm/run_mas_pipeline_selected10.slurm"
