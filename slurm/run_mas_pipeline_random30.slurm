#!/usr/bin/env bash
#SBATCH --job-name=code2video-mas-random30
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

# Fixed, reproducible sample drawn once from all 117 topics with Python
# random.Random(20260728).sample(topics, 30). This deliberately does not
# exclude the curated selected-ten pool.
export KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-json_files/mas_random_topics_30_seed_20260728.json}"
export MAX_TOPICS="${MAX_TOPICS:--1}"
export RUN_LABEL="${RUN_LABEL:-random-30}"

# Match the random-15 pipeline: run three independent topics concurrently.
# Section-level parallelism remains unset, retaining the MAS pipeline's
# existing concurrent section-rendering behavior within each topic.
export TOPIC_PARALLEL_WORKERS="${TOPIC_PARALLEL_WORKERS:-3}"

# Reuse the maintained MAS/container launcher. Its #SBATCH directives are
# ignored because this wrapper already owns the active allocation.
exec bash "$REPO_ROOT/slurm/run_mas_pipeline_selected10.slurm"
