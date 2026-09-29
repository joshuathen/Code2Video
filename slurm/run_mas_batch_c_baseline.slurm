#!/usr/bin/env bash
#SBATCH --job-name=mas-batch-c-baseline
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/pawsey1357/jthen/Code2Video"
if [ ! -f "$REPO_ROOT/slurm/run_mas_pipeline_selected10.slurm" ]; then
  echo "Batch C launcher could not find the Pawsey Code2Video repository: $REPO_ROOT" >&2
  exit 1
fi
export REPO_ROOT

export KNOWLEDGE_FILE="json_files/mas_random_topics_30_seed_20260904.json"
export QUESTIONS_JSON="json_files/questions_by_topic_10.json"
export RUN_LABEL="batch-c-seed-20260904-baseline"
export MAS_MODEL="gemini-3.1-flash-lite"
export CODE_MODEL="gemini-3.1-flash-lite"
export REPAIR_MODEL="gemini-3.1-flash-lite"
export EVAL_MODEL="gemini-2.5-pro"
export MAX_TOPICS=-1
export TOPIC_PARALLEL_WORKERS=3

for required_file in "$KNOWLEDGE_FILE" "$QUESTIONS_JSON"; do
  if [ ! -f "$REPO_ROOT/$required_file" ]; then
    echo "Required Batch C input not found: $REPO_ROOT/$required_file" >&2
    exit 1
  fi
done

exec bash "$REPO_ROOT/slurm/run_mas_pipeline_selected10.slurm"
