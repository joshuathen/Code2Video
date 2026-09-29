#!/usr/bin/env bash
#SBATCH --job-name=beliefs-batch-c
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/pawsey1357/jthen/Code2Video"
if [ ! -f "$REPO_ROOT/slurm/run_belief_generation_random30.slurm" ]; then
  echo "Batch C belief launcher could not find the Pawsey Code2Video repository: $REPO_ROOT" >&2
  exit 1
fi
export REPO_ROOT
export PIPELINE_DIR="${PIPELINE_DIR:-mas_logs/pipeline_20260921_074011}"
export BELIEF_OUTPUT_DIR="${BELIEF_OUTPUT_DIR:-$PIPELINE_DIR/belief_three_stage}"
export START_STAGE="${START_STAGE:-discovery}"
export RESUME_DISCOVERY="${RESUME_DISCOVERY:-0}"
# Match the model used to construct the Batch A and Batch B belief libraries.
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"

if [ ! -f "$REPO_ROOT/$PIPELINE_DIR/pipeline_summary.json" ]; then
  echo "Completed Batch C pipeline summary not found: $REPO_ROOT/$PIPELINE_DIR/pipeline_summary.json" >&2
  exit 1
fi

TOPIC_METRICS_COUNT=$(find "$REPO_ROOT/$PIPELINE_DIR" -mindepth 2 -maxdepth 2 -name topic_metrics.json | wc -l)
if [ "$TOPIC_METRICS_COUNT" -ne 30 ]; then
  echo "Batch C is incomplete: expected 30 topic_metrics.json files, found $TOPIC_METRICS_COUNT." >&2
  exit 1
fi

exec bash "$REPO_ROOT/slurm/run_belief_generation_random30.slurm" "$START_STAGE"
