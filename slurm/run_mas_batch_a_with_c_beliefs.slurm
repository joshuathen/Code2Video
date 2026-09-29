#!/usr/bin/env bash
#SBATCH --job-name=mas-c-to-a-beliefs
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="/scratch/pawsey1357/jthen/Code2Video"
TRANSFER_LAUNCHER="$REPO_ROOT/slurm/run_mas_transfer_reactive_repair.slurm"

if [ ! -f "$TRANSFER_LAUNCHER" ]; then
  echo "Reactive Repair transfer launcher not found: $TRANSFER_LAUNCHER" >&2
  exit 1
fi

# C -> A: generate the Batch A topics using beliefs learned from Batch C.
# All other settings are inherited from the common Reactive Repair launcher so
# this run is directly comparable with the earlier transfer experiments.
export REPO_ROOT
export BELIEF_ASSET_DIR="mas_logs/pipeline_20260921_074011/belief_three_stage"
export KNOWLEDGE_FILE="json_files/mas_random_topics_30_seed_20260728.json"
export QUESTIONS_JSON="json_files/questions_by_topic_10.json"
export RUN_LABEL="batch-c-to-a-reactive-repair"

for required_file in \
  "$BELIEF_ASSET_DIR/belief_library.json" \
  "$BELIEF_ASSET_DIR/bbn_parameters.json" \
  "$BELIEF_ASSET_DIR/belief_embeddings.npz" \
  "$BELIEF_ASSET_DIR/belief_embedding_metadata.json" \
  "$KNOWLEDGE_FILE" \
  "$QUESTIONS_JSON"; do
  if [ ! -f "$REPO_ROOT/$required_file" ]; then
    echo "Required C -> A input not found: $REPO_ROOT/$required_file" >&2
    exit 1
  fi
done

exec bash "$TRANSFER_LAUNCHER"
