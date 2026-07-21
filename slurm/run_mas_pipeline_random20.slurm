#!/usr/bin/env bash
#SBATCH --job-name=code2video-mas-random20
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Fixed, reproducible sample of 20 topics drawn from the 117-topic list after
# excluding the curated selected-ten topics (and equivalent duplicate titles).
export KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-json_files/mas_random_topics_20_excluding_selected10.json}"
export MAX_TOPICS="${MAX_TOPICS:--1}"
export RUN_LABEL="random-20"

# Reuse the maintained MAS/container launcher. Its #SBATCH directives are
# ignored here because this wrapper already owns the active allocation.
exec bash "$SCRIPT_DIR/run_mas_pipeline_selected10.slurm"
