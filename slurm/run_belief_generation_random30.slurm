#!/usr/bin/env bash
#SBATCH --job-name=code2video-beliefs-random30
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/src/mas_belief_three_stage.py" ]; then
  echo "Submit from the Code2Video repository root or export REPO_ROOT." >&2
  exit 1
fi
cd "$REPO_ROOT"

PIPELINE_DIR="${PIPELINE_DIR:-mas_logs/pipeline_20260728_101242}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/software/projects/pawsey1357/jthen/containers/manim-0.19.0.sif}"
PYTHON_BIN="${PYTHON_BIN:-/software/projects/pawsey1357/jthen/venvs/code2video-manim/bin/python}"
BELIEF_EMBEDDING_MODEL="${BELIEF_EMBEDDING_MODEL:-/software/projects/pawsey1357/jthen/models/bge-small-en-v1.5}"
BELIEF_OUTPUT_DIR="${BELIEF_OUTPUT_DIR:-$PIPELINE_DIR/belief_three_stage}"
START_STAGE="${START_STAGE:-${1:-}}"

if [ "$START_STAGE" != "discovery" ] && [ "$START_STAGE" != "consolidation" ] && [ "$START_STAGE" != "evidence" ]; then
  echo "Pass an explicit stage: sbatch $0 discovery|consolidation|evidence" >&2
  exit 1
fi
HF_HOME="${HF_HOME:-/software/projects/pawsey1357/jthen/huggingface-cache}"

if [ ! -d "$PIPELINE_DIR" ]; then
  echo "Pipeline directory not found: $PIPELINE_DIR" >&2
  exit 1
fi
if [ ! -f "$CONTAINER_IMAGE" ]; then
  echo "Container image not found: $CONTAINER_IMAGE" >&2
  exit 1
fi
if [ ! -d "$BELIEF_EMBEDDING_MODEL" ]; then
  echo "Local embedding model not found: $BELIEF_EMBEDDING_MODEL" >&2
  exit 1
fi

module load singularity/4.1.0-nompi
export HF_HOME
export SINGULARITYENV_HF_HOME="$HF_HOME"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"

CMD=(
  singularity exec
  "$CONTAINER_IMAGE"
  "$PYTHON_BIN"
  src/mas_belief_three_stage.py
  --pipeline-dir "$PIPELINE_DIR"
  --output-dir "$BELIEF_OUTPUT_DIR"
  --belief-embedding-model "$BELIEF_EMBEDDING_MODEL"
  --start-stage "$START_STAGE"
)
if [ "$START_STAGE" = "discovery" ]; then
  CMD+=(--fresh)
fi

echo "Starting three-stage 30-topic belief pipeline at $(date)"
echo "Repository: $REPO_ROOT"
echo "Pipeline: $PIPELINE_DIR"
echo "Reflection model: ${GEMINI_MODEL:-configured in src/api_config.json}"
echo "Embedding model: $BELIEF_EMBEDDING_MODEL"
echo "Output directory: $BELIEF_OUTPUT_DIR"
echo "Starting stage: $START_STAGE"
echo "Stages: candidate discovery -> global consolidation -> retrospective evidence"
if [ "$START_STAGE" = "discovery" ]; then
  echo "Belief library mode: fresh (prior three-stage outputs will be replaced)"
elif [ "$START_STAGE" = "consolidation" ]; then
  echo "Belief library mode: resume from completed candidate checkpoint"
else
  echo "Belief library mode: preserve frozen beliefs and restart evidence from topic 1"
fi
echo "Command: ${CMD[*]}"

"${CMD[@]}"

echo "Finished at $(date)"
