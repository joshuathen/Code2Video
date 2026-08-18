#!/usr/bin/env bash
#SBATCH --job-name=code2video-eval-retry
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/src/eval_video.py" ]; then
  echo "Submit from the Code2Video repository root or export REPO_ROOT." >&2
  exit 1
fi
cd "$REPO_ROOT"

if [ "$#" -lt 3 ]; then
  echo "Usage: sbatch $0 VIDEO_PATH TOPIC OUTPUT_PATH" >&2
  exit 1
fi

VIDEO_PATH="$1"
TOPIC="$2"
OUTPUT_PATH="$3"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/software/projects/pawsey1357/jthen/containers/manim-0.19.0.sif}"
PYTHON_BIN="${PYTHON_BIN:-/software/projects/pawsey1357/jthen/venvs/code2video-manim/bin/python}"
QUESTIONS_JSON="${QUESTIONS_JSON:-json_files/questions_by_topic_10.json}"

module load singularity/4.1.0-nompi
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"

echo "Video: $VIDEO_PATH"
echo "Topic: $TOPIC"
echo "Output: $OUTPUT_PATH"
echo "Evaluation model: ${EVAL_MODEL:-gemini-2.5-pro}"

singularity exec "$CONTAINER_IMAGE" "$PYTHON_BIN" src/eval_video.py \
  --video-path "$VIDEO_PATH" \
  --topic "$TOPIC" \
  --tq-concept "$TOPIC" \
  --questions-json "$QUESTIONS_JSON" \
  --per-question-workers 5 \
  --use-interactions \
  --output "$OUTPUT_PATH"

