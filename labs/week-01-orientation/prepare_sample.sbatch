#!/bin/bash
#SBATCH --job-name=week01-prepare
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

: "${LAB_WORKSPACE:?Submit with LAB_WORKSPACE exported as shown in the Week 1 guide}"
: "${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Lab workspace: $LAB_WORKSPACE"
echo "Started: $(date --iso-8601=seconds)"

cd "$SLURM_SUBMIT_DIR"
uv run python labs/week-01-orientation/prepare_sample.py \
  --workspace "$LAB_WORKSPACE" \
  --size 5

echo "Finished: $(date --iso-8601=seconds)"
