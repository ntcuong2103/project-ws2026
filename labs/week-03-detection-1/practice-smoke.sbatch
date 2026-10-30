#!/bin/bash
#SBATCH --job-name=w03-smoke
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Smallest non-MPS GPU slice available on the course GPU node.
#SBATCH --gres=gpu:a100_2g.10gb:1

set -euo pipefail
: "${LAB_WORKSPACE:?Export LAB_WORKSPACE before submitting this job}"
YOLO11N_WEIGHTS=${YOLO11N_WEIGHTS:-$LAB_WORKSPACE/models/pretrained/yolo11n.pt}
RUN_NAME=${RUN_NAME:-week03-practice-smoke}
test -f "$YOLO11N_WEIGHTS"

uv run python -m nom_ocr_lab.run_detector train \
  --data "$LAB_WORKSPACE/datasets/mth-practice/data.yaml" \
  --model "$YOLO11N_WEIGHTS" --name "$RUN_NAME" \
  --epochs 10 --imgsz 640 --batch 4 --patience 0 --device 0

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/mth-practice/data.yaml" \
  --model "$LAB_WORKSPACE/models/$RUN_NAME/weights/best.pt" \
  --split test --name "$RUN_NAME-test" \
  --imgsz 640 --batch 4 --device 0
