#!/bin/bash
#SBATCH --job-name=w03-train
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100_3g.20gb:1

set -euo pipefail
: "${LAB_WORKSPACE:?Export LAB_WORKSPACE before submitting this job}"
YOLO11N_WEIGHTS=${YOLO11N_WEIGHTS:-$LAB_WORKSPACE/models/pretrained/yolo11n.pt}
test -f "$YOLO11N_WEIGHTS"

uv run python -m nom_ocr_lab.run_detector train \
  --data "$LAB_WORKSPACE/datasets/mth-course-full/data.yaml" \
  --model "$YOLO11N_WEIGHTS" --name week03-mth1200 \
  --epochs 100 --imgsz 1280 --batch 8 --patience 10 --device 0
