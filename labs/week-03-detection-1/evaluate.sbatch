#!/bin/bash
#SBATCH --job-name=w03-eval
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100_3g.20gb:1

set -euo pipefail
: "${LAB_WORKSPACE:?Export LAB_WORKSPACE before submitting this job}"
YOLO11N_WEIGHTS=${YOLO11N_WEIGHTS:-$LAB_WORKSPACE/models/pretrained/yolo11n.pt}
test -f "$YOLO11N_WEIGHTS"

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/mth-course-full/data.yaml" \
  --model "$YOLO11N_WEIGHTS" --split test --name week03-coco-mth1000 \
  --imgsz 1280 --batch 8 --device 0

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/mth-course-full/data.yaml" \
  --model "$LAB_WORKSPACE/models/week03-mth1200/weights/best.pt" \
  --split test --name week03-trained-mth1000 \
  --imgsz 1280 --batch 8 --device 0
