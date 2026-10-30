#!/bin/bash
#SBATCH --job-name=w04-train
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100_3g.20gb:1

set -euo pipefail
: "${LAB_WORKSPACE:?Export LAB_WORKSPACE before submitting this job}"

uv run python -m nom_ocr_lab.run_detector train \
  --data "$LAB_WORKSPACE/datasets/nakagawa-character-detection/data.yaml" \
  --model "$LAB_WORKSPACE/models/week03-mth1200/weights/best.pt" \
  --name week04-nakagawa --epochs 100 --imgsz 1280 \
  --batch 8 --patience 10 --device 0
