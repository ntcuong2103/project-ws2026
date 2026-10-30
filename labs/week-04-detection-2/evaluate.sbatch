#!/bin/bash
#SBATCH --job-name=w04-eval
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

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/nakagawa-character-detection/data.yaml" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --split test --name week04-nakagawa-test \
  --imgsz 1280 --batch 8 --device 0

for model_name in week03-mth1200 week04-nakagawa; do
  uv run python -m nom_ocr_lab.run_detector evaluate \
    --data "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/data.yaml" \
    --model "$LAB_WORKSPACE/models/$model_name/weights/best.pt" \
    --split test --name "$model_name-nomnaocr" \
    --imgsz 1280 --batch 8 --device 0
done

uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/images/test" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --name week04-nomnaocr --imgsz 1280 --batch 8 --device 0
