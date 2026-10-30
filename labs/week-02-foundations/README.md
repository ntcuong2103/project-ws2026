# Week 2 — Foundations: Problem Framing, Dataset EDA, and Edit Distance

[← Back to course README](../../README.md) · Previous: [Week 1](../week-01-orientation/) · Next: [Week 3](../week-03-detection-1/)

**Session:** each group reviews Week 1 (5 min/group), then journal club and this lab.

## Week 2 checklist

- [ ] Read both required papers and answer the problem-clarification questions.
- [ ] Understand what the paper solves, why character-level annotation is needed, and what the team will reproduce.
- [ ] Verify that the five-page sample prepared in Week 1 is available.
- [ ] Distinguish line supervision from pretrained-detector reference output.
- [ ] Understand the full line-level-to-character-level pseudo-ground-truth pipeline; do not implement it this week.
- [ ] Run every notebook cell and retain its outputs.
- [ ] Audit pairing and labels; explain what would fail downstream if a check were skipped.
- [ ] Inspect all five sample pages for degradation, layout, reading order, and transcription conventions.
- [ ] Implement and sanity-check the four required edit-distance methods.
- [ ] Interpret OCR-style transcription errors with the four distances.

**Deliverable:** an executed EDA notebook with the team's observations and similarity
interpretation, including a concise, evidence-based statement of the problem and an
explanation of how character-position detection and line transcription eventually
produce character-level pseudo ground truth.

## Provided lab materials

The notebook audits the full read-only dataset and uses the five Week 1 triplets for
visual inspection and edit-distance examples.

## 1. Read and compare the papers

Read both papers before starting the notebook:

1. H.-Q. Dang et al., “NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script,” *2022 RIVF International Conference on Computing and Communication Technologies*, pp. 476–481, 2022. [DOI: 10.1109/RIVF55975.2022.10013842](https://doi.org/10.1109/RIVF55975.2022.10013842).
2. C. T. Nguyen et al., “Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer,” *Document Analysis and Recognition — ICDAR 2026*, LNCS 16974, pp. 469–485. [DOI: 10.1007/978-3-032-36039-7_28](https://doi.org/10.1007/978-3-032-36039-7_28).

You can refer to this guide to read the papers: [How to read paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)

As you read, distinguish the dataset introduced by the first paper from the annotation
problem and method presented by the second paper. The goal is not to propose a new
research program: it is to understand what must be reproduced and why each part is
necessary.

### Problem-clarification questions

1. What supervision does NomNaOCR provide, and what character-level information is
   missing?
2. Why is the missing character-level annotation needed, and why can it not be
   obtained by directly assigning the characters in a line transcription to detected
   boxes?
3. What output does the automated-annotation paper construct, and how do character
   detection, radical-based recognition, alignment, confidence estimation, and
   iterative updating contribute to it?
4. What result, metric, or observable behavior will the team compare with the paper
   to judge whether the reproduction is faithful?

### Pipeline story for Week 2

Every team must be able to explain this sequence:

> page image → character-position detection → box-to-line grouping and reading order
> → recognition/alignment with the line transcription → confidence filtering
> → character box + character identity pseudo ground truth

The supplied detection labels are pretrained-detector reference output: useful for
seeing likely box predictions, but not verified ground truth or character identities.

## 2. Verify the Week 1 sample

The notebook reuses the five persistent page triplets produced in Week 1. Set the workspace path
and check for the sample manifest:

```bash
# Enter the lab repository.
cd /home/projectgX/project-ws2026

# Set the persistent location for lab outputs.
export LAB_WORKSPACE=/data/projectgX/lab-exercise

# Confirm that the Week 1 sample manifest exists.
test -f "$LAB_WORKSPACE/datasets/nomnaocr-sample/manifest.json" \
  && echo "five-page sample is ready"
```

If the confirmation is not printed, return to Week 1 and follow
[Step 4: Create the persistent lab workspace](../week-01-orientation/#4-create-the-persistent-lab-workspace)
and [Step 5: Prepare the sample with a CPU-only batch job](../week-01-orientation/#5-prepare-the-sample-with-a-cpu-only-batch-job),
which submits [`prepare_sample.sbatch`](../week-01-orientation/prepare_sample.sbatch).

## 3. Run the notebook

Start a short CPU-only allocation and launch Jupyter from the repository:

```bash
# Start a 45-minute interactive shell for the notebook.
srun --partition=gpu --account=lab --qos=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:45:00 \
  --pty bash

# Enter the lab repository inside the interactive shell.
cd /home/projectgX/project-ws2026

# Restore the persistent lab-output location in this shell.
export LAB_WORKSPACE=/data/projectgX/lab-exercise

# Start Jupyter and open the Week 2 notebook.
uv run jupyter lab labs/week-02-foundations/nomnaocr_eda_and_edit_distance.ipynb
```

The notebook checks triplet completeness, label validity, and box/text count gaps;
inspects five overlays; then compares controlled OCR errors with a matching reference
line.

## 4. Complete and interpret the analysis

Explain each check's downstream consequence and which pipeline failure each
edit-distance example models.

## Materials

- [`nomnaocr_eda_and_edit_distance.ipynb`](nomnaocr_eda_and_edit_distance.ipynb): guided EDA and edit-distance lab.
- [Week 1 sample preparation](../week-01-orientation/)
- [Lab data contract](../../docs/DATA-CONTRACT.md)
