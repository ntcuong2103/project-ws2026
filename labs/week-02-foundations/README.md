# Week 2 — Foundations: Problem Framing, Dataset EDA, and Edit Distance

[← Back to course README](../../README.md) · Previous: [Week 1](../week-01-orientation/) · Next: [Week 3](../week-03-detection-1/)

**Session:** each group reviews Week 1 (5 min/group), then journal club and this lab.

## Week 2 checklist

- [ ] Read both required papers and answer the problem-clarification questions.
- [ ] Agree on what the paper solves, why character-level annotation is needed, and what the team will reproduce.
- [ ] Verify that the five-page sample prepared in Week 1 is available.
- [ ] Run every notebook cell and retain its outputs.
- [ ] Document the dataset structure, page/file counts, byte sizes, label format, and any discrepancies with the papers.
- [ ] Inspect all five sample pages for degradation, layout, reading order, and transcription conventions.
- [ ] Implement and sanity-check the four required edit-distance methods.
- [ ] Interpret the five-page similarity matrices and explain what they do not measure.

**Deliverable:** an executed EDA notebook with the team's observations and similarity
interpretation, including a concise, evidence-based statement of the problem and the
paper result that the team will reproduce.

## Provided lab materials

This lab provides a guided dataset-EDA and edit-distance notebook. It reuses the
persistent five-page sample prepared in Week 1 and reads the full shared NomNaOCR
dataset without modifying it.

## 1. Understand the week's objectives

By the end of the week, each team should be able to:

- formulate the character-level annotation problem, including its inputs, available supervision, desired outputs, and evaluation criteria;
- distinguish the NomNaOCR dataset contribution from the iterative radical-based annotation method;
- explain why the available line-level supervision is insufficient for the paper's character-level task;
- identify the paper's inputs, expected outputs, major processing stages, and evaluation evidence;
- explain the NomNaOCR filesystem, size, image and label formats, page quality, layouts, and transcription conventions from evidence; and
- implement and interpret character-level edit distances and normalized similarities on the five Week 1 pages.

## 2. Read and compare the papers

Read both papers before starting the notebook:

1. H.-Q. Dang et al., “NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script,” *2022 RIVF International Conference on Computing and Communication Technologies*, pp. 476–481, 2022. [DOI: 10.1109/RIVF55975.2022.10013842](https://doi.org/10.1109/RIVF55975.2022.10013842).
2. C. T. Nguyen et al., “Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer,” *Document Analysis and Recognition — ICDAR 2026*, LNCS 16974, pp. 469–485. [DOI: 10.1007/978-3-032-36039-7_28](https://doi.org/10.1007/978-3-032-36039-7_28).

You can refer to this guide to read the papers: [How to read paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)

As you read, distinguish the dataset introduced by the first paper from the annotation
problem and method presented by the second paper. The goal is not to propose a new
research program: it is to understand what must be reproduced and why each part is
necessary.

### Problem-clarification questions

Answer each question in one to three sentences in Section 1 of the notebook. Cite a
paper section, page number, figure, table, or notebook observation for factual claims.

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

Conclude with one synthesis statement:

> NomNaOCR provides **[available supervision]**, but not **[missing supervision]**.
> We therefore reproduce **[paper method/output]** because **[why it is necessary]**,
> and we will compare **[evidence]** with the paper.

## 3. Verify the Week 1 sample

The notebook reuses the five persistent page pairs produced in Week 1. Do not create a
second ad-hoc copy or repeat the preparation instructions here. Set the workspace path
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
and [Step 5: Prepare the sample with a CPU-only batch job](../week-01-orientation/#5-prepare-the-sample-with-a-cpu-only-batch-job).

## 4. Run the notebook

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

The notebook only reads `/data/shared/project2026/nomnaocr`. It inventories the full
dataset, inspects the five local sample pages in detail, visualizes line polygons, and
computes pairwise similarities between the five page transcriptions.

## 5. Complete and interpret the analysis

Complete the notebook and record brief observations about the dataset structure,
labels, and five sample pages. Compare the page transcriptions using the required edit
distances, then explain what the results show and what they do not show.

## Materials

- [`nomnaocr_eda_and_edit_distance.ipynb`](nomnaocr_eda_and_edit_distance.ipynb): guided EDA and edit-distance lab.
- [Week 1 sample preparation](../week-01-orientation/)
- [Lab data contract](../../docs/DATA-CONTRACT.md)
