# Week 2 — Foundations: Problem Framing, Dataset EDA, and Edit Distance

[← Back to course README](../../README.md) · Previous: [Week 1](../week-01-orientation/) · Next: [Week 3](../week-03-detection-1/)

**Session:** each group reviews Week 1 (5 min/group), then journal club and this lab.

## Learning outcomes

By the end of the week, each team should be able to:

- formulate the character-level annotation problem, including its inputs, available supervision, desired outputs, and evaluation criteria;
- distinguish the NomNaOCR dataset contribution from the iterative radical-based annotation method;
- state concrete project goals and non-goals, assign accountable team roles, and maintain a risk register;
- explain the NomNaOCR filesystem, size, image and label formats, page quality, layouts, and transcription conventions from evidence; and
- implement and interpret character-level edit distances and normalized similarities on the five Week 1 pages.

## Required reading

Read both papers before completing the project charter:

1. H.-Q. Dang et al., “NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script,” *2022 RIVF International Conference on Computing and Communication Technologies*, pp. 476–481, 2022. [DOI: 10.1109/RIVF55975.2022.10013842](https://doi.org/10.1109/RIVF55975.2022.10013842).
2. C. T. Nguyen et al., “Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer,” *Document Analysis and Recognition — ICDAR 2026*, LNCS 16974, pp. 469–485. [DOI: 10.1007/978-3-032-36039-7_28](https://doi.org/10.1007/978-3-032-36039-7_28).

As you read, prepare short answers to these questions:

| Question | NomNaOCR paper | Automated-annotation paper |
| --- | --- | --- |
| What problem is formulated? | | |
| What are the inputs and existing labels? | | |
| What output does the work produce? | | |
| What assumptions make the approach possible? | | |
| How is success measured? | | |
| What can fail or bias the result? | | |

Then write one synthesis sentence: **NomNaOCR gives us …, but the project must solve … because …**

## Before the lab: recover the Week 1 sample if needed

The notebook reuses the five persistent page pairs created in Week 1. Do not make a second ad-hoc copy. In a shell, replace `X` with the team's assigned group number, then run:

```bash
# Enter the lab repository.
cd /home/projectgX/project-ws2026

# Set the persistent location for lab outputs.
export LAB_WORKSPACE=/data/projectgX/lab-exercise

# Create any missing output folders.
mkdir -p "$LAB_WORKSPACE"/{datasets,models,predictions,logs}

# Rebuild the Week 1 sample in a batch job and capture Slurm's response.
submission=$(sbatch --parsable \
  --export=ALL,LAB_WORKSPACE="$LAB_WORKSPACE" \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-01-orientation/prepare_sample.sbatch)

# Extract the numeric job ID from the response.
job_id=${submission%%;*}

# Print the job ID for the checks below.
echo "Submitted job $job_id"
```

The preparation job is safe to rerun: it validates and reuses an intact sample rather than overwriting it. After it finishes, check:

```bash
# Check whether the batch job completed successfully.
sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode

# Confirm that the sample manifest was created.
test -f "$LAB_WORKSPACE/datasets/nomnaocr-sample/manifest.json" \
  && echo "five-page sample is ready"
```

## Run the notebook

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

The notebook only reads `/data/shared/project2026/nomnaocr`. It inventories the full dataset, inspects the five local sample pages in detail, visualizes line polygons, and computes pairwise similarities between the five page transcriptions.

## Lab tasks

- [ ] Complete the two-paper comparison and agree on a one-paragraph problem formulation.
- [ ] Run every notebook cell and retain its outputs.
- [ ] Draw the filesystem tree and explain how an image is paired with each available label file.
- [ ] Record page/file counts and byte sizes. Reconcile your observed counts with the papers; do not silently “fix” a mismatch.
- [ ] Inspect all five sample pages. Record at least one piece of evidence about degradation, layout, reading order, and label conventions per page.
- [ ] Explain what the eight coordinate fields and final text field in a line label mean. Find and report any `?` labels or malformed rows.
- [ ] Implement and sanity-check Hamming, Levenshtein, optimal-string-alignment Damerau–Levenshtein, and LCS-based distance/similarity.
- [ ] Interpret the five-page similarity matrices. State why they do **not** measure visual similarity or model quality.
- [ ] Complete the [project charter template](project-charter-template.md). Give every risk and goal an owner.

## Deliverables

1. The executed notebook, including the team's written EDA observations and similarity interpretation.
2. A one-page project charter covering problem formulation, goals/non-goals, success measures, roles, risks, and immediate actions.

## Materials

- [`nomnaocr_eda_and_edit_distance.ipynb`](nomnaocr_eda_and_edit_distance.ipynb): guided EDA and edit-distance lab.
- [`project-charter-template.md`](project-charter-template.md): one-page team charter scaffold.
- [Week 1 sample preparation](../week-01-orientation/)
- [Lab data contract](../../docs/DATA-CONTRACT.md)
