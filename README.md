# Confidence-Weighted Nôm OCR - Project Course

## Course Overview

Students design, build, and evaluate a full document-AI pipeline — character detection, radical-based recognition, IDS-level sequence alignment, and confidence-weighted pseudo-labeling — modeled on a published research system for historical Hán-Nôm document annotation.

- **Format:** 15 weeks, 4 contact hours/week (60 hours total), teams of 3–4 students
- **Structure:** the whole class moves through the pipeline stage by stage (detection → recognition → alignment → pseudo-labeling); each team builds the complete pipeline end-to-end on the whole NomNaOCR dataset, so results are comparable across teams at the final presentation
- **Prerequisites:** deep learning foundations (CNNs, training loops, PyTorch/TensorFlow), basic Python; no prior OCR/CV or CJK-script experience assumed
- **Resources:** lab GPUs and the datasets used in the source paper — the NomNaOCR corpus (2,953 pages), TUAT-Nakagawa detection annotations, MTHv2 pretraining set, and the CHISE/cjkvi-ids dictionary
- **Learning objectives:** by Week 15, teams can fine-tune an object detector for character-level localization; train a sequence model to predict compositional (IDS) character representations; implement a dynamic-programming alignment algorithm that recovers box-to-character correspondence with a graded confidence score; run an iterative confidence-weighted self-training loop; and evaluate results against a baseline using CER and precision/recall metrics

## Lab and Project Philosophy

This repository contains the instructor-maintained lab handouts, notebooks, fixtures,
and shared teaching utilities. Each week begins with a small, hand-checkable example
before scaling to real pages or the GPU cluster. Generated lab artifacts are kept in a
persistent team workspace and reused by later labs.

Teams develop the assessed pipeline in their own repository rather than inside this
lab repository. Teams choose when to create that repository and may follow the
[optional project contract](docs/PROJECT-CONTRACT.md). The lab and project repositories
remain separate so that lab updates do not overwrite or prescribe student work.

## Lecture Format (4 AHs per week)

Each weekly session is **4 academic hours (AHs)** and follows the same two-part structure (Weeks 8 and 15 are presentation weeks, see below):

| Part | Content | Time |
| --- | --- | --- |
| 1. Review of previous week | Each group presents the tasks it completed in the previous week and its deliverable: what worked, what failed, what is blocking. The instructor gives short feedback and flags issues that affect the whole class. | **5 min per group** (e.g. 8 groups ≈ 40 min) |
| 2. Lab / tutorial | The week's topic is introduced briefly, then the class works through that week's lab tutorial (see the [Lab Tutorials](#lab-tutorials) folders), hands-on in teams, with the instructor circulating. Teams start the week's tasks in class and finish them as homework, ready for next week's review. | Remainder of the 4 AHs |

Notes:

- **Week 1** has no previous-week review (course kickoff).
- **Week 8 (Midterm)** and **Week 15 (Final)** replace this format with team presentations (see [Assessment](#assessment-two-presentations)).
- The review in Part 1 is of the *previous* week's tasks, so each week's deliverable is presented at the start of the next session.

## Week 1 — Start Here

Complete the Week 1 onboarding before beginning the later labs:

1. Follow the [Week 1 orientation lab](labs/week-01-orientation/) to verify your team workspaces and access to the shared NomNaOCR benchmark.
2. Follow the [Slurm Quick Guide](https://github.com/ntcuong2103/project-ws2026/blob/main/SLURM-QUICK-GUIDE.md) to verify GPU access.
3. Read the repository's [AI workspace rules](AGENTS.md) before using a coding agent.

## Tentative 15-Week Schedule

The table gives the overview; the per-week details (reading, provided materials, tasks) are in [Weekly Task Details](#weekly-task-details). Each week has a **lab tutorial folder** linked in the *Lab* column.

| Week | Phase | Topic & In-Class Focus | Lab / Studio Work | Deliverable | Lab |
| --- | --- | --- | --- | --- | --- |
| 1 | Orientation | Course kickoff; journal-club read of the source paper; intro to Hán-Nôm OCR and IDS decomposition; team formation | Verify access; prepare five image/line-label/detection-label triplets; visualize line polygons and character boxes | Teams formed; environment verified; five sample triplets inspected and visualized | [labs/week-01-orientation](labs/week-01-orientation/) |
| 2 | Foundations | Two-paper problem formulation; NomNaOCR readiness; character-level edit distance | Explain how line supervision and detector output become pseudo ground truth; audit pairing/labels and five sample triplets; analyze OCR-style transcription errors | Executed readiness/edit-distance notebook with problem statement and pseudo-ground-truth pipeline explanation | [labs/week-02-foundations](labs/week-02-foundations/) |
| 3 | Detection I | YOLO dataset structure, single-class character detection, and metrics | Organize a raw 18/2/4 MTH1200 practice sample; verify box conversion; train on MTH1200; calculate IoU; compare COCO and trained checkpoints on MTH1000 | Validated practice data, MTH checkpoint, student-authored Slurm artifacts, and interpreted before/after metrics | [Week 3 lab](labs/week-03-detection-1/) |
| 4 | Detection II | Nôm adaptation and external evaluation | Inspect Nakagawa class collapse; fine-tune the Week 3 checkpoint; compare both checkpoints on external NomNaOCR data | Fine-tuned checkpoint, student-authored jobs, Nakagawa test metrics, NomNaOCR comparison, and prediction handoff | [Week 4 lab](labs/week-04-detection-2/) |
| 5 | Recognition I | IDS operators, direct/expanded decompositions, and recognizer token vocabulary | Validate Chinese examples; analyze one Nôm page; verify with zi.tools; export unlabeled detector crops | IDS converter, page coverage report, manual checks, and inference crops | [labs/week-05-recognition-1](labs/week-05-recognition-1/) |
| 6 | Recognition II | Recognizer logits, token vocabulary, and forward passes | Inspect synthetic, Chinese, Nôm, and real-box logits using the given checkpoint | Executed logits notebook + one-page top-token report | [labs/week-06-recognition-2](labs/week-06-recognition-2/) |
| 7 | Recognition III | Autoregressive greedy decoding, deduplication, IDS→Unicode composition | Implement the decoder from next-token logits; run the full detector→recognizer pipeline; prepare midterm slides | Decoder tests + full per-box output + combined demo | [labs/week-07-recognition-3](labs/week-07-recognition-3/) |
| 8 | Midterm | Midterm presentations: detection + recognition results, error analysis, alignment-stage plan | Team presentations (\~15 min + Q&A) | **Midterm presentation (graded)** | [labs/week-08-midterm](labs/week-08-midterm/) |
| 9 | Alignment I | Box-to-line assignment; formalizing correspondence recovery | Implement center-in-rectangle box-to-line test and grouping/sorting; validate on 2 synthetic pages | Box-to-line assignment module, validated then run on real data | [labs/week-09-alignment-1](labs/week-09-alignment-1/) |
| 10 | Alignment II | Two-level IDS-level DP alignment — the paper's Algorithm 1 | Implement outer DP + backtracking with inner IDS edit distance; verify against hand-computed toy example | DP alignment implementation verified on a toy example | [labs/week-10-alignment-2](labs/week-10-alignment-2/) |
| 11 | Alignment III | Confidence score s\_i; binary-flag baseline; running at scale | Add confidence score (Eq. 2); implement binary-flag variant; run both on whole dataset; plot s\_i distribution | Confidence-weighted alignment + binary-flag baseline run on whole dataset | [labs/week-11-alignment-3](labs/week-11-alignment-3/) |
| 12 | Pseudo-labeling I | Confidence-weighted training loop design | Implement confidence-weighted loss; wire recognize→align→weight→fine-tune loop; run one real iteration | Confidence-weighted fine-tuning loop run for one iteration | [labs/week-12-pseudo-labeling-1](labs/week-12-pseudo-labeling-1/) |
| 13 | Pseudo-labeling II & Metrics | Modified CER; precision/recall by confidence band | Implement both metrics on toy examples; build human-verified holdout with annotation app; run on holdout | Modified CER + confidence-band P/R, validated then run on holdout | [labs/week-13-pseudo-labeling-2](labs/week-13-pseudo-labeling-2/) |
| 14 | Integration | Robustness to detector segmentation noise; per-document-type analysis | Implement box duplicate/drop noise injection; run at 5%/10%; per-document-type table; dry-run final talk | Noise-injection implementation + robustness results + finalized report materials | [labs/week-14-integration](labs/week-14-integration/) |
| 15 | Final | Final presentations: full pipeline, results vs. baseline, robustness, lessons learned | Team presentations (\~20 min + Q&A/demo) | **Final presentation (graded) + written report + code repository** | [labs/week-15-final](labs/week-15-final/) |

## Lab Tutorials

Each week has its own folder under [`labs/`](labs/) holding that week's lab tutorial (`README.md`) and any supporting material (notebooks, toy fixtures, scripts, expected outputs).

| Week | Folder |
| --- | --- |
| 1 | [labs/week-01-orientation](labs/week-01-orientation/) |
| 2 | [labs/week-02-foundations](labs/week-02-foundations/) |
| 3 | [labs/week-03-detection-1](labs/week-03-detection-1/) |
| 4 | [labs/week-04-detection-2](labs/week-04-detection-2/) |
| 5 | [labs/week-05-recognition-1](labs/week-05-recognition-1/) |
| 6 | [labs/week-06-recognition-2](labs/week-06-recognition-2/) |
| 7 | [labs/week-07-recognition-3](labs/week-07-recognition-3/) |
| 8 | [labs/week-08-midterm](labs/week-08-midterm/) |
| 9 | [labs/week-09-alignment-1](labs/week-09-alignment-1/) |
| 10 | [labs/week-10-alignment-2](labs/week-10-alignment-2/) |
| 11 | [labs/week-11-alignment-3](labs/week-11-alignment-3/) |
| 12 | [labs/week-12-pseudo-labeling-1](labs/week-12-pseudo-labeling-1/) |
| 13 | [labs/week-13-pseudo-labeling-2](labs/week-13-pseudo-labeling-2/) |
| 14 | [labs/week-14-integration](labs/week-14-integration/) |
| 15 | [labs/week-15-final](labs/week-15-final/) |

## Weekly Task Details

### Week 1 — Orientation

**Topic:** Course kickoff; journal-club read of the source paper; intro to Hán-Nôm OCR and IDS decomposition; team formation.
**Reading:** Nguyen et al., "Confidence-Weighted Annotation of Historical Nôm Documents via IDS-Level Sequence Alignment," *PRL* (2026) — read in full.
**Provided lab materials:** workspace rules, a reusable sample-data utility, and a notebook for inspecting five NomNaOCR image/line-label/detection-label triplets.
**Lab:** [labs/week-01-orientation](labs/week-01-orientation/)

**Tasks:**

- [ ] Verify server workspace and shared-data access
- [ ] Clone this lab repository under `/home/projectgX`; run `uv sync`
- [ ] Create `/data/projectgX/lab-exercise`; prepare its persistent sample using the provided CPU-only `sbatch` job
- [ ] Inspect the Slurm job and its resource request
- [ ] Visualize line polygons and character-position boxes on all five sample pages from a short CPU-only interactive allocation
- [ ] Read this repository's `AGENTS.md`
- [ ] Read the paper's abstract + Figure 1; come ready to discuss
- [ ] Form teams of 3–4 and submit the team roster

**Deliverable:** Teams formed; environment and Slurm verified; five local sample triplets loaded with both annotation levels and visualized successfully.

### Week 2 — Foundations

**Topic:** Problem formulation from the dataset and automated-annotation papers; NomNaOCR EDA; character-level edit distance and normalized similarity.
**Reading:** Dang et al., "NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script," RIVF (2022); Nguyen et al., "Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer," ICDAR (2026).
**Starter kit (given):** full read-only NomNaOCR access, the persistent five-triplet Week 1 sample, and a guided EDA/edit-distance notebook.
**Lab:** [labs/week-02-foundations](labs/week-02-foundations/)

**Tasks:**

- [ ] Compare the two papers; explain the missing character-level supervision, why it is needed, and what paper result the team will reproduce
- [ ] Explain how character-position detection, box-to-line ordering, recognition/alignment, and confidence filtering convert line transcriptions into character-level pseudo ground truth
- [ ] Audit triplet pairing and label validity; explain the downstream failure prevented by each check
- [ ] Audit page quality, layout, reading order, line labels, and pretrained-detector reference boxes on all five Week 1 pages
- [ ] Implement four edit distances and use OCR-style errors on a matching reference/hypothesis transcription pair to interpret them

**Deliverable:** Executed readiness/edit-distance notebook with an evidence-based problem statement, reproduction target, and explanation of the line-level-to-character-level pseudo-ground-truth pipeline.

### Week 3 — Detection I

**Topic:** YOLOv11n single-class character detection, dataset organization, and transfer learning.
**Reading:** Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024); Ma et al., ICFHR (2020).
**Starter kit (given):** deterministic raw-sample staging and label conversion/validation helpers.
**You implement:** organize the small dataset, verify box conversion, write MTH1200 training/evaluation jobs, and interpret MTH1000 metrics.
**Lab:** [Week 3 lab](labs/week-03-detection-1/)

**Tasks:**

- [ ] Organize the staged 24-page MTH1200 sample into 18 training, 2 validation, and 4 practice-test pages
- [ ] Convert one pixel box by hand, convert the remaining raw labels, and pass the dataset validator
- [ ] Inspect the shared 90/10 MTH1200 train/validation and full MTH1000 test view
- [ ] Write and submit MTH1200 training and evaluation jobs with `sbatch`; inspect local Ultralytics artifacts
- [ ] Calculate IoU by hand; explain precision, recall, mAP50, and mAP50–95 before comparing untouched COCO weights with the selected MTH checkpoint on MTH1000

**Deliverable:** validated practice structure, MTH1200-trained detector with `best.pt` and `last.pt`, Slurm artifacts, and an interpreted MTH1000 before/after comparison.

### Week 4 — Detection II

**Topic:** Domain adaptation, external evaluation, and detector error modes.
**Starter kit (given):** Nakagawa and NomNaOCR preparation helpers.
**You implement:** verify Nakagawa's single-class conversion, write fine-tuning/evaluation jobs, and compare both checkpoints on NomNaOCR.
**Lab:** [Week 4 lab](labs/week-04-detection-2/)

**Tasks:**

- [ ] Write jobs to fine-tune the Week 3 detector on TUAT-Nakagawa and report held-out test metrics
- [ ] Prepare the supplied NomNaOCR archive as one external test set
- [ ] Evaluate the MTH and Nakagawa checkpoints on identical NomNaOCR data
- [ ] Interpret precision, recall, and localization changes without treating the paper numbers as a reproduction target

**Deliverable:** Nakagawa-fine-tuned detector, student-authored Slurm artifacts, Nakagawa test metrics, NomNaOCR before/after comparison, and the detector prediction handoff.

### Week 5 — Recognition I

**Topic:** IDS operators, direct versus recursively expanded decompositions, and text-level tokenization for Nôm recognition.
**Reading:** Morioka, "CHISE" (2008); Zhang et al., "Radical analysis network for zero-shot learning in printed Chinese character recognition" (2018).
**Starter kit (given):** pinned and checksum-verified `nom-ids` model resources, CJKVI source data, a runnable reference converter, and an unlabeled crop exporter.
**You implement:** an equivalent character-by-character text→IDS converter in the team repository, then audit the full read-only NomNaOCR corpus.
**Small experiment:** reproduce the exact expanded IDS for `休好字明林`, compare direct and expanded forms, and manually verify five Nôm characters with zi.tools.
**Lab:** [labs/week-05-recognition-1](labs/week-05-recognition-1/)

**Tasks:**

- [ ] Download and verify the pinned IDS resources without changing vocabulary order
- [ ] Validate text→IDS conversion on the five Chinese reference characters
- [ ] Report corpus-wide coverage and the ten most common OOV characters, each with one inspectable image/line-label example
- [ ] Record five zi.tools checks and explain raw/expanded or glyph-variant differences
- [ ] Inspect exported detector crops: run the Week 3–4 detector and your team crop exporter first. These boxes remain unlabeled; line transcription order is not a safe box-to-character assignment.

**Deliverable:** Validated IDS converter, corpus-wide coverage report with ten OOV examples, zi.tools verification table, and unlabeled recognizer inference crops.

### Week 6 — Recognition II

**Topic:** Encoder–decoder/attention-based recognizers (BTTR-style); logits and teacher-forced forward passes.
**Reading:** Zhao et al., "BTTR," ICDAR (2021).
**Starter kit (given):** a YAML-configured LitBTTR adapter that returns crop/prefix logits; training the recognizer itself is out of scope.
**You implement:** inspect logits, token IDs, softmax/top-*k* results, and prefix dependence. The decoder is Week 7 work.
**Small experiment:** calculate argmax and probabilities for hand-crafted logits before inspecting the checkpoint's Chinese and Nôm crop logits.
**Lab:** [labs/week-06-recognition-2](labs/week-06-recognition-2/)

**Tasks:**

- [ ] Inspect hand-crafted, Chinese, and Nôm logits with the configured vocabulary
- [ ] Explain why independent teacher-forced argmax values are not an inference decoder
- [ ] Run the forward adapter on one page of the full Week 4 detector handoff
- [ ] Save a per-box top-token report

**Deliverable:** Executed logits notebook, Chinese/Nôm inspection notes, and a one-page top-token report.

### Week 7 — Recognition III

**Topic:** Autoregressive greedy decoding, deduplication, IDS→Unicode composition, and midterm preparation.
**Starter kit (given):** next-token logits primitive and fixed decoder fixtures.
**You implement:** greedy decoding from SOS through EOS/max length, then deduplication and one-to-many IDS→Unicode candidate lookup.
**Small experiment:** exact-match normal EOS, immediate EOS, repeated-token, and maximum-length decoder fixtures.
**Lab:** [labs/week-07-recognition-3](labs/week-07-recognition-3/)

**Tasks:**

- [ ] Implement and exact-match test the greedy decoder
- [ ] Implement sequence deduplication and IDS→Unicode candidate lookup
- [ ] Run the full page → detector boxes → recognizer → decoder pipeline on the complete handoff
- [ ] Assemble a combined detection+recognition demo; draft midterm slides

**Deliverable:** Decoder module (validated on fixtures), full per-box recognition output, combined demo, and midterm slide draft.

### Week 8 — Midterm

**Topic:** Midterm presentations (\~15 min + Q&A per team): detection + recognition results, error analysis, alignment-stage plan.
**Lab:** [labs/week-08-midterm](labs/week-08-midterm/)

Read [the midterm presentation rules](docs/MIDTERM-PRESENTATION.md), including the required
team-membership slide and individual contribution statements.

**Tasks:**

- [ ] Deliver the midterm presentation
- [ ] Submit slides and current repo state before class

**Deliverable:** Midterm presentation (graded).

### Week 9 — Alignment I

**Topic:** Box-to-line assignment; formalizing correspondence recovery.
**Reading:** Nguyen et al., "Automated Character-Level Annotation...," ICDAR (2026) — the paper's own prior work.
**Starter kit (given):** none — this is core to the task.
**You implement:** the center-in-rectangle box-to-line test and the box-to-line grouping/sorting logic, from the paper's description alone.
**Small experiment:** 2 hand-built synthetic pages (a handful of boxes, 2–3 lines) where you know by hand which box belongs to which line; verify your function against that before running on real pages.
**Lab:** [labs/week-09-alignment-1](labs/week-09-alignment-1/)

**Tasks:**

- [ ] Implement box-to-line assignment from the paper's §3.2 description
- [ ] Validate on 2 hand-built synthetic pages
- [ ] Run it on the team's real detected boxes and ground-truth lines
- [ ] Write up 2–3 edge cases you had to decide on (e.g., a box on a line boundary)

**Deliverable:** Box-to-line assignment module, validated on synthetic pages, then run on real data.

### Week 10 — Alignment II (the core algorithm)

**Topic:** Two-level IDS-level dynamic-programming alignment — the paper's Algorithm 1.
**Starter kit (given):** a function **stub** with the paper's signature/docstring (`raise NotImplementedError` in place of the body); a plain Levenshtein `edit_distance()` utility as a building block (implement it yourself first if you don't already have it from Week 9).
**You implement:** the full two-level DP alignment — outer character-to-character alignment scored by inner IDS edit distance, length-proportional gap costs, and backtracking to recover the correspondence map (Algorithm 1, Eq. 1).
**Small experiment:** a hand-built toy example (5–6 predicted boxes' IDS strings vs. 5–6 ground-truth characters' IDS strings) where you compute the expected DP table and correspondence by hand first, then check your code reproduces it exactly.
**Lab:** [labs/week-10-alignment-2](labs/week-10-alignment-2/)

**Tasks:**

- [ ] Implement `edit_distance()` (if needed) and the outer DP table + backtracking per Algorithm 1
- [ ] Hand-compute the expected DP table and alignment for the toy example
- [ ] Verify your code matches the hand-computed result exactly; fix discrepancies
- [ ] Document the two-level cost model in your own words (outer vs. inner)

**Deliverable:** DP alignment implementation, verified against a hand-computed toy example.

### Week 11 — Alignment III

**Topic:** Confidence score s\_i; binary-flag baseline; running at scale.
**Starter kit (given):** none new.
**You implement:** the confidence score s\_i = 1 − ED(p\_i, g\_j)/max(|p\_i|,|g\_j|) (Eq. 2) on top of your Week 10 alignment, and a binary-flag variant (accept only on exact match) for the Table 2 comparison.
**Small experiment:** compute s\_i by hand for 3–4 aligned pairs in your Week 10 toy example and check your code's output matches.
**Lab:** [labs/week-11-alignment-3](labs/week-11-alignment-3/)

**Tasks:**

- [ ] Add confidence-score computation to your alignment; verify by hand on the toy example
- [ ] Implement the binary-flag variant of the same alignment
- [ ] Run both variants on the whole dataset; save both label sets
- [ ] Plot the distribution of s\_i values across the whole dataset

**Deliverable:** Confidence-weighted alignment + binary-flag baseline, both run on the whole dataset.

### Week 12 — Pseudo-labeling I

**Topic:** Confidence-weighted training loop design.
**Starter kit (given):** a minimal fine-tuning loop skeleton (data loading + checkpointing only — the loss function is a stub).
**You implement:** a confidence-weighted loss (`loss = sum(s_i * per_box_loss_i)`) and the retrain/re-infer/re-align iteration loop.
**Small experiment:** on a 5-example synthetic mini-batch with hand-picked s\_i values, verify your weighted loss computes the expected number by hand before running real fine-tuning.
**Lab:** [labs/week-12-pseudo-labeling-1](labs/week-12-pseudo-labeling-1/)

**Tasks:**

- [ ] Implement the confidence-weighted loss; verify on the synthetic mini-batch
- [ ] Wire it into a fine-tuning loop (recognize → align → weight → fine-tune)
- [ ] Run one real iteration on the whole dataset
- [ ] Log CER and labeled coverage before/after

**Deliverable:** Confidence-weighted fine-tuning loop, validated on a toy mini-batch, run for one real iteration.

### Week 13 — Pseudo-labeling II & Metrics

**Topic:** Modified CER; precision/recall by confidence band.
**Reading:** source paper §3.3–3.4, §4.2–4.4.3 (re-read).
**Starter kit (given):** the annotation review app, given as-is — building a web annotation tool is out of scope.
**You implement:** the modified CER metric (§3.4: expand each character to its Unicode-variant set before computing edit distance) and the confidence-band precision/recall computation.
**Small experiment:** hand-compute the modified CER on 2–3 short strings containing at least one known Unicode-variant pair, and hand-verify precision/recall on a 10-item toy confusion table — check both against your implementation before touching the real holdout.
**Lab:** [labs/week-13-pseudo-labeling-2](labs/week-13-pseudo-labeling-2/)

**Tasks:**

- [ ] Implement modified CER; verify against your hand-computed toy examples
- [ ] Implement precision/recall by confidence band; verify on the 10-item toy table
- [ ] Use the annotation app to build a small human-verified holdout for the whole dataset
- [ ] Compute both metrics on the real holdout; run 1–2 more self-training iterations if time allows

**Deliverable:** Modified CER + confidence-band precision/recall, validated on toy examples, then run against the team's holdout.

### Week 14 — Integration

**Topic:** Robustness to detector segmentation noise; per-document-type analysis.
**Reading:** source paper §4.4.5, §5 (Discussion) (re-read).
**Starter kit (given):** none.
**You implement:** a noise-injection function that duplicates or drops boxes at a controlled rate.
**Small experiment:** on a synthetic list of 10 boxes, inject 30% noise and verify the resulting count and which boxes were affected match what you expect, before running at the paper's 5%/10% rates on real data.
**Lab:** [labs/week-14-integration](labs/week-14-integration/)

**Tasks:**

- [ ] Implement box duplication/drop noise injection; verify on the 10-box synthetic list
- [ ] Run the noise experiment at 5% and 10% on the team's real detection output
- [ ] Build the per-document-type table (binary-flag vs. graded alignment) for all document types in the dataset
- [ ] Finalize report tables/figures; full dry-run of the final presentation

**Deliverable:** Noise-injection implementation (validated on a toy list) + robustness results + finalized report materials.

### Week 15 — Final

**Topic:** Final presentations (\~20 min + Q&A/demo per team): full pipeline, results vs. baseline, robustness findings, lessons learned.
**Lab:** [labs/week-15-final](labs/week-15-final/)

**Tasks:**

- [ ] Deliver the final presentation
- [ ] Submit the written report and the code repository

**Deliverable:** Final presentation (graded) + written report + code repository.

## Reading List

| Assign by | Reading | Focus |
| --- | --- | --- |
| Week 1 | Nguyen et al., "Confidence-Weighted Annotation of Historical Nôm Documents via IDS-Level Sequence Alignment," *Pattern Recognition Letters* (2026) | The paper the whole course builds toward; read in full for the Week 1 journal club |
| Week 2 | Dang et al., "NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script," RIVF (2022) | The dataset every team trains and evaluates on |
| Week 2; re-read Week 9 | Nguyen et al., "Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer," ICDAR (2026) | Problem formulation, detector–recognizer pipeline, and iterative pseudo-labeling baseline |
| Week 3–4 | Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024) | Detector architecture used in `detection.py` |
| Week 3–4 | Ma et al., "Joint Layout Analysis, Character Detection and Recognition for Historical Document Digitization," ICFHR (2020) | TUAT-Nakagawa detection annotations used to fine-tune the detector |
| Week 5–7 | Zhao et al., "Handwritten Mathematical Expression Recognition with A Bidirectionally Trained Transformer (BTTR)," ICDAR (2021) | Architecture behind `LitBTTR`, the recognizer in `ocr_inference.py` |
| Week 5 | Morioka, "CHISE: Character Processing Based on Character Ontology" (2008) | Character ontology and the limits of coded character sets for historical ideographs |
| Week 5 | Zhang et al., "Radical analysis network for zero-shot learning in printed Chinese character recognition," ICME (2018) | Why shared structural tokens enable recognition beyond atomic character classes |
| Week 12–13 | Source paper, §3.3–3.4 and §4.2–4.4.3 (re-read) | Confidence-weighted selection/training design and the modified CER metric, to implement directly |
| Week 14 | Source paper, §4.4.5 and §5 (Discussion) (re-read) | Robustness-to-noise experiment design and the paper's own stated limitations |

## Assessment: Two Presentations

- **Midterm presentation (Week 8, \~15 min + Q&A per team):** covers the detection and recognition stages — architecture choices, training metrics (P/R/mAP, per-character accuracy), error analysis, and the team's plan for the alignment stage. Feedback here directly shapes each team's Weeks 9–11 work.
- **Final presentation (Week 15, \~20 min + Q&A/demo per team):** covers the complete pipeline — the alignment algorithm, confidence-weighted pseudo-labeling results, evaluation against the binary-flag baseline (CER, coverage, precision/recall by confidence band), and robustness findings. Accompanied by a written report and code repository submission.

**Weighting:**

| Assessment | Weight |
| --- | --- |
| Midterm presentation | 20% |
| Final presentation | 60% |
| Code/report | 10% |
| Review session | 10% |

## Assumptions & Next Steps

- The paper's full iterative self-updating loop runs 6 rounds; Weeks 12–13 compress this to 1–2 rounds given time constraints, with room to extend if pacing allows.
- Each team will handle the whole NomNaOCR dataset, covering all document types, for comparable final results.
- Still open: grading rubrics per presentation.
