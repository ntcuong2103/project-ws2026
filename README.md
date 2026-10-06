# Confidence-Weighted Nôm OCR - Project Course

## Course Overview

Students design, build, and evaluate a full document-AI pipeline — character detection, radical-based recognition, IDS-level sequence alignment, and confidence-weighted pseudo-labeling — modeled on a published research system for historical Hán-Nôm document annotation (Nguyen et al., *Pattern Recognition Letters*, 2026).

- **Format:** 15 weeks, 4 contact hours/week (60 hours total), teams of 3–4 students
- **Structure:** the whole class moves through the pipeline stage by stage (detection → recognition → alignment → pseudo-labeling); each team builds the complete pipeline end-to-end on an assigned document subset, so results are comparable across teams at the final presentation
- **Prerequisites:** deep learning foundations (CNNs, training loops, PyTorch/TensorFlow), basic Python; no prior OCR/CV or CJK-script experience assumed
- **Resources:** lab GPUs and the datasets used in the source paper — the NomNaOCR corpus (2,953 pages), TUAT-Nakagawa detection annotations, MTHv2 pretraining set, and the CHISE/cjkvi-ids dictionary
- **Learning objectives:** by Week 15, teams can fine-tune an object detector for character-level localization; train a sequence model to predict compositional (IDS) character representations; implement a dynamic-programming alignment algorithm that recovers box-to-character correspondence with a graded confidence score; run an iterative confidence-weighted self-training loop; and evaluate results against a baseline using CER and precision/recall metrics

## Starter Kit and Lab Philosophy

Students get a **stripped-down starter kit** derived from [nom-ocr-inference](https://github.com/ntcuong2103/nom-ocr-inference/tree/claude-edit) and its companion `nom-ids` package. The surrounding plumbing (config, CLI, I/O, data loading, the pretrained detector/recognizer checkpoints) is given, but the functions that embody the paper's actual contributions — the IDS-level alignment, the confidence-weighted training, the modified CER, the noise-robustness test — ship as stubs (docstring + `raise NotImplementedError`).

Every stub comes with a tiny, hand-checkable synthetic example so a team can validate its own implementation cheaply before running it on real pages or the GPU cluster. The instructor's own working implementation stays private, used only to generate expected outputs for these toy examples and for grading.

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

## Tentative 15-Week Schedule

The table gives the overview; the per-week details (reading, starter kit, tasks) are in [Weekly Task Details](#weekly-task-details). Each week has a **lab tutorial folder** linked in the *Lab* column.

| Week | Phase | Topic & In-Class Focus | Lab / Studio Work | Deliverable | Lab |
| --- | --- | --- | --- | --- | --- |
| 1 | Orientation | Course kickoff; journal-club read of the source paper; intro to Hán-Nôm OCR and IDS decomposition; team formation | Clone starter repo + `nom-ids`; verify GPU and 5-page sample access | Teams formed; environment verified; one sample page loaded and printed | [labs/week-01-orientation](labs/week-01-orientation/) |
| 2 | Foundations | Object detection for character localization; DP sequence-alignment primer; weak supervision / pseudo-labeling concepts | Browse the assigned folder's images/labels; skim NW/edit-distance primer; draft project charter | Project charter; each team assigned a document subset | [labs/week-02-foundations](labs/week-02-foundations/) |
| 3 | Detection I | YOLOv11n architecture; annotation format; transfer learning from MTHv2 pretraining | Run given YOLO script on small MTHv2 sample; scale to full MTHv2; detect + visualize on NomNaOCR | Trained YOLO detector + detection output and visualizations on team's NomNaOCR subset | [labs/week-03-detection-1](labs/week-03-detection-1/) |
| 4 | Detection II | Detection metrics (Precision/Recall/mAP@50); detector error modes (merges, splits, spurious/missed boxes) | Fine-tune MTHv2-pretrained YOLO on TUAT-Nakagawa; evaluate with ultralytics; tag errors | Fine-tuned detector + metrics report + error-analysis gallery | [labs/week-04-detection-2](labs/week-04-detection-2/) |
| 5 | Recognition I | Ideographic Description Sequences (IDS); CHISE/cjkvi-ids dictionary; compositional character representation | Implement text→IDS mapping; validate on 5-char example; report vocabulary coverage; prepare crops | IDS decomposition utility + recognizer training set | [labs/week-05-recognition-1](labs/week-05-recognition-1/) |
| 6 | Recognition II | Encoder–decoder / attention-based recognizers (BTTR-style); greedy decoding | Implement greedy decoding from logits; validate on synthetic fixtures; run on real boxes with given checkpoint | Greedy-decode implementation + raw inference output | [labs/week-06-recognition-2](labs/week-06-recognition-2/) |
| 7 | Recognition III | Deduplication for speed; IDS→Unicode composition; midterm prep | Implement dedup + IDS→Unicode; full decode on subset; combined detection+recognition demo | Decoding module + combined demo + midterm slide draft | [labs/week-07-recognition-3](labs/week-07-recognition-3/) |
| 8 | Midterm | Midterm presentations: detection + recognition results, error analysis, alignment-stage plan | Team presentations (\~15 min + Q&A) | **Midterm presentation (graded)** | [labs/week-08-midterm](labs/week-08-midterm/) |
| 9 | Alignment I | Box-to-line assignment; formalizing correspondence recovery | Implement center-in-rectangle box-to-line test and grouping/sorting; validate on 2 synthetic pages | Box-to-line assignment module, validated then run on real data | [labs/week-09-alignment-1](labs/week-09-alignment-1/) |
| 10 | Alignment II | Two-level IDS-level DP alignment — the paper's Algorithm 1 | Implement outer DP + backtracking with inner IDS edit distance; verify against hand-computed toy example | DP alignment implementation verified on a toy example | [labs/week-10-alignment-2](labs/week-10-alignment-2/) |
| 11 | Alignment III | Confidence score s\_i; binary-flag baseline; running at scale | Add confidence score (Eq. 2); implement binary-flag variant; run both on full subset; plot s\_i distribution | Confidence-weighted alignment + binary-flag baseline run on team's subset | [labs/week-11-alignment-3](labs/week-11-alignment-3/) |
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
**Starter kit (given):** empty project scaffold — `config.py`, folder layout, a 5-page sample of images + line labels, and the IDS dictionary/vocab files. No pipeline code yet.
**Lab:** [labs/week-01-orientation](labs/week-01-orientation/)

**Tasks:**

- [ ] Clone the starter repo and `nom-ids` (release branch); `uv sync`
- [ ] Verify GPU access and read/write access to the 5-page sample
- [ ] Read `Readme.md`/`CLAUDE.md` and the paper's abstract + Figure 1; come ready to discuss
- [ ] Form teams of 3–4 and submit the team roster

**Deliverable:** Teams formed; environment verified; one sample page loaded and printed successfully.

### Week 2 — Foundations

**Topic:** Object detection for character localization; DP sequence-alignment primer; weak supervision/pseudo-labeling concepts.
**Reading:** Dang et al., "NomNaOCR: The First Dataset for OCR on Han-Nom Script," RIVF (2022).
**Starter kit (given):** full dataset access for the team's assigned folder; a blank project-charter template.
**Lab:** [labs/week-02-foundations](labs/week-02-foundations/)

**Tasks:**

- [ ] Browse the full images/line-labels for the team's assigned folder
- [ ] Skim a Needleman–Wunsch/edit-distance primer and one weak-supervision reference
- [ ] Team assigned one document folder/volume (from the paper's Table 4 list)
- [ ] Submit a one-page project charter (goals, roles, risks)

**Deliverable:** Project charter; team assigned a document subset.

### Week 3 — Detection I

**Topic:** YOLOv11n architecture; annotation format; transfer learning from MTHv2 pretraining.
**Reading:** Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024); Ma et al., ICFHR (2020).
**Starter kit (given):** YOLO training script, small number of samples from MTHv2, YOLO detection, visualization script.
**You implement:** YOLO training script for full samples of MTHv2, run detection and visualize on NomNaOCR.
**Lab:** [labs/week-03-detection-1](labs/week-03-detection-1/)

**Tasks:**

- [ ] Run the given YOLO training script on the small MTHv2 sample; confirm the training loop runs end-to-end
- [ ] Scale training up to the full set of MTHv2 samples
- [ ] Run the trained detector on the team's assigned NomNaOCR subset
- [ ] Visualize detections with the given visualization script and review a few pages

**Deliverable:** Trained YOLO detector (validated first on the small MTHv2 sample, then trained at full scale) + detection output and visualizations on the team's NomNaOCR subset.

### Week 4 — Detection II

**Topic:** Detection metrics; detector error modes (merges, splits, spurious/missed boxes).
**Starter kit (given):** Precision/Recall/mAP@50 using ultralytics.
**You implement:** fine-tuning the pretrained MTHv2 detector on the NakagawaLab dataset, then check the metrics after training.
**Lab:** [labs/week-04-detection-2](labs/week-04-detection-2/)

**Tasks:**

- [ ] Fine-tune the MTHv2-pretrained YOLO detector on the TUAT-Nakagawa (NakagawaLab) dataset
- [ ] Compute Precision/Recall/mAP@50 with ultralytics' built-in evaluation after training
- [ ] Visualize detections on 5–10 pages using your Week 3 visualization script
- [ ] Tag at least 10 detector errors by type (merge/split/spurious/missed); write a half-page error-analysis note

**Deliverable:** Fine-tuned detector + ultralytics metrics report + error-analysis gallery.

### Week 5 — Recognition I

**Topic:** Ideographic Description Sequences (IDS); CHISE/cjkvi-ids dictionary; compositional character representation.
**Reading:** Zhang, Du & Dai, "Radical analysis network..." (2020); Morioka, "CHISE" (2008).
**Starter kit (given):** the character→IDS dictionary and vocabulary files only — no decomposition code.
**You implement:** a function mapping ground-truth line text to IDS strings, character by character.
**Small experiment:** test on a hand-picked 5-character line where you know the expected IDS strings by dictionary lookup, before running over the full assigned subset.
**Lab:** [labs/week-05-recognition-1](labs/week-05-recognition-1/)

**Tasks:**

- [ ] Implement the text→IDS mapping function
- [ ] Validate it on the 5-character hand-picked example
- [ ] Run it over the team's full assigned subset; report vocabulary coverage %
- [ ] Prepare the cropped character-image set needed for recognizer inference

**Deliverable:** IDS decomposition utility (validated on a toy example) + recognizer training set.

### Week 6 — Recognition II

**Topic:** Encoder–decoder/attention-based recognizers (BTTR-style); greedy decoding.
**Reading:** Zhao et al., "BTTR," ICDAR (2021).
**Starter kit (given):** the pretrained LitBTTR checkpoint and its forward pass (crop in, per-step logits out) — training the recognizer itself is out of scope.
**You implement:** the greedy decoding loop that turns per-step logits into a token sequence (argmax + stopping condition), from scratch.
**Small experiment:** run your decode loop on 3–4 hand-crafted logits arrays with known expected output tokens before wiring it to the real model.
**Lab:** [labs/week-06-recognition-2](labs/week-06-recognition-2/)

**Tasks:**

- [ ] Implement greedy decoding from raw per-step logits
- [ ] Validate on the synthetic logits fixtures (exact match required)
- [ ] Run your decode loop on the team's real detected boxes using the given checkpoint
- [ ] Save raw token sequences for the team's subset

**Deliverable:** Greedy-decode implementation (validated on synthetic fixtures) + raw inference output.

### Week 7 — Recognition III

**Topic:** Deduplication for speed; IDS→Unicode composition; midterm prep.
**Starter kit (given):** none new.
**You implement:** deduplication of identical token sequences before decoding, and mapping a decoded IDS string back to a composed Unicode character where possible.
**Small experiment:** test both functions on a handful of hand-picked token sequences (including one repeated sequence and one IDS string with no direct Unicode match) with known expected outputs.
**Lab:** [labs/week-07-recognition-3](labs/week-07-recognition-3/)

**Tasks:**

- [ ] Implement sequence deduplication and IDS→Unicode composition
- [ ] Validate both on the hand-picked examples
- [ ] Run the full decode step on the team's subset to produce predicted text + IDS per box
- [ ] Assemble a combined detection+recognition demo; draft midterm slides

**Deliverable:** Decoding module (validated on toy examples) + combined demo + midterm slide draft.

### Week 8 — Midterm

**Topic:** Midterm presentations (\~15 min + Q&A per team): detection + recognition results, error analysis, alignment-stage plan.
**Lab:** [labs/week-08-midterm](labs/week-08-midterm/)

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
- [ ] Run both variants on the team's full assigned subset; save both label sets
- [ ] Plot the distribution of s\_i values across the team's subset

**Deliverable:** Confidence-weighted alignment + binary-flag baseline, both run on the team's subset.

### Week 12 — Pseudo-labeling I

**Topic:** Confidence-weighted training loop design.
**Starter kit (given):** a minimal fine-tuning loop skeleton (data loading + checkpointing only — the loss function is a stub).
**You implement:** a confidence-weighted loss (`loss = sum(s_i * per_box_loss_i)`) and the retrain/re-infer/re-align iteration loop.
**Small experiment:** on a 5-example synthetic mini-batch with hand-picked s\_i values, verify your weighted loss computes the expected number by hand before running real fine-tuning.
**Lab:** [labs/week-12-pseudo-labeling-1](labs/week-12-pseudo-labeling-1/)

**Tasks:**

- [ ] Implement the confidence-weighted loss; verify on the synthetic mini-batch
- [ ] Wire it into a fine-tuning loop (recognize → align → weight → fine-tune)
- [ ] Run one real iteration on the team's subset
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
- [ ] Use the annotation app to build a small human-verified holdout for your subset
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
- [ ] Build the per-document-type table (binary-flag vs. graded alignment) for the team's folder
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
| Week 1–2 | Dang et al., "NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script," RIVF (2022) | The dataset every team trains and evaluates on |
| Week 3–4 | Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024) | Detector architecture used in `detection.py` |
| Week 3–4 | Ma et al., "Joint Layout Analysis, Character Detection and Recognition for Historical Document Digitization," ICFHR (2020) | TUAT-Nakagawa detection annotations used to fine-tune the detector |
| Week 5–7 | Zhao et al., "Handwritten Mathematical Expression Recognition with A Bidirectionally Trained Transformer (BTTR)," ICDAR (2021) | Architecture behind `LitBTTR`, the recognizer in `ocr_inference.py` |
| Week 5–7 | Zhang, Du & Dai, "Radical analysis network for learning hierarchies of Chinese characters," Pattern Recognition (2020) | Why characters are represented as IDS/radical decompositions instead of atomic classes |
| Week 9–11 | Nguyen et al., "Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer," ICDAR (2026) | The binary-flag baseline this course's alignment work replaces |
| Week 9–11 | Morioka, "CHISE: Character Processing Based on Character Ontology" (2008) | The IDS decomposition dictionary (`nom-ids/ids_exp.txt`) |
| Week 12–13 | Source paper, §3.3–3.4 and §4.2–4.4.3 (re-read) | Confidence-weighted selection/training design and the modified CER metric, to implement directly |
| Week 14 | Source paper, §4.4.5 and §5 (Discussion) (re-read) | Robustness-to-noise experiment design and the paper's own stated limitations |

## Assessment: Two Presentations

- **Midterm presentation (Week 8, \~15 min + Q&A per team):** covers the detection and recognition stages — architecture choices, training metrics (P/R/mAP, per-character accuracy), error analysis, and the team's plan for the alignment stage. Feedback here directly shapes each team's Weeks 9–11 work.
- **Final presentation (Week 15, \~20 min + Q&A/demo per team):** covers the complete pipeline — the alignment algorithm, confidence-weighted pseudo-labeling results, evaluation against the binary-flag baseline (CER, coverage, precision/recall by confidence band), and robustness findings. Accompanied by a written report and code repository submission.
- Weighting: midterm presentation 20%, final presentation 60%, code/report/participation 20%.

## Assumptions & Next Steps

- The paper's full iterative self-updating loop runs 6 rounds; Weeks 12–13 compress this to 1–2 rounds given time constraints, with room to extend if pacing allows.
- Assumes each team can be assigned one document subset from the paper's per-document-type table (DVSKTT-1–5, Lục Vân Tiên, Kiều 1866/1871/1872) for comparable final results — 8 subsets support up to 8 teams of 3–4 students (\~24–32 students total).
- Still open: grading rubrics per presentation, and a detailed plan mapping specific teams to specific NomNaOCR folders.
