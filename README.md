# Confidence-Weighted Nôm OCR - Project Course

## Course Overview

Students design, build, and evaluate a full document-AI pipeline — character detection, radical-based recognition, IDS-level sequence alignment, and confidence-weighted pseudo-labeling — modeled on a published research system for historical Hán-Nôm document annotation.

- **Format:** 15 weeks, 4 contact hours/week (60 hours total), teams of 3–4 students
- **Structure:** the whole class moves through the pipeline stage by stage (detection → recognition → alignment → pseudo-labeling); each team builds the complete pipeline end-to-end on an assigned document subset, so results are comparable across teams at the final presentation
- **Prerequisites:** deep learning foundations (CNNs, training loops, PyTorch/TensorFlow), basic Python; no prior OCR/CV or CJK-script experience assumed
- **Resources:** lab GPUs and the datasets used in the source paper — the NomNaOCR corpus (2,953 pages), TUAT-Nakagawa detection annotations, MTHv2 pretraining set, and the CHISE/cjkvi-ids dictionary
- **Learning objectives:** by Week 15, teams can fine-tune an object detector for character-level localization; train a sequence model to predict compositional (IDS) character representations; implement a dynamic-programming alignment algorithm that recovers box-to-character correspondence with a graded confidence score; run an iterative confidence-weighted self-training loop; and evaluate results against a baseline using CER and precision/recall metrics

## Tentative 15-Week Schedule

| Week | Phase | Topic & In-Class Focus | Lab / Studio Work | Deliverable |
| --- | --- | --- | --- | --- |
| 1 | Orientation | Course kickoff; journal-club read of the source paper; intro to Hán-Nôm OCR and IDS decomposition; team formation | GPU/environment setup, repo scaffold, dataset access verification | Teams formed; environment verified |
| 2 | Foundations | Object detection for character localization; DP sequence-alignment primer; weak supervision / pseudo-labeling concepts | Explore NomNaOCR and TUAT-Nakagawa data; draft project charter | Project charter; each team assigned a document subset |
| 3 | Detection I | YOLOv11n architecture; character-level annotation format; transfer learning from MTHv2 pretraining | Set up training pipeline; begin detector fine-tuning | Training script + first training-run logs |
| 4 | Detection II | Detection metrics (Precision/Recall/mAP@50/mAP@50–95); detector error modes (merges, splits, spurious/missed boxes) | Complete training; evaluate on held-out split; error analysis | Detector checkpoint + evaluation report + error gallery |
| 5 | Recognition I | Ideographic Description Sequences (IDS); CHISE/cjkvi-ids dictionary; compositional character representation | Build IDS decomposition utility; prepare cropped character-image dataset | IDS decomposition utility + recognizer training set |
| 6 | Recognition II | Encoder–decoder / attention-based recognizers (BTTR-style); beam-search decoding | Train/fine-tune recognizer on assigned subset | Recognizer training run + per-character accuracy |
| 7 | Recognition III | Recognizer vs. detector error modes; decoding trade-offs; midterm prep | Combine detector + recognizer on sample pages; assemble slides | Combined detection+recognition demo + midterm slide draft |
| 8 | Midterm | — | Team presentations (\~15 min + Q&A) | **Midterm presentation (graded)** |
| 9 | Alignment I | Box-to-line assignment; formalizing correspondence recovery; walkthrough of the paper's DP alignment algorithm | Implement box-to-line assignment; set up inner IDS edit distance | Assignment module + unit tests |
| 10 | Alignment II | Two-level cost model (outer character alignment, inner IDS edit distance); gap-cost design rationale | Implement Needleman-Wunsch-style global alignment producing a per-box confidence score | Working alignment module + confidence scores on sample lines |
| 11 | Alignment III | Confidence-score interpretation; baseline comparison design | Run full-page alignment; implement a binary-flag baseline for comparison | Validated alignment module + binary-flag baseline |
| 12 | Pseudo-labeling I | Confidence-weighted selection/training; iterative self-training loop design | Implement pseudo-label emission + confidence-weighted fine-tuning loop | Pseudo-label pipeline + first self-training iteration results |
| 13 | Pseudo-labeling II & Metrics | Modified CER (Unicode-variant handling); pseudo-label precision/recall by confidence band | Run remaining self-training iterations; compute CER, coverage, precision/recall | Iteration results table + confidence-band analysis |
| 14 | Integration | Robustness to detector segmentation noise; per-document-type analysis; final report/poster guidelines | Optional noise-injection experiment; finalize result tables; polish repo, report, and slides | Final report draft + finalized results + final slide draft |
| 15 | Final | — | Team presentations (\~20 min + Q&A/demo) | **Final presentation (graded) + written report + code repository** |

## Assessment: Two Presentations

- **Midterm presentation (Week 8, \~15 min + Q&A per team):** covers the detection and recognition stages — architecture choices, training metrics (P/R/mAP, per-character accuracy), error analysis, and the team's plan for the alignment stage. Feedback here directly shapes each team's Weeks 9–11 work.
- **Final presentation (Week 15, \~20 min + Q&A/demo per team):** covers the complete pipeline — the alignment algorithm, confidence-weighted pseudo-labeling results, evaluation against the binary-flag baseline (CER, coverage, precision/recall by confidence band), and robustness findings. Accompanied by a written report and code repository submission.
- Weighting: midterm presentation 20%, final presentation 60%, code/report/participation 20%.

## Assumptions & Next Steps
- The paper's full iterative self-updating loop runs 6 rounds; Weeks 12–13 compress this to 1–2 rounds given time constraints, with room to extend if pacing allows.
