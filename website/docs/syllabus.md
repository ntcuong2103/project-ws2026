# Syllabus

## Overview

Students design, build, and evaluate a full document-AI pipeline (character detection,
radical-based recognition, IDS-level sequence alignment, and confidence-weighted
pseudo-labeling) modeled on a published research system for historical Hán-Nôm document
annotation.

- **Format:** 15 weeks, 4 academic hours per week (60 hours), teams of 3–4.
- **Structure:** the whole class moves through the pipeline stage by stage. Each team builds the
  complete pipeline on the whole NomNaOCR dataset, so results are comparable at the final
  presentation.
- **Prerequisites:** deep-learning foundations (CNNs, training loops, PyTorch) and Python. No
  prior OCR, computer-vision, or CJK-script experience is assumed.
- **Resources:** the lab GPU server and the datasets used in the source paper: NomNaOCR
  (2,953 pages), TUAT-Nakagawa detection annotations, MTHv2, and the CHISE/cjkvi-ids dictionary.

## Learning objectives

By Week 15, teams can:

1. fine-tune an object detector for character-level localization;
2. use a sequence model that predicts compositional (IDS) character representations, and decode
   its output;
3. implement a dynamic-programming alignment that recovers box-to-character correspondence with a
   graded confidence score;
4. run an iterative confidence-weighted self-training loop;
5. evaluate against a baseline with CER and precision/recall.

## Session format

| Part | Content | Time |
| --- | --- | --- |
| 1. Review | Each group presents the previous week's tasks and deliverable: what worked, what failed, what is blocking. | 5 min per group |
| 2. Lab | Short topic introduction, then the week's lab, hands-on in teams. Unfinished tasks become homework for next week's review. | Rest of the session |

Week 1 has no review. Weeks 8 and 15 are presentation weeks. See the full
[weekly schedule](weeks/index.md).

## Assessment

| Component | Weight |
| --- | --- |
| Midterm presentation (Week 8) | 20% |
| Final presentation (Week 15) | 60% |
| Code and report | 10% |
| Weekly review sessions | 10% |

Details are on the [assessments page](assessments.md).

## Reading list

| Assigned | Reading | Focus |
| --- | --- | --- |
| Week 1 | Nguyen et al., "Confidence-Weighted Annotation of Historical Nôm Documents via IDS-Level Sequence Alignment," *Pattern Recognition Letters* (2026) | The paper the course builds toward |
| Week 2 | Dang et al., "NomNaOCR: The First Dataset for Optical Character Recognition on Han-Nom Script," RIVF (2022) · [DOI](https://doi.org/10.1109/RIVF55975.2022.10013842) | The dataset every team uses |
| Week 2; re-read Week 9 | Nguyen et al., "Automated Character-Level Annotation for Historical Nom Documents via an Iterative Self-updating Radical-Based Recognizer," ICDAR (2026) · [DOI](https://doi.org/10.1007/978-3-032-36039-7_28) | Problem formulation and the binary-flag baseline |
| Weeks 3–4 | Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024) | The detector architecture |
| Weeks 3–4 | Ma et al., "Joint Layout Analysis, Character Detection and Recognition for Historical Document Digitization," ICFHR (2020) | The TUAT-Nakagawa annotations |
| Week 5 | Morioka, "CHISE: Character Processing Based on Character Ontology" (2008) · [Springer](https://link.springer.com/chapter/10.1007/978-3-540-78159-2_14) | Character ontology and IDS |
| Week 5 | Zhang et al., "Radical analysis network for zero-shot learning in printed Chinese character recognition," ICME (2018) · [arXiv](https://arxiv.org/abs/1711.01889) | Why structural tokens generalize |
| Weeks 5–7 | Zhao et al., "Handwritten Mathematical Expression Recognition with a Bidirectionally Trained Transformer (BTTR)," ICDAR (2021) | The recognizer and its bidirectional decoder |
| Weeks 12–13 | Source paper §3.3–3.4 and §4.2–4.4.3 (re-read) | Confidence-weighted training and the modified CER |
| Week 14 | Source paper §4.4.5 and §5 (re-read) | Robustness experiment and limitations |

## Policies

- **Two repositories.** The lab repository holds handouts and shared tools. Your team's assessed
  work lives in your own repository.
- **Data stays out of Git.** Datasets, checkpoints, predictions, logs, and generated figures go in
  `/data/projectgX`, never in a repository.
- **Shared data is read-only.** Never create, edit, move, or delete anything under
  `/data/shared/project2026`.
- **AI coding agents** are allowed within the [workspace rules](guides/workspace-rules.md). Each
  member must be able to explain the code their team submits.
- **Contributions are individual.** The midterm requires a slide listing every member's role
  and contribution, presented by that member.
