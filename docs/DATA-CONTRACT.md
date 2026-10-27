# Lab Data Contract

Lab source code and generated data are deliberately separated.

## Locations

Replace `X` with the assigned group number.

```text
/home/projectgX/project-ws2026/       # cloned lab repository
/data/projectgX/lab-exercise/
├── datasets/
│   ├── nomnaocr-sample/                 # durable Week 1 page triplets
│   │   ├── images/
│   │   ├── line_labels/
│   │   ├── detection_labels/
│   │   └── manifest.json
│   ├── mth1000-practice/                # Week 3 same-data fitting check
│   ├── mth1000-full/                    # deterministic 80/10/10 split
│   ├── nomnaocr-character-detection/    # nested character-box evaluation view
│   ├── nakagawa-character-detection/    # supplied train/val/test split
│   ├── ids/                              # pinned Week 5 text resources
│   │   ├── cjkvi/                        # raw/direct IDS references
│   │   ├── nom-ids/                      # expanded IDS + ordered vocabulary
│   │   └── manifest.json                 # source provenance and checksums
│   └── nomnaocr-recognition-crops/       # one-page unlabeled inference crops
│       └── <page>/
│           ├── crops/
│           ├── manifest.json
│           └── manifest.tsv
├── models/
├── metrics/
├── predictions/
│   └── <detector-run>/                    # complete Week 4 detector handoff
│       ├── images/<relative-page>          # detector overlays
│       ├── labels/<relative-page>.txt      # 6-field YOLO rows; empty if no box
│       └── manifest.json                   # every processed page + model/settings
└── logs/
```

Set the lab root once in each shell used to start commands or Jupyter:

```bash
export LAB_WORKSPACE=/data/projectgX/lab-exercise
```

The Week 1 sample is a durable input for later labs. Each selected page has an image,
a line polygon/transcription file, and a character-position detection file at matching
relative paths. Later exercises must read it from `datasets/nomnaocr-sample` instead
of creating another copy.

## Ownership and safety

- `/data/shared/project2026` is read-only source data.
- `/home/projectgX` contains Git repositories, scripts, tests, and documentation.
- `/data/projectgX/lab-exercise` contains lab datasets, models, predictions, and job logs.
- Do not commit generated data, model weights, or predictions.
- Scripts must accept configurable paths and must never embed a real group number.
- A team project uses a separate, team-chosen data root and documents that choice.

## Artifact rules

- Reuse a valid artifact rather than rebuilding it in a week-specific directory.
- Keep relative page paths when copying matched images and labels.
- A manifest records how a reusable sample was produced and checksums every image,
  line label, and detection label in the sample.
- Detection dataset views preserve document-relative paths, normalize every object to
  class `0` (`character`), and keep their generated `data.yaml` beside the manifest.
- IDS resources retain the exact `nom-ids` vocabulary order used by the recognizer checkpoint;
  never sort or regenerate that vocabulary.
- Recognition crops retain detector box IDs and optional confidence, but no character identity.
  Line transcriptions cannot label boxes safely before sequence alignment.
- The Week 4 full detector run is the canonical recognition input. Its `labels/` tree mirrors
  the source `images/` tree exactly and has an empty label file for every no-detection page;
  recognition code must use `page_id` (the relative image path without its suffix) and the
  zero-based row order as `box_id`.
- Recognition YAML may expand `${LAB_WORKSPACE}` but must not hard-code a team identifier.
  Checkpoints and the private, resolved configuration stay outside Git.
- Refuse conflicting or partial destinations; do not silently overwrite them.
