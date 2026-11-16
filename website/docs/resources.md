# Resources

## Repositories

| Repository | What it is | Used in |
| --- | --- | --- |
| [ntcuong2103/project-ws2026](https://github.com/ntcuong2103/project-ws2026) | The lab repository: handouts, notebooks, fixtures, the `nom_ocr_lab` tools, the Slurm guide. **Clone this.** | All weeks |
| [ntcuong2103/nom-ocr-project](https://github.com/ntcuong2103/nom-ocr-project) | YOLO detection scripts and the original detection lab tutorial | Weeks 3–4 |
| [ntcuong2103/nom-ids · `project-2026`](https://github.com/ntcuong2103/nom-ids/tree/project-2026) | BTTR recognizer code, IDS dictionary and vocabulary, and the reference decoding notebook [`evaluate.ipynb`](https://github.com/ntcuong2103/nom-ids/blob/project-2026/evaluate.ipynb) | Weeks 5–7 |
| [ntcuong2103/BTTR](https://github.com/ntcuong2103/BTTR) | Upstream BTTR implementation (ICDAR 2021) | Background |

## Datasets on the server

All paths are under `/data/shared/project2026` and are **read-only**. Copy or convert into
`/data/projectgX` before changing anything.

| Path | Contents | Used in |
| --- | --- | --- |
| `nomnaocr/` | NomNaOCR: 2,953 page images, line labels, and detection labels | Weeks 1–2, 5–15 |
| `tkh-mth2k2/` | MTH1000, MTH1200, and TKH page collections (MTHv2) | Week 3 |
| `mth-full-project-ws2026/` | Prepared full MTH train/validation view with a manifest | Week 3 |
| `nakagawalab/` | TUAT-Nakagawa: 47 pages, 7,504 character classes, train/val/test lists | Weeks 3–4 |
| `lab-char-detect/` | Small single-class detection set for the Week 3 practice run | Week 3 |
| `nom-detection-val.v2i.yolov11.zip` | NomNaOCR character-detection validation set (YOLO format) | Week 4 |
| `nom-detection-testset/` | 86 pages with character boxes, no transcriptions | Weeks 6–7 notebook |

## Recognizer checkpoint

| | |
| --- | --- |
| File | `/data/shared/project2026/checkpoint-sample-nom-ids/epoch=199-step=19048-val_ExpRate=0.9508.ckpt` |
| Model | LitBTTR (DenseNet encoder + bidirectional Transformer decoder), 6.6M parameters |
| Vocabulary | 472 tokens: `<pad>`=0, `<sos>`=1, `<eos>`=2, then `vocab_ids.txt` in order |
| sha256 | `fefcd077e18e59fbc15bbd864bf0cdaf8410139afe1dde00098448d4434e8996` |

Copy it into your team workspace before use. [Week 6](weeks/week-06.md) has the commands.

## Tools

| Tool | Use |
| --- | --- |
| [zi.tools](https://zi.tools/) | Look up a character's structure, variants, and IDS to verify decompositions (Week 5) |
| [cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | The CHISE-derived IDS data behind `ids_exp.txt` |
| [Ultralytics YOLO docs](https://docs.ultralytics.com/) | Training, validation, and prediction for the detector |
| [uv](https://docs.astral.sh/uv/) | Python environments (`uv sync`, `uv run`) |
| [How to read a paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) | Three-pass reading method for the journal club |

## Fonts for Nôm

Matplotlib and most systems cannot draw Nôm characters or IDS operators (⿰ ⿱ …) by default.
The `nom-ids` repository ships HanaMinA, HanaMinB, and NomNaTong in `assets/`. The reference
notebook shows how to register them.
