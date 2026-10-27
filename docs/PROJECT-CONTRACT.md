# Optional Team Project Contract

Teams may create their implementation repository whenever they are ready. This is a
recommended structure, not starter code and not a required choice of framework.

```text
team-nom-ocr/
├── README.md
├── pyproject.toml
├── configs/
├── src/
├── tests/
├── scripts/
└── reports/
```

The repository should document its separate project-data root and provide
reproducible commands for these stages:

1. prepare detector and recognizer datasets;
2. pretrain or fine-tune the character detector;
3. train or load the IDS recognizer;
4. run detection and recognition inference;
5. decode IDS predictions;
6. align boxes with line-level ground truth and generate pseudo-labels;
7. evaluate the pipeline and reproduce reported results.

Use stable `page_id` and `box_id` values across outputs. Keep configuration and paths
outside implementation code. Do not commit datasets, checkpoints, predictions, logs,
or secrets. The final README should state the environment, commands, input/output
formats, and the location of each reported result.

For recognition, a configuration should register the detector and recognizer checkpoints,
IDS resources, full detector-label root, image root, output root, special-token IDs, and
deterministic crop preprocessing. The full detector prediction manifest is the handoff
between detection and recognition; do not duplicate the complete crop dataset.
