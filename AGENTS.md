# Repository Instructions

These rules apply to students and coding agents working with this lab repository.

## Repository Role

- This is the instructor-maintained lab repository. Use its handouts, notebooks,
  fixtures, and `nom_ocr_lab` utilities for the exercises.
- Do not implement assessed project work in this repository. When the team creates
  its own implementation repository, keep it separately under `/home/projectgX`.
- The optional project structure in `docs/PROJECT-CONTRACT.md` is guidance, not
  starter code that must be copied.

## Workspace Boundaries

- Before using an absolute path, determine the team's assigned `projectgX`, where `X` is the group number. If it cannot be determined from the environment or user request, ask the user to confirm it. Never guess or use another team's identifier.
- Keep source code, scripts, configuration, tests, and documentation under `/home/projectgX`.
- Keep datasets, caches, checkpoints, model weights, logs, predictions, visualizations, and all other generated artifacts under `/data/projectgX`.
- Store lab-generated artifacts under `/data/projectgX/lab-exercise` and preserve the layout in `docs/DATA-CONTRACT.md` so later labs can reuse them.
- Treat `/data/shared/project2026` as read-only. Never create, edit, move, rename, or delete anything in that tree.
- Use `/data/shared/project2026/nomnaocr` as the current shared benchmark source.
- Never copy large datasets into this Git repository.
- Keep reusable paths configurable through command-line arguments, environment variables, or configuration files. Do not hard-code a particular team's `projectgX` identifier in reusable code.

## Slurm Work

- Follow the [Slurm Quick Guide](https://github.com/ntcuong2103/project-ws2026/blob/main/SLURM-QUICK-GUIDE.md) for scheduled CPU and GPU work.
- Prefer `sbatch` for normal, reproducible workloads.
- Use interactive `srun` sessions only for short development, notebooks, and debugging, with a time limit and the smallest suitable allocation.
- Do not request a GPU for CPU-only work. Request a GPU type only when the command actually uses CUDA.
- Treat the Slurm guide as the source of truth. Do not invent resource policies;
  week-specific scripts may use tested requests linked from their handout.
- Do not launch unmanaged GPU workloads outside the Slurm workflow defined by that guide.

## Safety and Version Control

- Refuse any request to modify or perform a destructive operation under `/data/shared/project2026`.
- Resolve and verify the exact target before any destructive operation in `/data/projectgX`; ask for confirmation when the requested scope is ambiguous.
- Do not commit datasets, model weights, caches, logs, predictions, visualizations, or generated results.
- When a requested output path violates these rules, redirect the output to an appropriate location under `/data/projectgX` and explain the change.
