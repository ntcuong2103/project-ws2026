# Help & FAQ

## Getting help

1. Check the FAQ below and the week's lab page.
2. Ask your team, then ask in the class group, where others may have hit the same problem.
3. For access problems (accounts, workspaces, GPU) or private matters, contact the teaching
   staff.

**Instructor:** Dr. Nguyễn Tuấn Cường · **Teaching assistants:** Minh, Dương

When you ask, include the command you ran, the full error message, your group number, and the
Slurm job ID if there is one.

## FAQ

??? question "`ImportError: libGL.so.1: cannot open shared object file`"
    The full `opencv-python` package got installed, and it needs a GUI library the server
    doesn't have. The lab environment uses the headless build. Run `uv sync` again in the lab
    repository. In your own project, replace `opencv-python` with `opencv-python-headless`.

??? question "`ModuleNotFoundError: No module named 'pytorch_lightning'` (or `einops`)"
    The recognizer dependencies are a separate group. Run `uv sync --group recognition` in the
    lab repository. See [Week 6](weeks/week-06.md).

??? question "The reference notebook fails with a CUDA or device error"
    Its first code cell sets `CUDA_VISIBLE_DEVICES` to a fixed MIG device ID from another
    session. Delete that line. Slurm already gives your job its GPU, and inside the job it
    appears as device 0.

??? question "My interactive session is still holding a GPU"
    Type `exit` in the `srun --pty` shell. If you lost the terminal, find the job with
    `squeue -u $USER` and cancel it with `scancel <jobid>`. Idle sessions block other teams.

??? question "My job is stuck in `PENDING`"
    The GPU type you asked for is busy, or you hit the limit of 2 running jobs. Check with
    `squeue`. Ask for a smaller MIG slice if your job fits (`a100_2g.10gb` for notebooks and
    small runs), or wait. See the [Slurm guide](guides/slurm.md).

??? question "Can I save files in `/data/shared/project2026`?"
    No. It is shared, read-only course data. Write datasets, crops, checkpoints, and outputs
    to `/data/projectgX/...`.

??? question "`Configured vocab_size … does not match …`"
    `vocab_size` must equal the number of lines in `vocab_ids.txt` plus 3 (`<pad>`, `<sos>`,
    `<eos>`). For the provided checkpoint that's 472. Don't reorder `vocab_ids.txt`: token IDs
    depend on its order.

??? question "Left-to-right and right-to-left give the same IDS, but different 'winners'"
    When both directions decode the same tokens, their scores are equal in exact arithmetic,
    so rounding decides the label. Count these crops as agreement. See
    [Week 7](weeks/week-07.md).

??? question "The decoded character shows two or three characters, like `來來`"
    Several Unicode characters share the same IDS in the dictionary, and the lookup returns all
    of them. Keep the whole candidate list in your output, as the Week 7 output contract asks.

??? question "Notebook cells show empty boxes instead of Nôm characters"
    The font can't draw them. Register the HanaMin or NomNaTong fonts from `nom-ids/assets`
    (see [Resources](resources.md#fonts-for-nom)).
