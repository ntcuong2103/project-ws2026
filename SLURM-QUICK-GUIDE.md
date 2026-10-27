# Slurm Quick Guide (server `local-a100`)

The preferred way to run work is to submit a batch job with `sbatch`. Slurm
then schedules the job when the requested resources are available, keeps it
running after you disconnect, and releases its resources when it finishes.

## 1. Ready to use: submit a batch job with `sbatch`

Create the log directory once, then submit:

```bash
mkdir -p logs
sbatch job.sh
```

Use this `job.sh` as a starting point:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal                 # or "long" (up to 7 days, 1 running job)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100_3g.20gb:1    # a100 | a100_3g.20gb | a100_2g.10gb | a100_2g.10gb_mps
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id
#SBATCH --error=logs/%x-%j.err
##SBATCH --mail-type=END,FAIL        # uncomment for email
##SBATCH --mail-user=you@example.com

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) started at $(date)"
nvidia-smi -L

# --- load your environment ---
# module load python/3.11
# source ~/venvs/myenv/bin/activate

# --- your work ---
python my_script.py

echo "Finished at $(date)"
```

For a CPU-only job, delete the `--gres` line.

### Several runs of the same script (job array)

```bash
#SBATCH --array=1-10%1               # 10 tasks, at most 1 at a time
python my_script.py --index "$SLURM_ARRAY_TASK_ID"
```

Keep the `%N` throttle low: each array task counts as a job and shares the
GPUs with other users.

## 2. This server at a glance

| Item | Value |
|---|---|
| Server name | `local-a100` |
| Node | `gpu-a240` (single node, there is no separate login node) |
| Partition | `gpu` (default, no partition time limit) |
| CPUs | 32 |
| RAM | ~82 GB (`RealMemory=84179` MB) |
| GPUs | 2 × NVIDIA A100-PCIE-40GB (one whole, one split with MIG) |
| Account | `lab` |
| QOS | `normal` (default), `long` |

Because everything runs on one shared machine, **always request the CPUs,
memory and GPU you need through Slurm** so that other users' jobs are not
disturbed. Memory is *not* limited by default (`DefMemPerNode=UNLIMITED`), so
always pass `--mem`.

### Am I inside a Slurm allocation?

This installation has one physical node and no separate login node, so `hostname`
normally reports `gpu-a240` both outside and inside a job. Check Slurm's environment
instead:

```bash
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Inside Slurm job $SLURM_JOB_ID"
  scontrol show job "$SLURM_JOB_ID"
else
  echo "Outside a Slurm allocation"
fi
```

The same rule applies to a notebook kernel. Inspect `SLURM_JOB_ID` from Python; do not
infer allocation status from the hostname.

### QOS limits

| QOS | Max wall time | Max running jobs / user | Max submitted jobs / user | Priority |
|---|---|---|---|---|
| `normal` (default) | 3 days | 2 | 8 | 100 |
| `long` | 7 days | 1 | 4 | 10 |

Use `--qos=long` only when you really need more than 3 days. It has lower
priority and allows only one running job.

### GPU resources (GRES)

GPU 0 is a full A100. GPU 1 is in MIG mode and is split into three slices.
Each type below has **1** unit available; request it with `--gres=gpu:<type>:1`.

| GRES type | What it is | GPU memory | Typical use |
|---|---|---|---|
| `a100` | Whole A100 (GPU 0) | 40 GB | Training, large models, full-speed work |
| `a100_3g.20gb` | MIG slice 3g.20gb (GPU 1) | 20 GB | Medium models, fine-tuning, inference |
| `a100_2g.10gb` | MIG slice 2g.10gb, exclusive (GPU 1) | 10 GB | Small models, debugging, notebooks |
| `a100_2g.10gb_mps` | MIG slice 2g.10gb for per-job MPS (GPU 1) | 10 GB | Small jobs that use CUDA MPS |

Notes:
- Request exactly one GPU type per job, e.g. `--gres=gpu:a100:1`. Plain
  `--gres=gpu:1` lets Slurm pick any free type, which may not be the one you
  want.
- A MIG slice is hard-isolated: your job sees only its slice and its memory.
  Pick a slice whose memory fits your model; otherwise you get out-of-memory errors.
- Check what is free right now: `scontrol show node gpu-a240 | grep -E "Gres|AllocTRES"`
  and `squeue` to see who holds what.
- Inside a job, `nvidia-smi -L` shows the GPU or MIG device you were given.

## 3. Why batch jobs are preferred

| | Interactive session | Batch job |
|---|---|---|
| Command | `srun --pty bash` | `sbatch script.sh` |
| You are | In a shell with your allocation | Disconnected; Slurm runs the script |
| Starts | Immediately if resources are free, otherwise you wait at the prompt | Queued, starts when resources are free |
| Survives logout / network drop | No (killed when the session ends) | Yes |
| Output | Terminal | File (`logs/<name>-<jobid>.out`) |

### Interactive session

**Pros**
- Immediate feedback: good for debugging, testing code, exploring data, notebooks.
- You can run, tweak and re-run commands without re-queuing.

**Cons**
- Resources (including the GPU) are reserved the whole time, even while you
  are thinking or idle. It is easy to forget and block other users.
- Counts against your running-job limit (2 on `normal`).
- Dies if your SSH connection drops (unless you use `tmux`/`screen`).
- If resources are busy you must wait at the keyboard.
- Not reproducible; hard to track what was run.

### Batch job

**Pros**
- Runs unattended, survives disconnects, and frees resources automatically
  when finished.
- Reproducible: the script documents exactly what was run and with which resources.
- Can queue several jobs, use dependencies and email notification.
- Better use of the shared GPUs: nothing sits idle waiting for you.

**Cons**
- Slower feedback loop: each change needs a resubmit and possibly a queue wait.
- Harder to debug interactively.
- Errors may only show up after the job starts, so check the log file.

**Default to `sbatch` for scheduled work.** Use an interactive session only
for short development and debugging with the smallest GPU slice that fits.

## 4. Interactive sessions for short debugging

```bash
# CPU only: 4 cores, 16 GB RAM, 2 hours
srun --partition=gpu --account=lab \
     --cpus-per-task=4 --mem=16G --time=02:00:00 \
     --pty bash

# Small GPU slice (10 GB): 4 cores, 16 GB RAM, 2 hours, good for debugging
srun --partition=gpu --account=lab \
     --gres=gpu:a100_2g.10gb:1 --cpus-per-task=4 --mem=16G --time=02:00:00 \
     --pty bash

# Medium GPU slice (20 GB): 8 cores, 32 GB RAM, 4 hours
srun --partition=gpu --account=lab \
     --gres=gpu:a100_3g.20gb:1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
     --pty bash

# Whole A100 (40 GB): 8 cores, 32 GB RAM, 4 hours
srun --partition=gpu --account=lab \
     --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
     --pty bash
```

Always set `--time` to what you really need. The session ends by itself at the
limit, which protects you if you forget to exit.

For a session that survives SSH drops, start `tmux` first, then run the
`srun ... --pty bash` command inside it.

### Ending an interactive session (release resources)

Resources are held until the session ends. Use **any** of these:

```bash
exit            # or press Ctrl+D  -> normal way, releases the allocation
```

If the terminal is stuck or you lost the session, cancel from another shell:

```bash
squeue -u $USER          # find the JOBID of your interactive job
scancel <JOBID>          # release it
scancel -u $USER         # cancel ALL your jobs (be careful)
```

Also remember to stop anything you started inside the session (notebooks,
background processes) before exiting. Check that nothing is left:

```bash
squeue -u $USER
```

## 5. Common commands

| Task | Command |
|---|---|
| Submit a job | `sbatch job.sh` |
| My jobs | `squeue -u $USER` |
| All jobs | `squeue` |
| Job details | `scontrol show job <JOBID>` |
| Cancel a job | `scancel <JOBID>` |
| Node / GPU state | `sinfo`, `scontrol show node gpu-a240` |
| Finished job info (time, memory used) | `sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode` |
| Efficiency of a finished job | `seff <JOBID>` |
| My account/QOS | `sacctmgr show assoc user=$USER format=account,qos,defaultqos` |

Job states in `squeue`: `PD` pending, `R` running, `CG` completing.
A job stuck in `PD` shows the reason in the last column (e.g. `Resources`,
`Priority`, `QOSMaxJobsPerUserLimit`).

## 6. Good practice

- Request only what you need; over-requesting makes you wait longer and
  blocks others. Use the smallest GPU slice that fits your model.
- Always set `--time` and `--mem`.
- Check `seff <JOBID>` after runs and adjust CPU/memory requests.
- Exit interactive sessions as soon as you are done. Don't leave them idle overnight.
- Run long work as batch jobs, not as interactive sessions kept open "just in case".
