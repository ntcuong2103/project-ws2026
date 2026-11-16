# Getting started

Do these steps once per team at the start of the course. The full walkthrough, with the
sample-preparation job and notebook, is in the [Week 1 lab](weeks/week-01.md).

## 1. Log in and find your group number

Log in to the course GPU server with the account the lecturer gave you. Your team has a group
number `X`, confirmed in the registration sheet shared in class. Replace `X` in every
`projectgX` below with it. Never use another team's workspace.

## 2. Know your three locations

| What | Where | Access |
| --- | --- | --- |
| Code: Git repositories, scripts, tests, docs | `/home/projectgX` | Your team |
| Data you produce: datasets, checkpoints, predictions, logs | `/data/projectgX` (labs use `/data/projectgX/lab-exercise`) | Your team |
| Shared course data | `/data/shared/project2026` | **Read-only.** Never write here. |

Check your access (each line should print `OK`):

```bash
groups | grep -qw students && echo "OK: course access"
test -w /home/projectgX && echo "OK: code workspace"
test -w /data/projectgX && echo "OK: data workspace"
test -r /data/shared/project2026/nomnaocr && echo "OK: shared dataset"
```

## 3. Clone the lab repository

```bash
cd /home/projectgX
git clone https://github.com/ntcuong2103/project-ws2026.git
cd project-ws2026
uv sync
export LAB_WORKSPACE=/data/projectgX/lab-exercise
```

Pull before each session (`git pull`) to get the latest labs. Build your assessed project in
your **own team repository**, not in this one. See the
[team project contract](guides/project-contract.md).

## 4. Run work through Slurm

The server has one node with 2 × A100 40 GB, one whole and one split into MIG slices. All work
goes through Slurm:

- Use `sbatch` for training, evaluation, and anything long.
- Use `srun --pty bash` only for short debugging or notebooks, and type `exit` as soon as you
  finish so the GPU is released.
- Request a GPU only when your code uses CUDA, and always pass `--mem`.

```bash
mkdir -p logs
sbatch --gres=gpu:a100_2g.10gb:1 --cpus-per-task=4 --mem=16G --time=01:00:00 job.sh
squeue -u $USER
```

Read the [Slurm quick guide](guides/slurm.md) for GPU types, limits, and ready-made job scripts.

## 5. Recognizer setup (Week 6 onward)

```bash
cd /home/projectgX/project-ws2026
uv sync --group recognition
git clone -b project-2026 https://github.com/ntcuong2103/nom-ids.git \
  "$LAB_WORKSPACE/models/nom-ids"
git -C "$LAB_WORKSPACE/models/nom-ids" checkout 64be114
```

Then follow [Week 6](weeks/week-06.md) to copy the checkpoint and write your private
configuration.

## 6. Using a coding agent

Read the [workspace rules](guides/workspace-rules.md) before letting an AI coding agent work in
your repositories. They define where code and data may live and what must never be committed.
