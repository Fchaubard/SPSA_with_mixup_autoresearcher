# autoresearch

This is an experiment to have an LLM do its own research on zero order, schedule-free diffusion.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer (must be zero order, no use of full gradients ever, can you backprop only for baseline), training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains ImageNet data. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 1 hour** (wall clock training time, excluding startup/compilation). You launch it as: `uv run train.py` (backprop baseline) or `uv run train.py --solver spsa` (zero-order SPSA). All hyperparameters are exposed as CLI flags — run `uv run train.py --help` for the full list.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, loss equation, per layer loss and global loss, sum of loss per timestep or loss calculated only at end, MSE vs. cross entropy vs. FID direct loss vs. something else, type of zero order, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, flow matching, and training constants (time budget, image resolution, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_fid` function in `prepare.py` is the ground truth metric. You also can not add more data.

**The goal is simple: get the lowest val_fid.** Since the time budget is fixed, you don't need to worry about training time — it's always 1 hour. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_fid gains, but it should not blow up dramatically. Zero order is much more memory efficient and you should keep that in mind. Its able to train 'in place' with no solver bloat. You should consider this as a plus and keep it if possible. Caching of the probe speeds things up which you can do if you must.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_fid improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_fid improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
solver:           spsa
val_fid:          125.4321
training_seconds: 3600.0
total_seconds:    3615.2
peak_vram_mb:     8042.1
mfu_percent:      12.50
total_images_M:   2.1
num_steps:        8192
num_params_M:     13.2
depth:            1
denoising_steps:  20
n_perts:          40
final_lr:         1.00e-04
final_epsilon:    1.00e-04
```

(Note: `denoising_steps` through `final_epsilon` only printed for SPSA solver)

Note that the script is configured to always stop after 1 hour wallclock, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_fid:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_fid	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_fid achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_fid	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_fid:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_fid improved (lower), you "advance" the branch, keeping the git commit
9. If val_fid is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take 1 hour total (+ a few seconds for startup and eval overhead). If a run exceeds 1 hour and 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~1 hour then you can run approx 1/hour * # of GPUs available (which is 8 A100s on this node), for a total of about 64 over the duration of the average human sleep (8 hours). The user then wakes up to experimental results, all completed by you while they slept! You should NEVER STOP THOUGH! Keep making the solver, the architecture, and the diffusion better and better.

## Batch loader experiment scope

**For this run**, the experimental surface is narrowed to **batch construction only**. The backbone, loss, diffusion process, SPSA math, and evaluation are frozen. The only thing that changes is which images are fed into the solver at each step.

### Frozen after calibration (do not change)

- Architecture (depth, width, channels, patch size)
- Denoising steps, loss definition
- SPSA solver math, number of perturbations, epsilon/LR schedule

### Available batch policies

All policies are selectable via `--batch-policy` CLI flag:

```
--batch-policy iid_raw                                                # baseline (default)
--batch-policy iid_replay   --replay-factor {2,4,8}                   # replay same batch r times
--batch-policy random_bucket_mean   --compression {2,4,8,16}          # average random buckets
--batch-policy random_bucket_medoid --compression {2,4,8,16}          # medoid of random buckets
--batch-policy random_bucket_mix    --compression {2,4,8} --bucket-alpha {0.25,0.5,0.75}
--batch-policy local_similarity_bucket --compression {2,4,8} --oversample {1,2} --bucket-repr {mean,medoid,mix}
--batch-policy ema_slot_bank --ema-beta {0.9,0.99,0.995} --ema-emit {ema_only,current_mix}
--batch-policy prototype_bank_ema --bank-size-mult {4,8,16} --proto-beta {0.95,0.99}
```

Any policy can be combined with `--replay-factor {2,4}` for hybrid replay.
Use `--debug-minutes N` for quick smoke tests before committing to full 1-hour runs.

### Experimental ladder

- **Phase 0**: Baseline + replay (iid_raw, iid_replay r=2, r=4)
- **Phase 1**: Random bucket compression (mean, medoid, mix at various m)
- **Phase 2**: Similarity-aware compression (local_similarity_bucket variants)
- **Phase 3**: Online rolling-average (ema_slot_bank variants)
- **Phase 4**: Larger compressed memory (prototype_bank_ema variants)
- **Phase 5**: Combine the best policy with replay

## Fixed training configuration (MANDATORY for all runs)

Every single run MUST use these exact settings. No exceptions.

```
--solver spsa
--depth 1
--denoising-steps 5
--use-curvature
--saturating-alpha 0.1
--lr 1e-4
--n-perts 40
--device-batch-size 64
```

LR can be tuned slightly (e.g. 5e-5 to 3e-4) if needed for a specific batch policy, but must be justified and logged. Everything else is locked.

## What you CANNOT do (batch loader experiment)

These are **hard constraints** that override the general "everything is fair game" guidance above. For this experiment:

- **DO NOT** change the model architecture (depth, width, channels, patch size, attention heads). It stays at depth=1, n_embd=768, patch_size=4.
- **DO NOT** change the diffusion schedule. Denoising steps = 5, constant, for all runs.
- **DO NOT** change the loss function. MSE denoising loss only.
- **DO NOT** change the SPSA solver math, number of perturbations (40), or saturating alpha (0.1).
- **DO NOT** add Adam, momentum, or any other optimizer. Pure 1.5-SPSA only.
- **DO NOT** add backprop baselines or gradient-based training of any kind.
- **DO NOT** use class labels or class-aware grouping in batch construction.
- **DO NOT** change the evaluation harness, prepare.py, or install new packages.
- **DO NOT** change warmup/warmdown schedule ratios.

## What you SHOULD do (batch loader experiment)

- **DO** experiment with different `--batch-policy` settings and their hyperparameters.
- **DO** try different compression factors, replay factors, EMA betas, bank sizes, etc.
- **DO** combine policies with replay (`--replay-factor`).
- **DO** log batch policy diagnostics (raw images seen, virtual images seen, CPU time, compression ratio).
- **DO** use `--debug-minutes 5` for quick smoke tests before full 1-hour runs.
- **DO** follow the experimental ladder systematically (Phase 0 through Phase 5).
- **DO** keep the code simple — batch policy logic should be clean and readable.
