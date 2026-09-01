# MoE tuning on a second machine: Qwen3.6-35B-A3B on an RTX 3060 12GB

First real-hardware run of the 0.4.0 MoE search, on hardware and a model
class the project had never been tested against: an **NVIDIA RTX 3060
12GB** (compute capability 8.6, WDDM) with an **i5-12400F** and 32GB
DDR4-3200, Windows, tuning **Qwen3.6-35B-A3B** (`lmstudio-community`,
Q4_K_M, 19.7GB on disk, 40 layers, 256 experts / 8 active,
architecture `qwen35moe`).

Everything before this was measured on one machine (RTX 5060 8GB) with
dense 9B models. This report exists because the README asks for exactly
this, and because the run changed the code: **four defects were found
here that the unit tests did not catch**, three of them in the 0.4.0
changes themselves.

## Reference point: the same tuning done by hand

Before FluxInfer could tune MoE models at all, the same machine was tuned
manually by sweeping `--n-cpu-moe` with `llama-server` at `-c 32768
--no-mmap`, timing real completions. That sweep is the yardstick here:

| `--n-cpu-moe` | peak VRAM | generation | prompt processing |
| --- | --- | --- | --- |
| 36 | 5.7 GB | 33.6 tok/s | 202-265 tok/s |
| 32 | 7.6 GB | 37.4 tok/s | 430-519 tok/s |
| 28 | 9.4 GB | 41.1 tok/s | 461-579 tok/s |
| **24** | 11.2 GB | **45.4 tok/s** | 491-654 tok/s |
| 20 | 12.0 GB | 28.4 tok/s | **77 tok/s** |

The last row is the failure mode that shaped most of 0.4.0: at ~260MB
free, throughput collapsed while every run still reported success. No
CUDA OOM, no error — WDDM had started backing allocations with system
RAM. A search that only reacts to OOM would have accepted that
configuration on the grounds that it worked.

## Run 1 — `-c 4096` (default context)

```
[baseline (cpu-only)]     timed out
[n_cpu_moe=40]            prompt=111.9  gen=27.1
[n_cpu_moe=30]            prompt=130.9  gen=34.6
[n_cpu_moe=20]            prompt=160.3  gen=43.3
[n_cpu_moe=10]            timed out
[n_cpu_moe=0]             timed out
[batch=128 ubatch=64]     timed out
[batch=512 ubatch=256]    prompt=266.4  gen=45.2
[batch=2048 ubatch=512]   prompt=431.4  gen=44.5
[threads=12]              prompt=434.3  gen=44.8   <- selected
```

The MoE stage engaged correctly and replaced the dense sweep. But note
what is missing: **no bisection ran**. Refinement was gated on a known
failing value, and only a confirmed OOM or spill set one — while on
Windows an over-VRAM MoE configuration stalls rather than reporting OOM.
On the platform where the search is needed most, refinement never ran at
all (fixed in `a772daa`).

## Run 2 — `-c 32768`, matching how the model is actually served

Every one of eleven candidates was killed with "went silent". The idle
timeout introduced in 0.4.0 rested on a false premise — *"loading is slow
but never silent"* — when llama-bench in JSON mode prints nothing until
the entire run completes. At 4096 the prefill was short enough to hide
this; at 32768 it failed the whole tune.

Fixed in `689f8e3` by passing `--progress` (so llama-bench announces each
phase and silence becomes meaningful) and by sizing both the idle window
and the total cap to include a context-scaled prefill allowance. Worth
noting that this was only diagnosable because the previous commit had
started reporting *which* limit fired; with the earlier bare "timed out"
it would have looked like a VRAM problem.

## Run 3 — `-c 32768`, after the fixes

```
[baseline (cpu-only)]        prompt=84.7   gen=2.9    VRAM peak=2.2 GB
[n_cpu_moe=40]               prompt=101.2  gen=26.4   VRAM peak=3.7 GB
[n_cpu_moe=30]               prompt=113.2  gen=32.6   VRAM peak=8.7 GB
[n_cpu_moe=20]               prompt=46.2   gen=20.4   VRAM peak=11.9 GB  <- rejected: silent spill
[n_cpu_moe=25 (bisection)]   prompt=136.5  gen=37.9   VRAM peak=10.5 GB
[n_cpu_moe=22 (bisection)]   prompt=140.9  gen=37.7   VRAM peak=11.8 GB
[n_cpu_moe=21 (bisection)]   prompt=138.7  gen=38.7   VRAM peak=12.0 GB
[batch=2048 ubatch=512]      prompt=376.4  gen=40.7   VRAM peak=11.8 GB
[threads=9]                  prompt=364.9  gen=42.3   VRAM peak=11.8 GB  <- selected
```

Selected: `threads=9 gpu_layers=40 batch=2048 ubatch=512 n_cpu_moe=21` at
**42.3 tok/s generation / 364.9 tok/s prompt processing**, against the
manual optimum of ~40-45 tok/s at `--n-cpu-moe 24`. The automatic search
took under an hour; the manual sweep took an afternoon of a person's
attention.

> **Note added later in this branch.** This run predates the VRAM-headroom
> guard. The selected configuration peaks at 11.8 GB on a 12 GB card, leaving
> roughly 0.2 GB free, and the guard added afterwards rejects exactly that:
> it served measurably worse than a lower setting once a browser was open.
> Re-run with the guard in place, the search stops at `--n-cpu-moe 27`
> (9.6 GB peak, ~2.4 GB free) and reports 37.7 tok/s generation / 340 tok/s
> prompt processing. The ~11% of generation throughput between the two
> numbers is the price of a configuration that still fits while the machine
> is being used, which is the tradeoff this branch argues for.

Three things are worth reading carefully here:

**The measured VRAM curve is monotonic and lands exactly on the card's
limit**: 3.7 → 8.7 → 10.5 → 11.8 → 12.0 GB. The boundary was found by
measurement, not estimated.

**The spill guard fired at `n_cpu_moe=20`** (11.9GB peak, prompt
throughput at 41% of the previous candidate) and that rejection is *why*
the bisection had a floor to work against — it probed 25, then 22, then
21, exactly as a floor at 20 implies. It also did so **silently**: the
rejection message was attached to the progress callback, which fires
before the search reaches its verdict, so the log showed only an
unremarkable slow result. Correct decision, invisible reasoning (fixed in
`13bb250`).

**Stage ordering is a real limitation, not a theoretical one.** In an
earlier 32k run the batch stage selected `batch=2048` *after* the expert
placement had left the card nearly full, and generation halved (41 → 18.9
tok/s) with the same placement — the larger compute buffers took the
remaining VRAM. The spill guard now covers the batch and thread stages
too, but guarding only stops a bad candidate from winning; it does not
make a greedy, one-axis-at-a-time search jointly optimal. Note also that
run 3 did *not* reproduce the collapse at the same setting: the boundary
moves by a few hundred MB depending on what else holds VRAM at the time.

## What this run says about the tool

- MoE detection, stage substitution, bisection, profile persistence and
  replay all work end to end on hardware the author has never used.
- The tuned answer is close to a careful manual optimum, on a model where
  the previous dense search was not merely suboptimal but non-monotonic
  and misleading.
- **The profile is only valid for the context it was tuned at.** Run 1
  and run 3 disagree about the best `--n-cpu-moe` (20 vs 21, with very
  different margins) purely because the KV cache at 32768 changes the
  VRAM budget. Tune at the context you intend to serve at.
- Four defects in three runs, none caught by 91 passing unit tests. The
  tests are worth keeping, but on this kind of tool they verify logic,
  not reality.
