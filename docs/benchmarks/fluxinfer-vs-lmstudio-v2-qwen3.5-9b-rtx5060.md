# FluxInfer vs LM Studio — after the fixes

Real-world run, 2026-07-07, same machine/model/GPU as the
[first comparison](fluxinfer-vs-lmstudio-qwen3.5-9b-rtx5060.md): Intel Core
i5-14600KF, 32 GB RAM, NVIDIA GeForce RTX 5060 8 GB, Qwen3.5-9B-Aggressive
Q8_0.

The first head-to-head found FluxInfer close on generation speed but ~5.9x
slower than LM Studio on prompt processing, and identified three concrete
causes: an overly conservative memory-pressure penalty, a batch/ubatch
search settling on a small batch from a single noisy sample, and a coarse
gpu_layers grid with no refinement near the OOM boundary. All three were
fixed (see [README.md](../../README.md#scoring) and the `tuner`/`scoring`
changes), then this comparison was re-run end to end.

## Result

| | FluxInfer v1 (`ngl=16`) | **FluxInfer v2** (`ngl=24, b=2048`) | LM Studio (auto) |
| --- | --- | --- | --- |
| Prompt tok/s | 27.17 | **184.25** | 159.91 |
| Generation tok/s | 12.69 | **19.48** | 13.74 |
| VRAM used | ~5.7 GB | ~7.82 GB (peak, sampled) | ~7.31 GB (steady-state) |

**FluxInfer v2 now beats LM Studio's own default auto-offload on both
metrics**, not just closes the gap:

- Prompt processing: **+15.2%** over LM Studio (184.25 vs 159.91 tok/s)
- Generation: **+41.8%** over LM Studio (19.48 vs 13.74 tok/s)
- Both confirmed by non-overlapping 5-repetition sample ranges within
  FluxInfer's own comparison report (see
  [`qwen3.5-9b-q8_0-rtx5060-v2.md`](qwen3.5-9b-q8_0-rtx5060-v2.md)) — this
  document only adds the side-by-side against LM Studio's numbers, which
  were captured manually the same way as in the first comparison (isolated
  pp/gen HTTP requests against LM Studio's own warm server, 3 measured
  requests each after a discarded warm-up).

Versus its own v1 baseline, this is +578% prompt tok/s and +53% generation
tok/s from the same search infrastructure, same hardware, same model —
entirely from fixing how the search decides between candidates, not from
any new capability.

## What actually changed the result, and why

The single biggest factor was **batch size**. FluxInfer v1's search
compared `b=256` against `b=512` on one noisy sample and picked 256 by a
razor-thin, statistically meaningless margin. v2 does three things
differently:

1. Every search candidate is now benchmarked with `llama-bench -r 3`
   (three repetitions in one warm process) and scored on the *median*, not
   a single sample — the exact noise source from v1 no longer decides
   close candidates.
2. The batch/ubatch search now always includes llama-bench's own default,
   `2048/512`, instead of assuming a smaller batch is the safe/sane
   choice. It won by a landslide once actually measured properly: batch
   size dominates prompt-processing throughput far more than gpu_layers
   does (see the full candidate table in the v2 report — batch=2048 alone
   took the score from ~49 to ~57 at the *same* gpu_layers).
3. The memory-pressure penalty (which in v1 suppressed `ngl=24` despite it
   measuring faster than `ngl=16`) now only activates past 90% estimated
   usage at a much lower weight, so it no longer overrides large real
   throughput differences based on an imprecise VRAM estimate.

## A bug found and fixed along the way (didn't change this result, but is real)

While re-implementing the OOM-boundary refinement, a real correctness bug
turned up: a candidate that **timed out** (rather than being classified as
a confirmed OOM) was incorrectly treated as "last known good" in the
gpu_layers search-boundary bookkeeping, because the check only looked at
`result.oom`, not general failure. On Windows, an over-VRAM allocation is
frequently reported by the WDDM driver as a hang/timeout rather than a
clean CUDA OOM error (already observed in the v1 comparison) — meaning
this bug could have let a genuinely-failed run corrupt the search's
"where's the boundary" tracking on exactly the platform where it matters
most. Fixed by only promoting genuinely `usable()` results to
last-known-good, and by having the binary-search refinement narrow the
range (without promoting the candidate) on any inconclusive result, not
just a confirmed OOM. Traced and confirmed via code review that this bug
happened not to change v2's outcome for *this* run (the model file's
timed-out candidates were bookended by genuinely successful ones that
overwrote the bad state before it could matter) — but it was a live
correctness bug, not a hypothetical one, and would matter in a run where
the OOM/timeout falls at the edge of the candidate list.

## Remaining honest caveats

- The 90%/0.3 memory-pressure recalibration and the added 2048/512
  candidate are still just parameter/grid changes, not a different
  algorithm — a sufficiently different model/GPU/VRAM combination could
  still expose a case where the search picks a worse config than a hand-
  tuned one. This result is real but is one data point, not a proof of
  general superiority.
- LM Studio's "auto" offload logic is a black box from FluxInfer's side;
  this comparison doesn't know exactly how it picked its own ~29-30 layer,
  ~7.3 GB configuration, only that FluxInfer's search (with a real,
  measured 24-layer + large-batch config) now outperforms it on this
  model.
- The batch=2048/ubatch=512 win is likely specific to this GPU having
  enough compute+VRAM headroom to benefit from a large batch at `ngl=24`;
  smaller GPUs or larger models could still find the memory-constrained
  smaller-batch candidates winning, which is exactly why all four
  batch/ubatch candidates are still searched rather than hardcoding 2048.
