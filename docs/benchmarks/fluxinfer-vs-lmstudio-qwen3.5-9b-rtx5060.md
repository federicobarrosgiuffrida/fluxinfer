# FluxInfer vs LM Studio — real-world head-to-head

Real-world run, 2026-07-07. Same machine, same model, same GPU as
[`qwen3.5-9b-q8_0-rtx5060.md`](qwen3.5-9b-q8_0-rtx5060.md): Intel Core
i5-14600KF, 32 GB RAM, NVIDIA GeForce RTX 5060 8 GB.

This is **not** a FluxInfer feature (no code changes) — a manual comparison
run against LM Studio (llama.cpp `2.23.1` CUDA12 backend, headless via its
`lms` CLI) using the same model file as the FluxInfer benchmark:
`Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`.

## Method

- FluxInfer numbers: from the checked-in report, `fluxinfer tune --compare-repeats 3`
  (separate `llama-bench.exe` process per measured run, `-b 256 -ub 128 -ngl 16 -t 14`).
- LM Studio numbers: `lms load <model> -y` (auto GPU-offload ratio, no
  manual tuning — LM Studio's actual default UX), then 1 discarded warm-up
  + 3 measured HTTP requests to its OpenAI-compatible `/v1/completions`
  endpoint, isolating prompt-processing (long prompt, `max_tokens=1`) and
  generation (1-token prompt, `max_tokens=128`) the same way llama-bench
  separates `pp`/`tg` tests.

## Results

| | FluxInfer-selected (`ngl=16`) | LM Studio (auto) |
| --- | --- | --- |
| Prompt tok/s (mean of 3) | 27.17 | 159.91 |
| Generation tok/s (mean of 3) | 12.69 (median 13.43) | 13.74 |
| VRAM used | ~5.73 GB (peak, sampled) | ~7.31 GB (steady-state, `nvidia-smi`) |

**Generation speed is close** (FluxInfer within ~8% of LM Studio, and
FluxInfer's own median run was actually faster than its mean, i.e. one
noisy run pulled the mean down). **Prompt processing is not** — LM Studio
is ~5.9x faster. This is a real result, not favorable spin, and it comes
with three concrete, honest explanations, found by actually digging into
why rather than just reporting the gap:

1. **Different GPU offload.** LM Studio's "auto" ratio used ~7.3 GB of the
   8 GB budget; FluxInfer's search stopped at `ngl=16` (5.73 GB) because
   its scoring formula's memory-pressure penalty downweighted `ngl=24`
   (measured *faster* in the search: 19.6 gen tok/s vs 13.7, see the other
   report) for sitting closer to the VRAM ceiling, and its coarse
   0/25/50/75/100% grid jumped straight from 24 to the OOM'd 32 with
   nothing in between. LM Studio's offload heuristic is evidently more
   aggressive/sophisticated here than FluxInfer's current staged search +
   conservative scoring — a genuine gap, not a wash.
2. **Batch size.** FluxInfer's batch/ubatch search picked `b=256/ub=128`
   over `b=512/ub=256` by a razor-thin, single-sample margin (score 29.82
   vs 29.79 — statistical noise, not a real difference). Prompt processing
   throughput is heavily batch-size sensitive; llama-bench's own default
   is `b=2048`. FluxInfer likely left significant prompt-processing
   performance on the table by settling on a small batch based on one
   noisy sample.
3. **Cold process vs warm server.** FluxInfer's "3 measured runs" are 3
   separate `llama-bench.exe` process launches — each gets its own fresh
   CUDA context. LM Studio runs one persistent server process; my 3
   measured HTTP requests all landed on an already-warm context (its own
   log shows prompt processing jumping from 87.9 tok/s on the very first,
   cold request to steady-state once CUDA graphs were captured and
   reused). Comparing "N cold process launches" against "N requests to one
   warm server" is not perfectly apples-to-apples, and that gap is
   structural to how `tune`'s repeated-measurement currently works, not
   something either config file changes.

## An unrelated discovery along the way

LM Studio's server log shows: `layer 0 is assigned to device CPU but the
fused Gated Delta Net tensor is assigned to device CUDA0 (usually due to
missing support)` / `fused Gated Delta Net (chunked) not supported, set to
disabled`. Qwen3.5 apparently mixes standard transformer layers with a
linear-attention/SSM-style "Gated Delta Net" mechanism, and at least one
such tensor gets forced onto CPU by llama.cpp regardless of `-ngl`. This
means "N GPU layers" isn't a perfectly uniform, fully-offloadable count for
this architecture — a real limitation of the "block_count is the whole
story" assumption FluxInfer's GPU-layers search currently makes, on top of
the already-documented "layers aren't equal-sized" caveat.

## Takeaway

FluxInfer's real-metadata-driven search picked a *safe, working*
configuration and correctly avoided the OOM at full offload — the core
0.3.1 goal. But this head-to-head shows it isn't yet matching a mature
tool's hand-tuned-by-a-team defaults on raw throughput, specifically on
prompt processing. The concrete next steps this suggests (not implemented
here, out of scope for this milestone): finer-grained gpu_layers probing
near the OOM boundary instead of fixed 25% steps, re-testing near-tied
batch/ubatch candidates with more than one sample before picking a winner,
and reconsidering how aggressively the memory-pressure penalty should
discourage configs that actually didn't fail.
