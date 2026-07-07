# FluxInfer benchmark comparison report (v2, after scoring/search fixes)

Real-world run, 2026-07-07, same machine/model/llama.cpp build as the [v1
report](qwen3.5-9b-q8_0-rtx5060.md), after applying the fixes described in
[`fluxinfer-vs-lmstudio-v2-qwen3.5-9b-rtx5060.md`](fluxinfer-vs-lmstudio-v2-qwen3.5-9b-rtx5060.md):
memory-pressure penalty recalibrated (90% ramp-start, 0.3 weight, was
70%/1.0), median-of-3 scoring per search candidate, iterative OOM-boundary
refinement, a llama-bench-default-matching 2048/512 batch/ubatch candidate
added, and the comparison phase switched to a single warm `llama-bench -r
N` process per side instead of N separate cold process launches.

Command used: `fluxinfer tune <model> --llama-dir <dir> --timeout 60
--search-repetitions 3 --compare-repeats 5 --warmup-runs 1 --report-out
report.md`.

Note: the "Arguments" section below shows `-r 1` because it was generated
before a same-day fix that makes the displayed arguments match the actual
`-r <compare-repeats>` used (5 here, not 1) -- the *numbers* in this report
are correct and were genuinely measured with `-r 5` in a single warm
process (see `tune_full_log_v2.txt`), only this cosmetic display bug in
the argument string itself has since been corrected for future reports.

## Model

- Path: `C:/Users/Utente/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`
- Architecture: qwen35
- Quantization: Q8_0
- Layer count: 32
- llama-bench: `C:\Users\Utente\AppData\Local\Temp\claude\C--Users-Utente-Desktop-nerual\6887fa3d-d685-4f1a-a65a-a822bcee9d9e\scratchpad\llama_release\llama\llama-bench.exe` (unknown)

## Arguments

- Baseline: `-m C:/Users/Utente/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf -p 512 -n 128 -t 14 -ngl 0 -b 256 -ub 128 -r 5 -o json`
- Selected: `-m C:/Users/Utente/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf -p 512 -n 128 -t 14 -ngl 24 -b 2048 -ub 512 -r 5 -o json`

## Results

### Baseline

Repetitions: 5 successful, 0 failed. Measured via a single warm `llama-bench -r N` process (its own internal warmup pass excluded, not a separately-launched process).

| Metric | Mean | Median | Min | Max | Stddev |
| --- | --- | --- | --- | --- | --- |
| Prompt processing tok/s | 14.5384 | 14.5377 | 14.5326 | 14.5451 | 0.00447415 |
| Generation tok/s | 7.88958 | 7.90587 | 7.82179 | 7.91567 | 0.0386393 |

Peak VRAM observed: 2431.23 MiB

### Selected (FluxInfer-tuned)

Repetitions: 5 successful, 0 failed. Measured via a single warm `llama-bench -r N` process (its own internal warmup pass excluded, not a separately-launched process).

| Metric | Mean | Median | Min | Max | Stddev |
| --- | --- | --- | --- | --- | --- |
| Prompt processing tok/s | 184.252 | 184.406 | 183.528 | 184.469 | 0.406327 |
| Generation tok/s | 19.4833 | 19.4817 | 19.4409 | 19.5169 | 0.0280291 |

Peak VRAM observed: 7823.23 MiB

## Improvement

- Generation tok/s: 146.95%
- Prompt processing tok/s: 1167.35%
- **Confirmed**: every selected-config run outperformed every baseline-config run in this sample (non-overlapping ranges).

## All configurations attempted during the search

| Config | Outcome | Prompt tok/s | Gen tok/s | Score |
| --- | --- | --- | --- | --- |
| baseline (cpu-only) | timed out | 0 | 0 | -1000 |
| gpu_layers=0 | timed out | 0 | 0 | -1000 |
| gpu_layers=8 | ok | 18.7078 | 10.5966 | 22.5294 |
| gpu_layers=16 | ok | 27.3909 | 14.0033 | 30.3806 |
| gpu_layers=24 | ok | 51.1029 | 20.2586 | 43.7937 |
| gpu_layers=32 | timed out | 0 | 0 | -1000 |
| batch=128 ubatch=64 | ok | 25.9184 | 20.3173 | 41.2025 |
| batch=512 ubatch=256 | ok | 98.9939 | 20.2745 | 48.7093 |
| batch=2048 ubatch=512 | ok | 186.292 | 20.2768 | 57.4911 |
| threads=17 | ok | 186.366 | 19.4562 | 55.8573 |
| threads=20 | ok | 186.378 | 19.4827 | 55.9115 |

