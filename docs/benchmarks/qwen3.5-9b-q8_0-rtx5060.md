# FluxInfer benchmark comparison report

Real-world run, 2026-07-07. Hardware: Intel Core i5-14600KF, 32 GB RAM,
NVIDIA GeForce RTX 5060 8 GB (driver reporting CUDA 13.3). llama.cpp
release [b9895](https://github.com/ggml-org/llama.cpp/releases/tag/b9895)
(`llama-b9895-bin-win-cuda-13.3-x64.zip`), CUDA backend, compute capability
12.0 (sm_120 / Blackwell) confirmed via `--list-devices`. Exact commands
used are in the main write-up; paths below are local to the machine this
ran on.

Note on `llama-bench: ... (unknown)`: this specific llama-bench build
exposes no version string via either `--version` (returns "invalid
parameter for argument: --version") or `--help` — only `llama-cli` prints
one (`version: 9895 (defa95c30)`). `detect_llama_version()` was fixed
during this milestone to also accept a "version:" banner (previously only
"build:"), but llama-bench here simply has neither, so "unknown" is
correct/expected rather than a parsing failure.

## Model

- Path: `C:/Users/Utente/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`
- Architecture: qwen35
- Quantization: Q8_0
- Layer count: 32
- llama-bench: `C:\Users\Utente\AppData\Local\Temp\claude\C--Users-Utente-Desktop-nerual\6887fa3d-d685-4f1a-a65a-a822bcee9d9e\scratchpad\llama_release\llama\llama-bench.exe` (unknown)

## Arguments

- Baseline: `-m C:/Users/Utente/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf -p 512 -n 128 -t 14 -ngl 0 -b 256 -ub 128 -r 1 -o json`
- Selected: `-m C:/Users/Utente/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf -p 512 -n 128 -t 14 -ngl 16 -b 256 -ub 128 -r 1 -o json`

## Results

### Baseline

Runs: 3 successful, 0 failed (of 3 measured; 1 warm-up run(s) excluded)

| Metric | Mean | Median | Min | Max | Stddev |
| --- | --- | --- | --- | --- | --- |
| Prompt processing tok/s | 14.5452 | 14.5424 | 14.5422 | 14.5511 | 0.00507243 |
| Generation tok/s | 7.90266 | 7.90481 | 7.88322 | 7.91995 | 0.0184556 |

Peak VRAM observed: 2439.57 MiB

### Selected (FluxInfer-tuned)

Runs: 3 successful, 0 failed (of 3 measured; 1 warm-up run(s) excluded)

| Metric | Mean | Median | Min | Max | Stddev |
| --- | --- | --- | --- | --- | --- |
| Prompt processing tok/s | 27.173 | 27.1994 | 27.1086 | 27.211 | 0.0560642 |
| Generation tok/s | 12.6916 | 13.4256 | 11.1542 | 13.4949 | 1.33186 |

Peak VRAM observed: 5727.95 MiB

## Improvement

- Generation tok/s: 60.5991%
- Prompt processing tok/s: 86.8172%
- **Confirmed**: every selected-config run outperformed every baseline-config run in this sample (non-overlapping ranges).

## All configurations attempted during the search

| Config | Outcome | Prompt tok/s | Gen tok/s | Score |
| --- | --- | --- | --- | --- |
| baseline (cpu-only) | ok | 14.5403 | 7.80511 | 16.3765 |
| gpu_layers=0 | ok | 14.5347 | 7.91554 | 16.5965 |
| gpu_layers=8 | ok | 18.5956 | 10.3324 | 21.9865 |
| gpu_layers=16 | ok | 27.2378 | 13.7493 | 29.8161 |
| gpu_layers=24 | ok | 50.6449 | 19.6239 | 26.316 |
| gpu_layers=32 | timed out | 0 | 0 | -1000 |
| batch=128 ubatch=64 | ok | 13.7537 | 13.5759 | 27.761 |
| batch=512 ubatch=256 | ok | 53.0968 | 12.3555 | 29.7931 |
| threads=17 | ok | 27.2092 | 12.7351 | 27.7844 |
| threads=20 | ok | 27.2146 | 12.1143 | 26.5435 |

## Observations (not part of the generated report, added afterward)

- **`gpu_layers=32` (full offload) failed, as expected.** The Q8_0 file is
  9.5 GB against 8 GB of VRAM; llama-bench did not raise a clean CUDA OOM
  within FluxInfer's 120s per-run timeout, it just hung/ran far slower than
  normal (classified as `timed out`, not `oom`). On Windows, this is
  consistent with the WDDM driver silently falling back to system-memory
  spillover instead of failing an over-budget allocation outright. The
  staged search still did the right thing operationally (stopped climbing
  after this point), but the *classification* is "timed out" rather than
  "OOM" — worth knowing if you're specifically looking for OOM log
  messages on Windows+WDDM.
- **`gpu_layers=24` scored lower than `gpu_layers=16` despite higher raw
  throughput** (19.62 vs 13.75 gen tok/s) in the single-sample search
  stage. This is the memory-pressure penalty in the scoring formula doing
  its job: at 24/32 layers the estimated VRAM usage is close enough to the
  8 GB budget that the quadratic penalty term dominates, so FluxInfer
  preferred the safer 16-layer config. Whether 24 layers is actually more
  practical day-to-day (it did complete without OOM/timeout in this run)
  is a legitimate judgment call the current scoring weights resolve
  conservatively; not changed in this milestone since it wasn't one of its
  goals, but worth reconsidering later with real usage data.
- One measured `gpu_layers=16` run during the comparison phase had a
  noticeably lower generation tok/s (11.15 vs ~13.4-13.5 for the other two)
  — visible in the min/max/stddev columns above. This is exactly the kind
  of run-to-run variance repeated measurement is meant to surface, and is
  why the report states mean/median/min/max/stddev rather than a single
  number.

