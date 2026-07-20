# FluxInfer

[![CI](https://github.com/federicobarrosgiuffrida/fluxinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/federicobarrosgiuffrida/fluxinfer/actions/workflows/ci.yml)

FluxInfer is a multiplatform wrapper/autotuner for [llama.cpp](https://github.com/ggml-org/llama.cpp).
It detects your hardware, benchmarks a small number of `llama-bench`
configurations, picks the best one for a given `.gguf` model, saves it as a
reusable profile, and then uses that profile to launch `llama-cli` or
`llama-server` for you.

FluxInfer does **not** reimplement llama.cpp, does not link against it, and
does not modify it. It only locates and invokes prebuilt
`llama-bench` / `llama-cli` / `llama-server` binaries as external processes.

## Status: experimental (milestone 0.4.0)

Hardware detection, real GGUF metadata discovery, a staged benchmark search
driven by the model's actual layer count, a MoE-aware expert-placement
search (`--n-cpu-moe`), measured peak-VRAM sampling with silent-spill
rejection, profile persistence, a reproducible repeated-measurement
comparison report, a pre-flight fit estimate, an interactive menu, and the
six subcommands (`menu`, `inspect`, `doctor`, `tune`, `run`, `serve`). It has not been used in
production. Interfaces (CLI flags, profile JSON schema) may still change.

**Tested on:** one machine, two models, one GPU — Qwen3.5-9B (dense, Q8_0)
on an NVIDIA RTX 5060 8GB, Windows and Qwythos-9B-Claude-Mythos-5-1M-MTP-Q8_0.
See
[`docs/benchmarks/`](docs/benchmarks/) for the real numbers, including a
head-to-head against LM Studio's own default auto-offload on the same
hardware. **This is one data point, not a general performance guarantee.**
Results on other GPUs (different VRAM budgets, architectures — e.g. RTX
3060, 4060 Ti), other models (especially MoE or other hybrid
architectures), and other OSes are unverified. Reports from testing on
different hardware are very welcome — please open an issue with your
`fluxinfer tune --compare-repeats 3 --report-out report.md` output and
hardware specs attached.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for noncommercial
use (personal, research, hobby, nonprofit/educational/government use); a
separate commercial license is required for commercial use. Open an issue
or contact the maintainer to discuss commercial licensing.

## Dependencies

All C++ dependencies are vendored as single-header/amalgamated sources under
[`third_party/`](third_party/), so a normal build does not need network
access or a package manager:

- [nlohmann/json](https://github.com/nlohmann/json) 3.11.3 — profile JSON (de)serialization.
- [CLI11](https://github.com/CLIUtils/CLI11) 2.4.2 — command-line parsing.
- [Catch2](https://github.com/catchorg/Catch2) 3.6.0 — unit tests (`FLUXINFER_BUILD_TESTS`, on by default).

Otherwise, only a C++20 compiler and CMake ≥ 3.20 are required. On Windows,
GPU detection additionally links `pdh.lib` (part of the Windows SDK); on
Linux it links `pthread` and `dl` for dynamic NVML loading.

## Building

### Windows

Requires a Visual Studio 2022+ installation with the "Desktop development
with C++" workload (provides MSVC and, optionally, a bundled CMake/Ninja
under `Common7\IDE\CommonExtensions\Microsoft\CMake`).

```bat
:: from a "x64 Native Tools Command Prompt for VS"
cmake -S . -B build -G Ninja
cmake --build build -j
```

The `fluxinfer.exe` CLI and `fluxinfer_tests.exe` test binary are written to
`build/`.

### Linux

```bash
sudo apt install build-essential cmake ninja-build   # or your distro's equivalent
cmake -S . -B build -G Ninja
cmake --build build -j
```

### Running the tests

```bash
ctest --test-dir build --output-on-failure
# or directly:
./build/tests/fluxinfer_tests
```

Tests do not require a GPU or a real llama.cpp build: process-runner and
benchmark-classification tests re-invoke the test binary itself (via hidden
`--fluxinfer-*` flags handled in `tests/main.cpp`) as a stand-in child
process, and parser/scoring/profile/GGUF tests use fixture strings and
small synthetic files.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds and runs the
full test suite on every push/PR, on:

- **Windows** (MSVC, Visual Studio generator)
- **Ubuntu** (GCC, Ninja)

macOS is intentionally not covered yet (see [Current limits](#current-limits)).

## llama.cpp integration

FluxInfer never builds llama.cpp itself. Point it at prebuilt binaries in
one of three ways (checked in this order):

1. `--llama-dir <path>` on any subcommand.
2. the `FLUXINFER_LLAMA_DIR` environment variable.
3. `PATH`, plus a few common local build output directories relative to the
   current working directory (`external/llama.cpp/build/bin`,
   `llama.cpp/build/bin`, `build/bin`).

If you want to keep a llama.cpp checkout inside this repository (e.g. as a
submodule) for convenience, `external/llama.cpp/` is reserved for that:

```bash
git submodule add https://github.com/ggml-org/llama.cpp external/llama.cpp
cmake -S external/llama.cpp -B external/llama.cpp/build -DGGML_CUDA=ON
cmake --build external/llama.cpp/build -j
```

FluxInfer will then find the resulting binaries automatically under
`external/llama.cpp/build/bin`.

Before using any llama.cpp flag, FluxInfer runs the target binary with
`--help` and only uses flags it actually finds in the output — it never
assumes a flag exists just because a particular llama.cpp version supports
it.

## Usage

```bash
fluxinfer menu     # interactive: pick a model and an action, no flags to remember
fluxinfer inspect
fluxinfer doctor
fluxinfer tune  path/to/model.gguf [--timeout SECONDS] [--idle-timeout SECONDS] [--llama-dir DIR]
                                    [--profiles-dir DIR] [--context N] [--search-repetitions N]
                                    [--vram-headroom-mb N] [--no-moe-tune]
                                    [--compare-repeats N] [--warmup-runs N] [--report-out FILE]
fluxinfer run   path/to/model.gguf [-- extra llama-cli args]
fluxinfer serve path/to/model.gguf [--host HOST] [--port PORT] [-- extra llama-server args]
```

### `fluxinfer inspect`

```
$ fluxinfer inspect
CPU:
  Name: Intel Core i5-14600KF
  Physical cores: 14
  Logical threads: 20

Memory:
  Total RAM: 32 GB
  Available RAM: 21 GB

GPU:
  Name: NVIDIA GeForce RTX 5060
  Total VRAM: 8 GB
  Available VRAM: 7.3 GB
  Backend: CUDA

llama.cpp:
  llama-bench: found (/usr/local/bin/llama-bench)
  llama-cli: found (/usr/local/bin/llama-cli)
  llama-server: found (/usr/local/bin/llama-server)
```

If NVML can't be loaded (no NVIDIA driver, non-NVIDIA GPU, ...), the `GPU`
section prints a reason instead of crashing:

```
GPU:
  Not available (NVML shared library not found (no NVIDIA driver installed, or non-NVIDIA GPU))
```

### `fluxinfer tune`

```
$ fluxinfer tune qwen-model.gguf
Using llama-bench: /usr/local/bin/llama-bench
Detected 23 supported llama-bench flags.
Model metadata: architecture=qwen35 layers=32 context_length=262144 quantization=Q8_0
Running staged benchmark search (this can take a while)...
  [baseline (cpu-only)] prompt=612.3 tok/s, gen=9.8 tok/s, score=39.5
  [gpu_layers=0] prompt=612.3 tok/s, gen=9.8 tok/s, score=39.5
  [gpu_layers=8] prompt=780.1 tok/s, gen=13.2 tok/s, score=52.4
  [gpu_layers=16] OOM
  [batch=128 ubatch=64] prompt=770.0 tok/s, gen=13.0 tok/s, score=51.9
  [threads=20] prompt=782.4 tok/s, gen=13.4 tok/s, score=53.0

Best configuration: threads=20
  threads=20 gpu_layers=8 batch=256 ubatch=128
  prompt=782.4 tok/s, generation=13.4 tok/s, score=53.0
Profile saved to profiles/qwen-model-5cbad49f5acb922c.json
```

Runs a staged search rather than an exhaustive grid. Every candidate is
benchmarked with `llama-bench -r <search-repetitions>` (default 3, in a
single warm process/model-load) and scored on the *median* sample, not a
single run — a lone noisy measurement was found in practice to flip which
of two close candidates looked better:

1. **Baseline** — one conservative, CPU-only configuration.
2. **GPU layers** — probes ~0/25/50/75/100% of the model's *real* transformer
   layer count, read from its own GGUF metadata (see
   [Model metadata discovery](#model-metadata-discovery) below), always
   including 0 (no offload) and the full count (full offload). Once a value
   OOMs, no larger value is tried; an iterative binary search (up to 4 more
   probes) then refines the boundary between the highest known-good value
   and the lowest known-OOM value, converging much closer to the true
   offload ceiling than a single midpoint probe would. This stage is
   skipped entirely — never estimated — if there's no GPU or the layer
   count couldn't be read.
3. **Batch / ubatch** — a handful of `(batch, ubatch)` pairs including
   llama-bench's own default (2048/512), trimmed based on available RAM.
   Prompt-processing throughput is strongly batch-size sensitive, so the
   search doesn't assume a smaller batch is safer by default.
4. **Threads** — physical core count, logical thread count, and their
   midpoint.

The highest-scoring *usable* configuration (no crash/OOM/timeout/invalid
output) is saved as the profile. See [Scoring](#scoring) and
[Profiles](#profiles) below.

`--search-repetitions N` controls the per-candidate repetition count
(default 3); higher is more resistant to noisy single-sample decisions at
some extra search time (repetitions reuse the same model load, so the
added cost is just N× the compute passes, not N× the load time).

#### Context size (`--context`)

`llama-bench` has no context-size flag of its own — it only allocates as
much KV cache as `-p`/`-n` need for the test (a few hundred tokens). But
`llama-cli`/`llama-server` (used by `run`/`serve`) *do* have one
(`-c`/`--ctx-size`), and if it's left unset they default to **the model's
full native training context** — which for a modern long-context model can
be 262144 tokens or more. Left unaddressed, this is a real train/serve
mismatch: a config benchmarked with a tiny effective context could behave
very differently once actually run with a much larger one, since KV cache
size competes with GPU-offloaded model layers for the same VRAM budget.
This was found via a real regression report (a tuned config performing
*worse* in practice than in the benchmark) rather than found by inspection.

FluxInfer now tunes and serves at a single, explicit context size,
consistent between the two phases: `--context N` (default 4096) is passed
to `llama-bench` as `-d N` (its "pre-fill this many tokens of KV cache
before measuring" flag — the closest equivalent to `-c` it has) during
`tune`, is saved into the profile, and is passed as `-c N` to
`llama-cli`/`llama-server` by `run`/`serve`. Profiles saved before this
field existed have `context_length == 0` and fall back to the old
behavior (no explicit `-c`) rather than guessing a value.

#### Model metadata discovery

FluxInfer reads the model's real transformer layer count (and a few other
fields) directly from its own GGUF key-value metadata, rather than
estimating it from file size. Implemented as a minimal, read-only parser
([`gguf_metadata.hpp`](include/fluxinfer/llama/gguf_metadata.hpp)) that:

- reads only the GGUF header and key-value metadata section — never tensor
  data, and never the full (often tens-of-GB) file;
- validates every length/offset against the known file size and rejects
  malformed or truncated files with a clear error instead of crashing;
- supports GGUF versions 2 and 3, and all 12 scalar value types plus arrays
  (walked and discarded, since FluxInfer doesn't need e.g. tokenizer vocab
  arrays);
- looks up architecture-specific keys by convention
  (`<architecture>.block_count`, `.context_length`, `.embedding_length`,
  `.expert_count`, `.expert_used_count`) after first reading
  `general.architecture`, since GGUF doesn't guarantee key ordering.

If a value can't be found (missing key, wrong type, unreadable file),
FluxInfer reports it as unknown rather than guessing — and the GPU-layers
search stage is simply skipped in that case, per the same "never invent a
number" policy.

An existing llama.cpp tool/log output was considered as the primary source
(per the general preference for reusing llama.cpp over reimplementing it),
but `llama-bench`/`llama-cli` only print this as unstructured,
version-dependent log text (see `llama_model_loader: - kv N: ...` lines),
and llama.cpp ships no dedicated `--dump-metadata`-style binary among the
tools FluxInfer already depends on. Parsing GGUF directly — a small,
stable, documented binary format — was more reliable and testable than
scraping log lines that could change wording between versions.

#### Reproducible benchmark comparison (`--compare-repeats`)

```bash
fluxinfer tune qwen-model.gguf --compare-repeats 3 --warmup-runs 1 --report-out report.md
```

After the staged search picks a configuration, this re-benchmarks *both*
the search's baseline (CPU-only) configuration and the selected
configuration via a single `llama-bench -r N` process each (not `N`
separate process launches): all `N` repetitions, and llama-bench's own
internal warmup pass, share one warm model load/CUDA context, which is
both faster (one model load per side, not `N+1`) and more representative
of how a long-lived `fluxinfer run`/`serve` process actually behaves than
repeatedly cold-starting a new process would be. `--warmup-runs 0` passes
`--no-warmup` through to llama-bench (it supports on/off, not a
configurable count); any nonzero value leaves its default warmup enabled.
The report includes:

- model path, architecture, quantization, and layer count;
- the exact `llama-bench` arguments used for each side;
- prompt-processing and generation tok/s: mean, median, min, max, stddev,
  computed from llama-bench's own per-repetition `samples_ts` values
  (not just its aggregate average);
- peak VRAM sampled via NVML while each side runs (see
  [`vram_sampler.hpp`](include/fluxinfer/hardware/vram_sampler.hpp)), when
  available;
- percentage improvement — but only presented as **confirmed** if every
  selected-config repetition outperformed every baseline-config repetition
  in the sample (non-overlapping ranges); otherwise the report explicitly
  says the improvement is *not* conclusively demonstrated by that sample;
- every configuration attempted during the search, and any OOM/failed runs
  on either side.

Real reports generated this way, against a real 9.5 GB Qwen3.5-9B Q8_0
model and real `llama-bench` on an RTX 5060 8 GB, are checked in under
[`docs/benchmarks/`](docs/benchmarks/) as worked examples — including a
head-to-head against LM Studio's own default auto-offload on the same
model/GPU that surfaced real gaps (not just a flattering percentage) and
directly motivated the scoring/search changes described above: see
[`docs/benchmarks/fluxinfer-vs-lmstudio-qwen3.5-9b-rtx5060.md`](docs/benchmarks/fluxinfer-vs-lmstudio-qwen3.5-9b-rtx5060.md)
for the original findings and
[`docs/benchmarks/fluxinfer-vs-lmstudio-v2-qwen3.5-9b-rtx5060.md`](docs/benchmarks/fluxinfer-vs-lmstudio-v2-qwen3.5-9b-rtx5060.md)
for the re-run after the fixes.

### `fluxinfer run`

```bash
fluxinfer run qwen-model.gguf -- -p "Scrivi un plugin Paper"
```

Loads the profile for `qwen-model.gguf`, builds the `llama-cli` argument
list from it, and launches `llama-cli` with stdin/stdout/stderr inherited
(so it behaves like running `llama-cli` directly). Anything after `--` is
forwarded verbatim as extra `llama-cli` arguments. Ctrl+C / SIGTERM sent to
FluxInfer are forwarded to the child process.

### `fluxinfer serve`

```bash
fluxinfer serve qwen-model.gguf --host 0.0.0.0 --port 8080 -- --api-key secret
```

Same idea as `run`, but launches `llama-server` and prints the host/port it
was started with. Extra arguments after `--` are forwarded to
`llama-server`.

## Scoring

```
score = generation_tokens_per_second * 2.0
      + prompt_tokens_per_second     * 0.10
      - first_token_latency_ms       * 0.01
      - memory_pressure_penalty      * 0.3
```

Configurations that crash, OOM, time out, or produce output FluxInfer can't
parse score a flat `-1000` instead (see `ScoringWeights::instability_penalty`
in [`scoring.hpp`](include/fluxinfer/tuner/scoring.hpp)), so they never win
over a slower-but-working configuration. All weights are adjustable via
`fluxinfer::tuner::ScoringWeights`.

`memory_pressure_penalty` only ramps up once *estimated* usage exceeds 90%
of the available budget (was 70%, weighted 1.0, until a real-world
head-to-head against LM Studio — see below — showed it heavily penalizing
a ~75%-estimated-usage config that had already completed successfully
without OOM, purely because the estimate itself runs somewhat hot). Real
OOM/crash/timeout is always detected from the benchmark process's actual
exit behaviour (`usable()`), never from this estimate — the penalty exists
only as a tie-breaking safety margin between already-successful configs,
which is why both its ramp-start and its weight were deliberately lowered
rather than tuned to "feel" more aggressive.

`first_token_latency_ms` and `estimated_ram_bytes`/`estimated_vram_bytes`
are **estimates**: llama-bench doesn't report actual memory usage or
time-to-first-token, so FluxInfer approximates them from prompt-processing
throughput and from `gpu_layers` relative to the model's real layer count
(from GGUF metadata — see [Model metadata discovery](#model-metadata-discovery)),
respectively; this still assumes roughly equal-sized layers, which doesn't
hold for every architecture. Actual OOM/crash detection is always based on
the benchmark process's real exit code and stderr output, never on these
estimates.

## Profiles

Saved under `profiles/<model-stem>-<fingerprint-prefix>.json` (directory
configurable with `--profiles-dir`), matching this schema:

```json
{
  "schema_version": 1,
  "model": { "path": "...", "size_bytes": 0, "fingerprint": "..." },
  "hardware": { "cpu": "...", "logical_threads": 0, "ram_bytes": 0, "gpu": "...", "vram_bytes": 0 },
  "llama": { "version": "...", "binary_path": "...", "supported_flags": [] },
  "best_config": { "threads": 0, "gpu_layers": 0, "batch_size": 0, "ubatch_size": 0, "kv_cache_type": null, "context_length": 4096 },
  "results": { "prompt_tps": 0.0, "generation_tps": 0.0, "duration_ms": 0, "score": 0.0 }
}
```

A profile is considered invalid (and `run`/`serve` will refuse to use it,
pointing you back to `tune`) if any of the following changed since it was
created: the model file's size or fingerprint, the GPU, the VRAM capacity,
the llama.cpp version, or the set of supported flags.

**Model fingerprint**: file size + last-write-time + a hash of the first and
last 1 MiB of the file, rather than a full-file hash. GGUF models are
commonly tens of gigabytes; hashing the entire file on every `tune`/`run`
invocation would cost minutes of disk I/O for a local dev tool. This trades
exhaustiveness (a file deliberately crafted to keep the same size, mtime,
and prefix/suffix while changing interior bytes would evade detection) for
near-instant fingerprinting, which is the right tradeoff for this threat
model. See `compute_model_info()` in
[`profile.hpp`](include/fluxinfer/profiles/profile.hpp).

## Current limits

- The GGUF parser reads only KV metadata, not tensor descriptors or tensor
  data, so it has no way to detect true per-layer size variation (e.g.
  uneven quantization across layers, or a non-uniform architecture) — the
  GPU-layers search still assumes roughly equal-sized layers when
  estimating VRAM usage for scoring. Only GGUF versions 2 and 3 are
  supported (version 1 is rejected outright). It also has no visibility
  into architecture-specific device-placement quirks: on a real hybrid
  model (Qwen3.5, which mixes standard attention with a
  linear-attention/SSM-style "Gated Delta Net" block), llama.cpp forces at
  least one tensor onto CPU regardless of `-ngl` — so "N GPU layers" isn't
  always as uniform a knob as the search assumes.
- The staged search optimises one axis at a time and the axes are not
  independent: batch size and thread count both move VRAM usage, so an
  offload value chosen before them can be invalidated by them. Since
  0.4.0 a bounded second pass re-probes the offload value's immediate
  neighbours under the final batch/thread settings, which narrows the
  problem but does not make the search jointly optimal.
- The fit estimate printed before tuning is arithmetic on weights, KV
  cache and a flat overhead allowance -- not a model of llama.cpp's
  allocator. It is meant to catch the obviously-impossible case, warns
  rather than blocks, and says so when the model's metadata is too
  incomplete to estimate at all.
- No Bayesian optimization or other adaptive search — the staged search is
  fixed and coarse by design (see [Roadmap](#roadmap)). It does now run
  each candidate with `-r <search-repetitions>` (default 3) and score the
  median rather than a single sample, and refine the GPU-layers OOM
  boundary iteratively (see [`fluxinfer tune`](#fluxinfer-tune) above) —
  both added after a real head-to-head against LM Studio (see
  [`docs/benchmarks/`](docs/benchmarks/)) showed single-sample decisions
  and a coarse boundary probe leaving real performance on the table.
- MoE tuning searches `--n-cpu-moe` at *whole-layer* granularity, which is
  all llama.cpp exposes: GGUF stores a layer's routed experts as one fused
  tensor, so individual experts cannot be placed separately from outside
  the engine (see
  [`docs/moe-vram-cache-research.md`](docs/moe-vram-cache-research.md)).
  The search also picks layers by index, not by measured routing
  popularity — informed placement is `0.5`, and requires per-layer
  activation profiling that does not exist here yet. On MoE models the
  dense `--n-gpu-layers` sweep is replaced rather than complemented
  (`--no-moe-tune` forces the old behaviour), because that sweep is
  non-monotonic on such models and its results are misleading.
- The peak-VRAM sampler (`VramSampler`) polls NVML on a fixed interval from
  a background thread; it can miss short spikes between polls and reports
  GPU-wide usage (other processes included), not llama-bench's allocations
  specifically.
- On Windows, an over-VRAM allocation does not fail cleanly. WDDM either
  stalls the process (reported as a hang/timeout rather than a CUDA OOM)
  or, worse, silently backs the allocation with system RAM so the run
  *succeeds* at a fraction of the speed. FluxInfer handles both, but
  neither perfectly:
  - Timeouts are still treated as *inconclusive* rather than a confirmed
    OOM (it won't select such a config, but won't necessarily stop
    climbing on it either), so an unrelated slow run isn't misread as a
    VRAM ceiling. Since `0.4.0` a run is only killed for being *silent*
    (`--idle-timeout`), and the total budget scales with model size, so
    healthy runs on large models are no longer cut short.
  - Silent spills are caught by comparing measured peak VRAM and
    throughput against the last good configuration
    (`looks_like_vram_spill`). This is a heuristic with a deliberately
    conservative threshold: a spill that costs less than ~40% of
    throughput will not be flagged, and the guard depends on NVML samples
    being available at all.
  - `--vram-headroom-mb` (default 1.5GB on Windows, 1GB elsewhere) keeps
    the search away from that regime in the first place.
- `fluxinfer run`/`serve` re-detect supported flags from `llama-cli` /
  `llama-server`'s own `--help` output (which may differ from
  `llama-bench`'s), but profile *validity* is checked against
  `llama-bench`'s snapshot for consistency with `tune`. If `llama-bench` is
  missing at `run`/`serve` time, version/flag validation is skipped with a
  warning.
- No GUI; no Electron; CLI-only.
- The POSIX process runner (`src/process/process_runner_posix.cpp`) builds
  and passes the full test suite on Ubuntu via GitHub Actions (see
  [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and the badge
  above) — it's no longer just syntax-checked. Not run on macOS; a macOS CI
  job will be added once that's been verified rather than assumed.

## Roadmap

```
0.1   hardware detection + launcher
0.2   benchmark parser + profiles
0.3   automatic tuning
0.3.1 real GGUF metadata discovery + reproducible benchmark comparison
0.4.0 static MoE expert-layer placement search (--n-cpu-moe as a tuned      <- current
      dimension), measured VRAM + silent-spill rejection, idle-based
      timeouts, `doctor`
0.5   informed layer placement from offline expert-activation profiling (Linux/eBPF, opt-in)
0.6   orchestration for a dynamic GPU expert cache, if/when one exists upstream
```

Everything through `0.4.0` exists in an initial form; `0.5` onward is not
implemented. `0.4`–`0.6` were reordered and re-scoped from the original
plan after dedicated research — see
[`docs/moe-vram-cache-research.md`](docs/moe-vram-cache-research.md) for
the full investigation, what's realistic without touching llama.cpp,
what would require forking it, and why. In short: real per-individual-
expert dynamic caching (the biggest win, ~10x in early third-party
reports) requires changes inside llama.cpp's backend scheduler that don't
exist upstream yet, so it's listed as orchestration-readiness, not
something FluxInfer will implement itself.
