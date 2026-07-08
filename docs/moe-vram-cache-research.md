# MoE VRAM/RAM placement for FluxInfer — research and design

**Status:** research + design, not yet implemented. **Scope:** how FluxInfer
could make very large MoE models (e.g. Qwen3.6-35B-A3B) more usable on
consumer 8GB-class GPUs by tuning where expert weights live, without
changing FluxInfer's identity as a wrapper/benchmarker/profile manager
around llama.cpp.

## 0. Where this fits in FluxInfer today

FluxInfer already has every piece this feature needs *except* the
MoE-specific parameter itself:

- [`gguf_metadata.hpp`](../include/fluxinfer/llama/gguf_metadata.hpp) /
  [`gguf_metadata.cpp`](../src/llama/gguf_metadata.cpp) already read
  `<architecture>.expert_count` and `<architecture>.expert_used_count` from
  the model's own GGUF metadata — MoE detection is a solved problem here,
  it's just unused for search decisions yet (`tune` only prints a note:
  *"MoE model: this MVP does not do MoE-aware tuning yet"*, see
  [`main.cpp`](../src/main.cpp)).
- [`parameter_space.hpp`](../include/fluxinfer/tuner/parameter_space.hpp) /
  `.cpp` already implement the exact staged-search pattern this feature
  needs: generate candidates, run each through `llama-bench`, let real
  OOM/timeout detection (not estimates) decide what's usable, refine near a
  failure boundary via binary search. `gpu_layers_candidates()` is
  structurally the template to copy.
- [`tuner.cpp`](../src/tuner/tuner.cpp)'s `Tuner::run()` staged loop
  (baseline → GPU layers → batch/ubatch → threads) is exactly where a new
  stage would slot in, and `build_arguments()` is exactly where a new flag
  gets added to the constructed `llama-bench` argv.
- [`profile.hpp`](../include/fluxinfer/profiles/profile.hpp)'s `BestConfig`
  is exactly where a new tuned value gets persisted and later replayed by
  `run`/`serve` (see how `context_length` was added there and in
  `main.cpp`'s `build_config_arguments()` — the MoE dimension follows the
  identical pattern).
- [`comparison.hpp`](../include/fluxinfer/tuner/comparison.hpp)'s repeated-
  measurement A/B report is exactly the tool needed to *honestly validate*
  any claimed improvement from smarter expert placement, rather than
  asserting it — this project's own history (see
  [`docs/benchmarks/`](benchmarks/)) is a live example of why that
  validation step matters: an earlier claimed improvement turned out to be
  partly a self-inflicted scoring bug, and a real-world regression report
  from different hardware caught a second real issue. The same discipline
  applies here.

Nothing about this feature requires a new module family. It's a new
parameter-space dimension plus, optionally, a new *profiling* capability
that produces data for that dimension to consume.

## 1. What the best current approaches are

Research (llama.cpp's own issue tracker, three peer-reviewed/preprint
papers, and llama.cpp's current shipped flags) converges on a clear
hierarchy of approaches, from weakest to strongest:

| Approach | Granularity | Adaptivity | Requires engine changes? |
| --- | --- | --- | --- |
| `--n-cpu-moe N` (current llama.cpp) | whole layer, top-N contiguous by index | static, chosen at load time | no — already shipped |
| `--override-tensor` regex placement (current llama.cpp) | whole layer, arbitrary selection | static, chosen at load time | no — already shipped |
| Offline-profiled informed static placement (Fiddler, ICLR 2025) | whole layer (see §0 tensor-layout note below) | static, but *informed* by measured expert popularity | no, if built as an external profiler |
| Dynamic runtime expert cache with eviction policy (llama.cpp issue [#20757](https://github.com/ggml-org/llama.cpp/issues/20757); similar to a Dec-2025 CPU-GPU collaborative-inference paper, [arXiv:2512.16473](https://arxiv.org/abs/2512.16473)) | individual expert, per forward pass | fully dynamic, LRU/SLRU eviction | **yes** — modifies the backend scheduler |

The dynamic-cache approach is the clear performance ceiling. The llama.cpp
issue's own proof-of-concept reports **12–14 tok/s at steady state with a
~98–100% cache hit rate, versus 0.5–1 tok/s without caching** — roughly an
order of magnitude — by observing that **15–20% of experts handle 80% of
token traffic** (real, measured routing locality, not a hypothesis). The
Dec-2025 paper reports 4.4×/4.3× speedups over prefetching baselines with
a similar LRU-based design, and explicitly credits *dynamic, runtime*
observation over offline profiling: "our approach dynamically observes
expert reuse patterns and adapts accordingly," citing 44% cross-layer and
40–60% cross-token expert reuse.

**Critical technical fact that shapes everything below:** GGUF stores all
of a layer's routed experts as **one fused tensor** per role (e.g.
`blk.5.ffn_gate_exps.weight`, shape `(n_expert, n_ff, n_embd)`), not as
separate per-expert tensors. `ggml_compute_forward_mul_mat_id` does the
per-token expert *selection* at compute time by indexing into that single
tensor. This means `--override-tensor`/`--n-cpu-moe` can only choose
**which whole layers'** expert blocks go to CPU vs GPU — never individual
experts *within* a layer. Only the dynamic-cache approach, which remaps
global expert IDs to a smaller set of GPU-resident slots inside the
scheduler itself, can achieve individual-expert residency. This is not a
missing feature FluxInfer could work around from outside; it's a
consequence of the on-disk/in-memory tensor layout.

## 2. What's realistic for FluxInfer without modifying llama.cpp

Two things, both bounded by the "whole layer" granularity limit above:

**A. Search `--n-cpu-moe` (and/or a curated `--override-tensor` layer set)
the same way FluxInfer already searches `--n-gpu-layers`.** This is a pure
parameter-space extension: reuse the existing candidate-generation +
OOM-detection + binary-search-refinement machinery verbatim, on a new
axis. Zero new subsystems, zero new dependencies, works identically on
Windows and Linux. This is realistic *today*.

**B. Optionally *inform* which layers to prioritize for GPU residency
using real, measured per-layer routing skew**, rather than the naive
"top-N contiguous layers by index" that `--n-cpu-moe` does. Fiddler
(ICLR 2025) validates this general idea — offline-profiled expert
popularity → static placement — and reports running the 90GB+ uncompressed
Mixtral-8x7B at >3 tok/s on a single 24GB GPU, "an order of magnitude
improvement over existing methods" at the time. The open question is
*how* FluxInfer observes routing behavior without touching llama.cpp's
source. The only non-invasive technique found in current research is an
**eBPF uprobe attached to `ggml_compute_forward_mul_mat_id`** (used by a
recent inference profiler, [arXiv:2601.20755](https://arxiv.org/abs/2601.20755)),
which reads the selected expert IDs directly out of the running process's
memory at that function's entry, with no recompilation or source patch
needed. Constraints this brings:

- **eBPF is a Linux kernel feature.** No Windows or macOS equivalent found
  with comparable maturity for this exact use case (uprobing a specific
  user-space function and reading tensor contents). This capability would
  be Linux-only, full stop — a real platform split for a project that has
  so far been developed and validated primarily on Windows.
- It adds measurable overhead per the source paper, described as
  acceptable for **offline analysis, not production serving** — which
  maps cleanly onto FluxInfer's existing `tune` phase (an explicit, one-time,
  offline step) and is inappropriate for `run`/`serve`.
- It only gets FluxInfer *layer-level* routing-skew data (per the tensor-
  layout fact above), which then informs a *static* `--override-tensor`
  choice — still no runtime adaptivity.

Both A and B keep FluxInfer's identity fully intact: it profiles, decides,
and orchestrates; llama.cpp still does 100% of the actual inference.

## 3. What requires modifying or forking llama.cpp

Genuine per-individual-expert dynamic residency with an eviction policy —
the approach with the ~10× reported ceiling — requires changes inside
`ggml_backend_sched_compute_splits()` (per the llama.cpp issue's own
technical proposal): maintaining a persistent expert-ID→GPU-slot mapping
across forward passes, remapping IDs for the `MUL_MAT_ID` kernel, evicting
slots on cache misses, and using pinned host memory for async transfers.
This is backend-scheduler surgery, not something reachable from outside
the process. **As of this research, the issue is open with no
implementation** — the requester explicitly says they need a C++
contributor familiar with the scheduler internals. There is no existing
patch or fork to integrate against today.

Two other alternatives were considered and rejected for the same
"requires touching the engine" reason, or worse:

- **A separate process proxying/intercepting llama-server and managing its
  own VRAM cache.** Doesn't work: the expert weight tensors live inside
  llama.cpp's own process address space and CUDA context. Reaching into
  that from outside would need CUDA IPC or shared-memory tricks
  considerably more fragile than just patching ggml, and still couldn't
  change *which* weights are resident without reloading the model, which
  defeats the purpose.
- **FluxInfer implementing its own inference/caching engine.** Explicitly
  out of scope per this project's own constraints (no new inference
  engine, don't duplicate llama.cpp), and would duplicate ggml's already-
  mature CPU/CUDA kernels for no benefit.

## 4 & 5. The chosen architecture, and why

**FluxInfer extends its existing staged tuner with a MoE-aware layer-
placement search, delivered in two concrete phases it can build now
(cross-platform static search, then optional Linux-only informed
profiling), plus explicit, non-faked orchestration-readiness for a
dynamic backend if/when one exists upstream.** No new product, no new
identity — the *exact same* "generate candidates → benchmark with
llama-bench → let real OOM detection decide → refine near the boundary →
persist to a profile → replay via run/serve" loop this project already
validated for `--n-gpu-layers`, `--batch-size`/`--ubatch-size`, and (most
recently) `--ctx-size`.

This beats the alternatives considered:

- Over "do nothing / recommend `--cpu-moe` blindly": naive `--n-cpu-moe`
  picks contiguous layers from the top by index, with no regard to which
  layers are actually worth the VRAM. FluxInfer already has the
  infrastructure to *measure* this instead of guessing — it would be
  strange not to use it, and it's exactly the kind of grunt work a
  benchmarking wrapper should be doing.
- Over "implement dynamic caching inside FluxInfer": technically
  impossible without forking llama.cpp (§3), and even if it weren't, it
  would mean FluxInfer stops being a wrapper and starts being an inference
  engine — a explicit non-goal.
- Over "only do the Linux/eBPF-informed version": would abandon Windows
  users (this project's primary validated platform so far) entirely for
  the MoE case. The static search-based MVP works everywhere the rest of
  FluxInfer already works; the informed profiling is worth having but must
  be additive, not a hard requirement.
- Over "promise the 10× number from the dynamic-cache research": that
  number describes a different, not-yet-existing implementation. Citing
  it as what FluxInfer delivers would repeat the exact mistake already
  made and corrected once this session (an unvalidated performance claim)
  — see §6 for what FluxInfer can *honestly* expect.

## 6. Expected performance benefits and failure modes

**MVP (static `--n-cpu-moe` / curated `--override-tensor` search):**
expected gains are in the same rough range already observed for
`--n-gpu-layers` tuning on dense models in this project's own real
benchmarks (`docs/benchmarks/`) — meaningful (double-digit percent), not
transformative, because it's still choosing *which* layers, not changing
what's fundamentally possible on an 8GB card. Failure modes: same as the
existing gpu-layers search already handles — OOM, and on Windows
specifically the already-documented WDDM behavior of timing out instead
of failing cleanly on over-VRAM allocation (see `README.md`'s "Current
limits"). No new failure mode class, just a new dimension exposed to the
same detection.

**v2 (eBPF-informed layer selection, Linux only):** benefit is
*conditional on real per-layer routing skew existing* for the specific
model being tuned — an empirical question, not a given. MoE pretraining
commonly uses load-balancing auxiliary losses specifically to keep expert
usage roughly even; if a model's routing is close to uniform across
layers, profiled placement could tie with or barely beat the naive search,
in which case the extra complexity buys little. This must be verified per
model using FluxInfer's own `--compare-repeats` A/B mechanism, not
assumed. Failure mode: profiling on an unrepresentative prompt set
produces a placement that's tuned for the profiling sample and worse for
the user's real workload — this needs to be surfaced clearly in the CLI
output (e.g. "profiled on N prompts of type X; results may not generalize"),
not hidden.

**v3 (dynamic cache, requires upstream/fork):** not something FluxInfer
can deliver or measure today. Listed for roadmap completeness only; see
§9.

## 7. Implementation complexity

- MVP: **low.** New `MoeConfig`-style fields on `TuneConfig`/`BestConfig`
  (mirroring how `context_length` was just added), a new
  `moe_layer_candidates()` in `parameter_space.cpp` mirroring
  `gpu_layers_candidates()`, a new stage in `Tuner::run()`, new argv
  wiring in `build_arguments()`/`build_config_arguments()`, gated entirely
  on `expert_count > 0` from GGUF metadata (already read). No new
  dependencies. Directly testable with the same synthetic-fixture pattern
  already used throughout `tests/`.
- v2: **medium–high.** A genuinely new subsystem: an eBPF/libbpf (or bcc)
  dependency, a way to attach to the spawned `llama-bench`/`llama-server`
  process (needs `CAP_BPF`/root on most distros — a real deployment
  friction point), a symbol-resolution step to find
  `ggml_compute_forward_mul_mat_id` in the target binary, aggregation
  logic for the activation histogram, and a decision function turning that
  histogram into an `--override-tensor` argument list. Linux-only by
  construction; needs its own dedicated CI job and cannot reuse the
  Windows-tested code paths.
- v3: **N/A for FluxInfer itself** — orchestration-only, and only once a
  concrete backend exists to orchestrate.

## 8. Risks specific to 8GB VRAM GPUs

- The attention tensors, any dense/shared-expert FFN, and the KV cache
  (now correctly sized via `--context`, see the recent context-size fix)
  are effectively mandatory GPU residents for acceptable speed — they're
  used on every token. For a model in the Qwen3.6-35B-A3B class, that
  "always resident" set plus even a modest number of GPU-resident expert
  layers can consume most or all of 8GB before the KV cache is accounted
  for. Realistic expectation-setting matters here: for a genuinely
  35B-class MoE model, an 8GB card will likely still need to keep *most*
  routed-expert layers on CPU RAM no matter how well they're chosen —
  FluxInfer's job is to find the best outcome *within* that ceiling, not
  to remove the ceiling.
- The same Windows/WDDM timeout-vs-clean-OOM ambiguity already found and
  handled for `--n-gpu-layers` (a real, previously-observed behavior, not
  theoretical) applies identically to MoE layer placement search — an
  over-budget layer selection is more likely to hang/time out than to
  fail cleanly on Windows. The existing "treat timeout as inconclusive,
  narrow the search without promoting the candidate" logic in
  `Tuner::run()` already generalizes to this without modification.
- CPU RAM budget is a second, easy-to-overlook constraint: a 35B-class
  MoE model with most experts on CPU needs that capacity in system RAM
  too (on top of whatever's already resident for the OS and other apps).
  FluxInfer already probes total/available RAM (`hardware/memory_probe`)
  but doesn't yet fail fast if a MoE model's CPU-resident portion won't
  fit in available RAM — worth adding as an explicit pre-flight check
  before spending a full search cycle on a configuration that can't work.

## 9. Phased roadmap

**MVP — static MoE-aware search (cross-platform, no new dependencies)**
- Detect MoE models via existing GGUF `expert_count`/`expert_used_count`.
- Add `--n-cpu-moe` as a new staged-search dimension (percentage-based
  candidates + OOM-boundary refinement, mirroring `gpu_layers_candidates`).
- Persist the chosen value in the profile; replay it in `run`/`serve`.
- Report it in the existing `--compare-repeats` comparison report so any
  claimed improvement is measured, not asserted.

**v2 — informed layer selection (Linux-only, opt-in)**
- New optional `--profile-experts <prompt-file>` flag on `tune`.
- eBPF/uprobe-based per-layer expert-activation histogram collection
  during a dedicated profiling pass, clearly separated from the timed
  benchmark passes.
- Turn the histogram into a routing-skew score per layer (e.g. normalized
  entropy of the activation distribution); use it to pick a
  non-contiguous, skew-ranked `--override-tensor` layer set instead of
  naive top-N.
- A/B this against the MVP's naive `--n-cpu-moe` choice using the existing
  comparison-report mechanism before ever recommending it as the default.
- Requires a dedicated Linux CI job; must degrade gracefully (fall back to
  MVP behavior) on Windows/macOS or when eBPF attachment isn't permitted.

**v3 — dynamic-cache orchestration (blocked on upstream/fork)**
- Not implementable today. If/when llama.cpp (or a maintained fork) ships
  something matching issue #20757's proposal, FluxInfer's job is exactly
  what it already does for every other flag: detect the new CLI surface,
  add it as tunable dimensions (e.g. cache size, eviction policy choice),
  benchmark and persist the result. No inference logic of FluxInfer's own.
  This phase should not begin until a concrete backend exists to target —
  building against a moving/nonexistent target risks the same
  "don't fake it" problem this research was explicitly asked to avoid.

## 10. Metrics FluxInfer must collect

Already collected today (reusable as-is): prompt/generation tok/s
(mean/median/min/max/stddev via `comparison.hpp`), peak VRAM via
`VramSampler`, GGUF `expert_count`/`expert_used_count`/`block_count`.

**New for the MVP:** the `n_cpu_moe` value tested per candidate (parallel
to `gpu_layers` in `BenchmarkResult`/report tables), and peak-VRAM
sampling applied to the MoE search stage the same way it's already applied
to the comparison phase.

**New for v2:** a per-layer expert-activation histogram (counts per
expert ID per layer, aggregated over the profiling prompt set), a derived
per-layer routing-skew score, and — critically, matching this project's
existing profile-invalidation philosophy — a hash/fingerprint of the
profiling prompt set saved into the profile, so a profile silently reused
against a very different workload can be flagged as potentially stale the
same way a changed model file or GPU already invalidates a profile today.

**New for v3 readiness:** cache hit-rate and eviction-count metrics,
exact shape TBD until a concrete backend exists to report them.

## Sources

- [llama.cpp issue #20757 — Two-tier GPU+RAM expert cache](https://github.com/ggml-org/llama.cpp/issues/20757)
- [llama.cpp discussion #18049 — GPU/tensor-split/override-tensor automation](https://github.com/ggml-org/llama.cpp/discussions/18049)
- [Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems, arXiv:2512.16473](https://arxiv.org/abs/2512.16473)
- [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models, ICLR 2025, arXiv:2402.07033](https://arxiv.org/abs/2402.07033)
- [ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler, arXiv:2601.20755](https://arxiv.org/abs/2601.20755)
- [Performant local MoE CPU inference with GPU acceleration in llama.cpp (community guide)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [Guide to optimizing MoE inference across CPU+GPU with llama.cpp (gist)](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)
