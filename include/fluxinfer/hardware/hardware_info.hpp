#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fluxinfer::hardware {

struct CpuInfo {
    std::string name = "unknown";
    unsigned physical_cores = 0;
    unsigned logical_threads = 0;
};

struct MemoryInfo {
    std::uint64_t total_bytes = 0;
    std::uint64_t available_bytes = 0;
};

struct GpuInfo {
    bool available = false;
    std::string name;
    std::uint64_t total_vram_bytes = 0;
    std::uint64_t available_vram_bytes = 0;
    std::string backend;              // e.g. "CUDA"
    std::string unavailable_reason;   // populated when available == false

    // NVML device index, as used by CUDA_VISIBLE_DEVICES and llama.cpp's own
    // device ordering. 0 for the single-GPU case.
    unsigned index = 0;

    // CUDA compute capability, e.g. 8.6 for an RTX 3060. Both 0 when the
    // driver did not report it; FluxInfer treats "unknown" as "make no
    // capability-dependent decision" rather than assuming a floor.
    int compute_capability_major = 0;
    int compute_capability_minor = 0;

    bool has_compute_capability() const { return compute_capability_major > 0; }
};

struct HardwareInfo {
    CpuInfo cpu;
    MemoryInfo memory;

    // Primary device (NVML index 0). Every existing tuning decision is made
    // against this one: FluxInfer benchmarks and serves on a single GPU, and
    // does not tune multi-GPU splits.
    GpuInfo gpu;

    // Every NVIDIA device NVML reported, in index order, including the
    // primary one. Empty when no GPU is available. Detection only: knowing a
    // second card exists explains VRAM figures that would otherwise look
    // wrong, and is reported by `inspect`/`doctor`, but no split is tuned.
    std::vector<GpuInfo> gpus;
};

// Probes CPU, RAM and (optionally) GPU. Never throws: any probe that fails
// leaves its section at default values with a human-readable reason attached
// where applicable (see GpuInfo::unavailable_reason).
HardwareInfo probe_hardware();

} // namespace fluxinfer::hardware
