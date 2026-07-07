#pragma once

#include <cstdint>
#include <string>

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
};

struct HardwareInfo {
    CpuInfo cpu;
    MemoryInfo memory;
    GpuInfo gpu;
};

// Probes CPU, RAM and (optionally) GPU. Never throws: any probe that fails
// leaves its section at default values with a human-readable reason attached
// where applicable (see GpuInfo::unavailable_reason).
HardwareInfo probe_hardware();

} // namespace fluxinfer::hardware
