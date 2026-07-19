#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"

#include <vector>

namespace fluxinfer::hardware {

// Detects an NVIDIA GPU via NVML, loaded dynamically at runtime (no NVML SDK
// / CUDA toolkit required to build FluxInfer). If the NVML shared library is
// not present, no NVIDIA driver is installed, or no device is found, returns
// GpuInfo with available == false and unavailable_reason set to a
// human-readable explanation. Never throws.
GpuInfo probe_gpu();

// Same detection as probe_gpu(), but returns every NVIDIA device NVML
// reports, in index order. Returns an empty vector when no usable device was
// found (the reason is then available via probe_gpu().unavailable_reason).
// Never throws.
std::vector<GpuInfo> probe_gpus();

} // namespace fluxinfer::hardware
