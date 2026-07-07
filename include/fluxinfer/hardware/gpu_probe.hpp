#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"

namespace fluxinfer::hardware {

// Detects an NVIDIA GPU via NVML, loaded dynamically at runtime (no NVML SDK
// / CUDA toolkit required to build FluxInfer). If the NVML shared library is
// not present, no NVIDIA driver is installed, or no device is found, returns
// GpuInfo with available == false and unavailable_reason set to a
// human-readable explanation. Never throws.
GpuInfo probe_gpu();

} // namespace fluxinfer::hardware
