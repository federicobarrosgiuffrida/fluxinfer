#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"

namespace fluxinfer::hardware {

// Detects CPU brand string, physical core count and logical thread count.
// Falls back to std::thread::hardware_concurrency() for the logical thread
// count if platform-specific enumeration fails.
CpuInfo probe_cpu();

} // namespace fluxinfer::hardware
