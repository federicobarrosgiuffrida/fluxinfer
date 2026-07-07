#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"

namespace fluxinfer::hardware {

// Detects total and currently available system RAM.
MemoryInfo probe_memory();

} // namespace fluxinfer::hardware
