#include "fluxinfer/hardware/hardware_info.hpp"

#include "fluxinfer/hardware/cpu_probe.hpp"
#include "fluxinfer/hardware/gpu_probe.hpp"
#include "fluxinfer/hardware/memory_probe.hpp"

namespace fluxinfer::hardware {

HardwareInfo probe_hardware() {
    HardwareInfo info;
    info.cpu = probe_cpu();
    info.memory = probe_memory();
    info.gpu = probe_gpu();
    return info;
}

} // namespace fluxinfer::hardware
