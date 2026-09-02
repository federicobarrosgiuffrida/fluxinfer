#include "fluxinfer/hardware/hardware_info.hpp"

#include "fluxinfer/hardware/cpu_probe.hpp"
#include "fluxinfer/hardware/gpu_probe.hpp"
#include "fluxinfer/hardware/memory_probe.hpp"

namespace fluxinfer::hardware {

HardwareInfo probe_hardware() {
    HardwareInfo info;
    info.cpu = probe_cpu();
    info.memory = probe_memory();
    // One NVML pass for both fields: probe_gpus() enumerates every device,
    // and the primary one keeps the exact meaning it had before.
    info.gpus = probe_gpus();
    info.gpu = info.gpus.empty() ? probe_gpu() : info.gpus.front();
    return info;
}

} // namespace fluxinfer::hardware
