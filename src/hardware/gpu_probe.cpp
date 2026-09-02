#include "fluxinfer/hardware/gpu_probe.hpp"

#include "nvml_shim.hpp"

#include <array>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fluxinfer::hardware {

namespace {

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle open_library() {
    // NVML ships with the NVIDIA driver, not the CUDA toolkit, so this is
    // present on any machine with a working NVIDIA GPU driver installed.
    return LoadLibraryA("nvml.dll");
}

void close_library(LibraryHandle handle) {
    if (handle) {
        FreeLibrary(handle);
    }
}

template <typename Fn>
Fn load_symbol(LibraryHandle handle, const char* name) {
    return reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(handle, name)));
}
#else
using LibraryHandle = void*;

LibraryHandle open_library() {
    LibraryHandle handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        handle = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    }
    return handle;
}

void close_library(LibraryHandle handle) {
    if (handle) {
        dlclose(handle);
    }
}

template <typename Fn>
Fn load_symbol(LibraryHandle handle, const char* name) {
    return reinterpret_cast<Fn>(dlsym(handle, name));
}
#endif

struct NvmlBindings {
    nvmlInit_v2_t init = nullptr;
    nvmlShutdown_t shutdown = nullptr;
    nvmlDeviceGetCount_v2_t get_count = nullptr;
    nvmlDeviceGetHandleByIndex_v2_t get_handle = nullptr;
    nvmlDeviceGetName_t get_name = nullptr;
    nvmlDeviceGetMemoryInfo_t get_memory_info = nullptr;
    // Optional: missing on very old drivers, in which case compute
    // capability is simply reported as unknown rather than failing detection.
    nvmlDeviceGetCudaComputeCapability_t get_compute_capability = nullptr;

    bool all_resolved() const {
        return init && shutdown && get_count && get_handle && get_name && get_memory_info;
    }
};

NvmlBindings resolve_bindings(LibraryHandle handle) {
    NvmlBindings bindings;
    bindings.init = load_symbol<nvmlInit_v2_t>(handle, "nvmlInit_v2");
    bindings.shutdown = load_symbol<nvmlShutdown_t>(handle, "nvmlShutdown");
    bindings.get_count = load_symbol<nvmlDeviceGetCount_v2_t>(handle, "nvmlDeviceGetCount_v2");
    bindings.get_handle = load_symbol<nvmlDeviceGetHandleByIndex_v2_t>(handle, "nvmlDeviceGetHandleByIndex_v2");
    bindings.get_name = load_symbol<nvmlDeviceGetName_t>(handle, "nvmlDeviceGetName");
    bindings.get_memory_info = load_symbol<nvmlDeviceGetMemoryInfo_t>(handle, "nvmlDeviceGetMemoryInfo");
    bindings.get_compute_capability =
        load_symbol<nvmlDeviceGetCudaComputeCapability_t>(handle, "nvmlDeviceGetCudaComputeCapability");
    return bindings;
}

GpuInfo describe_device(const NvmlBindings& nvml, nvmlDevice_t device, unsigned int index) {
    GpuInfo info;
    info.index = index;

    std::array<char, nvml_detail::kNvmlDeviceNameBufferSize> name_buffer{};
    if (nvml.get_name(device, name_buffer.data(), static_cast<unsigned int>(name_buffer.size())) == NVML_SUCCESS) {
        info.name = std::string(name_buffer.data());
    } else {
        info.name = "NVIDIA GPU (name unavailable)";
    }

    nvmlMemory_t memory{};
    if (nvml.get_memory_info(device, &memory) == NVML_SUCCESS) {
        info.total_vram_bytes = memory.total;
        info.available_vram_bytes = memory.free;
    }

    if (nvml.get_compute_capability != nullptr) {
        int major = 0;
        int minor = 0;
        if (nvml.get_compute_capability(device, &major, &minor) == NVML_SUCCESS && major > 0) {
            info.compute_capability_major = major;
            info.compute_capability_minor = minor;
        }
    }

    info.backend = "CUDA";
    info.available = true;
    return info;
}

// Shared implementation: enumerates every NVML device. On failure, returns an
// empty vector and writes the human-readable cause to *reason.
std::vector<GpuInfo> enumerate_devices(std::string* reason) {
    auto fail = [&](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return std::vector<GpuInfo>{};
    };

    LibraryHandle handle = open_library();
    if (!handle) {
        return fail("NVML shared library not found (no NVIDIA driver installed, or non-NVIDIA GPU)");
    }

    NvmlBindings nvml = resolve_bindings(handle);
    if (!nvml.all_resolved()) {
        close_library(handle);
        return fail("NVML library found but required symbols are missing (unexpected driver version)");
    }

    if (nvml.init() != NVML_SUCCESS) {
        close_library(handle);
        return fail("nvmlInit_v2 failed (driver present but NVML could not initialize)");
    }

    unsigned int device_count = 0;
    if (nvml.get_count(&device_count) != NVML_SUCCESS || device_count == 0) {
        nvml.shutdown();
        close_library(handle);
        return fail("no NVIDIA GPU detected by NVML");
    }

    std::vector<GpuInfo> devices;
    devices.reserve(device_count);
    for (unsigned int index = 0; index < device_count; ++index) {
        nvmlDevice_t device{};
        if (nvml.get_handle(index, &device) != NVML_SUCCESS) {
            continue; // skip a device we cannot open rather than failing the whole probe
        }
        devices.push_back(describe_device(nvml, device, index));
    }

    nvml.shutdown();
    close_library(handle);

    if (devices.empty()) {
        return fail("NVML could not open a handle to any reported device");
    }
    return devices;
}

} // namespace

std::vector<GpuInfo> probe_gpus() {
    std::string reason;
    return enumerate_devices(&reason);
}

GpuInfo probe_gpu() {
    std::string reason;
    std::vector<GpuInfo> devices = enumerate_devices(&reason);
    if (devices.empty()) {
        GpuInfo info;
        info.backend = "none";
        info.unavailable_reason = reason;
        return info;
    }
    return devices.front();
}

} // namespace fluxinfer::hardware
