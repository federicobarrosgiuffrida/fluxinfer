#include "fluxinfer/hardware/gpu_probe.hpp"

#include "nvml_shim.hpp"

#include <array>
#include <string>

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
    return bindings;
}

} // namespace

GpuInfo probe_gpu() {
    GpuInfo info;
    info.backend = "none";

    LibraryHandle handle = open_library();
    if (!handle) {
        info.unavailable_reason = "NVML shared library not found (no NVIDIA driver installed, or non-NVIDIA GPU)";
        return info;
    }

    NvmlBindings nvml = resolve_bindings(handle);
    if (!nvml.all_resolved()) {
        info.unavailable_reason = "NVML library found but required symbols are missing (unexpected driver version)";
        close_library(handle);
        return info;
    }

    if (nvml.init() != NVML_SUCCESS) {
        info.unavailable_reason = "nvmlInit_v2 failed (driver present but NVML could not initialize)";
        close_library(handle);
        return info;
    }

    unsigned int device_count = 0;
    if (nvml.get_count(&device_count) != NVML_SUCCESS || device_count == 0) {
        info.unavailable_reason = "no NVIDIA GPU detected by NVML";
        nvml.shutdown();
        close_library(handle);
        return info;
    }

    nvmlDevice_t device{};
    if (nvml.get_handle(0, &device) != NVML_SUCCESS) {
        info.unavailable_reason = "NVML could not open a handle to device 0";
        nvml.shutdown();
        close_library(handle);
        return info;
    }

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

    info.backend = "CUDA";
    info.available = true;

    nvml.shutdown();
    close_library(handle);
    return info;
}

} // namespace fluxinfer::hardware
