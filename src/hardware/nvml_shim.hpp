#pragma once

// Minimal, self-contained subset of the NVML C API used by gpu_probe.cpp.
//
// FluxInfer does not link against the real nvml.h / nvml.lib from the CUDA
// toolkit: doing so would make the CUDA toolkit a hard build dependency just
// to detect whether an NVIDIA GPU is present. Instead we declare the tiny
// slice of the ABI we need here and load nvml.dll / libnvidia-ml.so.1 at
// runtime with LoadLibrary/dlopen, resolving each symbol individually. If
// the library is missing (no NVIDIA driver installed) the loader simply
// reports the GPU as unavailable instead of failing to build or crashing.
//
// The struct layouts and function signatures below match NVML's stable
// public ABI and are safe to redeclare like this.

#include <cstddef>
#include <cstdint>

extern "C" {

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
} nvmlReturn_t;

typedef struct nvmlMemory_st {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

using nvmlInit_v2_t = nvmlReturn_t (*)();
using nvmlShutdown_t = nvmlReturn_t (*)();
using nvmlDeviceGetCount_v2_t = nvmlReturn_t (*)(unsigned int*);
using nvmlDeviceGetHandleByIndex_v2_t = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using nvmlDeviceGetName_t = nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int);
using nvmlDeviceGetMemoryInfo_t = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);

} // extern "C"

namespace fluxinfer::hardware::nvml_detail {

constexpr unsigned int kNvmlDeviceNameBufferSize = 96;

} // namespace fluxinfer::hardware::nvml_detail
