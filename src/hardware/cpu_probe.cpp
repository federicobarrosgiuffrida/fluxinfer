#include "fluxinfer/hardware/cpu_probe.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

namespace fluxinfer::hardware {

namespace {

#if defined(_WIN32)

std::string brand_string_from_cpuid() {
    std::array<int, 4> regs{};
    __cpuid(regs.data(), 0x80000000);
    unsigned max_ext = static_cast<unsigned>(regs[0]);
    if (max_ext < 0x80000004) {
        return {};
    }

    char brand[49] = {};
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(regs.data(), static_cast<int>(0x80000002 + i));
        std::memcpy(brand + i * 16, regs.data(), sizeof(regs));
    }
    std::string name(brand);
    // Trim leading/trailing whitespace some vendors pad the string with.
    const auto first = name.find_first_not_of(' ');
    const auto last = name.find_last_not_of(' ');
    if (first == std::string::npos) {
        return {};
    }
    return name.substr(first, last - first + 1);
}

unsigned physical_core_count() {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) {
        return 0;
    }

    std::vector<char> buffer(length);
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
        return 0;
    }

    unsigned cores = 0;
    DWORD offset = 0;
    while (offset < length) {
        auto* entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (entry->Relationship == RelationProcessorCore) {
            ++cores;
        }
        offset += entry->Size;
    }
    return cores;
}

#else // POSIX

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return {};
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string brand_string_from_proc_cpuinfo() {
    std::ifstream file("/proc/cpuinfo");
    if (!file) {
        return {};
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string value = line.substr(colon + 1);
                const auto first = value.find_first_not_of(' ');
                if (first != std::string::npos) {
                    return value.substr(first);
                }
            }
        }
    }
    return {};
}

unsigned physical_core_count_linux() {
    std::ifstream file("/proc/cpuinfo");
    if (!file) {
        return 0;
    }
    // Count unique (physical id, core id) pairs. Falls back to counting
    // "processor" entries (i.e. logical threads) if the fields are absent,
    // e.g. inside some containers/VMs.
    std::string line;
    int current_physical_id = 0;
    int current_core_id = -1;
    std::vector<std::pair<int, int>> pairs;
    bool saw_core_id = false;

    while (std::getline(file, line)) {
        if (line.rfind("physical id", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                current_physical_id = std::atoi(line.c_str() + colon + 1);
            }
        } else if (line.rfind("core id", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                current_core_id = std::atoi(line.c_str() + colon + 1);
                saw_core_id = true;
            }
        } else if (line.empty() && current_core_id != -1) {
            pairs.emplace_back(current_physical_id, current_core_id);
            current_core_id = -1;
        }
    }

    if (!saw_core_id) {
        return 0;
    }

    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    return static_cast<unsigned>(pairs.size());
}

#endif

} // namespace

CpuInfo probe_cpu() {
    CpuInfo info;

#if defined(_WIN32)
    std::string brand = brand_string_from_cpuid();
    info.name = brand.empty() ? "unknown" : brand;

    unsigned physical = physical_core_count();
    info.physical_cores = physical > 0 ? physical
                                        : std::thread::hardware_concurrency();

    SYSTEM_INFO sys_info{};
    GetSystemInfo(&sys_info);
    info.logical_threads = sys_info.dwNumberOfProcessors > 0
                                ? sys_info.dwNumberOfProcessors
                                : std::thread::hardware_concurrency();
#else
    std::string brand = brand_string_from_proc_cpuinfo();
    info.name = brand.empty() ? "unknown" : brand;

    unsigned physical = physical_core_count_linux();
    info.physical_cores = physical > 0 ? physical
                                        : std::thread::hardware_concurrency();

    long logical = sysconf(_SC_NPROCESSORS_ONLN);
    info.logical_threads = logical > 0 ? static_cast<unsigned>(logical)
                                        : std::thread::hardware_concurrency();
#endif

    if (info.physical_cores == 0) {
        info.physical_cores = 1;
    }
    if (info.logical_threads == 0) {
        info.logical_threads = 1;
    }

    return info;
}

} // namespace fluxinfer::hardware
