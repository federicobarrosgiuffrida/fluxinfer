#include "fluxinfer/hardware/memory_probe.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#endif

namespace fluxinfer::hardware {

#if defined(_WIN32)

MemoryInfo probe_memory() {
    MemoryInfo info;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        info.total_bytes = status.ullTotalPhys;
        info.available_bytes = status.ullAvailPhys;
    }
    return info;
}

#else

namespace {

// Parses a line of the form "MemTotal:       32797236 kB" and returns the
// value in bytes, or 0 if the line doesn't match.
std::uint64_t parse_meminfo_line_kb(const std::string& line) {
    std::istringstream iss(line);
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    iss >> key >> value >> unit;
    return value * 1024ULL;
}

} // namespace

MemoryInfo probe_memory() {
    MemoryInfo info;

    std::ifstream file("/proc/meminfo");
    if (file) {
        std::string line;
        std::uint64_t mem_total = 0;
        std::uint64_t mem_available = 0;
        bool has_available = false;
        while (std::getline(file, line)) {
            if (line.rfind("MemTotal:", 0) == 0) {
                mem_total = parse_meminfo_line_kb(line);
            } else if (line.rfind("MemAvailable:", 0) == 0) {
                mem_available = parse_meminfo_line_kb(line);
                has_available = true;
            }
        }
        info.total_bytes = mem_total;
        info.available_bytes = has_available ? mem_available : mem_total;
    }

    if (info.total_bytes == 0) {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            info.total_bytes = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
        }
        long avail_pages = sysconf(_SC_AVPHYS_PAGES);
        if (avail_pages > 0 && page_size > 0) {
            info.available_bytes = static_cast<std::uint64_t>(avail_pages) * static_cast<std::uint64_t>(page_size);
        } else {
            info.available_bytes = info.total_bytes;
        }
    }

    return info;
}

#endif

} // namespace fluxinfer::hardware
