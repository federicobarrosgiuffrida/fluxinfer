#include "fluxinfer/llama/llama_locator.hpp"

#include <cstdlib>
#include <system_error>

namespace fluxinfer::llama {

namespace {

#if defined(_WIN32)
constexpr char kPathSeparator = ';';
constexpr const char* kExeSuffix = ".exe";
#else
constexpr char kPathSeparator = ':';
constexpr const char* kExeSuffix = "";
#endif

std::vector<std::filesystem::path> split_path_env() {
    std::vector<std::filesystem::path> dirs;
    const char* path_env = std::getenv("PATH");
    if (!path_env) {
        return dirs;
    }
    std::string path(path_env);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t sep = path.find(kPathSeparator, start);
        std::string entry = path.substr(start, sep == std::string::npos ? std::string::npos : sep - start);
        if (!entry.empty()) {
            dirs.emplace_back(entry);
        }
        if (sep == std::string::npos) {
            break;
        }
        start = sep + 1;
    }
    return dirs;
}

} // namespace

LlamaLocator::LlamaLocator(std::optional<std::filesystem::path> explicit_dir) : explicit_dir_(std::move(explicit_dir)) {}

std::vector<std::filesystem::path> LlamaLocator::search_directories() const {
    std::vector<std::filesystem::path> dirs;
    if (explicit_dir_) {
        dirs.push_back(*explicit_dir_);
    }

    for (auto& dir : split_path_env()) {
        dirs.push_back(dir);
    }

    std::error_code ec;
    std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        dirs.push_back(cwd / "external" / "llama.cpp" / "build" / "bin");
        dirs.push_back(cwd / "external" / "llama.cpp" / "build");
        dirs.push_back(cwd / "llama.cpp" / "build" / "bin");
        dirs.push_back(cwd / "llama.cpp" / "build");
        dirs.push_back(cwd / "build" / "bin");
        dirs.push_back(cwd);
    }

    return dirs;
}

std::optional<std::filesystem::path> LlamaLocator::find_binary(const std::string& base_name) const {
    const std::string filename = base_name + kExeSuffix;
    for (const auto& dir : search_directories()) {
        std::error_code ec;
        std::filesystem::path candidate = dir / filename;
        if (std::filesystem::is_regular_file(candidate, ec)) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }
    return std::nullopt;
}

LlamaBinaries LlamaLocator::locate() const {
    LlamaBinaries binaries;
    binaries.llama_bench = find_binary("llama-bench");
    binaries.llama_cli = find_binary("llama-cli");
    binaries.llama_server = find_binary("llama-server");
    return binaries;
}

} // namespace fluxinfer::llama
