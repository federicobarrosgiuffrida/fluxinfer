#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fluxinfer::profiles {

struct ModelInfo {
    std::string path;
    std::uint64_t size_bytes = 0;
    std::string fingerprint;
};

struct HardwareSnapshot {
    std::string cpu;
    unsigned logical_threads = 0;
    std::uint64_t ram_bytes = 0;
    std::string gpu;      // empty string means "no GPU"
    std::uint64_t vram_bytes = 0;
};

struct LlamaSnapshot {
    std::string version;
    std::string binary_path;
    std::vector<std::string> supported_flags;
};

struct BestConfig {
    unsigned threads = 0;
    int gpu_layers = 0;
    unsigned batch_size = 0;
    unsigned ubatch_size = 0;
    std::optional<std::string> kv_cache_type;
};

struct ProfileResults {
    double prompt_tps = 0.0;
    double generation_tps = 0.0;
    std::int64_t duration_ms = 0;
    double score = 0.0;
};

struct Profile {
    int schema_version = 1;
    ModelInfo model;
    HardwareSnapshot hardware;
    LlamaSnapshot llama;
    BestConfig best_config;
    ProfileResults results;
};

nlohmann::json to_json(const Profile& profile);

// Returns std::nullopt (with *error set) if `json` is not a well-formed
// profile document (wrong/missing schema_version, wrong field types, ...).
std::optional<Profile> profile_from_json(const nlohmann::json& json, std::string* error = nullptr);

// Computes a fingerprint for `model_path` from file size, last-write-time,
// and a partial hash of the first and last 1 MiB of the file. GGUF models
// are commonly tens of gigabytes; hashing the whole file on every `tune`/
// `run` invocation would cost minutes of disk I/O for no real benefit here,
// so this trades exhaustiveness for near-instant fingerprinting. It will
// not detect a file deliberately crafted to keep the same size, mtime, and
// prefix/suffix while changing interior bytes -- an irrelevant threat model
// for a local benchmarking tool, but worth knowing if this code is reused
// elsewhere.
std::optional<ModelInfo> compute_model_info(const std::filesystem::path& model_path, std::string* error = nullptr);

// True if `profile` still applies to the given current model/hardware/
// llama.cpp snapshot (i.e. none of: model file, model size/fingerprint,
// GPU, VRAM, llama.cpp version, or supported flags have changed). On
// mismatch, *reason (if non-null) is set to a human-readable explanation.
bool profile_is_valid_for(const Profile& profile, const ModelInfo& current_model, const HardwareSnapshot& current_hardware,
                           const LlamaSnapshot& current_llama, std::string* reason = nullptr);

} // namespace fluxinfer::profiles
