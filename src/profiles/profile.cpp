#include "fluxinfer/profiles/profile.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <vector>

namespace fluxinfer::profiles {

using nlohmann::json;

namespace {

std::uint64_t fnv1a_update(std::uint64_t hash, const void* data, std::size_t len) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL; // FNV prime
    }
    return hash;
}

std::string to_hex(std::uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << value;
    return oss.str();
}

} // namespace

json to_json(const Profile& profile) {
    json j;
    j["schema_version"] = profile.schema_version;

    j["model"] = {
        {"path", profile.model.path},
        {"size_bytes", profile.model.size_bytes},
        {"fingerprint", profile.model.fingerprint},
    };

    j["hardware"] = {
        {"cpu", profile.hardware.cpu},
        {"logical_threads", profile.hardware.logical_threads},
        {"ram_bytes", profile.hardware.ram_bytes},
        {"gpu", profile.hardware.gpu},
        {"vram_bytes", profile.hardware.vram_bytes},
    };

    j["llama"] = {
        {"version", profile.llama.version},
        {"binary_path", profile.llama.binary_path},
        {"supported_flags", profile.llama.supported_flags},
    };

    j["best_config"] = {
        {"threads", profile.best_config.threads},
        {"gpu_layers", profile.best_config.gpu_layers},
        {"batch_size", profile.best_config.batch_size},
        {"ubatch_size", profile.best_config.ubatch_size},
        {"kv_cache_type", profile.best_config.kv_cache_type ? json(*profile.best_config.kv_cache_type) : json(nullptr)},
        {"context_length", profile.best_config.context_length},
    };

    j["results"] = {
        {"prompt_tps", profile.results.prompt_tps},
        {"generation_tps", profile.results.generation_tps},
        {"duration_ms", profile.results.duration_ms},
        {"score", profile.results.score},
    };

    return j;
}

std::optional<Profile> profile_from_json(const json& j, std::string* error) {
    try {
        if (!j.contains("schema_version") || !j["schema_version"].is_number_integer()) {
            if (error) *error = "missing or invalid 'schema_version'";
            return std::nullopt;
        }

        Profile profile;
        profile.schema_version = j.at("schema_version").get<int>();
        if (profile.schema_version != 1) {
            if (error) *error = "unsupported schema_version " + std::to_string(profile.schema_version);
            return std::nullopt;
        }

        const auto& model = j.at("model");
        profile.model.path = model.at("path").get<std::string>();
        profile.model.size_bytes = model.at("size_bytes").get<std::uint64_t>();
        profile.model.fingerprint = model.at("fingerprint").get<std::string>();

        const auto& hw = j.at("hardware");
        profile.hardware.cpu = hw.at("cpu").get<std::string>();
        profile.hardware.logical_threads = hw.at("logical_threads").get<unsigned>();
        profile.hardware.ram_bytes = hw.at("ram_bytes").get<std::uint64_t>();
        profile.hardware.gpu = hw.value("gpu", std::string());
        profile.hardware.vram_bytes = hw.value("vram_bytes", std::uint64_t{0});

        const auto& llama = j.at("llama");
        profile.llama.version = llama.at("version").get<std::string>();
        profile.llama.binary_path = llama.at("binary_path").get<std::string>();
        profile.llama.supported_flags = llama.value("supported_flags", std::vector<std::string>());

        const auto& best = j.at("best_config");
        profile.best_config.threads = best.at("threads").get<unsigned>();
        profile.best_config.gpu_layers = best.at("gpu_layers").get<int>();
        profile.best_config.batch_size = best.at("batch_size").get<unsigned>();
        profile.best_config.ubatch_size = best.at("ubatch_size").get<unsigned>();
        if (best.contains("kv_cache_type") && !best["kv_cache_type"].is_null()) {
            profile.best_config.kv_cache_type = best["kv_cache_type"].get<std::string>();
        }
        // value() default (0) keeps older profiles saved before this field
        // existed loadable: they simply behave as before (no explicit -c).
        profile.best_config.context_length = best.value("context_length", std::uint64_t{0});

        const auto& results = j.at("results");
        profile.results.prompt_tps = results.at("prompt_tps").get<double>();
        profile.results.generation_tps = results.at("generation_tps").get<double>();
        profile.results.duration_ms = results.at("duration_ms").get<std::int64_t>();
        profile.results.score = results.at("score").get<double>();

        return profile;
    } catch (const json::exception& e) {
        if (error) *error = std::string("malformed profile JSON: ") + e.what();
        return std::nullopt;
    }
}

std::optional<ModelInfo> compute_model_info(const std::filesystem::path& model_path, std::string* error) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(model_path, ec)) {
        if (error) *error = "model file not found: " + model_path.string();
        return std::nullopt;
    }

    const std::uint64_t size = std::filesystem::file_size(model_path, ec);
    if (ec) {
        if (error) *error = "could not read model file size: " + ec.message();
        return std::nullopt;
    }

    const auto mtime = std::filesystem::last_write_time(model_path, ec);
    const auto mtime_count = ec ? std::int64_t{0} : mtime.time_since_epoch().count();

    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        if (error) *error = "could not open model file for reading: " + model_path.string();
        return std::nullopt;
    }

    constexpr std::size_t kChunkSize = 1024 * 1024;
    std::uint64_t hash = 1469598103934665603ULL; // FNV offset basis
    hash = fnv1a_update(hash, &size, sizeof(size));
    hash = fnv1a_update(hash, &mtime_count, sizeof(mtime_count));

    // Heap-allocated: kChunkSize is 1 MiB, too large to safely put on the
    // stack (default thread stack sizes, e.g. 1 MiB on Windows, would
    // overflow immediately).
    std::vector<char> buffer(kChunkSize);

    file.read(buffer.data(), static_cast<std::streamsize>(std::min<std::uint64_t>(kChunkSize, size)));
    hash = fnv1a_update(hash, buffer.data(), static_cast<std::size_t>(file.gcount()));

    if (size > kChunkSize) {
        const std::uint64_t tail_offset = size > 2 * kChunkSize ? size - kChunkSize : kChunkSize;
        file.seekg(static_cast<std::streamoff>(tail_offset), std::ios::beg);
        if (file) {
            file.read(buffer.data(), static_cast<std::streamsize>(kChunkSize));
            hash = fnv1a_update(hash, buffer.data(), static_cast<std::size_t>(file.gcount()));
        }
    }

    ModelInfo info;
    info.path = model_path.string();
    info.size_bytes = size;
    info.fingerprint = to_hex(hash);
    return info;
}

bool profile_is_valid_for(const Profile& profile, const ModelInfo& current_model, const HardwareSnapshot& current_hardware,
                           const LlamaSnapshot& current_llama, std::string* reason) {
    auto fail = [&](const std::string& why) {
        if (reason) *reason = why;
        return false;
    };

    if (profile.model.size_bytes != current_model.size_bytes || profile.model.fingerprint != current_model.fingerprint) {
        return fail("model file has changed since the profile was created");
    }
    if (profile.hardware.gpu != current_hardware.gpu) {
        return fail("GPU has changed (was '" + profile.hardware.gpu + "', now '" + current_hardware.gpu + "')");
    }
    if (profile.hardware.vram_bytes != current_hardware.vram_bytes) {
        return fail("VRAM capacity has changed");
    }
    if (profile.llama.version != current_llama.version) {
        return fail("llama.cpp version has changed (was '" + profile.llama.version + "', now '" + current_llama.version + "')");
    }
    if (profile.llama.supported_flags != current_llama.supported_flags) {
        return fail("llama.cpp supported flag set has changed");
    }

    return true;
}

} // namespace fluxinfer::profiles
