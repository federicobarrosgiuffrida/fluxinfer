#include "fluxinfer/profiles/profile_store.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>

namespace fluxinfer::profiles {

namespace {

std::string sanitize_filename_component(const std::string& input) {
    std::string result = input;
    for (char& c : result) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_' && c != '.') {
            c = '_';
        }
    }
    return result.empty() ? "model" : result;
}

} // namespace

ProfileStore::ProfileStore(std::filesystem::path profiles_directory) : profiles_directory_(std::move(profiles_directory)) {}

std::filesystem::path ProfileStore::profile_path_for(const ModelInfo& model) const {
    const std::string stem = std::filesystem::path(model.path).stem().string();
    const std::string fingerprint_prefix = model.fingerprint.substr(0, std::min<std::size_t>(16, model.fingerprint.size()));
    const std::string filename = sanitize_filename_component(stem) + "-" + fingerprint_prefix + ".json";
    return profiles_directory_ / filename;
}

bool ProfileStore::save(const Profile& profile, std::string* error) const {
    std::error_code ec;
    std::filesystem::create_directories(profiles_directory_, ec);
    if (ec) {
        if (error) *error = "could not create profiles directory: " + ec.message();
        return false;
    }

    const std::filesystem::path final_path = profile_path_for(profile.model);
    const std::filesystem::path tmp_path = final_path.string() + ".tmp";

    {
        std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            if (error) *error = "could not open profile file for writing: " + tmp_path.string();
            return false;
        }
        file << to_json(profile).dump(2);
        if (!file) {
            if (error) *error = "failed writing profile file: " + tmp_path.string();
            return false;
        }
    }

    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        // rename() can fail across filesystems/mount points; fall back to a
        // copy + remove instead of leaving the profile unwritten.
        std::filesystem::copy_file(tmp_path, final_path, std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmp_path);
        if (ec) {
            if (error) *error = "could not finalize profile file: " + ec.message();
            return false;
        }
    }

    return true;
}

std::optional<Profile> ProfileStore::load_valid(const ModelInfo& current_model, const HardwareSnapshot& current_hardware,
                                                 const LlamaSnapshot& current_llama, std::string* reason) const {
    const std::filesystem::path path = profile_path_for(current_model);

    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        if (reason) *reason = "no profile found at " + path.string();
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (reason) *reason = "could not open profile file: " + path.string();
        return std::nullopt;
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(buffer.str());
    } catch (const nlohmann::json::exception& e) {
        if (reason) *reason = std::string("profile file is not valid JSON: ") + e.what();
        return std::nullopt;
    }

    std::string parse_error;
    std::optional<Profile> profile = profile_from_json(json, &parse_error);
    if (!profile) {
        if (reason) *reason = parse_error;
        return std::nullopt;
    }

    std::string invalid_reason;
    if (!profile_is_valid_for(*profile, current_model, current_hardware, current_llama, &invalid_reason)) {
        if (reason) *reason = invalid_reason;
        return std::nullopt;
    }

    return profile;
}

} // namespace fluxinfer::profiles
