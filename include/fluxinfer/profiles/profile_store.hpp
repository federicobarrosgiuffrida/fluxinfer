#pragma once

#include "fluxinfer/profiles/profile.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace fluxinfer::profiles {

// Persists and retrieves tuning profiles as JSON files on disk, one per
// model, keyed by the model's filename stem and current fingerprint.
class ProfileStore {
public:
    explicit ProfileStore(std::filesystem::path profiles_directory);

    // Deterministic path for the profile associated with `model`. Does not
    // touch the filesystem.
    std::filesystem::path profile_path_for(const ModelInfo& model) const;

    bool save(const Profile& profile, std::string* error = nullptr) const;

    // Verifies that a profile could actually be written here, by creating
    // the directory and writing (then removing) a probe file. Meant to be
    // called *before* a tuning run: a tune costs tens of minutes, and
    // discovering only at the end that the directory is not writable throws
    // the whole result away. On Windows the usual cause is Controlled
    // Folder Access, which silently denies writes to Documents/Desktop/
    // Pictures for executables it does not recognise.
    bool check_writable(std::string* error = nullptr) const;

    // Loads the profile for `current_model` (if any) and validates it
    // against the given current hardware/llama.cpp snapshot via
    // profile_is_valid_for(). Returns std::nullopt (with *reason set) if no
    // profile exists, it can't be parsed, or it has been invalidated.
    std::optional<Profile> load_valid(const ModelInfo& current_model, const HardwareSnapshot& current_hardware,
                                       const LlamaSnapshot& current_llama, std::string* reason = nullptr) const;

private:
    std::filesystem::path profiles_directory_;
};

} // namespace fluxinfer::profiles
