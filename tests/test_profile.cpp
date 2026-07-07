#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/profiles/profile.hpp"
#include "fluxinfer/profiles/profile_store.hpp"

#include <fstream>
#include <random>

using namespace fluxinfer::profiles;

namespace {

Profile make_sample_profile() {
    Profile profile;
    profile.schema_version = 1;
    profile.model.path = "qwen-model.gguf";
    profile.model.size_bytes = 123456789;
    profile.model.fingerprint = "deadbeefcafef00d";

    profile.hardware.cpu = "Intel Core i5-14600KF";
    profile.hardware.logical_threads = 20;
    profile.hardware.ram_bytes = 34359738368ULL;
    profile.hardware.gpu = "NVIDIA GeForce RTX 5060";
    profile.hardware.vram_bytes = 8589934592ULL;

    profile.llama.version = "build 1234 (abcdef1)";
    profile.llama.binary_path = "/usr/local/bin/llama-bench";
    profile.llama.supported_flags = {"--batch-size", "--n-gpu-layers", "--threads"};

    profile.best_config.threads = 14;
    profile.best_config.gpu_layers = 31;
    profile.best_config.batch_size = 512;
    profile.best_config.ubatch_size = 256;
    profile.best_config.kv_cache_type = "q8_0";
    profile.best_config.context_length = 4096;

    profile.results.prompt_tps = 1200.5;
    profile.results.generation_tps = 18.7;
    profile.results.duration_ms = 12000;
    profile.results.score = 47.3;

    return profile;
}

std::filesystem::path make_scratch_dir() {
    static std::mt19937_64 rng(std::random_device{}());
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                 ("fluxinfer_test_" + std::to_string(rng()));
    std::filesystem::create_directories(dir);
    return dir;
}

std::filesystem::path write_scratch_file(const std::filesystem::path& dir, const std::string& name, const std::string& content) {
    std::filesystem::path path = dir / name;
    std::ofstream file(path, std::ios::binary);
    file << content;
    return path;
}

} // namespace

TEST_CASE("profile JSON round-trips through to_json/profile_from_json", "[profile]") {
    Profile original = make_sample_profile();
    nlohmann::json json = to_json(original);

    std::string error;
    std::optional<Profile> parsed = profile_from_json(json, &error);
    REQUIRE(parsed.has_value());
    CHECK(error.empty());

    CHECK(parsed->schema_version == original.schema_version);
    CHECK(parsed->model.path == original.model.path);
    CHECK(parsed->model.size_bytes == original.model.size_bytes);
    CHECK(parsed->model.fingerprint == original.model.fingerprint);
    CHECK(parsed->hardware.cpu == original.hardware.cpu);
    CHECK(parsed->hardware.vram_bytes == original.hardware.vram_bytes);
    CHECK(parsed->llama.version == original.llama.version);
    CHECK(parsed->llama.supported_flags == original.llama.supported_flags);
    CHECK(parsed->best_config.threads == original.best_config.threads);
    CHECK(parsed->best_config.gpu_layers == original.best_config.gpu_layers);
    CHECK(parsed->best_config.kv_cache_type == original.best_config.kv_cache_type);
    CHECK(parsed->best_config.context_length == original.best_config.context_length);
    CHECK(parsed->results.score == Catch::Approx(original.results.score));
}

TEST_CASE("profile_from_json defaults context_length to 0 for a profile saved before the field existed",
          "[profile][compat]") {
    Profile original = make_sample_profile();
    nlohmann::json json = to_json(original);
    json["best_config"].erase("context_length"); // simulate an older, pre-context_length profile file

    std::optional<Profile> parsed = profile_from_json(json);
    REQUIRE(parsed.has_value());
    CHECK(parsed->best_config.context_length == 0);
}

TEST_CASE("profile JSON round-trips a null kv_cache_type", "[profile]") {
    Profile original = make_sample_profile();
    original.best_config.kv_cache_type = std::nullopt;

    nlohmann::json json = to_json(original);
    std::optional<Profile> parsed = profile_from_json(json);
    REQUIRE(parsed.has_value());
    CHECK_FALSE(parsed->best_config.kv_cache_type.has_value());
}

TEST_CASE("profile_from_json rejects missing schema_version", "[profile]") {
    nlohmann::json json = to_json(make_sample_profile());
    json.erase("schema_version");

    std::string error;
    std::optional<Profile> parsed = profile_from_json(json, &error);
    CHECK_FALSE(parsed.has_value());
    CHECK_FALSE(error.empty());
}

TEST_CASE("profile_from_json rejects an unsupported schema_version", "[profile]") {
    nlohmann::json json = to_json(make_sample_profile());
    json["schema_version"] = 99;

    std::string error;
    std::optional<Profile> parsed = profile_from_json(json, &error);
    CHECK_FALSE(parsed.has_value());
    CHECK_FALSE(error.empty());
}

TEST_CASE("profile_from_json rejects a document missing required fields", "[profile]") {
    nlohmann::json json = to_json(make_sample_profile());
    json["model"].erase("fingerprint");

    std::string error;
    std::optional<Profile> parsed = profile_from_json(json, &error);
    CHECK_FALSE(parsed.has_value());
    CHECK_FALSE(error.empty());
}

TEST_CASE("compute_model_info is stable across repeated calls and reacts to content changes", "[profile]") {
    std::filesystem::path dir = make_scratch_dir();
    std::filesystem::path model_path = write_scratch_file(dir, "model.gguf", "GGUF-fake-content-v1-aaaaaaaaaa");

    std::optional<ModelInfo> first = compute_model_info(model_path);
    std::optional<ModelInfo> second = compute_model_info(model_path);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(first->fingerprint == second->fingerprint);
    CHECK(first->size_bytes == second->size_bytes);

    // Sleep isn't reliable for mtime granularity in CI; change size directly
    // instead, which is always reflected in the fingerprint.
    write_scratch_file(dir, "model.gguf", "GGUF-fake-content-v2-DIFFERENT-and-longer-content");
    std::optional<ModelInfo> third = compute_model_info(model_path);
    REQUIRE(third.has_value());
    CHECK(third->fingerprint != first->fingerprint);

    std::filesystem::remove_all(dir);
}

TEST_CASE("compute_model_info reports a clear error for a missing file", "[profile]") {
    std::string error;
    std::optional<ModelInfo> info = compute_model_info("/nonexistent/path/model.gguf", &error);
    CHECK_FALSE(info.has_value());
    CHECK_FALSE(error.empty());
}

TEST_CASE("profile_is_valid_for accepts a matching snapshot", "[profile][invalidation]") {
    Profile profile = make_sample_profile();
    ModelInfo model{profile.model.path, profile.model.size_bytes, profile.model.fingerprint};
    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         profile.hardware.gpu, profile.hardware.vram_bytes};
    LlamaSnapshot llama{profile.llama.version, profile.llama.binary_path, profile.llama.supported_flags};

    std::string reason;
    CHECK(profile_is_valid_for(profile, model, hw, llama, &reason));
    CHECK(reason.empty());
}

TEST_CASE("profile_is_valid_for rejects a changed model fingerprint", "[profile][invalidation]") {
    Profile profile = make_sample_profile();
    ModelInfo model{profile.model.path, profile.model.size_bytes, "totally-different-fingerprint"};
    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         profile.hardware.gpu, profile.hardware.vram_bytes};
    LlamaSnapshot llama{profile.llama.version, profile.llama.binary_path, profile.llama.supported_flags};

    std::string reason;
    CHECK_FALSE(profile_is_valid_for(profile, model, hw, llama, &reason));
    CHECK_FALSE(reason.empty());
}

TEST_CASE("profile_is_valid_for rejects a changed GPU", "[profile][invalidation]") {
    Profile profile = make_sample_profile();
    ModelInfo model{profile.model.path, profile.model.size_bytes, profile.model.fingerprint};
    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         "A Different GPU", profile.hardware.vram_bytes};
    LlamaSnapshot llama{profile.llama.version, profile.llama.binary_path, profile.llama.supported_flags};

    CHECK_FALSE(profile_is_valid_for(profile, model, hw, llama));
}

TEST_CASE("profile_is_valid_for rejects a changed VRAM capacity", "[profile][invalidation]") {
    Profile profile = make_sample_profile();
    ModelInfo model{profile.model.path, profile.model.size_bytes, profile.model.fingerprint};
    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         profile.hardware.gpu, profile.hardware.vram_bytes / 2};
    LlamaSnapshot llama{profile.llama.version, profile.llama.binary_path, profile.llama.supported_flags};

    CHECK_FALSE(profile_is_valid_for(profile, model, hw, llama));
}

TEST_CASE("profile_is_valid_for rejects a changed llama.cpp version", "[profile][invalidation]") {
    Profile profile = make_sample_profile();
    ModelInfo model{profile.model.path, profile.model.size_bytes, profile.model.fingerprint};
    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         profile.hardware.gpu, profile.hardware.vram_bytes};
    LlamaSnapshot llama{"build 9999 (newer)", profile.llama.binary_path, profile.llama.supported_flags};

    CHECK_FALSE(profile_is_valid_for(profile, model, hw, llama));
}

TEST_CASE("profile_is_valid_for rejects a changed supported-flag set", "[profile][invalidation]") {
    Profile profile = make_sample_profile();
    ModelInfo model{profile.model.path, profile.model.size_bytes, profile.model.fingerprint};
    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         profile.hardware.gpu, profile.hardware.vram_bytes};
    LlamaSnapshot llama{profile.llama.version, profile.llama.binary_path, {"--only-one-flag"}};

    CHECK_FALSE(profile_is_valid_for(profile, model, hw, llama));
}

TEST_CASE("ProfileStore saves and loads a valid profile, invalidates on model change", "[profile][store]") {
    std::filesystem::path dir = make_scratch_dir();
    std::filesystem::path model_path = write_scratch_file(dir, "model.gguf", "GGUF-fake-content-for-store-test");

    std::optional<ModelInfo> model_info = compute_model_info(model_path);
    REQUIRE(model_info.has_value());

    Profile profile = make_sample_profile();
    profile.model = *model_info;

    ProfileStore store(dir / "profiles");
    std::string save_error;
    REQUIRE(store.save(profile, &save_error));
    CHECK(save_error.empty());

    HardwareSnapshot hw{profile.hardware.cpu, profile.hardware.logical_threads, profile.hardware.ram_bytes,
                         profile.hardware.gpu, profile.hardware.vram_bytes};
    LlamaSnapshot llama{profile.llama.version, profile.llama.binary_path, profile.llama.supported_flags};

    std::string reason;
    std::optional<Profile> loaded = store.load_valid(*model_info, hw, llama, &reason);
    REQUIRE(loaded.has_value());
    CHECK(loaded->best_config.threads == profile.best_config.threads);

    // Overwrite the model file: the profile on disk is unchanged but should
    // no longer validate against the new fingerprint.
    write_scratch_file(dir, "model.gguf", "GGUF-fake-content-for-store-test-BUT-DIFFERENT-NOW");
    std::optional<ModelInfo> changed_model_info = compute_model_info(model_path);
    REQUIRE(changed_model_info.has_value());

    std::optional<Profile> stale = store.load_valid(*changed_model_info, hw, llama, &reason);
    CHECK_FALSE(stale.has_value());
    CHECK_FALSE(reason.empty());

    std::filesystem::remove_all(dir);
}

TEST_CASE("ProfileStore::load_valid reports a clear reason when no profile exists", "[profile][store]") {
    std::filesystem::path dir = make_scratch_dir();
    ProfileStore store(dir / "profiles");

    ModelInfo model{"missing.gguf", 1, "fingerprint"};
    HardwareSnapshot hw;
    LlamaSnapshot llama;

    std::string reason;
    std::optional<Profile> loaded = store.load_valid(model, hw, llama, &reason);
    CHECK_FALSE(loaded.has_value());
    CHECK_FALSE(reason.empty());

    std::filesystem::remove_all(dir);
}
