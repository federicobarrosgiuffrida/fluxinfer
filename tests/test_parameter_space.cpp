#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/tuner/parameter_space.hpp"

using namespace fluxinfer::tuner;

namespace {
constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;

ParameterSpaceInput make_input(unsigned physical, unsigned logical, std::uint64_t ram_bytes, bool gpu, std::uint64_t vram_bytes,
                                std::optional<std::uint64_t> real_layer_count) {
    ParameterSpaceInput input;
    input.hardware.cpu.physical_cores = physical;
    input.hardware.cpu.logical_threads = logical;
    input.hardware.memory.total_bytes = ram_bytes;
    input.hardware.memory.available_bytes = ram_bytes;
    input.hardware.gpu.available = gpu;
    input.hardware.gpu.total_vram_bytes = vram_bytes;
    input.hardware.gpu.available_vram_bytes = vram_bytes;
    input.model_size_bytes = 4ULL * kGiB;
    input.real_layer_count = real_layer_count;
    return input;
}
} // namespace

TEST_CASE("baseline_config is CPU-only and uses physical core count", "[param-space]") {
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, true, 8 * kGiB, 32);
    TuneConfig baseline = baseline_config(input);
    CHECK(baseline.threads == 8);
    CHECK(baseline.gpu_layers == 0);
}

TEST_CASE("gpu_layers_candidates is empty without a GPU", "[param-space]") {
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, false, 0, 32);
    TuneConfig baseline = baseline_config(input);
    CHECK(gpu_layers_candidates(input, baseline).empty());
}

TEST_CASE("gpu_layers_candidates is empty when the real layer count is unavailable", "[param-space]") {
    // A GPU is present, but FluxInfer never invents a layer count when
    // GGUF metadata couldn't be read.
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, true, 8 * kGiB, std::nullopt);
    TuneConfig baseline = baseline_config(input);
    CHECK(gpu_layers_candidates(input, baseline).empty());
}

TEST_CASE("gpu_layers_candidates covers 0/25/50/75/100 percent of the real layer count", "[param-space]") {
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, true, 8 * kGiB, 32);
    TuneConfig baseline = baseline_config(input);
    std::vector<TuneConfig> candidates = gpu_layers_candidates(input, baseline);

    std::set<int> layers;
    for (const auto& c : candidates) layers.insert(c.gpu_layers);

    CHECK(layers.count(0) == 1);
    CHECK(layers.count(8) == 1);  // 25%
    CHECK(layers.count(16) == 1); // 50%
    CHECK(layers.count(24) == 1); // 75%
    CHECK(layers.count(32) == 1); // 100%
    CHECK(candidates.size() == 5); // no duplicates for a count divisible by 4
}

TEST_CASE("gpu_layers_candidates always includes 0 and full offload", "[param-space]") {
    // 7 layers: none of 0/25/50/75/100% round to exactly 0 and 7 together
    // by chance, so this specifically exercises the "always include both
    // endpoints" guarantee rather than relying on rounding.
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, true, 8 * kGiB, 7);
    TuneConfig baseline = baseline_config(input);
    std::vector<TuneConfig> candidates = gpu_layers_candidates(input, baseline);

    std::set<int> layers;
    for (const auto& c : candidates) layers.insert(c.gpu_layers);
    CHECK(layers.count(0) == 1);
    CHECK(layers.count(7) == 1);
    for (int l : layers) {
        CHECK(l >= 0);
        CHECK(l <= 7);
    }
}

TEST_CASE("gpu_layers_candidates deduplicates when percentages collide on a small layer count", "[param-space]") {
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, true, 1 * kGiB, 1);
    TuneConfig baseline = baseline_config(input);
    std::vector<TuneConfig> candidates = gpu_layers_candidates(input, baseline);
    // With a real layer count of 1, 0%/25% both round to 0 and
    // 50%/75%/100% all round to 1: only two distinct values should remain.
    CHECK(candidates.size() == 2);
}

TEST_CASE("batch_ubatch_candidates trims to smaller combinations on constrained RAM", "[param-space]") {
    TuneConfig base;
    base.batch_size = 999; // sentinel so nothing is filtered as "already the base"
    base.ubatch_size = 999;

    ParameterSpaceInput plenty = make_input(8, 16, 32 * kGiB, false, 0, std::nullopt);
    ParameterSpaceInput tight = make_input(8, 16, 6 * kGiB, false, 0, std::nullopt);
    ParameterSpaceInput very_tight = make_input(8, 16, 2 * kGiB, false, 0, std::nullopt);

    ParameterSpaceInput moderate = make_input(8, 16, 12 * kGiB, false, 0, std::nullopt);

    CHECK(batch_ubatch_candidates(plenty, base).size() == 4); // includes llama-bench's own 2048/512 default
    CHECK(batch_ubatch_candidates(moderate, base).size() == 3);
    CHECK(batch_ubatch_candidates(tight, base).size() == 2);
    CHECK(batch_ubatch_candidates(very_tight, base).size() == 1);
}

TEST_CASE("batch_ubatch_candidates skips a combination already equal to the base", "[param-space]") {
    TuneConfig base;
    base.batch_size = 128;
    base.ubatch_size = 64;

    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, false, 0, std::nullopt);
    std::vector<TuneConfig> candidates = batch_ubatch_candidates(input, base);
    for (const auto& c : candidates) {
        CHECK_FALSE((c.batch_size == 128 && c.ubatch_size == 64));
    }
    CHECK(candidates.size() == 3); // the (256,128), (512,256) and (2048,512) pairs remain
}

TEST_CASE("thread_candidates deduplicates when physical == logical cores", "[param-space]") {
    ParameterSpaceInput input = make_input(8, 8, 32 * kGiB, false, 0, std::nullopt);
    TuneConfig base = baseline_config(input);
    std::vector<TuneConfig> candidates = thread_candidates(input, base);
    CHECK(candidates.size() == 1);
    CHECK(candidates.front().threads == 8);
}

TEST_CASE("thread_candidates covers physical, logical, and midpoint on SMT hardware", "[param-space]") {
    ParameterSpaceInput input = make_input(8, 16, 32 * kGiB, false, 0, std::nullopt);
    TuneConfig base = baseline_config(input);
    std::vector<TuneConfig> candidates = thread_candidates(input, base);

    std::set<unsigned> threads;
    for (const auto& c : candidates) threads.insert(c.threads);
    CHECK(threads.count(8) == 1);
    CHECK(threads.count(16) == 1);
    CHECK(threads.count(12) == 1);
}
