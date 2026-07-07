#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/tuner/scoring.hpp"

using namespace fluxinfer::tuner;

namespace {

BenchmarkResult make_usable_result(double prompt_tps, double gen_tps, std::uint64_t est_ram, std::uint64_t est_vram) {
    BenchmarkResult result;
    result.ran = true;
    result.output_valid = true;
    result.prompt_tokens_per_second = prompt_tps;
    result.generation_tokens_per_second = gen_tps;
    result.first_token_latency_ms = prompt_tps > 0 ? 1000.0 / prompt_tps : 0.0;
    result.estimated_ram_bytes = est_ram;
    result.estimated_vram_bytes = est_vram;
    return result;
}

} // namespace

TEST_CASE("compute_memory_pressure_penalty is zero under 90% usage", "[scoring]") {
    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    CHECK(compute_memory_pressure_penalty(4 * kGiB, 16 * kGiB, 0, 0) == Catch::Approx(0.0));
    CHECK(compute_memory_pressure_penalty(0, 0, 0, 0) == Catch::Approx(0.0)); // unknown budget: no penalty
}

TEST_CASE("compute_memory_pressure_penalty grows as usage approaches/exceeds the budget", "[scoring]") {
    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    const double near_full = compute_memory_pressure_penalty(15 * kGiB, 16 * kGiB, 0, 0);
    const double over_full = compute_memory_pressure_penalty(20 * kGiB, 16 * kGiB, 0, 0);
    const double comfortable = compute_memory_pressure_penalty(8 * kGiB, 16 * kGiB, 0, 0);

    CHECK(comfortable == Catch::Approx(0.0));
    CHECK(near_full > comfortable);
    CHECK(over_full > near_full);
}

TEST_CASE("compute_memory_pressure_penalty does not penalize a config that safely completed at ~75% estimated usage",
          "[scoring][regression]") {
    // Regression test for a real-world finding: a config estimated at
    // ~75% of the VRAM budget (e.g. 24 of 32 layers offloaded) that
    // actually ran successfully (no OOM) was previously penalized enough
    // by the old 70%-ramp-start to score below a config using barely any
    // GPU offload, despite measuring ~40% higher real generation
    // throughput. At 90% ramp-start, 75% usage must be penalty-free.
    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    CHECK(compute_memory_pressure_penalty(6 * kGiB, 8 * kGiB, 0, 0) == Catch::Approx(0.0)); // 75%
}

TEST_CASE("compute_score matches the documented formula for a usable result", "[scoring]") {
    ScoringWeights weights;
    BenchmarkResult result = make_usable_result(1000.0, 20.0, 0, 0);

    const double expected = result.generation_tokens_per_second * weights.generation_tps_weight +
                             result.prompt_tokens_per_second * weights.prompt_tps_weight -
                             result.first_token_latency_ms * weights.first_token_latency_weight;

    CHECK(compute_score(result, 0, 0, weights) == Catch::Approx(expected));
}

TEST_CASE("compute_score penalizes unusable results regardless of throughput", "[scoring]") {
    ScoringWeights weights;

    BenchmarkResult oom_result = make_usable_result(9999.0, 999.0, 0, 0);
    oom_result.oom = true;

    BenchmarkResult crashed_result = make_usable_result(9999.0, 999.0, 0, 0);
    crashed_result.crashed = true;

    BenchmarkResult timed_out_result = make_usable_result(9999.0, 999.0, 0, 0);
    timed_out_result.timed_out = true;

    BenchmarkResult invalid_output_result;
    invalid_output_result.ran = true;
    invalid_output_result.output_valid = false;

    CHECK(compute_score(oom_result, 0, 0, weights) == Catch::Approx(-weights.instability_penalty));
    CHECK(compute_score(crashed_result, 0, 0, weights) == Catch::Approx(-weights.instability_penalty));
    CHECK(compute_score(timed_out_result, 0, 0, weights) == Catch::Approx(-weights.instability_penalty));
    CHECK(compute_score(invalid_output_result, 0, 0, weights) == Catch::Approx(-weights.instability_penalty));
}

TEST_CASE("compute_score ranks a slow-but-usable config above a fast-but-OOMing one", "[scoring]") {
    ScoringWeights weights;
    BenchmarkResult slow_but_usable = make_usable_result(200.0, 5.0, 0, 0);
    BenchmarkResult fast_but_oom = make_usable_result(9999.0, 999.0, 0, 0);
    fast_but_oom.oom = true;

    CHECK(compute_score(slow_but_usable, 0, 0, weights) > compute_score(fast_but_oom, 0, 0, weights));
}

TEST_CASE("compute_score applies the memory pressure penalty with the configured weight", "[scoring]") {
    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    ScoringWeights weights;
    weights.memory_pressure_weight = 2.0;

    BenchmarkResult result = make_usable_result(0.0, 10.0, 20 * kGiB, 0);
    const double penalty = compute_memory_pressure_penalty(20 * kGiB, 16 * kGiB, 0, 0);
    const double expected = result.generation_tokens_per_second * weights.generation_tps_weight - penalty * weights.memory_pressure_weight;

    CHECK(compute_score(result, 16 * kGiB, 0, weights) == Catch::Approx(expected));
}
