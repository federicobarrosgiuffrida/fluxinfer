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

namespace {
constexpr std::uint64_t kMiB = 1024ULL * 1024;

// Shaped after the real RTX 3060 12GB / Qwen3.6-35B-A3B sweep recorded in
// docs/: each entry is a usable run with a measured VRAM peak.
BenchmarkResult make_run(double prompt_tps, double generation_tps, std::uint64_t peak_mib) {
    BenchmarkResult result;
    result.ran = true;
    result.exit_code = 0;
    result.output_valid = true;
    result.prompt_tokens_per_second = prompt_tps;
    result.generation_tokens_per_second = generation_tps;
    result.measured_peak_vram_bytes = peak_mib * kMiB;
    return result;
}

constexpr std::uint64_t kTotalVram = 12288 * kMiB; // RTX 3060 12GB
constexpr std::uint64_t kHeadroom = 1536 * kMiB;   // Windows default
} // namespace

TEST_CASE("looks_like_vram_spill flags the measured silent-spill case", "[scoring][spill]") {
    // Real numbers: the setting that left ~260MB free ran at 77 tok/s
    // prompt / 28.4 tok/s generation, against 654 / 45.4 one step lower.
    const BenchmarkResult healthy = make_run(654.0, 45.4, 11235);
    const BenchmarkResult spilling = make_run(77.0, 28.4, 12026);

    CHECK(looks_like_vram_spill(spilling, healthy, kTotalVram, kHeadroom));
}

TEST_CASE("looks_like_vram_spill does not flag healthy configurations", "[scoring][spill]") {
    SECTION("normal improvement as more work moves to the GPU") {
        const BenchmarkResult slower = make_run(202.0, 33.6, 5724);
        const BenchmarkResult faster = make_run(519.0, 37.4, 7585);
        CHECK_FALSE(looks_like_vram_spill(faster, slower, kTotalVram, kHeadroom));
    }

    SECTION("slower run with plenty of VRAM left is not a spill") {
        // Whatever made this configuration slower, it was not the card
        // running out of room -- over 6GB was still free.
        const BenchmarkResult reference = make_run(600.0, 45.0, 5724);
        const BenchmarkResult slow = make_run(60.0, 10.0, 5900);
        CHECK_FALSE(looks_like_vram_spill(slow, reference, kTotalVram, kHeadroom));
    }

    SECTION("full card but only a mild slowdown is not a spill") {
        const BenchmarkResult reference = make_run(600.0, 45.0, 11235);
        const BenchmarkResult mild = make_run(560.0, 41.0, 12026);
        CHECK_FALSE(looks_like_vram_spill(mild, reference, kTotalVram, kHeadroom));
    }

    SECTION("without a VRAM measurement no accusation is made") {
        BenchmarkResult unmeasured = make_run(77.0, 28.4, 12026);
        unmeasured.measured_peak_vram_bytes.reset();
        const BenchmarkResult healthy = make_run(654.0, 45.4, 11235);
        CHECK_FALSE(looks_like_vram_spill(unmeasured, healthy, kTotalVram, kHeadroom));
    }

    SECTION("a crashed or OOM run is handled by its own detection, not this") {
        BenchmarkResult crashed = make_run(0.0, 0.0, 12026);
        crashed.crashed = true;
        const BenchmarkResult healthy = make_run(654.0, 45.4, 11235);
        CHECK_FALSE(looks_like_vram_spill(crashed, healthy, kTotalVram, kHeadroom));
    }
}

TEST_CASE("exceeds_vram_headroom rejects configurations with no margin", "[scoring][spill]") {
    // Real case: --n-cpu-moe 21 peaked at 12.0GB of a 12GB card during the
    // search, benchmarked fastest, and then served ~10% slower than a
    // configuration with margin once the machine was actually in use.
    CHECK(exceeds_vram_headroom(make_run(138.7, 38.7, 12280), kTotalVram, kHeadroom));

    SECTION("a configuration with margin is accepted") {
        CHECK_FALSE(exceeds_vram_headroom(make_run(136.5, 37.9, 10500), kTotalVram, kHeadroom));
    }
    SECTION("exactly at the headroom boundary is still acceptable") {
        CHECK_FALSE(exceeds_vram_headroom(make_run(136.5, 37.9, 12288 - 1536), kTotalVram, kHeadroom));
    }
    SECTION("no measurement means no verdict") {
        BenchmarkResult unmeasured = make_run(138.7, 38.7, 12280);
        unmeasured.measured_peak_vram_bytes.reset();
        CHECK_FALSE(exceeds_vram_headroom(unmeasured, kTotalVram, kHeadroom));
    }
    SECTION("headroom disabled means no verdict") {
        CHECK_FALSE(exceeds_vram_headroom(make_run(138.7, 38.7, 12280), kTotalVram, 0));
    }
}
