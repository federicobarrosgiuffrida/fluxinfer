#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/tuner/feasibility.hpp"

using namespace fluxinfer::tuner;
using fluxinfer::llama::GgufMetadata;

namespace {
constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;

// Qwen3.6-35B-A3B as actually measured: 40 layers, 4096 embedding,
// grouped-query attention with 4 KV heads out of 32.
GgufMetadata qwen_moe_metadata() {
    GgufMetadata metadata;
    metadata.architecture = "qwen35moe";
    metadata.block_count = 40;
    metadata.embedding_length = 4096;
    metadata.attention_head_count = 32;
    metadata.attention_head_count_kv = 4;
    return metadata;
}

fluxinfer::hardware::HardwareInfo rtx3060() {
    fluxinfer::hardware::HardwareInfo hardware;
    hardware.gpu.available = true;
    hardware.gpu.total_vram_bytes = 12 * kGiB;
    hardware.gpu.available_vram_bytes = 11 * kGiB;
    hardware.memory.total_bytes = 32 * kGiB;
    hardware.memory.available_bytes = 20 * kGiB;
    return hardware;
}
} // namespace

TEST_CASE("KV cache accounts for grouped-query attention", "[feasibility]") {
    const GgufMetadata metadata = qwen_moe_metadata();

    // 4096 embedding x (4/32 GQA) = 512, x 40 layers x 4 bytes (K+V, f16)
    // = 81920 bytes per token, i.e. 2.5GB of KV cache at 32k context.
    const std::optional<std::uint64_t> kv = estimate_kv_cache_bytes(metadata, 32768);
    REQUIRE(kv.has_value());
    CHECK(*kv == 81920ULL * 32768);

    SECTION("without head counts it falls back to the no-GQA upper bound") {
        GgufMetadata no_heads = metadata;
        no_heads.attention_head_count.reset();
        no_heads.attention_head_count_kv.reset();
        const std::optional<std::uint64_t> bound = estimate_kv_cache_bytes(no_heads, 32768);
        REQUIRE(bound.has_value());
        CHECK(*bound == *kv * 8); // exactly the 32/4 GQA ratio
    }

    SECTION("missing layer count yields no estimate rather than a guess") {
        GgufMetadata incomplete = metadata;
        incomplete.block_count.reset();
        CHECK_FALSE(estimate_kv_cache_bytes(incomplete, 32768).has_value());
    }
}

TEST_CASE("estimate_fit classifies the three real situations", "[feasibility]") {
    const GgufMetadata metadata = qwen_moe_metadata();
    const auto hardware = rtx3060();
    const std::uint64_t headroom = kGiB + kGiB / 2;

    SECTION("the measured case: 19.7GB model on a 12GB card needs offload") {
        const FitEstimate fit = estimate_fit(21166757728ULL, metadata, 32768, hardware, headroom);
        CHECK(fit.verdict == FitEstimate::Verdict::NeedsCpuOffload);
        CHECK_FALSE(fit.is_problem());
    }

    SECTION("a small model fits entirely in VRAM") {
        GgufMetadata small = metadata;
        small.block_count = 32;
        small.embedding_length = 2048;
        const FitEstimate fit = estimate_fit(5 * kGiB, small, 8192, hardware, headroom);
        CHECK(fit.verdict == FitEstimate::Verdict::FitsInVram);
    }

    SECTION("a model larger than VRAM plus RAM is impossible, and says so") {
        const FitEstimate fit = estimate_fit(80 * kGiB, metadata, 32768, hardware, headroom);
        CHECK(fit.verdict == FitEstimate::Verdict::ExceedsTotalMemory);
        CHECK(fit.is_problem());
        CHECK(fit.explanation.find("does not fit") != std::string::npos);
    }

    SECTION("an enormous context can make even a modest model impossible") {
        const FitEstimate fit = estimate_fit(8 * kGiB, metadata, 1000000, hardware, headroom);
        CHECK(fit.verdict == FitEstimate::Verdict::ExceedsTotalMemory);
    }

    SECTION("unreadable metadata produces no verdict and no accusation") {
        GgufMetadata empty;
        const FitEstimate fit = estimate_fit(20 * kGiB, empty, 32768, hardware, headroom);
        CHECK(fit.verdict == FitEstimate::Verdict::Unknown);
        CHECK_FALSE(fit.is_problem());
        CHECK(fit.explanation.find("not estimated") != std::string::npos);
    }
}
