#include "fluxinfer/tuner/feasibility.hpp"

#include <algorithm>
#include <sstream>

namespace fluxinfer::tuner {

namespace {

constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;

// Activations, compute buffers and allocator slack. Deliberately a flat
// figure rather than a model of llama.cpp's allocator: the point is to
// catch the obviously-impossible case, and a wrong constant here cannot
// turn a comfortable fit into an impossible one.
constexpr std::uint64_t kOverheadBytes = 1024ULL * 1024 * 1024;

std::string format_gib(std::uint64_t bytes) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << static_cast<double>(bytes) / static_cast<double>(kGiB) << " GB";
    return out.str();
}

} // namespace

std::optional<std::uint64_t> estimate_kv_cache_bytes(const llama::GgufMetadata& metadata, std::uint64_t context_length) {
    if (!metadata.block_count || !metadata.embedding_length || context_length == 0) {
        return std::nullopt;
    }
    const std::uint64_t layers = *metadata.block_count;
    const std::uint64_t embedding = *metadata.embedding_length;
    if (layers == 0 || embedding == 0) {
        return std::nullopt;
    }

    // With grouped-query attention only head_count_kv heads carry K and V,
    // so the per-token cache is embedding * (head_count_kv / head_count).
    // Without both head counts, fall back to the no-GQA case (ratio 1),
    // which overestimates -- the safe direction for a fit check, and
    // flagged as such by the caller's wording.
    double kv_ratio = 1.0;
    if (metadata.attention_head_count && metadata.attention_head_count_kv && *metadata.attention_head_count > 0) {
        kv_ratio = static_cast<double>(*metadata.attention_head_count_kv) / static_cast<double>(*metadata.attention_head_count);
        kv_ratio = std::clamp(kv_ratio, 0.0, 1.0);
    }

    // 2 tensors (K and V) x 2 bytes per f16 element.
    constexpr std::uint64_t kBytesPerElementBothTensors = 4;
    const double per_token = static_cast<double>(embedding) * kv_ratio * static_cast<double>(layers) *
                              static_cast<double>(kBytesPerElementBothTensors);
    return static_cast<std::uint64_t>(per_token * static_cast<double>(context_length));
}

FitEstimate estimate_fit(std::uint64_t model_size_bytes, const llama::GgufMetadata& metadata,
                          std::uint64_t context_length, const hardware::HardwareInfo& hardware,
                          std::uint64_t vram_headroom_bytes) {
    FitEstimate estimate;
    estimate.weights_bytes = model_size_bytes;
    estimate.overhead_bytes = kOverheadBytes;

    const std::optional<std::uint64_t> kv = estimate_kv_cache_bytes(metadata, context_length);
    if (model_size_bytes == 0 || !kv) {
        estimate.explanation =
            "not estimated: the model's GGUF metadata does not report enough to size the KV cache (layer count and "
            "embedding length are required). Tuning will proceed and measure the real behaviour instead.";
        return estimate;
    }

    estimate.kv_cache_bytes = *kv;
    estimate.total_required_bytes = model_size_bytes + *kv + kOverheadBytes;

    const std::uint64_t vram = hardware.gpu.available ? hardware.gpu.total_vram_bytes : 0;
    const std::uint64_t ram = hardware.memory.total_bytes;
    const std::uint64_t usable_vram = vram > vram_headroom_bytes ? vram - vram_headroom_bytes : 0;

    std::ostringstream out;
    out << "model weights " << format_gib(model_size_bytes) << " + KV cache at " << context_length << " tokens "
        << format_gib(*kv) << " + ~" << format_gib(kOverheadBytes) << " overhead = "
        << format_gib(estimate.total_required_bytes) << " required";

    if (estimate.total_required_bytes <= usable_vram) {
        estimate.verdict = FitEstimate::Verdict::FitsInVram;
        out << ", against " << format_gib(vram) << " of VRAM: fits on the GPU.";
    } else if (estimate.total_required_bytes <= usable_vram + ram) {
        estimate.verdict = FitEstimate::Verdict::NeedsCpuOffload;
        out << ", against " << format_gib(vram) << " VRAM + " << format_gib(ram)
            << " RAM: fits only with part of the model in system RAM, which is what the offload search is for. "
               "Expect throughput well below what this GPU could do with a model that fits entirely in VRAM.";
    } else {
        estimate.verdict = FitEstimate::Verdict::ExceedsTotalMemory;
        out << ", against " << format_gib(vram) << " VRAM + " << format_gib(ram)
            << " RAM: this does not fit in VRAM and RAM combined. A smaller quantization or a smaller context is "
               "needed; no amount of tuning changes it.";
    }

    if (!metadata.attention_head_count || !metadata.attention_head_count_kv) {
        out << " (KV cache assumed without grouped-query attention -- the model does not report head counts -- so the "
               "figure above is an upper bound.)";
    }

    estimate.explanation = out.str();
    return estimate;
}

} // namespace fluxinfer::tuner
