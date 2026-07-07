#include "fluxinfer/hardware/vram_sampler.hpp"

#include "fluxinfer/hardware/gpu_probe.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace fluxinfer::hardware {

struct VramSampler::Impl {
    std::atomic<bool> running{false};
    std::atomic<bool> have_sample{false};
    std::atomic<std::uint64_t> peak_used_bytes{0};
    std::thread worker;
};

VramSampler::VramSampler() : impl_(std::make_unique<Impl>()) {}

VramSampler::~VramSampler() { stop(); }

void VramSampler::start(unsigned interval_ms) {
    if (impl_->running.exchange(true)) {
        return; // already running
    }
    impl_->have_sample = false;
    impl_->peak_used_bytes = 0;

    Impl* impl = impl_.get();
    impl->worker = std::thread([impl, interval_ms] {
        while (impl->running.load(std::memory_order_relaxed)) {
            GpuInfo gpu = probe_gpu();
            if (gpu.available && gpu.total_vram_bytes >= gpu.available_vram_bytes) {
                const std::uint64_t used = gpu.total_vram_bytes - gpu.available_vram_bytes;
                std::uint64_t previous_peak = impl->peak_used_bytes.load(std::memory_order_relaxed);
                while (used > previous_peak &&
                       !impl->peak_used_bytes.compare_exchange_weak(previous_peak, used, std::memory_order_relaxed)) {
                    // retry with the updated previous_peak
                }
                impl->have_sample = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    });
}

void VramSampler::stop() {
    if (!impl_->running.exchange(false)) {
        return; // wasn't running
    }
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
}

std::optional<std::uint64_t> VramSampler::peak_used_bytes() const {
    if (!impl_->have_sample.load(std::memory_order_relaxed)) {
        return std::nullopt;
    }
    return impl_->peak_used_bytes.load(std::memory_order_relaxed);
}

} // namespace fluxinfer::hardware
