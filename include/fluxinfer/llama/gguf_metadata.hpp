#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <optional>
#include <string>

namespace fluxinfer::llama {

struct GgufMetadata {
    std::uint32_t gguf_version = 0;
    std::string architecture; // general.architecture
    std::string model_name;   // general.name

    // All of these come from "<architecture>.<key>" entries and are only
    // populated when both the architecture string and the corresponding
    // key were found and correctly typed. A missing value means exactly
    // that: FluxInfer never invents a number here.
    std::optional<std::uint64_t> block_count;       // <arch>.block_count (transformer layer count)
    std::optional<std::uint64_t> context_length;     // <arch>.context_length
    std::optional<std::uint64_t> embedding_length;    // <arch>.embedding_length
    std::optional<std::uint64_t> expert_count;        // <arch>.expert_count
    std::optional<std::uint64_t> expert_used_count;   // <arch>.expert_used_count

    std::optional<std::int64_t> file_type; // general.file_type (llama.cpp quantization enum)
    std::string quantization_label;        // best-effort human label derived from file_type
};

struct GgufParseResult {
    bool valid = false;
    GgufMetadata metadata;
    std::string error; // populated when valid == false
};

// Reads only the GGUF header and key-value metadata section of
// `model_path` -- never tensor data, and never the full file into memory.
// Rejects malformed, truncated, or version-unsupported files by returning
// valid == false with a human-readable error; never throws or crashes on
// untrusted input.
GgufParseResult parse_gguf_metadata(const std::filesystem::path& model_path);

// Same parsing logic applied to an already-open input stream of known
// total size. Exposed primarily for tests, which use small synthetic
// fixtures instead of multi-gigabyte real model files, but also usable
// directly if a caller already has a stream open.
GgufParseResult parse_gguf_metadata(std::istream& stream, std::uint64_t stream_size);

// Best-effort mapping from a llama.cpp `general.file_type` integer to a
// human-readable quantization label (e.g. 15 -> "Q4_K_M"). This mirrors a
// snapshot of llama.cpp's `llama_ftype` enum and is not guaranteed to be
// exhaustive or current -- unknown values return "unknown (ftype N)"
// rather than a guess.
std::string describe_file_type(std::int64_t file_type);

} // namespace fluxinfer::llama
