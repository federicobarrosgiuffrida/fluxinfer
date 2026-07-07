#include "fluxinfer/llama/gguf_metadata.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <type_traits>
#include <unordered_map>

namespace fluxinfer::llama {

namespace {

// GGUF value type tags, per the GGUF spec (ggml-org/ggml gguf.h). Values
// are stable across GGUF versions 2 and 3.
enum class GgufValueType : std::uint32_t {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
};

constexpr std::uint32_t kMaxKnownValueType = 12;

// Defensive ceilings, not spec limits: real models never come close to
// these. They exist purely so a corrupted/malicious file fails fast with a
// clear error instead of attempting a huge allocation or a very long loop.
constexpr std::uint64_t kMaxKeyBytes = 4 * 1024;               // metadata keys are short dotted identifiers
constexpr std::uint64_t kMaxStringBytes = 16ULL * 1024 * 1024; // generous for e.g. chat templates
constexpr std::uint64_t kMaxKvCount = 1'000'000;
constexpr std::uint64_t kMaxArrayElements = 100'000'000;

// Returns the fixed on-disk size for a scalar GGUF value type, or
// std::nullopt for String (variable length) or Array (not a scalar).
std::optional<std::size_t> fixed_size_for_type(GgufValueType type) {
    switch (type) {
        case GgufValueType::UInt8:
        case GgufValueType::Int8:
        case GgufValueType::Bool:
            return 1;
        case GgufValueType::UInt16:
        case GgufValueType::Int16:
            return 2;
        case GgufValueType::UInt32:
        case GgufValueType::Int32:
        case GgufValueType::Float32:
            return 4;
        case GgufValueType::UInt64:
        case GgufValueType::Int64:
        case GgufValueType::Float64:
            return 8;
        case GgufValueType::String:
        case GgufValueType::Array:
            return std::nullopt;
    }
    return std::nullopt;
}

// A parsed scalar value (never an array -- arrays are walked and discarded
// without being materialized, see GgufReader::skip_array below).
struct GgufScalarValue {
    GgufValueType type = GgufValueType::UInt8;
    std::uint64_t u = 0;
    std::int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;

    std::optional<std::uint64_t> as_uint64() const {
        switch (type) {
            case GgufValueType::UInt8:
            case GgufValueType::UInt16:
            case GgufValueType::UInt32:
            case GgufValueType::UInt64:
                return u;
            case GgufValueType::Int8:
            case GgufValueType::Int16:
            case GgufValueType::Int32:
            case GgufValueType::Int64:
                return i >= 0 ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(i)) : std::nullopt;
            case GgufValueType::Bool:
                return b ? 1 : 0;
            default:
                return std::nullopt;
        }
    }

    std::optional<std::int64_t> as_int64() const {
        switch (type) {
            case GgufValueType::Int8:
            case GgufValueType::Int16:
            case GgufValueType::Int32:
            case GgufValueType::Int64:
                return i;
            case GgufValueType::UInt8:
            case GgufValueType::UInt16:
            case GgufValueType::UInt32:
            case GgufValueType::UInt64:
                return static_cast<std::int64_t>(u);
            default:
                return std::nullopt;
        }
    }

    std::optional<std::string> as_string() const {
        if (type == GgufValueType::String) return s;
        return std::nullopt;
    }
};

// Bounded, fail-safe reader over a stream of known total size. Every read
// validates the stream state and, for skips/seeks, the resulting position
// against the known file size, so a truncated or malformed file is
// detected immediately rather than producing garbage or an out-of-bounds
// access.
class GgufReader {
public:
    GgufReader(std::istream& stream, std::uint64_t size) : stream_(stream), size_(size) {}

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    bool read_raw(void* dest, std::size_t n) {
        if (!ok_) return false;
        if (position() + n > size_) {
            return fail("unexpected end of file");
        }
        stream_.read(static_cast<char*>(dest), static_cast<std::streamsize>(n));
        if (!stream_) {
            return fail("read error (truncated file?)");
        }
        return true;
    }

    template <typename T>
    bool read_pod(T& out) {
        static_assert(std::is_trivially_copyable_v<T>);
        return read_raw(&out, sizeof(T));
    }

    // Reads a GGUF string: u64 length prefix + that many raw bytes.
    bool read_string(std::string& out, std::uint64_t max_len) {
        std::uint64_t len = 0;
        if (!read_pod(len)) return false;
        if (len > max_len) {
            return fail("string length " + std::to_string(len) + " exceeds sanity limit");
        }
        out.resize(static_cast<std::size_t>(len));
        if (len == 0) return true;
        return read_raw(out.data(), static_cast<std::size_t>(len));
    }

    // Advances the stream by `n` bytes without materializing them, used to
    // discard array/tensor-adjacent data we don't care about. Bounded by
    // the known file size regardless of what seekg() itself would allow.
    bool skip_bytes(std::uint64_t n) {
        if (!ok_) return false;
        if (position() + n > size_) {
            return fail("unexpected end of file while skipping");
        }
        stream_.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        if (!stream_) {
            return fail("seek error (truncated file?)");
        }
        return true;
    }

    std::uint64_t position() {
        const auto pos = stream_.tellg();
        if (pos < 0) {
            fail("could not determine stream position");
            return size_; // force subsequent bounds checks to fail closed
        }
        return static_cast<std::uint64_t>(pos);
    }

    bool read_scalar(GgufValueType type, GgufScalarValue& out) {
        out.type = type;
        switch (type) {
            case GgufValueType::UInt8: {
                std::uint8_t v;
                if (!read_pod(v)) return false;
                out.u = v;
                return true;
            }
            case GgufValueType::Int8: {
                std::int8_t v;
                if (!read_pod(v)) return false;
                out.i = v;
                return true;
            }
            case GgufValueType::UInt16: {
                std::uint16_t v;
                if (!read_pod(v)) return false;
                out.u = v;
                return true;
            }
            case GgufValueType::Int16: {
                std::int16_t v;
                if (!read_pod(v)) return false;
                out.i = v;
                return true;
            }
            case GgufValueType::UInt32: {
                std::uint32_t v;
                if (!read_pod(v)) return false;
                out.u = v;
                return true;
            }
            case GgufValueType::Int32: {
                std::int32_t v;
                if (!read_pod(v)) return false;
                out.i = v;
                return true;
            }
            case GgufValueType::Float32: {
                float v;
                if (!read_pod(v)) return false;
                out.f = v;
                return true;
            }
            case GgufValueType::Bool: {
                std::uint8_t v;
                if (!read_pod(v)) return false;
                out.b = v != 0;
                return true;
            }
            case GgufValueType::UInt64: {
                std::uint64_t v;
                if (!read_pod(v)) return false;
                out.u = v;
                return true;
            }
            case GgufValueType::Int64: {
                std::int64_t v;
                if (!read_pod(v)) return false;
                out.i = v;
                return true;
            }
            case GgufValueType::Float64: {
                double v;
                if (!read_pod(v)) return false;
                out.f = v;
                return true;
            }
            case GgufValueType::String:
                return read_string(out.s, kMaxStringBytes);
            case GgufValueType::Array:
                return fail("nested arrays are not valid GGUF");
        }
        return fail("unreachable value type");
    }

    // Reads an array's element type + count, then discards every element
    // without storing it: FluxInfer has no need for array-valued metadata
    // (tokenizer vocabularies etc.), but must still walk past them
    // correctly to keep the stream position valid for subsequent keys.
    bool skip_array() {
        std::uint32_t raw_element_type = 0;
        if (!read_pod(raw_element_type)) return false;
        if (raw_element_type > kMaxKnownValueType) {
            return fail("array element type " + std::to_string(raw_element_type) + " is not a known GGUF value type");
        }
        const auto element_type = static_cast<GgufValueType>(raw_element_type);
        if (element_type == GgufValueType::Array) {
            return fail("nested arrays are not valid GGUF");
        }

        std::uint64_t count = 0;
        if (!read_pod(count)) return false;
        if (count > kMaxArrayElements) {
            return fail("array element count " + std::to_string(count) + " exceeds sanity limit");
        }

        if (element_type == GgufValueType::String) {
            for (std::uint64_t i = 0; i < count; ++i) {
                std::string discard;
                if (!read_string(discard, kMaxStringBytes)) return false;
            }
            return true;
        }

        const std::optional<std::size_t> element_size = fixed_size_for_type(element_type);
        if (!element_size) {
            return fail("array element type has no fixed size");
        }
        return skip_bytes(count * static_cast<std::uint64_t>(*element_size));
    }

private:
    bool fail(const std::string& message) {
        if (ok_) {
            ok_ = false;
            error_ = message;
        }
        return false;
    }

    std::istream& stream_;
    std::uint64_t size_;
    bool ok_ = true;
    std::string error_;
};

// Best-effort mirror of llama.cpp's `enum llama_ftype` (llama.h), covering
// the common quantization schemes. Not exhaustive by design -- FluxInfer
// does not reimplement llama.cpp's model loading or quantization logic,
// this is purely a display label.
const std::unordered_map<std::int64_t, std::string>& file_type_labels() {
    static const std::unordered_map<std::int64_t, std::string> labels = {
        {0, "F32"},        {1, "F16"},        {2, "Q4_0"},       {3, "Q4_1"},
        {7, "Q8_0"},        {8, "Q5_0"},        {9, "Q5_1"},       {10, "Q2_K"},
        {11, "Q3_K_S"},     {12, "Q3_K_M"},     {13, "Q3_K_L"},    {14, "Q4_K_S"},
        {15, "Q4_K_M"},     {16, "Q5_K_S"},     {17, "Q5_K_M"},    {18, "Q6_K"},
        {19, "IQ2_XXS"},    {20, "IQ2_XS"},     {21, "Q2_K_S"},    {22, "IQ3_XS"},
        {23, "IQ3_XXS"},    {24, "IQ1_S"},      {25, "IQ4_NL"},    {26, "IQ3_S"},
        {27, "IQ3_M"},      {28, "IQ2_S"},      {29, "IQ2_M"},     {30, "IQ4_XS"},
        {31, "IQ1_M"},      {32, "BF16"},       {36, "TQ1_0"},     {37, "TQ2_0"},
    };
    return labels;
}

} // namespace

std::string describe_file_type(std::int64_t file_type) {
    const auto& labels = file_type_labels();
    const auto it = labels.find(file_type);
    if (it != labels.end()) {
        return it->second;
    }
    return "unknown (ftype " + std::to_string(file_type) + ")";
}

GgufParseResult parse_gguf_metadata(std::istream& stream, std::uint64_t stream_size) {
    GgufParseResult result;
    GgufReader reader(stream, stream_size);

    std::array<char, 4> magic{};
    if (!reader.read_raw(magic.data(), magic.size())) {
        result.error = reader.error();
        return result;
    }
    if (std::memcmp(magic.data(), "GGUF", 4) != 0) {
        result.error = "not a GGUF file (bad magic)";
        return result;
    }

    std::uint32_t version = 0;
    if (!reader.read_pod(version)) {
        result.error = reader.error();
        return result;
    }
    if (version != 2 && version != 3) {
        result.error = "unsupported GGUF version " + std::to_string(version) + " (only 2 and 3 are supported)";
        return result;
    }
    result.metadata.gguf_version = version;

    std::uint64_t tensor_count = 0;
    if (!reader.read_pod(tensor_count)) {
        result.error = reader.error();
        return result;
    }

    std::uint64_t kv_count = 0;
    if (!reader.read_pod(kv_count)) {
        result.error = reader.error();
        return result;
    }
    if (kv_count > kMaxKvCount) {
        result.error = "metadata_kv_count " + std::to_string(kv_count) + " exceeds sanity limit (possibly corrupt file)";
        return result;
    }

    std::map<std::string, GgufScalarValue> scalars;

    for (std::uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        if (!reader.read_string(key, kMaxKeyBytes)) {
            result.error = reader.error();
            return result;
        }

        std::uint32_t raw_type = 0;
        if (!reader.read_pod(raw_type)) {
            result.error = reader.error();
            return result;
        }
        if (raw_type > kMaxKnownValueType) {
            result.error = "key '" + key + "' has unknown value type " + std::to_string(raw_type);
            return result;
        }
        const auto type = static_cast<GgufValueType>(raw_type);

        if (type == GgufValueType::Array) {
            if (!reader.skip_array()) {
                result.error = reader.error();
                return result;
            }
            continue;
        }

        GgufScalarValue value;
        if (!reader.read_scalar(type, value)) {
            result.error = reader.error();
            return result;
        }
        scalars.emplace(std::move(key), std::move(value));
    }

    // We deliberately stop here: the tensor info section that follows
    // (names, shapes, per-tensor data offsets) is not needed for any of
    // the metadata FluxInfer cares about, and reading it would mean
    // walking `tensor_count` more variable-length records for no benefit.
    (void)tensor_count;

    auto find_string = [&](const std::string& key) -> std::optional<std::string> {
        const auto it = scalars.find(key);
        if (it == scalars.end()) return std::nullopt;
        return it->second.as_string();
    };
    auto find_uint = [&](const std::string& key) -> std::optional<std::uint64_t> {
        const auto it = scalars.find(key);
        if (it == scalars.end()) return std::nullopt;
        return it->second.as_uint64();
    };
    auto find_int = [&](const std::string& key) -> std::optional<std::int64_t> {
        const auto it = scalars.find(key);
        if (it == scalars.end()) return std::nullopt;
        return it->second.as_int64();
    };

    result.metadata.architecture = find_string("general.architecture").value_or(std::string());
    result.metadata.model_name = find_string("general.name").value_or(std::string());
    result.metadata.file_type = find_int("general.file_type");
    if (result.metadata.file_type) {
        result.metadata.quantization_label = describe_file_type(*result.metadata.file_type);
    }

    if (!result.metadata.architecture.empty()) {
        const std::string prefix = result.metadata.architecture + ".";
        result.metadata.block_count = find_uint(prefix + "block_count");
        result.metadata.context_length = find_uint(prefix + "context_length");
        result.metadata.embedding_length = find_uint(prefix + "embedding_length");
        result.metadata.expert_count = find_uint(prefix + "expert_count");
        result.metadata.expert_used_count = find_uint(prefix + "expert_used_count");
    }

    result.valid = true;
    return result;
}

GgufParseResult parse_gguf_metadata(const std::filesystem::path& model_path) {
    GgufParseResult result;

    std::error_code ec;
    if (!std::filesystem::is_regular_file(model_path, ec)) {
        result.error = "model file not found: " + model_path.string();
        return result;
    }

    const std::uint64_t size = std::filesystem::file_size(model_path, ec);
    if (ec) {
        result.error = "could not read model file size: " + ec.message();
        return result;
    }

    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        result.error = "could not open model file for reading: " + model_path.string();
        return result;
    }

    return parse_gguf_metadata(file, size);
}

} // namespace fluxinfer::llama
