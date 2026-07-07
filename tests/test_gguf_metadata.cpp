#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/llama/gguf_metadata.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <random>

using namespace fluxinfer::llama;

namespace {

// Minimal builder for synthetic GGUF byte streams, used to exercise the
// parser without needing real (multi-gigabyte) model files. Mirrors the
// GGUF binary layout directly rather than going through the parser's own
// types, so a bug in the parser can't also hide in the fixture.
class GgufBuilder {
public:
    explicit GgufBuilder(std::uint32_t version = 3) {
        data_.append("GGUF", 4);
        append_pod(version);
        append_pod(std::uint64_t{0}); // tensor_count: unused by the parser
        kv_count_offset_ = data_.size();
        append_pod(std::uint64_t{0}); // placeholder, patched in build()
    }

    void add_u8(const std::string& key, std::uint8_t v) { add_scalar(key, 0, v); }
    void add_i8(const std::string& key, std::int8_t v) { add_scalar(key, 1, v); }
    void add_u16(const std::string& key, std::uint16_t v) { add_scalar(key, 2, v); }
    void add_i16(const std::string& key, std::int16_t v) { add_scalar(key, 3, v); }
    void add_u32(const std::string& key, std::uint32_t v) { add_scalar(key, 4, v); }
    void add_i32(const std::string& key, std::int32_t v) { add_scalar(key, 5, v); }
    void add_f32(const std::string& key, float v) { add_scalar(key, 6, v); }
    void add_bool(const std::string& key, bool v) { add_scalar(key, 7, static_cast<std::uint8_t>(v ? 1 : 0)); }
    void add_u64(const std::string& key, std::uint64_t v) { add_scalar(key, 10, v); }
    void add_i64(const std::string& key, std::int64_t v) { add_scalar(key, 11, v); }
    void add_f64(const std::string& key, double v) { add_scalar(key, 12, v); }

    void add_string(const std::string& key, const std::string& value) {
        append_string(key);
        append_pod(std::uint32_t{8}); // String
        append_string(value);
        ++kv_count_;
    }

    void add_string_array(const std::string& key, const std::vector<std::string>& values) {
        append_string(key);
        append_pod(std::uint32_t{9}); // Array
        append_pod(std::uint32_t{8}); // element type: String
        append_pod(static_cast<std::uint64_t>(values.size()));
        for (const auto& v : values) append_string(v);
        ++kv_count_;
    }

    void add_u32_array(const std::string& key, const std::vector<std::uint32_t>& values) {
        append_string(key);
        append_pod(std::uint32_t{9}); // Array
        append_pod(std::uint32_t{4}); // element type: UInt32
        append_pod(static_cast<std::uint64_t>(values.size()));
        for (auto v : values) append_pod(v);
        ++kv_count_;
    }

    // Escape hatch used by the "kv_count wildly larger than the file"
    // fixture below to desynchronize the declared count from reality.
    void override_kv_count(std::uint64_t n) { kv_count_override_ = n; }

    std::string build() const {
        std::string out = data_;
        const std::uint64_t kv_count = kv_count_override_.value_or(kv_count_);
        std::memcpy(out.data() + kv_count_offset_, &kv_count, sizeof(kv_count));
        return out;
    }

private:
    template <typename T>
    void add_scalar(const std::string& key, std::uint32_t type, T value) {
        append_string(key);
        append_pod(type);
        append_pod(value);
        ++kv_count_;
    }

    void append_string(const std::string& s) {
        append_pod(static_cast<std::uint64_t>(s.size()));
        data_.append(s);
    }

    template <typename T>
    void append_pod(T v) {
        data_.append(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    std::string data_;
    std::size_t kv_count_offset_;
    std::uint64_t kv_count_ = 0;
    std::optional<std::uint64_t> kv_count_override_;
};

GgufParseResult parse(const std::string& bytes) {
    std::istringstream stream(bytes, std::ios::binary);
    return parse_gguf_metadata(stream, bytes.size());
}

} // namespace

TEST_CASE("parses a minimal valid fixture with qwen-style architecture keys", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    b.add_string("general.name", "Test Model");
    b.add_u32("testarch.block_count", 32);
    b.add_u32("testarch.context_length", 4096);
    b.add_u32("testarch.embedding_length", 2048);
    b.add_i32("general.file_type", 15);

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    CHECK(result.error.empty());
    CHECK(result.metadata.gguf_version == 3);
    CHECK(result.metadata.architecture == "testarch");
    CHECK(result.metadata.model_name == "Test Model");
    REQUIRE(result.metadata.block_count.has_value());
    CHECK(result.metadata.block_count.value() == 32);
    REQUIRE(result.metadata.context_length.has_value());
    CHECK(result.metadata.context_length.value() == 4096);
    REQUIRE(result.metadata.embedding_length.has_value());
    CHECK(result.metadata.embedding_length.value() == 2048);
    REQUIRE(result.metadata.file_type.has_value());
    CHECK(result.metadata.file_type.value() == 15);
    CHECK(result.metadata.quantization_label == "Q4_K_M");
}

TEST_CASE("supports GGUF version 2", "[gguf]") {
    GgufBuilder b(2);
    b.add_string("general.architecture", "testarch");
    b.add_u32("testarch.block_count", 12);

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    CHECK(result.metadata.gguf_version == 2);
    CHECK(result.metadata.block_count == 12);
}

TEST_CASE("reads MoE expert fields when present", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.architecture", "mixtral");
    b.add_u32("mixtral.block_count", 32);
    b.add_u32("mixtral.expert_count", 8);
    b.add_u32("mixtral.expert_used_count", 2);

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    REQUIRE(result.metadata.expert_count.has_value());
    CHECK(result.metadata.expert_count.value() == 8);
    REQUIRE(result.metadata.expert_used_count.has_value());
    CHECK(result.metadata.expert_used_count.value() == 2);
}

TEST_CASE("handles every scalar value type without desyncing subsequent keys", "[gguf]") {
    GgufBuilder b;
    b.add_u8("k_u8", 200);
    b.add_i8("k_i8", -100);
    b.add_u16("k_u16", 60000);
    b.add_i16("k_i16", -30000);
    b.add_u32("k_u32", 4000000000u);
    b.add_i32("k_i32", -2000000000);
    b.add_f32("k_f32", 3.5f);
    b.add_bool("k_bool", true);
    b.add_u64("k_u64", 18000000000000000000ULL);
    b.add_i64("k_i64", -9000000000000000000LL);
    b.add_f64("k_f64", 2.718281828);
    b.add_string("general.architecture", "testarch");
    b.add_u32("testarch.block_count", 7); // must still parse correctly after all the above

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    REQUIRE(result.metadata.block_count.has_value());
    CHECK(result.metadata.block_count.value() == 7);
}

TEST_CASE("skips a string array without desyncing subsequent keys", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    b.add_string_array("tokenizer.ggml.tokens", {"<s>", "</s>", "hello", "world", ""});
    b.add_u32("testarch.block_count", 40);

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    REQUIRE(result.metadata.block_count.has_value());
    CHECK(result.metadata.block_count.value() == 40);
}

TEST_CASE("skips a numeric array without desyncing subsequent keys", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    b.add_u32_array("tokenizer.ggml.token_type", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    b.add_u32("testarch.block_count", 24);

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    REQUIRE(result.metadata.block_count.has_value());
    CHECK(result.metadata.block_count.value() == 24);
}

TEST_CASE("block_count is nullopt (not invented) when the architecture-specific key is missing", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    b.add_string("general.name", "No Layer Info Model");
    // No testarch.block_count key at all.

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid); // a well-formed file missing optional metadata is not a parse error
    CHECK_FALSE(result.metadata.block_count.has_value());
    CHECK_FALSE(result.metadata.expert_count.has_value());
}

TEST_CASE("architecture and dependent fields are empty/nullopt when general.architecture is absent", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.name", "Nameless Architecture Model");

    GgufParseResult result = parse(b.build());
    REQUIRE(result.valid);
    CHECK(result.metadata.architecture.empty());
    CHECK_FALSE(result.metadata.block_count.has_value());
}

TEST_CASE("rejects a bad magic", "[gguf][malformed]") {
    GgufBuilder b;
    std::string bytes = b.build();
    bytes[0] = 'X'; // corrupt "GGUF" -> "XGUF"

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
    CHECK(result.error.find("magic") != std::string::npos);
}

TEST_CASE("rejects an unsupported GGUF version", "[gguf][malformed]") {
    GgufBuilder b(99);
    GgufParseResult result = parse(b.build());
    CHECK_FALSE(result.valid);
    CHECK(result.error.find("version") != std::string::npos);
}

TEST_CASE("rejects GGUF version 1 explicitly (unsupported layout)", "[gguf][malformed]") {
    GgufBuilder b(1);
    GgufParseResult result = parse(b.build());
    CHECK_FALSE(result.valid);
}

TEST_CASE("rejects a file truncated right after the header", "[gguf][malformed][truncated]") {
    // Just "GGUF" + version, nothing else.
    std::string bytes = "GGUF";
    std::uint32_t version = 3;
    bytes.append(reinterpret_cast<const char*>(&version), sizeof(version));

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("rejects a file truncated in the middle of a key string", "[gguf][malformed][truncated]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    std::string bytes = b.build();
    bytes.resize(bytes.size() - 20); // cut off mid-value

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("rejects a declared kv_count wildly larger than the file could hold", "[gguf][malformed]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    b.override_kv_count(5'000'000);

    GgufParseResult result = parse(b.build());
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("rejects an absurd string length instead of allocating it", "[gguf][malformed]") {
    // Hand-crafted: magic + version + tensor_count(0) + kv_count(1), then a
    // key whose declared length claims to be enormous.
    std::string bytes = "GGUF";
    std::uint32_t version = 3;
    std::uint64_t tensor_count = 0;
    std::uint64_t kv_count = 1;
    std::uint64_t huge_len = 0xFFFFFFFFFFFFFFFFULL;
    bytes.append(reinterpret_cast<const char*>(&version), sizeof(version));
    bytes.append(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));
    bytes.append(reinterpret_cast<const char*>(&kv_count), sizeof(kv_count));
    bytes.append(reinterpret_cast<const char*>(&huge_len), sizeof(huge_len));
    bytes.append("only a few bytes follow, nowhere near huge_len");

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("rejects an absurd array element count instead of looping forever", "[gguf][malformed]") {
    GgufBuilder b;
    // Manually build a key with an Array value whose declared count is
    // huge, since GgufBuilder's helpers always write a matching number of
    // real elements.
    std::string key = "bad.array";
    std::string bytes = b.build(); // header + kv_count(0) so far, we'll append raw KV below and fix count

    std::string kv;
    auto append_pod = [&](auto v) { kv.append(reinterpret_cast<const char*>(&v), sizeof(v)); };
    append_pod(static_cast<std::uint64_t>(key.size()));
    kv += key;
    append_pod(std::uint32_t{9});                    // Array
    append_pod(std::uint32_t{4});                    // element type UInt32
    append_pod(static_cast<std::uint64_t>(0xFFFFFFFFFFFFFFFFULL)); // element count: absurd

    bytes += kv;
    // Patch kv_count to 1 at its known offset (16 bytes into the header: 4 magic + 4 version + 8 tensor_count).
    std::uint64_t kv_count = 1;
    std::memcpy(bytes.data() + 16, &kv_count, sizeof(kv_count));

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("rejects nested arrays", "[gguf][malformed]") {
    std::string bytes;
    bytes.append("GGUF", 4);
    auto append_pod = [&](auto v) { bytes.append(reinterpret_cast<const char*>(&v), sizeof(v)); };
    append_pod(std::uint32_t{3});
    append_pod(std::uint64_t{0}); // tensor_count
    append_pod(std::uint64_t{1}); // kv_count

    std::string key = "bad.nested";
    append_pod(static_cast<std::uint64_t>(key.size()));
    bytes += key;
    append_pod(std::uint32_t{9}); // Array
    append_pod(std::uint32_t{9}); // element type: Array (invalid, arrays can't nest)
    append_pod(std::uint64_t{0}); // count

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
}

TEST_CASE("rejects an unknown value type", "[gguf][malformed]") {
    std::string bytes;
    bytes.append("GGUF", 4);
    auto append_pod = [&](auto v) { bytes.append(reinterpret_cast<const char*>(&v), sizeof(v)); };
    append_pod(std::uint32_t{3});
    append_pod(std::uint64_t{0});
    append_pod(std::uint64_t{1});

    std::string key = "bad.type";
    append_pod(static_cast<std::uint64_t>(key.size()));
    bytes += key;
    append_pod(std::uint32_t{255}); // not a valid GGUF value type

    GgufParseResult result = parse(bytes);
    CHECK_FALSE(result.valid);
}

TEST_CASE("parse_gguf_metadata(path) reads a real temp file and reports missing files clearly", "[gguf]") {
    GgufBuilder b;
    b.add_string("general.architecture", "testarch");
    b.add_u32("testarch.block_count", 16);

    static std::mt19937_64 rng(std::random_device{}());
    std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("fluxinfer_gguf_test_" + std::to_string(rng()));
    std::filesystem::create_directories(dir);
    std::filesystem::path model_path = dir / "fixture.gguf";

    {
        std::ofstream file(model_path, std::ios::binary);
        std::string bytes = b.build();
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    GgufParseResult result = parse_gguf_metadata(model_path);
    REQUIRE(result.valid);
    CHECK(result.metadata.block_count == 16);

    GgufParseResult missing = parse_gguf_metadata(dir / "does-not-exist.gguf");
    CHECK_FALSE(missing.valid);
    CHECK_FALSE(missing.error.empty());

    std::filesystem::remove_all(dir);
}

TEST_CASE("describe_file_type maps known values and reports unknown ones honestly", "[gguf]") {
    CHECK(describe_file_type(7) == "Q8_0");
    CHECK(describe_file_type(15) == "Q4_K_M");
    CHECK(describe_file_type(0) == "F32");
    CHECK(describe_file_type(9999).find("unknown") != std::string::npos);
}
