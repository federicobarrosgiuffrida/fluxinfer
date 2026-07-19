#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/llama/llama_runner.hpp"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

using namespace fluxinfer::llama;

TEST_CASE("extract_supported_flags finds long flags in realistic --help text", "[flags]") {
    const std::string help_text = R"(
usage: llama-bench [options]

options:
  -h, --help
  -m, --model <filename>       model path (default: models/7B/ggml-model-q4_0.gguf)
  -p, --n-prompt <n>           number of prompt tokens
  -n, --n-gen <n>              number of generated tokens
  -b, --batch-size <n>         batch size for prompt processing
  -ub, --ubatch-size <n>       physical maximum batch size
  -t, --threads <n>            number of threads
  -ngl, --n-gpu-layers <n>     number of layers to offload to the GPU
  -ctk, --cache-type-k <t>     KV cache data type for K
  -ctv, --cache-type-v <t>     KV cache data type for V
  -o, --output <csv|json|md|sql>   output format
  -r, --repetitions <n>        number of repetitions per test
)";

    std::set<std::string> flags = extract_supported_flags(help_text);

    CHECK(flags.count("--help") == 1);
    CHECK(flags.count("--model") == 1);
    CHECK(flags.count("--n-prompt") == 1);
    CHECK(flags.count("--n-gen") == 1);
    CHECK(flags.count("--batch-size") == 1);
    CHECK(flags.count("--ubatch-size") == 1);
    CHECK(flags.count("--threads") == 1);
    CHECK(flags.count("--n-gpu-layers") == 1);
    CHECK(flags.count("--cache-type-k") == 1);
    CHECK(flags.count("--cache-type-v") == 1);
    CHECK(flags.count("--output") == 1);
    CHECK(flags.count("--repetitions") == 1);

    // Never assume a flag exists just because it's common elsewhere.
    CHECK(flags.count("--flash-attn") == 0);
    CHECK(flags.count("--split-mode") == 0);
}

TEST_CASE("extract_supported_flags on an older build without -o/json support", "[flags]") {
    const std::string help_text = R"(
options:
  -m, --model <filename>
  -p, --n-prompt <n>
  -n, --n-gen <n>
  -t, --threads <n>
  -ngl, --n-gpu-layers <n>
)";

    std::set<std::string> flags = extract_supported_flags(help_text);
    CHECK(flags.count("--output") == 0);
    CHECK(flags.count("--cache-type-k") == 0);
    CHECK(flags.count("--n-gpu-layers") == 1);
}

TEST_CASE("extract_supported_flags on empty text returns an empty set", "[flags]") {
    CHECK(extract_supported_flags("").empty());
}

TEST_CASE("supports_flag looks up membership in a flag set", "[flags]") {
    std::set<std::string> flags = {"--n-gpu-layers", "--batch-size"};
    CHECK(supports_flag(flags, "--n-gpu-layers"));
    CHECK_FALSE(supports_flag(flags, "--flash-attn"));
}

TEST_CASE("merge_user_overrides replaces profile flags instead of duplicating them", "[flags]") {
    // Shaped like build_config_arguments() output for a tuned profile.
    const std::vector<std::string> profile_args = {"-m",  "model.gguf", "-t", "9",    "-ngl", "20",
                                                    "-b",  "2048",       "-ub", "512", "-c",   "4096"};

    SECTION("short-form override drops the profile's own value") {
        std::vector<std::string> overridden;
        std::vector<std::string> merged = merge_user_overrides(profile_args, {"-ngl", "99"}, &overridden);

        // Exactly one -ngl on the final command line, and it is the user's.
        CHECK(std::count(merged.begin(), merged.end(), std::string("-ngl")) == 1);
        auto it = std::find(merged.begin(), merged.end(), std::string("-ngl"));
        REQUIRE(it != merged.end());
        REQUIRE(std::next(it) != merged.end());
        CHECK(*std::next(it) == "99");
        CHECK(overridden == std::vector<std::string>{"-ngl"});
    }

    SECTION("long-form user flag overrides the short-form profile flag") {
        std::vector<std::string> merged = merge_user_overrides(profile_args, {"--n-gpu-layers", "99"});

        CHECK(std::count(merged.begin(), merged.end(), std::string("-ngl")) == 0);
        CHECK(std::count(merged.begin(), merged.end(), std::string("--n-gpu-layers")) == 1);
    }

    SECTION("--flag=value syntax is recognised as an override") {
        std::vector<std::string> merged = merge_user_overrides(profile_args, {"--ctx-size=32768"});

        CHECK(std::count(merged.begin(), merged.end(), std::string("-c")) == 0);
        CHECK(std::count(merged.begin(), merged.end(), std::string("--ctx-size=32768")) == 1);
    }

    SECTION("unrelated user flags leave the profile configuration intact") {
        std::vector<std::string> overridden;
        std::vector<std::string> merged =
            merge_user_overrides(profile_args, {"--jinja", "--no-mmap"}, &overridden);

        CHECK(overridden.empty());
        // Whole profile preserved, user flags appended after it.
        REQUIRE(merged.size() == profile_args.size() + 2);
        CHECK(std::equal(profile_args.begin(), profile_args.end(), merged.begin()));
        CHECK(merged[merged.size() - 2] == "--jinja");
        CHECK(merged.back() == "--no-mmap");
    }

    SECTION("several overrides at once, mixed with pass-through flags") {
        std::vector<std::string> overridden;
        std::vector<std::string> merged = merge_user_overrides(
            profile_args, {"-ngl", "99", "--ctx-size", "32768", "--jinja"}, &overridden);

        CHECK(overridden == std::vector<std::string>{"-ngl", "-c"});
        CHECK(std::count(merged.begin(), merged.end(), std::string("-c")) == 0);
        // Untouched profile flags survive.
        CHECK(std::count(merged.begin(), merged.end(), std::string("-t")) == 1);
        CHECK(std::count(merged.begin(), merged.end(), std::string("-ub")) == 1);
    }

    SECTION("empty user arguments are a no-op") {
        CHECK(merge_user_overrides(profile_args, {}) == profile_args);
    }
}
