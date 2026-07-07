#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/llama/llama_runner.hpp"

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
