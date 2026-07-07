#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/llama/benchmark_parser.hpp"

using namespace fluxinfer::llama;

TEST_CASE("parse_llama_bench_json parses a realistic pp+tg JSON array", "[parser]") {
    const std::string json_output = R"(
some warmup log line that isn't JSON
[
  {
    "build_commit": "abc1234", "build_number": 1234, "cpu_info": "some cpu",
    "gpu_info": "NVIDIA GeForce RTX 5060", "model_filename": "model.gguf",
    "model_type": "llama 7B", "model_size": 4000000000, "model_n_params": 7000000000,
    "n_batch": 512, "n_ubatch": 256, "n_threads": 8, "type_k": "f16", "type_v": "f16",
    "n_gpu_layers": 31, "n_prompt": 512, "n_gen": 0,
    "avg_ns": 415000000, "stddev_ns": 5000000, "avg_ts": 1234.5, "stddev_ts": 12.3
  },
  {
    "build_commit": "abc1234", "build_number": 1234, "n_gpu_layers": 31,
    "n_prompt": 0, "n_gen": 128, "avg_ns": 2800000000, "stddev_ts": 1.2,
    "avg_ts": 45.6, "stddev_ns": 30000000
  }
]
)";

    BenchmarkParseResult result = parse_llama_bench_json(json_output);
    REQUIRE(result.valid);
    REQUIRE(result.prompt_tokens_per_second.has_value());
    REQUIRE(result.generation_tokens_per_second.has_value());
    CHECK(result.prompt_tokens_per_second.value() == Catch::Approx(1234.5));
    CHECK(result.generation_tokens_per_second.value() == Catch::Approx(45.6));
    CHECK(result.measurements.size() == 2);
}

TEST_CASE("parse_llama_bench_json extracts per-repetition samples_ts for -r N runs", "[parser]") {
    const std::string json_output = R"([
  {
    "n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 5.0,
    "samples_ns": [ 5000000000, 5200000000, 4900000000 ],
    "samples_ts": [ 102.4, 98.5, 99.1 ]
  },
  {
    "n_prompt": 0, "n_gen": 128, "avg_ts": 13.7, "stddev_ts": 0.3,
    "samples_ns": [ 9300000000, 9350000000, 9280000000 ],
    "samples_ts": [ 13.76, 13.69, 13.79 ]
  }
])";

    BenchmarkParseResult result = parse_llama_bench_json(json_output);
    REQUIRE(result.valid);
    REQUIRE(result.prompt_tps_samples.size() == 3);
    REQUIRE(result.generation_tps_samples.size() == 3);
    CHECK(result.prompt_tps_samples[0] == Catch::Approx(102.4));
    CHECK(result.generation_tps_samples[2] == Catch::Approx(13.79));
}

TEST_CASE("parse_llama_bench_json leaves samples empty for a single-repetition run", "[parser]") {
    const std::string json_output = R"([
  {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 0.0},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 13.7, "stddev_ts": 0.0}
])";

    BenchmarkParseResult result = parse_llama_bench_json(json_output);
    REQUIRE(result.valid);
    CHECK(result.prompt_tps_samples.empty());
    CHECK(result.generation_tps_samples.empty());
}

TEST_CASE("parse_llama_bench_json rejects garbage input", "[parser]") {
    BenchmarkParseResult result = parse_llama_bench_json("not json at all, no brackets here");
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.parse_error.empty());
}

TEST_CASE("parse_llama_bench_json rejects malformed JSON between brackets", "[parser]") {
    BenchmarkParseResult result = parse_llama_bench_json("[ { \"n_prompt\": , broken } ]");
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.parse_error.empty());
}

TEST_CASE("parse_llama_bench_markdown parses the default table format", "[parser]") {
    const std::string markdown_output =
        "| model                          |       size |     params | backend    | ngl | test       |              t/s |\n"
        "| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ----------------: |\n"
        "| llama 7B mostly Q4_0            |   3.56 GiB |     6.74 B | CUDA       |  31 | pp512      |   1234.56 \xC2\xB1 12.34 |\n"
        "| llama 7B mostly Q4_0            |   3.56 GiB |     6.74 B | CUDA       |  31 | tg128      |     45.67 \xC2\xB1 1.23 |\n"
        "\n"
        "build: abc1234 (1234)\n";

    BenchmarkParseResult result = parse_llama_bench_markdown(markdown_output);
    REQUIRE(result.valid);
    REQUIRE(result.prompt_tokens_per_second.has_value());
    REQUIRE(result.generation_tokens_per_second.has_value());
    CHECK(result.prompt_tokens_per_second.value() == Catch::Approx(1234.56));
    CHECK(result.generation_tokens_per_second.value() == Catch::Approx(45.67));
}

TEST_CASE("parse_llama_bench_markdown fails gracefully with no matching table", "[parser]") {
    BenchmarkParseResult result = parse_llama_bench_markdown("just some plain text\nwith no tables at all\n");
    CHECK_FALSE(result.valid);
    CHECK_FALSE(result.parse_error.empty());
}

TEST_CASE("parse_llama_bench_output falls back from JSON to markdown", "[parser]") {
    const std::string markdown_only =
        "| model | size | params | backend | ngl | test  |    t/s |\n"
        "| ----- | ---: | -----: | ------- | --: | ----- | -----: |\n"
        "| llama | 1 GB |   1.0B | CPU     |   0 | pp128 | 500.00 |\n"
        "| llama | 1 GB |   1.0B | CPU     |   0 | tg64  |  20.00 |\n";

    BenchmarkParseResult result = parse_llama_bench_output(markdown_only);
    REQUIRE(result.valid);
    CHECK(result.prompt_tokens_per_second.value() == Catch::Approx(500.00));
    CHECK(result.generation_tokens_per_second.value() == Catch::Approx(20.00));
}
