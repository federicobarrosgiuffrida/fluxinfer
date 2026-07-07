#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/tuner/comparison.hpp"

#include <cmath>

using namespace fluxinfer::tuner;

TEST_CASE("Stats::compute on an empty sample is all zeros", "[stats]") {
    Stats stats = Stats::compute({});
    CHECK(stats.mean == 0.0);
    CHECK(stats.median == 0.0);
    CHECK(stats.min == 0.0);
    CHECK(stats.max == 0.0);
    CHECK(stats.stddev == 0.0);
}

TEST_CASE("Stats::compute on a single sample has zero stddev", "[stats]") {
    Stats stats = Stats::compute({42.0});
    CHECK(stats.mean == Catch::Approx(42.0));
    CHECK(stats.median == Catch::Approx(42.0));
    CHECK(stats.min == Catch::Approx(42.0));
    CHECK(stats.max == Catch::Approx(42.0));
    CHECK(stats.stddev == 0.0);
}

TEST_CASE("Stats::compute matches hand-computed values for an odd-sized sample", "[stats]") {
    // 10, 12, 14 -> mean 12, median 12, min 10, max 14
    Stats stats = Stats::compute({14.0, 10.0, 12.0});
    CHECK(stats.mean == Catch::Approx(12.0));
    CHECK(stats.median == Catch::Approx(12.0));
    CHECK(stats.min == Catch::Approx(10.0));
    CHECK(stats.max == Catch::Approx(14.0));
    // sample stddev (n-1): sqrt(((10-12)^2+(12-12)^2+(14-12)^2)/2) = sqrt(8/2) = 2
    CHECK(stats.stddev == Catch::Approx(2.0));
}

TEST_CASE("Stats::compute matches hand-computed values for an even-sized sample", "[stats]") {
    // 1, 2, 3, 4 -> mean 2.5, median (2+3)/2=2.5
    Stats stats = Stats::compute({4.0, 1.0, 3.0, 2.0});
    CHECK(stats.mean == Catch::Approx(2.5));
    CHECK(stats.median == Catch::Approx(2.5));
    CHECK(stats.min == Catch::Approx(1.0));
    CHECK(stats.max == Catch::Approx(4.0));
    // sample stddev: sqrt(((1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2)/3) = sqrt(5/3)
    CHECK(stats.stddev == Catch::Approx(std::sqrt(5.0 / 3.0)));
}

TEST_CASE("format_comparison_report renders a confirmed improvement without overlapping ranges", "[stats][report]") {
    ComparisonReport report;
    report.baseline.successful_runs = 3;
    report.baseline.generation_tps_stats = Stats::compute({10.0, 10.5, 9.5});
    report.baseline.prompt_tps_stats = Stats::compute({200.0, 205.0, 195.0});
    report.selected.successful_runs = 3;
    report.selected.generation_tps_stats = Stats::compute({20.0, 21.0, 19.0});
    report.selected.prompt_tps_stats = Stats::compute({400.0, 410.0, 390.0});
    report.generation_tps_improvement_pct = 100.0;
    report.prompt_tps_improvement_pct = 100.0;
    report.improvement_confirmed =
        report.selected.generation_tps_stats.min > report.baseline.generation_tps_stats.max;

    REQUIRE(report.improvement_confirmed);

    ReportContext context;
    context.model_path = "model.gguf";
    context.architecture = "testarch";
    context.quantization_label = "Q8_0";
    context.layer_count = 32;
    context.llama_bench_path = "llama-bench";
    context.llama_version = "build 1";
    context.baseline_arguments = {"-m", "model.gguf", "-ngl", "0"};
    context.selected_arguments = {"-m", "model.gguf", "-ngl", "32"};

    std::string text = format_comparison_report(report, context);
    CHECK(text.find("Confirmed") != std::string::npos);
    CHECK(text.find("testarch") != std::string::npos);
    CHECK(text.find("Q8_0") != std::string::npos);
}

TEST_CASE("format_comparison_report does not claim a confirmed improvement on overlapping ranges", "[stats][report]") {
    ComparisonReport report;
    report.baseline.successful_runs = 3;
    report.baseline.generation_tps_stats = Stats::compute({10.0, 15.0, 20.0});
    report.selected.successful_runs = 3;
    report.selected.generation_tps_stats = Stats::compute({12.0, 16.0, 18.0}); // overlaps with baseline's range
    report.generation_tps_improvement_pct = 10.0;
    report.improvement_confirmed =
        report.selected.generation_tps_stats.min > report.baseline.generation_tps_stats.max;

    REQUIRE_FALSE(report.improvement_confirmed);

    ReportContext context;
    std::string text = format_comparison_report(report, context);
    CHECK(text.find("Not confirmed") != std::string::npos);
}
