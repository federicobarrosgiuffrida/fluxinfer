#include "fluxinfer/llama/benchmark_parser.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <sstream>

namespace fluxinfer::llama {

namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::string> split_markdown_row(const std::string& line) {
    std::vector<std::string> cells;
    std::string trimmed = trim(line);
    if (trimmed.empty() || trimmed.front() != '|') {
        return cells;
    }
    // Strip leading/trailing pipe so split doesn't produce empty edge cells.
    if (!trimmed.empty() && trimmed.front() == '|') trimmed.erase(0, 1);
    if (!trimmed.empty() && trimmed.back() == '|') trimmed.pop_back();

    std::stringstream ss(trimmed);
    std::string cell;
    while (std::getline(ss, cell, '|')) {
        cells.push_back(trim(cell));
    }
    return cells;
}

bool is_separator_row(const std::vector<std::string>& cells) {
    if (cells.empty()) {
        return false;
    }
    for (const auto& cell : cells) {
        if (cell.find_first_not_of("-: ") != std::string::npos) {
            return false;
        }
    }
    return true;
}

// Parses the leading numeric token of strings like "1234.56 ± 12.34" and
// "1234.56" into (value, stddev-or-0).
std::pair<double, double> parse_tps_cell(const std::string& cell) {
    double value = 0.0;
    double stddev = 0.0;
    std::istringstream iss(cell);
    iss >> value;
    std::string maybe_plusminus;
    if (iss >> maybe_plusminus) {
        if (maybe_plusminus == "\xC2\xB1" || maybe_plusminus == "+/-") {
            iss >> stddev;
        }
    }
    return {value, stddev};
}

} // namespace

BenchmarkParseResult parse_llama_bench_json(const std::string& stdout_text) {
    BenchmarkParseResult result;

    const auto array_start = stdout_text.find('[');
    const auto array_end = stdout_text.rfind(']');
    if (array_start == std::string::npos || array_end == std::string::npos || array_end < array_start) {
        result.parse_error = "no JSON array found in llama-bench output";
        return result;
    }

    const std::string json_slice = stdout_text.substr(array_start, array_end - array_start + 1);

    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(json_slice);
    } catch (const nlohmann::json::exception& e) {
        result.parse_error = std::string("JSON parse error: ") + e.what();
        return result;
    }

    if (!parsed.is_array() || parsed.empty()) {
        result.parse_error = "llama-bench JSON output was not a non-empty array";
        return result;
    }

    for (const auto& entry : parsed) {
        BenchmarkMeasurement measurement;
        measurement.n_prompt = entry.value("n_prompt", 0ULL);
        measurement.n_gen = entry.value("n_gen", 0ULL);
        measurement.avg_tokens_per_second = entry.value("avg_ts", 0.0);
        measurement.stddev_tokens_per_second = entry.value("stddev_ts", 0.0);
        measurement.test_name = measurement.n_prompt > 0 ? ("pp" + std::to_string(measurement.n_prompt))
                                                           : ("tg" + std::to_string(measurement.n_gen));

        if (entry.contains("samples_ts") && entry["samples_ts"].is_array()) {
            for (const auto& sample : entry["samples_ts"]) {
                if (sample.is_number()) {
                    measurement.samples_tokens_per_second.push_back(sample.get<double>());
                }
            }
        }

        result.measurements.push_back(measurement);

        if (measurement.n_prompt > 0 && measurement.n_gen == 0 && !result.prompt_tokens_per_second) {
            result.prompt_tokens_per_second = measurement.avg_tokens_per_second;
            result.prompt_tps_samples = measurement.samples_tokens_per_second;
        }
        if (measurement.n_gen > 0 && measurement.n_prompt == 0 && !result.generation_tokens_per_second) {
            result.generation_tokens_per_second = measurement.avg_tokens_per_second;
            result.generation_tps_samples = measurement.samples_tokens_per_second;
        }
    }

    result.valid = !result.measurements.empty();
    if (!result.valid) {
        result.parse_error = "llama-bench JSON array contained no usable rows";
    }
    return result;
}

BenchmarkParseResult parse_llama_bench_markdown(const std::string& stdout_text) {
    BenchmarkParseResult result;

    std::istringstream lines(stdout_text);
    std::string line;

    std::vector<std::string> header;
    bool header_found = false;
    int test_col = -1;
    int tps_col = -1;

    while (std::getline(lines, line)) {
        std::vector<std::string> cells = split_markdown_row(line);
        if (cells.empty()) {
            continue;
        }

        if (!header_found) {
            header = cells;
            for (std::size_t i = 0; i < header.size(); ++i) {
                const std::string lower = to_lower(header[i]);
                if (lower == "test") {
                    test_col = static_cast<int>(i);
                } else if (lower.find("t/s") != std::string::npos) {
                    tps_col = static_cast<int>(i);
                }
            }
            if (test_col >= 0 && tps_col >= 0) {
                header_found = true;
            }
            continue;
        }

        if (is_separator_row(cells)) {
            continue;
        }

        if (static_cast<int>(cells.size()) <= std::max(test_col, tps_col)) {
            continue; // malformed/short row, skip rather than crash
        }

        BenchmarkMeasurement measurement;
        measurement.test_name = cells[static_cast<std::size_t>(test_col)];
        auto [value, stddev] = parse_tps_cell(cells[static_cast<std::size_t>(tps_col)]);
        measurement.avg_tokens_per_second = value;
        measurement.stddev_tokens_per_second = stddev;

        const std::string lower_test = to_lower(measurement.test_name);
        if (lower_test.rfind("pp", 0) == 0) {
            measurement.n_prompt = 1;
            if (!result.prompt_tokens_per_second) {
                result.prompt_tokens_per_second = value;
            }
        } else if (lower_test.rfind("tg", 0) == 0) {
            measurement.n_gen = 1;
            if (!result.generation_tokens_per_second) {
                result.generation_tokens_per_second = value;
            }
        }

        result.measurements.push_back(measurement);
    }

    if (!header_found) {
        result.parse_error = "no markdown table with 'test' and 't/s' columns found";
        return result;
    }

    result.valid = !result.measurements.empty();
    if (!result.valid) {
        result.parse_error = "markdown table had a header but no data rows";
    }
    return result;
}

BenchmarkParseResult parse_llama_bench_output(const std::string& stdout_text) {
    BenchmarkParseResult json_result = parse_llama_bench_json(stdout_text);
    if (json_result.valid) {
        return json_result;
    }
    return parse_llama_bench_markdown(stdout_text);
}

} // namespace fluxinfer::llama
