#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fluxinfer::process {

// Arguments are always passed as an argv-style vector and handed to the OS
// process-creation API directly (CreateProcess on Windows, posix_spawn on
// POSIX). Nothing is ever concatenated into a shell command line, so paths
// or arguments containing spaces, quotes or shell metacharacters cannot
// cause command injection or be misinterpreted by a shell (none is
// involved).
struct ProcessOptions {
    std::filesystem::path executable;
    std::vector<std::string> arguments;
    std::optional<std::filesystem::path> working_directory;

    // Extra/overriding environment variables. If inherit_environment is
    // true (default) these are merged on top of the current process
    // environment; otherwise the child receives only these variables.
    std::map<std::string, std::string> extra_environment;
    bool inherit_environment = true;

    // Hard upper bound on total run time.
    std::optional<std::chrono::milliseconds> timeout;

    // Kill the child if it produces no output at all for this long, even
    // when the total timeout above has not been reached. A process that is
    // still writing progress is working, however slowly; one that has gone
    // quiet is the only kind that is actually stuck. Only used by
    // run_captured(). Unset means "no idle limit".
    std::optional<std::chrono::milliseconds> idle_timeout;

    // Only used by run_captured(). When set, invoked once per complete
    // line of stdout/stderr as it is produced (in addition to the full
    // output still being collected into ProcessResult).
    std::function<void(std::string_view)> on_stdout_line;
    std::function<void(std::string_view)> on_stderr_line;
};

enum class ProcessOutcome {
    Exited,        // process ran to completion; exit_code is valid
    TimedOut,      // exceeded options.timeout and was force-terminated
    FailedToStart, // executable could not be launched
};

struct ProcessResult {
    ProcessOutcome outcome = ProcessOutcome::FailedToStart;
    int exit_code = -1;
    std::string stdout_data;
    std::string stderr_data;
    std::chrono::milliseconds duration{0};
    std::string start_error; // populated only when outcome == FailedToStart

    // Set together with outcome == TimedOut to say *which* limit fired:
    // true for the idle limit (the child went silent -- a genuine hang),
    // false for the total-time limit (the child was still producing output,
    // it was simply too slow overall). Callers use this to tell "stuck"
    // from "slow", which are very different signals when the child is a
    // benchmark being pushed towards a hardware limit.
    bool idle_timed_out = false;
};

// Runs a process to completion (or until timeout), capturing stdout/stderr.
// Blocking. Never throws. Intended for llama-bench invocations.
ProcessResult run_captured(const ProcessOptions& options);

// A child process launched with inherited stdio (console passthrough),
// used by `fluxinfer run` / `fluxinfer serve`. The caller owns the returned
// handle and must eventually call wait() or terminate().
class InteractiveProcess {
public:
    virtual ~InteractiveProcess() = default;
    virtual bool is_running() = 0;
    virtual int wait() = 0;       // blocks until the child exits, returns its exit code
    virtual void terminate() = 0; // best-effort graceful signal, escalates to a forceful kill
};

// Launches a child process with stdin/stdout/stderr inherited from
// FluxInfer itself. options.timeout, on_stdout_line and on_stderr_line are
// ignored in this mode. Returns nullptr and sets *error_message (if
// non-null) if the process could not be started.
std::unique_ptr<InteractiveProcess> launch_interactive(const ProcessOptions& options,
                                                        std::string* error_message = nullptr);

} // namespace fluxinfer::process
