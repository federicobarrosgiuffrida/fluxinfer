#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "fluxinfer/process/process_runner.hpp"

#include "line_buffer.hpp"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <map>
#include <vector>

extern char** environ;

namespace fluxinfer::process {

namespace {

std::vector<std::string> build_environment_strings(const ProcessOptions& options) {
    std::map<std::string, std::string> merged;
    if (options.inherit_environment) {
        for (char** entry = environ; entry && *entry; ++entry) {
            std::string_view kv(*entry);
            const auto eq = kv.find('=');
            if (eq != std::string_view::npos) {
                merged[std::string(kv.substr(0, eq))] = std::string(kv.substr(eq + 1));
            }
        }
    }
    for (const auto& [key, value] : options.extra_environment) {
        merged[key] = value;
    }

    std::vector<std::string> result;
    result.reserve(merged.size());
    for (const auto& [key, value] : merged) {
        result.push_back(key + "=" + value);
    }
    return result;
}

std::vector<char*> to_argv(const std::vector<std::string>& strings) {
    std::vector<char*> argv;
    argv.reserve(strings.size() + 1);
    for (const auto& s : strings) {
        argv.push_back(const_cast<char*>(s.c_str()));
    }
    argv.push_back(nullptr);
    return argv;
}

// Child-side setup executed after fork(), before execve(). Kept to
// async-signal-safe calls only (dup2/close/chdir/execve), as required when
// running in the child of a multithreaded process.
[[noreturn]] void child_exec(const std::filesystem::path& executable,
                              std::vector<char*>& argv,
                              std::vector<char*>& envp,
                              const std::optional<std::filesystem::path>& cwd,
                              int stdout_write_fd,  // -1 to inherit
                              int stderr_write_fd,  // -1 to inherit
                              int stdin_read_fd) {  // -1 to inherit
    // If the source fd differs from its target, the original must be closed
    // after dup2: otherwise the exec'd process inherits an extra open copy
    // of the pipe write end, which keeps it alive even after this process
    // exits and prevents the parent from ever observing EOF while reading.
    if (stdout_write_fd >= 0) {
        dup2(stdout_write_fd, STDOUT_FILENO);
        if (stdout_write_fd != STDOUT_FILENO) {
            close(stdout_write_fd);
        }
    }
    if (stderr_write_fd >= 0) {
        dup2(stderr_write_fd, STDERR_FILENO);
        if (stderr_write_fd != STDERR_FILENO) {
            close(stderr_write_fd);
        }
    }
    if (stdin_read_fd >= 0) {
        dup2(stdin_read_fd, STDIN_FILENO);
        if (stdin_read_fd != STDIN_FILENO) {
            close(stdin_read_fd);
        }
    }

    if (cwd) {
        if (chdir(cwd->c_str()) != 0) {
            _exit(126);
        }
    }

    execve(executable.c_str(), argv.data(), envp.data());
    _exit(127); // execve only returns on failure
}

struct SpawnedPipes {
    int stdout_read = -1;
    int stderr_read = -1;
};

class PosixInteractiveProcess final : public InteractiveProcess {
public:
    explicit PosixInteractiveProcess(pid_t pid) : pid_(pid) {}

    bool is_running() override {
        if (exited_) {
            return false;
        }
        int status = 0;
        pid_t result = waitpid(pid_, &status, WNOHANG);
        if (result == pid_) {
            exited_ = true;
            exit_code_ = decode_exit_code(status);
            return false;
        }
        return true;
    }

    int wait() override {
        if (!exited_) {
            int status = 0;
            waitpid(pid_, &status, 0);
            exited_ = true;
            exit_code_ = decode_exit_code(status);
        }
        return exit_code_;
    }

    void terminate() override {
        if (exited_) {
            return;
        }
        kill(pid_, SIGTERM);
        for (int i = 0; i < 50 && is_running(); ++i) { // ~5s grace period
            struct timespec ts{0, 100'000'000};
            nanosleep(&ts, nullptr);
        }
        if (!exited_) {
            kill(pid_, SIGKILL);
            wait();
        }
    }

private:
    static int decode_exit_code(int status) {
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        if (WIFSIGNALED(status)) {
            return 128 + WTERMSIG(status);
        }
        return -1;
    }

    pid_t pid_;
    bool exited_ = false;
    int exit_code_ = -1;
};

} // namespace

ProcessResult run_captured(const ProcessOptions& options) {
    ProcessResult result;
    const auto start_time = std::chrono::steady_clock::now();

    int stdout_fds[2] = {-1, -1};
    int stderr_fds[2] = {-1, -1};
    if (pipe(stdout_fds) != 0 || pipe(stderr_fds) != 0) {
        result.start_error = std::string("pipe() failed: ") + std::strerror(errno);
        return result;
    }

    int devnull_fd = open("/dev/null", O_RDONLY);

    std::vector<std::string> arg_strings;
    arg_strings.push_back(options.executable.string());
    arg_strings.insert(arg_strings.end(), options.arguments.begin(), options.arguments.end());
    std::vector<char*> argv = to_argv(arg_strings);

    std::vector<std::string> env_strings = build_environment_strings(options);
    std::vector<char*> envp = to_argv(env_strings);

    pid_t pid = fork();
    if (pid < 0) {
        result.start_error = std::string("fork() failed: ") + std::strerror(errno);
        close(stdout_fds[0]);
        close(stdout_fds[1]);
        close(stderr_fds[0]);
        close(stderr_fds[1]);
        if (devnull_fd >= 0) close(devnull_fd);
        return result;
    }

    if (pid == 0) {
        // Child.
        close(stdout_fds[0]);
        close(stderr_fds[0]);
        child_exec(options.executable, argv, envp, options.working_directory,
                   stdout_fds[1], stderr_fds[1], devnull_fd);
    }

    // Parent.
    close(stdout_fds[1]);
    close(stderr_fds[1]);
    if (devnull_fd >= 0) close(devnull_fd);

    std::string stdout_pending, stderr_pending;
    bool stdout_done = false, stderr_done = false;
    bool timed_out = false;

    while (!stdout_done || !stderr_done) {
        std::array<pollfd, 2> pfds{};
        int nfds = 0;
        int stdout_idx = -1, stderr_idx = -1;
        if (!stdout_done) {
            pfds[nfds] = {stdout_fds[0], POLLIN, 0};
            stdout_idx = nfds++;
        }
        if (!stderr_done) {
            pfds[nfds] = {stderr_fds[0], POLLIN, 0};
            stderr_idx = nfds++;
        }

        int poll_timeout_ms = -1;
        if (options.timeout) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time);
            auto remaining = *options.timeout - elapsed;
            if (remaining.count() <= 0) {
                timed_out = true;
                break;
            }
            poll_timeout_ms = static_cast<int>(remaining.count());
        }

        int ready = poll(pfds.data(), static_cast<nfds_t>(nfds), poll_timeout_ms);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ready == 0) {
            timed_out = true;
            break;
        }

        std::array<char, 4096> buffer{};
        if (stdout_idx >= 0 && (pfds[stdout_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t n = read(stdout_fds[0], buffer.data(), buffer.size());
            if (n > 0) {
                result.stdout_data.append(buffer.data(), static_cast<std::size_t>(n));
                detail::feed_lines(stdout_pending, std::string_view(buffer.data(), static_cast<std::size_t>(n)),
                                    options.on_stdout_line);
            } else {
                stdout_done = true;
            }
        }
        if (stderr_idx >= 0 && (pfds[stderr_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
            ssize_t n = read(stderr_fds[0], buffer.data(), buffer.size());
            if (n > 0) {
                result.stderr_data.append(buffer.data(), static_cast<std::size_t>(n));
                detail::feed_lines(stderr_pending, std::string_view(buffer.data(), static_cast<std::size_t>(n)),
                                    options.on_stderr_line);
            } else {
                stderr_done = true;
            }
        }
    }

    detail::flush_pending_line(stdout_pending, options.on_stdout_line);
    detail::flush_pending_line(stderr_pending, options.on_stderr_line);
    close(stdout_fds[0]);
    close(stderr_fds[0]);

    if (timed_out) {
        kill(pid, SIGKILL);
        int status = 0;
        waitpid(pid, &status, 0);
        result.outcome = ProcessOutcome::TimedOut;
        result.exit_code = -1;
    } else {
        int status = 0;
        waitpid(pid, &status, 0);
        result.outcome = ProcessOutcome::Exited;
        if (WIFEXITED(status)) {
            result.exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            result.exit_code = 128 + WTERMSIG(status);
        }
    }

    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    return result;
}

std::unique_ptr<InteractiveProcess> launch_interactive(const ProcessOptions& options, std::string* error_message) {
    std::vector<std::string> arg_strings;
    arg_strings.push_back(options.executable.string());
    arg_strings.insert(arg_strings.end(), options.arguments.begin(), options.arguments.end());
    std::vector<char*> argv = to_argv(arg_strings);

    std::vector<std::string> env_strings = build_environment_strings(options);
    std::vector<char*> envp = to_argv(env_strings);

    pid_t pid = fork();
    if (pid < 0) {
        if (error_message) {
            *error_message = std::string("fork() failed: ") + std::strerror(errno);
        }
        return nullptr;
    }

    if (pid == 0) {
        child_exec(options.executable, argv, envp, options.working_directory, -1, -1, -1);
    }

    return std::make_unique<PosixInteractiveProcess>(pid);
}

} // namespace fluxinfer::process
