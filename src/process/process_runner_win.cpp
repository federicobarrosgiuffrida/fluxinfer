#include "fluxinfer/process/process_runner.hpp"

#include "line_buffer.hpp"

#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cwchar>
#include <cwctype>
#include <map>
#include <string>
#include <thread>

namespace fluxinfer::process {

namespace {

std::wstring utf8_to_wide(const std::string& utf8) {
    if (utf8.empty()) {
        return {};
    }
    int size = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), result.data(), size);
    return result;
}

std::string wide_to_utf8(const std::wstring& wide) {
    if (wide.empty()) {
        return {};
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), result.data(), size, nullptr, nullptr);
    return result;
}

// Standard MSVC-CRT-compatible argv quoting. Never delegates to a shell, so
// this only has to satisfy CommandLineToArgvW-style re-parsing by the
// child, not shell metacharacter rules.
std::wstring quote_argument(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }

    std::wstring result = L"\"";
    for (auto it = arg.begin();; ++it) {
        std::size_t backslashes = 0;
        while (it != arg.end() && *it == L'\\') {
            ++it;
            ++backslashes;
        }

        if (it == arg.end()) {
            result.append(backslashes * 2, L'\\');
            break;
        }
        if (*it == L'"') {
            result.append(backslashes * 2 + 1, L'\\');
            result.push_back(L'"');
        } else {
            result.append(backslashes, L'\\');
            result.push_back(*it);
        }
    }
    result.push_back(L'"');
    return result;
}

std::wstring build_command_line(const std::filesystem::path& executable, const std::vector<std::string>& args) {
    std::wstring command_line = quote_argument(executable.wstring());
    for (const auto& arg : args) {
        command_line.push_back(L' ');
        command_line += quote_argument(utf8_to_wide(arg));
    }
    return command_line;
}

std::map<std::wstring, std::wstring> current_environment() {
    std::map<std::wstring, std::wstring> env;
    LPWCH block = GetEnvironmentStringsW();
    if (!block) {
        return env;
    }
    for (const wchar_t* p = block; *p != L'\0';) {
        std::wstring entry(p);
        p += entry.size() + 1;
        if (entry.empty() || entry.front() == L'=') {
            continue; // skip drive-letter pseudo-variables like "=C:"
        }
        const auto eq = entry.find(L'=');
        if (eq == std::wstring::npos) {
            continue;
        }
        env[entry.substr(0, eq)] = entry.substr(eq + 1);
    }
    FreeEnvironmentStringsW(block);
    return env;
}

struct CaseInsensitiveLess {
    bool operator()(const std::wstring& a, const std::wstring& b) const {
        return _wcsicmp(a.c_str(), b.c_str()) < 0;
    }
};

std::vector<wchar_t> build_environment_block(const ProcessOptions& options) {
    std::map<std::wstring, std::wstring, CaseInsensitiveLess> merged;
    if (options.inherit_environment) {
        for (auto& [key, value] : current_environment()) {
            merged[key] = value;
        }
    }
    for (const auto& [key, value] : options.extra_environment) {
        merged[utf8_to_wide(key)] = utf8_to_wide(value);
    }

    std::vector<wchar_t> block;
    for (const auto& [key, value] : merged) {
        std::wstring entry = key + L'=' + value;
        block.insert(block.end(), entry.begin(), entry.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}

struct PipeHandles {
    HANDLE read = nullptr;
    HANDLE write = nullptr;
};

bool make_inheritable_output_pipe(PipeHandles& pipe) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    if (!CreatePipe(&pipe.read, &pipe.write, &sa, 0)) {
        return false;
    }
    // Only the write end is handed to the child; the parent's read end must
    // not be inherited or it would leak into the child (and stay open if we
    // ever spawn a grandchild), preventing EOF detection.
    return SetHandleInformation(pipe.read, HANDLE_FLAG_INHERIT, 0) != 0;
}

class ReaderThread {
public:
    ReaderThread(HANDLE handle, std::string* out, std::function<void(std::string_view)> on_line)
        : handle_(handle), out_(out), on_line_(std::move(on_line)) {
        thread_ = std::thread([this] { run(); });
    }

    ~ReaderThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() {
        std::array<char, 4096> buffer{};
        DWORD bytes_read = 0;
        std::string pending;
        while (ReadFile(handle_, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read, nullptr) &&
               bytes_read > 0) {
            out_->append(buffer.data(), bytes_read);
            detail::feed_lines(pending, std::string_view(buffer.data(), bytes_read), on_line_);
        }
        detail::flush_pending_line(pending, on_line_);
    }

    HANDLE handle_;
    std::string* out_;
    std::function<void(std::string_view)> on_line_;
    std::thread thread_;
};

class WindowsInteractiveProcess final : public InteractiveProcess {
public:
    explicit WindowsInteractiveProcess(PROCESS_INFORMATION pi) : pi_(pi) {}

    ~WindowsInteractiveProcess() override {
        if (pi_.hThread) {
            CloseHandle(pi_.hThread);
        }
        if (pi_.hProcess) {
            CloseHandle(pi_.hProcess);
        }
    }

    bool is_running() override {
        DWORD exit_code = STILL_ACTIVE;
        if (!GetExitCodeProcess(pi_.hProcess, &exit_code)) {
            return false;
        }
        return exit_code == STILL_ACTIVE;
    }

    int wait() override {
        WaitForSingleObject(pi_.hProcess, INFINITE);
        DWORD exit_code = 0;
        GetExitCodeProcess(pi_.hProcess, &exit_code);
        return static_cast<int>(exit_code);
    }

    void terminate() override {
        // Best-effort graceful shutdown: send Ctrl+Break to the child's
        // process group, then escalate to TerminateProcess if it hasn't
        // exited shortly after. GenerateConsoleCtrlEvent requires the child
        // to share our console (true for llama-cli/llama-server launched
        // without CREATE_NEW_PROCESS_GROUP), so we fall back directly to
        // TerminateProcess if that's not applicable.
        if (is_running()) {
            TerminateProcess(pi_.hProcess, 1);
            WaitForSingleObject(pi_.hProcess, 5000);
        }
    }

private:
    PROCESS_INFORMATION pi_;
};

} // namespace

ProcessResult run_captured(const ProcessOptions& options) {
    ProcessResult result;
    const auto start_time = std::chrono::steady_clock::now();

    PipeHandles stdout_pipe;
    PipeHandles stderr_pipe;
    if (!make_inheritable_output_pipe(stdout_pipe) || !make_inheritable_output_pipe(stderr_pipe)) {
        result.start_error = "failed to create stdout/stderr pipes";
        return result;
    }

    HANDLE stdin_null = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (stdin_null != INVALID_HANDLE_VALUE) {
        SetHandleInformation(stdin_null, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
    }

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = stdin_null;
    si.hStdOutput = stdout_pipe.write;
    si.hStdError = stderr_pipe.write;

    std::wstring command_line = build_command_line(options.executable, options.arguments);
    std::vector<wchar_t> env_block = build_environment_block(options);
    std::wstring cwd_wide = options.working_directory ? options.working_directory->wstring() : std::wstring();

    PROCESS_INFORMATION pi{};
    BOOL created = CreateProcessW(
        options.executable.c_str(),
        command_line.data(),
        nullptr, nullptr,
        /*bInheritHandles=*/TRUE,
        CREATE_UNICODE_ENVIRONMENT,
        env_block.data(),
        options.working_directory ? cwd_wide.c_str() : nullptr,
        &si, &pi);

    // These write ends must be closed in the parent regardless of outcome,
    // or the reader threads (or CreateProcess failure path) would hang / we
    // would leak handles: the child owns its own duplicated copies.
    CloseHandle(stdout_pipe.write);
    CloseHandle(stderr_pipe.write);
    if (stdin_null != INVALID_HANDLE_VALUE) {
        CloseHandle(stdin_null);
    }

    if (!created) {
        DWORD err = GetLastError();
        result.start_error = "CreateProcessW failed with error " + std::to_string(err);
        CloseHandle(stdout_pipe.read);
        CloseHandle(stderr_pipe.read);
        return result;
    }
    CloseHandle(pi.hThread);

    {
        ReaderThread stdout_reader(stdout_pipe.read, &result.stdout_data, options.on_stdout_line);
        ReaderThread stderr_reader(stderr_pipe.read, &result.stderr_data, options.on_stderr_line);

        DWORD wait_ms = options.timeout ? static_cast<DWORD>(options.timeout->count()) : INFINITE;
        DWORD wait_result = WaitForSingleObject(pi.hProcess, wait_ms);

        if (wait_result == WAIT_TIMEOUT) {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            result.outcome = ProcessOutcome::TimedOut;
        } else {
            result.outcome = ProcessOutcome::Exited;
        }
        // Reader threads join here (ReaderThread destructor) once the pipes'
        // write ends close as part of process teardown, giving them EOF.
    }
    CloseHandle(stdout_pipe.read);
    CloseHandle(stderr_pipe.read);

    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    result.exit_code = static_cast<int>(exit_code);
    CloseHandle(pi.hProcess);

    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    return result;
}

std::unique_ptr<InteractiveProcess> launch_interactive(const ProcessOptions& options, std::string* error_message) {
    std::wstring command_line = build_command_line(options.executable, options.arguments);
    std::vector<wchar_t> env_block = build_environment_block(options);
    std::wstring cwd_wide = options.working_directory ? options.working_directory->wstring() : std::wstring();

    STARTUPINFOW si{};
    si.cb = sizeof(si);

    PROCESS_INFORMATION pi{};
    BOOL created = CreateProcessW(
        options.executable.c_str(),
        command_line.data(),
        nullptr, nullptr,
        /*bInheritHandles=*/TRUE,
        CREATE_UNICODE_ENVIRONMENT,
        env_block.data(),
        options.working_directory ? cwd_wide.c_str() : nullptr,
        &si, &pi);

    if (!created) {
        if (error_message) {
            *error_message = "CreateProcessW failed with error " + std::to_string(GetLastError());
        }
        return nullptr;
    }
    CloseHandle(pi.hThread);
    pi.hThread = nullptr; // avoid a double-close in the WindowsInteractiveProcess destructor
    return std::make_unique<WindowsInteractiveProcess>(pi);
}

} // namespace fluxinfer::process
