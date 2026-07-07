#include "line_buffer.hpp"

namespace fluxinfer::process::detail {

void feed_lines(std::string& pending, std::string_view chunk,
                 const std::function<void(std::string_view)>& on_line) {
    pending.append(chunk.data(), chunk.size());
    if (!on_line) {
        return;
    }

    std::size_t start = 0;
    while (true) {
        const std::size_t newline = pending.find('\n', start);
        if (newline == std::string::npos) {
            break;
        }
        std::size_t end = newline;
        if (end > start && pending[end - 1] == '\r') {
            --end;
        }
        on_line(std::string_view(pending).substr(start, end - start));
        start = newline + 1;
    }

    pending.erase(0, start);
}

void flush_pending_line(std::string& pending, const std::function<void(std::string_view)>& on_line) {
    if (!pending.empty() && on_line) {
        on_line(pending);
    }
    pending.clear();
}

} // namespace fluxinfer::process::detail
