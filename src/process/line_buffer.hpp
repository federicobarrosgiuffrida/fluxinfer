#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace fluxinfer::process::detail {

// Appends `chunk` to `pending` and invokes `on_line` once for each complete
// line (split on '\n', trailing '\r' stripped) found. Any trailing partial
// line stays buffered in `pending` for the next call. `on_line` may be
// empty, in which case this just does line splitting bookkeeping cheaply
// skipped (still safe to call).
void feed_lines(std::string& pending, std::string_view chunk,
                 const std::function<void(std::string_view)>& on_line);

// Flushes any remaining partial line in `pending` (with no trailing
// newline) to `on_line`, e.g. once the process has exited. Clears pending.
void flush_pending_line(std::string& pending, const std::function<void(std::string_view)>& on_line);

} // namespace fluxinfer::process::detail
