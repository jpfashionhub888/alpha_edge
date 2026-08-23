# atomic_io.py
"""
P3-3 FIX: shared atomic JSON write helper.

Previously this existed only inside main.py (atomic_json_write), while
~30 other call sites across the codebase wrote state with raw json.dump()
directly to the target file. A crash or kill mid-write on any of those
leaves a truncated/corrupt JSON file; the next read raises
JSONDecodeError, which — given how many exception handlers in this
codebase log-and-continue rather than halt — can silently be treated as
"no data" (empty positions, no signals, etc.) rather than surfaced as
the corruption it actually is.

This is not a hypothetical: this exact project has a documented incident
(CRITICAL_state_overwrite_loop.md) where corrupted/overwritten state
files fed the reconciler bad data and it "worked correctly" on corrupt
input — silently deleting real position tracking. Non-atomic writes are
a different failure mode with the same shape of consequence.

Usage:
    from atomic_io import atomic_json_write
    atomic_json_write('logs/some_state.json', data)

Safe for concurrent processes writing DIFFERENT files (temp file lives
in the same directory as the target, so os.replace() is atomic on both
POSIX and Windows). NOT a lock — two processes writing the SAME file
concurrently will still race, just never see a half-written result.
"""

import json
import os
import tempfile


def atomic_json_write(filepath: str, data: object, default=None) -> None:
    """Write JSON atomically: write to temp file, then rename.
    Prevents corrupted JSON if process is killed mid-write.

    `default`: optional callable passed through to json.dump() for
    serializing non-JSON-native types (numpy scalars, DataFrames, etc.)
    — same purpose as json.dump's own `default` parameter.
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(filepath) or '.', suffix='.tmp'
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2, default=default)
            # FIX: write-then-rename was atomic against partial writes, but
            # without fsync a completed write can still be lost (not
            # corrupted, just lost) on a crash between file close and the
            # OS's own flush to disk. This is the shared helper ~30 call
            # sites route through, so this one fix covers all of them —
            # see risk_circuit_breaker.py for the equivalent fix applied to
            # its separate, non-shared implementation of this same pattern.
            f.flush()
            os.fsync(f.fileno())
        # REGRESSION FIX: tempfile.mkstemp() always creates the temp file at
        # mode 0600 (owner-only), regardless of umask — that's its security
        # property. os.replace() then makes that 0600 file become the final
        # file, silently tightening permissions on every file this helper
        # writes vs. the plain open(path, 'w') it replaced (which honored
        # the process umask, typically 0644 for root cron/systemd jobs).
        # This broke alphaedge-audit.service (runs as a separate unprivileged
        # `alphaedge` user) reading logs/paper_trades_stocks_only.json,
        # written by merge_trades.py running as root — PermissionError,
        # failing the daily audit for 2+ days before being caught.
        # Restore the prior effective permissions explicitly.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, filepath)  # atomic on POSIX and Windows
    except Exception:
        os.unlink(tmp_path)
        raise
