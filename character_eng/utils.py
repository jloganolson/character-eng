import subprocess
import threading
import time
from collections.abc import Callable


def ts():
    """Wall-clock timestamp for diagnostic output."""
    t = time.time()
    lt = time.localtime(t)
    ms = int((t % 1) * 1000)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"


def start_prefixed_output_thread(
    proc: subprocess.Popen,
    *,
    prefix: str = "",
    sink: Callable[[str], None] | None = None,
) -> threading.Thread | None:
    """Stream a child process stdout line-by-line with a stable prefix."""
    if proc.stdout is None:
        return None
    if sink is None:
        sink = lambda line: print(line, flush=True)

    def _pump() -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                sink(f"{prefix}{line}")
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    return thread
