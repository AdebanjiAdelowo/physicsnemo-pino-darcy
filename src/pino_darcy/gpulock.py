"""An inter-process lock around accelerator work, shared by the PhysicsNeMo jobs of one machine.

Every job that trains or evaluates on MPS or CUDA takes an exclusive ``flock`` on one file
before it touches the device and keeps it until the work is done. The kernel releases the
lock when the holding process exits for any reason, so a crashed or killed job never leaves
it held. The file is ``~/.cache/physicsnemo/gpu.lock`` unless ``PHYSICSNEMO_GPU_LOCK`` names
another; other projects take the same lock by locking the same path.
"""

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_PATH = Path.home() / ".cache" / "physicsnemo" / "gpu.lock"


def lock_path() -> Path:
    return Path(os.environ.get("PHYSICSNEMO_GPU_LOCK", DEFAULT_PATH))


def needs_lock(device: str) -> bool:
    """Whether a config ``device`` would run on an accelerator (``auto`` resolves as in ``engine``)."""
    if device == "auto":
        import torch

        return torch.cuda.is_available() or torch.backends.mps.is_available()
    return not device.startswith("cpu")


@contextmanager
def gpu_lock(device: str = "auto", path: Path | None = None, poll_seconds: float = 30.0, timeout: float | None = None):
    """Hold the machine-wide accelerator lock for the body of the ``with`` block.

    Waits, printing who holds the lock, until it is free or ``timeout`` seconds have passed
    (``TimeoutError``). A no-op for CPU work.
    """
    if not needs_lock(device):
        yield None
        return
    path = Path(path or lock_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout is not None and time.monotonic() - start >= timeout:
                    raise TimeoutError(f"accelerator lock {path} still held after {timeout:g} s") from None
                holder = os.pread(fd, 512, 0).decode(errors="replace").strip() or "unknown"
                print(f"[gpu-lock] waiting for {path} (held by {holder})", flush=True)
                time.sleep(poll_seconds)
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"pid={os.getpid()} cwd={os.getcwd()} since={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n".encode(), 0)
        print(f"[gpu-lock] acquired {path}", flush=True)
        yield path
    finally:
        # closing the descriptor releases the lock; the file itself is left for the next job
        os.close(fd)
