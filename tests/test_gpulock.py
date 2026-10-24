"""The accelerator lock: exclusive across processes, released on exit and on failure."""

import subprocess
import sys
import textwrap

import pytest

from pino_darcy.gpulock import gpu_lock, needs_lock

HOLD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {src!r})
    from pino_darcy.gpulock import gpu_lock
    with gpu_lock("mps", path={path!r}):
        print("held", flush=True)
        time.sleep(30)
    """
)


def holder(path, src):
    proc = subprocess.Popen([sys.executable, "-c", HOLD.format(path=str(path), src=src)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "[gpu-lock] acquired " + str(path)
    assert proc.stdout.readline().strip() == "held"
    return proc


@pytest.fixture
def src():
    import pino_darcy

    return str(__import__("pathlib").Path(pino_darcy.__file__).parents[1])


def test_cpu_work_takes_no_lock(tmp_path):
    assert not needs_lock("cpu")
    with gpu_lock("cpu", path=tmp_path / "gpu.lock") as held:
        assert held is None
    assert not (tmp_path / "gpu.lock").exists()


def test_lock_is_exclusive_across_processes_and_freed_when_the_holder_dies(tmp_path, src):
    path = tmp_path / "gpu.lock"
    proc = holder(path, src)
    try:
        with pytest.raises(TimeoutError):
            with gpu_lock("mps", path=path, poll_seconds=0.05, timeout=0.2):
                pass
    finally:
        proc.kill()  # no clean-up code runs in the holder
        proc.wait()
    with gpu_lock("mps", path=path, timeout=1) as held:
        assert held == path


def test_lock_is_released_when_the_body_raises(tmp_path):
    path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError):
        with gpu_lock("mps", path=path):
            raise RuntimeError("training failed")
    with gpu_lock("mps", path=path, timeout=0) as held:
        assert held == path
