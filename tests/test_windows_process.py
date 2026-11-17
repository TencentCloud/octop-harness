"""Standalone Windows integration tests; each case has an external watchdog."""

import importlib.util
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "src" / "octop_harness" / "backends"
spec = importlib.util.spec_from_file_location("issue004_windows_process", BASE / "windows_process.py")
executor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = executor
spec.loader.exec_module(executor)


def worker(case):
    def run(code, timeout=4, limit=4096):
        command = subprocess.list2cmdline([sys.executable, "-c", code])
        return executor.execute_windows(
            command, cwd=str(BASE), env=dict(os.environ), timeout=timeout, max_output_bytes=limit
        )

    started = time.monotonic()
    if case == "normal":
        result = run("import sys; print('hello'); print('error', file=sys.stderr)")
        assert result.returncode == 0 and not result.timed_out, result
        assert b"hello" in result.stdout and b"error" in result.stderr, result
    elif case == "timeout":
        result = run("import time; print('before', flush=True); time.sleep(60)", timeout=0.3)
        assert result.timed_out and result.returncode == 124, result
        assert b"before" in result.stdout, result
    elif case == "descendant":
        child = "import time; print('child', flush=True); time.sleep(60)"
        code = (
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
            + repr(child)
            + "], close_fds=False); time.sleep(0.3); print('root', flush=True)"
        )
        result = run(code)
        assert result.returncode == 0 and not result.timed_out, result
        assert b"root" in result.stdout and b"child" in result.stdout, result
    elif case == "bounded":
        result = run("print('x'*100000)", limit=128)
        assert len(result.stdout) == 128 and result.snapshot_truncated, result
    else:
        raise AssertionError(case)
    assert result.cleanup.root_reaped, result
    assert result.cleanup.job_empty, result
    assert not result.cleanup.deadline_exceeded, result
    assert not result.cleanup.errors, result
    assert time.monotonic() - started < 8, result


@unittest.skipUnless(os.name == "nt", "Windows integration tests")
class WindowsExecutorTests(unittest.TestCase):
    def watchdog(self, case):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", case],
            stdin=subprocess.DEVNULL,
            timeout=12,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, case)

    def test_normal(self):
        self.watchdog("normal")

    def test_timeout(self):
        self.watchdog("timeout")

    def test_descendant_keeps_output_handles(self):
        self.watchdog("descendant")

    def test_bounded_snapshot(self):
        self.watchdog("bounded")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
