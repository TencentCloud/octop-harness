"""Integration checks for the opt-in Windows v2 local-shell executor."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from octop_harness.backends.local_shell import HarnessLocalShellBackend

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows integration test")
class LocalShellV2Tests(unittest.TestCase):
    def make_backend(self) -> HarnessLocalShellBackend:
        return HarnessLocalShellBackend(
            root_dir=str(ROOT),
            workspace_dir=str(ROOT),
            virtual_mode=False,
            windows_executor="v2_file_job",
        )

    def test_environment_switch(self) -> None:
        previous = os.environ.get("OCTOP_WINDOWS_EXECUTOR")
        os.environ["OCTOP_WINDOWS_EXECUTOR"] = "v2_file_job"
        try:
            backend = HarnessLocalShellBackend(
                root_dir=str(ROOT),
                workspace_dir=str(ROOT),
                virtual_mode=False,
            )
            self.assertEqual(backend._windows_executor, "v2_file_job")
        finally:
            if previous is None:
                os.environ.pop("OCTOP_WINDOWS_EXECUTOR", None)
            else:
                os.environ["OCTOP_WINDOWS_EXECUTOR"] = previous

    def test_execute_normal(self) -> None:
        backend = self.make_backend()
        result = backend.execute(
            subprocess.list2cmdline([sys.executable, "-c", "print('v2-normal')"]),
            timeout=5,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("v2-normal", result.output)

    def test_execute_timeout_with_descendant_holding_output(self) -> None:
        child = "import time; print('v2-child', flush=True); time.sleep(60)"
        code = (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c'," + repr(child) + "], close_fds=False); "
            "time.sleep(60)"
        )
        command = subprocess.list2cmdline([sys.executable, "-c", code])
        backend = self.make_backend()
        started = time.monotonic()
        result = backend.execute(command, timeout=0.5)
        self.assertEqual(result.exit_code, 124)
        self.assertIn("timed out", result.output.lower())
        self.assertLess(time.monotonic() - started, 8)
        cleanup = backend._last_windows_cleanup
        self.assertIsNotNone(cleanup)
        self.assertTrue(cleanup.root_reaped, cleanup)
        self.assertTrue(cleanup.job_empty, cleanup)
        self.assertFalse(cleanup.deadline_exceeded, cleanup)


if __name__ == "__main__":
    unittest.main()
