"""Windows file-backed execution with suspended job assignment (no pipe readers)."""

from __future__ import annotations

import ctypes as c
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from ctypes import wintypes as w
from dataclasses import dataclass, field
from threading import Event
from typing import IO, Any

logger = logging.getLogger(__name__)


@dataclass
class CleanupStatus:
    terminated: bool = False
    root_reaped: bool = False
    job_empty: bool = False
    deadline_exceeded: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class WindowsResult:
    args: str
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    cleanup: CleanupStatus
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def snapshot_truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated


class STARTUPINFO(c.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("lpReserved", w.LPWSTR),
        ("lpDesktop", w.LPWSTR),
        ("lpTitle", w.LPWSTR),
        ("dwX", w.DWORD),
        ("dwY", w.DWORD),
        ("dwXSize", w.DWORD),
        ("dwYSize", w.DWORD),
        ("dwXCountChars", w.DWORD),
        ("dwYCountChars", w.DWORD),
        ("dwFillAttribute", w.DWORD),
        ("dwFlags", w.DWORD),
        ("wShowWindow", w.WORD),
        ("cbReserved2", w.WORD),
        ("lpReserved2", c.POINTER(c.c_byte)),
        ("hStdInput", w.HANDLE),
        ("hStdOutput", w.HANDLE),
        ("hStdError", w.HANDLE),
    ]


class STARTUPINFOEX(c.Structure):
    _fields_ = [("StartupInfo", STARTUPINFO), ("lpAttributeList", c.c_void_p)]


class ProcessInformation(c.Structure):
    _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE), ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]


class BasicLimit(c.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", c.c_longlong),
        ("PerJobUserTimeLimit", c.c_longlong),
        ("LimitFlags", w.DWORD),
        ("MinimumWorkingSetSize", c.c_size_t),
        ("MaximumWorkingSetSize", c.c_size_t),
        ("ActiveProcessLimit", w.DWORD),
        ("Affinity", c.c_size_t),
        ("PriorityClass", w.DWORD),
        ("SchedulingClass", w.DWORD),
    ]


class IoCounters(c.Structure):
    _fields_ = [
        (name, c.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class ExtendedLimit(c.Structure):
    _fields_ = [
        ("BasicLimitInformation", BasicLimit),
        ("IoInfo", IoCounters),
        ("ProcessMemoryLimit", c.c_size_t),
        ("JobMemoryLimit", c.c_size_t),
        ("PeakProcessMemoryUsed", c.c_size_t),
        ("PeakJobMemoryUsed", c.c_size_t),
    ]


class ACCOUNTING(c.Structure):
    _fields_ = [
        (name, c.c_longlong)
        for name in ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")
    ] + [
        (name, w.DWORD)
        for name in (
            "TotalPageFaultCount",
            "TotalProcesses",
            "ActiveProcesses",
            "TotalTerminatedProcesses",
        )
    ]


if sys.platform == "win32":

    class _WindowsApi:
        def __init__(self) -> None:
            library = c.WinDLL("kernel32", use_last_error=True)

            def bind(name: str, restype: type[Any] | None, *argtypes: type[Any]) -> Callable[..., Any]:
                function = library[name]
                function.restype = restype
                function.argtypes = argtypes
                return function

            self.create_job = bind("CreateJobObjectW", w.HANDLE, c.c_void_p, w.LPCWSTR)
            self.set_job = bind("SetInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
            self.assign = bind("AssignProcessToJobObject", w.BOOL, w.HANDLE, w.HANDLE)
            self.create = bind(
                "CreateProcessW",
                w.BOOL,
                w.LPCWSTR,
                w.LPWSTR,
                c.c_void_p,
                c.c_void_p,
                w.BOOL,
                w.DWORD,
                c.c_void_p,
                w.LPCWSTR,
                c.POINTER(STARTUPINFOEX),
                c.POINTER(ProcessInformation),
            )
            self.resume = bind("ResumeThread", w.DWORD, w.HANDLE)
            self.wait = bind("WaitForSingleObject", w.DWORD, w.HANDLE, w.DWORD)
            self.terminate = bind("TerminateJobObject", w.BOOL, w.HANDLE, w.UINT)
            self.terminate_process = bind("TerminateProcess", w.BOOL, w.HANDLE, w.UINT)
            self.close = bind("CloseHandle", w.BOOL, w.HANDLE)
            self.get_exit = bind("GetExitCodeProcess", w.BOOL, w.HANDLE, c.POINTER(w.DWORD))
            self.query = bind("QueryInformationJobObject", w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p)
            self.init_attr = bind(
                "InitializeProcThreadAttributeList", w.BOOL, c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)
            )
            self.update_attr = bind(
                "UpdateProcThreadAttribute",
                w.BOOL,
                c.c_void_p,
                w.DWORD,
                c.c_size_t,
                c.c_void_p,
                c.c_size_t,
                c.c_void_p,
                c.c_void_p,
            )
            self.delete_attr = bind("DeleteProcThreadAttributeList", None, c.c_void_p)

    def _check(ok: object) -> None:
        if not ok:
            raise c.WinError(c.get_last_error())

    def _create_suspended(
        api: _WindowsApi,
        command: str,
        cwd: str,
        *,
        env: dict[str, str],
        handles: list[int],
        process: ProcessInformation,
    ) -> None:
        startup = STARTUPINFOEX()
        size = c.c_size_t()
        api.init_attr(None, 1, 0, c.byref(size))
        attributes = c.create_string_buffer(size.value)
        startup.lpAttributeList = c.cast(attributes, c.c_void_p)
        _check(api.init_attr(startup.lpAttributeList, 1, 0, c.byref(size)))
        try:
            handle_list = (w.HANDLE * 3)(*handles)
            _check(api.update_attr(startup.lpAttributeList, 0, 0x20002, handle_list, c.sizeof(handle_list), None, None))
            startup.StartupInfo.cb = c.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x100
            startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = handles
            shell = (
                env.get("COMSPEC")
                or os.environ.get("COMSPEC")
                or os.path.join(os.environ["SYSTEMROOT"], "System32", "cmd.exe")
            )
            cmdline = c.create_unicode_buffer(subprocess.list2cmdline([shell]) + ' /d /s /c "' + command + '"')
            variables = sorted(env.items(), key=lambda item: item[0].upper())
            environment = c.create_unicode_buffer("\0".join(f"{key}={value}" for key, value in variables) + "\0\0")
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            try:
                _check(
                    api.create(
                        shell,
                        cmdline,
                        None,
                        None,
                        True,
                        0x4 | 0x400 | 0x80000,
                        environment,
                        cwd,
                        c.byref(startup),
                        c.byref(process),
                    )
                )
            finally:
                for handle in handles:
                    os.set_handle_inheritable(handle, False)
        finally:
            api.delete_attr(startup.lpAttributeList)

    def _resume_and_wait(
        api: _WindowsApi, process: ProcessInformation, timeout: float, cancel_event: Event | None
    ) -> int | None:
        if cancel_event is not None and cancel_event.is_set():
            return None
        if api.resume(process.hThread) == 0xFFFFFFFF:
            raise c.WinError(c.get_last_error())

        deadline = time.monotonic() + timeout
        while cancel_event is None or not cancel_event.is_set():
            remaining_ms = max(0, math.ceil((deadline - time.monotonic()) * 1000))
            # Keep the synchronous fast path and poll only when a caller can cancel.
            wait_ms = min(remaining_ms, 50 if cancel_event is not None else 0xFFFFFFFE)
            outcome = int(api.wait(process.hProcess, wait_ms))
            if outcome not in (0, 258):
                raise c.WinError(c.get_last_error())
            if outcome == 0 or time.monotonic() >= deadline:
                return outcome
        return None

    def _cleanup_process(
        api: _WindowsApi,
        process: ProcessInformation,
        job: int | None,
        *,
        assigned: bool,
        termination_code: int,
        timeout: float,
    ) -> CleanupStatus:
        status = CleanupStatus()
        deadline = time.monotonic() + timeout
        if process.hProcess:
            ok = api.terminate(job, termination_code) if assigned else api.terminate_process(process.hProcess, 1)
            status.terminated = bool(ok)
            if not ok:
                status.errors.append(f"terminate: {c.get_last_error()}")
            remaining_ms = max(0, math.floor((deadline - time.monotonic()) * 1000))
            root_wait = api.wait(process.hProcess, remaining_ms)
            if root_wait == 0xFFFFFFFF:
                status.errors.append(f"wait for root: {c.get_last_error()}")
            status.root_reaped = root_wait == 0
            if assigned:
                while True:
                    accounting = ACCOUNTING()
                    if not api.query(job, 1, c.byref(accounting), c.sizeof(accounting), None):
                        status.errors.append(f"query: {c.get_last_error()}")
                        break
                    status.job_empty = accounting.ActiveProcesses == 0
                    if status.job_empty or time.monotonic() >= deadline:
                        break
                    time.sleep(min(0.01, max(0, deadline - time.monotonic())))
            incomplete = not status.root_reaped or (assigned and not status.job_empty)
            status.deadline_exceeded = incomplete and (root_wait == 258 or time.monotonic() >= deadline)
        for handle in (process.hThread, process.hProcess, job):
            if handle and not api.close(handle):
                status.errors.append(f"close: {c.get_last_error()}")
        if status.errors or status.deadline_exceeded:
            logger.warning(
                "Incomplete Windows shell cleanup (pid=%s, errors=%s, deadline_exceeded=%s)",
                process.dwProcessId or None,
                status.errors,
                status.deadline_exceeded,
                extra={
                    "cleanup_errors": status.errors,
                    "deadline_exceeded": status.deadline_exceeded,
                    "terminated": status.terminated,
                    "root_reaped": status.root_reaped,
                    "job_empty": status.job_empty,
                },
            )
        return status

    def _snapshot(stream: IO[bytes], limit: int) -> tuple[bytes, bool]:
        size = os.fstat(stream.fileno()).st_size
        stream.seek(0)
        return stream.read(min(size, limit)), size > limit

    def _execute_windows(
        command: str,
        *,
        cwd: str,
        env: dict[str, str],
        timeout: float,
        max_output_bytes: int,
        cleanup_timeout: float,
        cancel_event: Event | None,
    ) -> WindowsResult:
        """Run cmd.exe and terminate remaining job members after root exit.

        Cleanup waits share one monotonic deadline. Snapshots read at most the
        supplied byte limit per stream and never wait for a descendant's EOF.
        Win32 setup failures propagate without falling back to the pipe executor.
        """
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(cleanup_timeout) or cleanup_timeout < 0:
            raise ValueError("invalid execution or cleanup timeout")
        if max_output_bytes < 0:
            raise ValueError("negative output limit")
        import msvcrt

        if cancel_event is not None and cancel_event.is_set():
            return WindowsResult(command, 130, b"", b"", False, CleanupStatus())

        api = _WindowsApi()
        process = ProcessInformation()
        job = None
        assigned = False
        termination_code = 1
        code = w.DWORD()
        with (
            tempfile.TemporaryFile("w+b", dir=cwd) as out,
            tempfile.TemporaryFile("w+b", dir=cwd) as err,
            open(os.devnull, "rb") as devnull,
        ):
            handles = [msvcrt.get_osfhandle(stream.fileno()) for stream in (devnull, out, err)]
            try:
                job = api.create_job(None, None)
                _check(job)
                limits = ExtendedLimit()
                limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                _check(api.set_job(job, 9, c.byref(limits), c.sizeof(limits)))
                _create_suspended(api, command, cwd, env=env, handles=handles, process=process)
                _check(api.assign(job, process.hProcess))
                assigned = True
                outcome = _resume_and_wait(api, process, timeout, cancel_event)
                if outcome == 0:
                    _check(api.get_exit(process.hProcess, c.byref(code)))
                else:
                    termination_code = 130 if outcome is None else 124
            finally:
                cleanup = _cleanup_process(
                    api, process, job, assigned=assigned, termination_code=termination_code, timeout=cleanup_timeout
                )
            stdout, out_cut = _snapshot(out, max_output_bytes)
            stderr, err_cut = _snapshot(err, max_output_bytes)
        return WindowsResult(
            command,
            code.value if outcome == 0 else termination_code,
            stdout,
            stderr,
            outcome == 258,
            cleanup,
            out_cut,
            err_cut,
        )


def execute_windows(
    command: str,
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    max_output_bytes: int,
    cleanup_timeout: float = 2.0,
    cancel_event: Event | None = None,
) -> WindowsResult:
    """Run a Windows command with file-backed output and bounded Job cleanup."""
    if sys.platform == "win32":
        return _execute_windows(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            cleanup_timeout=cleanup_timeout,
            cancel_event=cancel_event,
        )
    raise OSError("Windows executor requires Windows")
