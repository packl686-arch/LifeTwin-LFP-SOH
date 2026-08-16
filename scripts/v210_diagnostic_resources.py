"""Result-blind resource sampling for synthetic V2.10 diagnostics."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import sys
import threading


_SAMPLE_INTERVAL_SECONDS = 0.2


if sys.platform == "win32":
    _TH32CS_SNAPPROCESS = 0x00000002
    _PROCESS_QUERY_INFORMATION = 0x0400
    _PROCESS_VM_READ = 0x0010
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class _ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]


def _disk_free_bytes() -> int | None:
    try:
        anchor = Path.cwd().anchor or str(Path.cwd())
        return int(shutil.disk_usage(anchor).free)
    except OSError:
        return None


def _available_physical_memory_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def _windows_process_parents() -> dict[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    create_snapshot.restype = wintypes.HANDLE
    process_first = kernel32.Process32FirstW
    process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
    process_first.restype = wintypes.BOOL
    process_next = kernel32.Process32NextW
    process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
    process_next.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    snapshot = create_snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot == _INVALID_HANDLE_VALUE:
        return {}
    parents: dict[int, int] = {}
    try:
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        present = bool(process_first(snapshot, ctypes.byref(entry)))
        while present:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            present = bool(process_next(snapshot, ctypes.byref(entry)))
    finally:
        close_handle(snapshot)
    return parents


def _descendant_process_ids(root_process_id: int) -> set[int]:
    parents = _windows_process_parents()
    descendants = {root_process_id}
    changed = True
    while changed:
        changed = False
        for process_id, parent_id in parents.items():
            if parent_id in descendants and process_id not in descendants:
                descendants.add(process_id)
                changed = True
    return descendants


def _windows_process_memory(process_id: int) -> tuple[int, int] | None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    get_memory = psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL

    handle = open_process(
        _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ,
        False,
        process_id,
    )
    if not handle:
        return None
    try:
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not get_memory(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)
    finally:
        close_handle(handle)


def _process_tree_sample() -> tuple[int, int, int]:
    if sys.platform != "win32":
        return 1, 0, 0
    process_ids = _descendant_process_ids(os.getpid())
    working_set = 0
    private_bytes = 0
    observed = 0
    for process_id in process_ids:
        memory = _windows_process_memory(process_id)
        if memory is None:
            continue
        observed += 1
        working_set += memory[0]
        private_bytes += memory[1]
    return observed, working_set, private_bytes


@dataclass
class ResourceSampler:
    """Sample only aggregate process-tree and host capacity statistics."""

    interval_seconds: float = _SAMPLE_INTERVAL_SECONDS
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _sample_count: int = field(init=False, default=0)
    _sampling_error_count: int = field(init=False, default=0)
    _peak_process_count: int = field(init=False, default=0)
    _peak_working_set_bytes: int = field(init=False, default=0)
    _peak_private_bytes: int = field(init=False, default=0)
    _minimum_available_memory_bytes: int | None = field(init=False, default=None)
    _disk_free_start_bytes: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()

    def _sample(self) -> None:
        try:
            process_count, working_set, private_bytes = _process_tree_sample()
            available = _available_physical_memory_bytes()
        except (OSError, ValueError):
            self._sampling_error_count += 1
            return
        self._sample_count += 1
        self._peak_process_count = max(self._peak_process_count, process_count)
        self._peak_working_set_bytes = max(
            self._peak_working_set_bytes,
            working_set,
        )
        self._peak_private_bytes = max(self._peak_private_bytes, private_bytes)
        if available is not None:
            if self._minimum_available_memory_bytes is None:
                self._minimum_available_memory_bytes = available
            else:
                self._minimum_available_memory_bytes = min(
                    self._minimum_available_memory_bytes,
                    available,
                )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Resource sampler already started")
        self._disk_free_start_bytes = _disk_free_bytes()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | str | None]:
        if self._thread is None:
            raise RuntimeError("Resource sampler was not started")
        self._stop_event.set()
        self._thread.join()
        self._sample()
        return {
            "backend": (
                "windows_toolhelp_psapi_v1"
                if sys.platform == "win32"
                else "portable_parent_only_v1"
            ),
            "sample_count": self._sample_count,
            "sampling_error_count": self._sampling_error_count,
            "peak_process_tree_process_count": self._peak_process_count,
            "peak_worker_process_count": max(self._peak_process_count - 1, 0),
            "peak_process_tree_working_set_bytes": self._peak_working_set_bytes,
            "peak_process_tree_private_bytes": self._peak_private_bytes,
            "minimum_available_physical_memory_bytes": (
                self._minimum_available_memory_bytes
            ),
            "disk_free_start_bytes": self._disk_free_start_bytes,
            "disk_free_end_bytes": _disk_free_bytes(),
        }
