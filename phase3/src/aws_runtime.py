"""Runtime environment, memory measurement, hashing and S3 archival.

Memory follows the Phase 2.5 method (phase2/tools/memwatch.ps1): the main-process peak working set /
resident set, and the peak of the summed memory of every python process sampled every 2 s (joblib
workers included). Works on Windows (the development workstation) and Linux (EC2).
"""
import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Where the run executed: EC2 is recognised from the DMI vendor string (no network call)."""
    vendor = Path("/sys/devices/virtual/dmi/id/board_vendor")
    ec2 = vendor.exists() and "Amazon" in vendor.read_text()
    return {"host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
            "logical_cpus": os.cpu_count(), "total_memory_mb": _total_memory_mb(),
            "execution": "aws_ec2" if ec2 else "local_workstation"}


def _total_memory_mb() -> int:
    if sys.platform == "win32":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "tp", "ap", "tv", "av", "ae")]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.total / 2**20)
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20)


def main_peak_mb() -> int:
    """Exact OS-recorded peak of this process (Windows PeakWorkingSetSize, Linux ru_maxrss)."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "a", "b", "c", "d",
                                               "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize / 2**20)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def all_python_mb() -> int:
    """Current summed working set / RSS of every python process on the machine."""
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        kb = [int(line.rsplit('","', 1)[-1].strip('"\r K').replace(",", "").replace("\xa0", "").replace(".", ""))
              for line in out.splitlines() if line.startswith('"python')]
        return sum(kb) // 1024
    total = 0
    for p in Path("/proc").iterdir():
        try:
            if p.name.isdigit() and (p / "comm").read_text().startswith("python"):
                total += next(int(l.split()[1]) for l in (p / "status").read_text().splitlines() if l.startswith("VmRSS"))
        except (OSError, StopIteration):  # the process exited between listing and reading
            pass
    return total // 1024


class MemorySampler(threading.Thread):
    """Samples all_python_mb() every ``every`` seconds until stop(); peak() returns both peaks."""

    def __init__(self, every: float = 2.0):
        super().__init__(daemon=True)
        self.every, self.all_peak, self._stop = every, 0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.all_peak = max(self.all_peak, all_python_mb())
            self._stop.wait(self.every)

    def stop(self) -> dict:
        self._stop.set()
        self.join()
        self.all_peak = max(self.all_peak, all_python_mb())
        return {"main_process_peak_mb": main_peak_mb(), "all_python_processes_peak_mb": self.all_peak,
                "method": "main: OS peak working set / ru_maxrss (exact); all: sum over python processes sampled "
                          f"every {self.every:g} s"}


def archive_to_s3(local_dir: Path, s3_uri: str) -> list:
    """Upload every file under local_dir to s3://bucket/prefix/<relative path> (the experiment's own
    prefix; objects are never deleted). Called only on AWS with an existing, approved bucket."""
    import boto3
    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    s3, keys = boto3.client("s3"), []
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}"
            s3.upload_file(str(p), bucket, key)
            keys.append(key)
    return keys
