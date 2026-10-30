#!/usr/bin/env python3
"""Long-lived kernel-load canary for I04 (optimisation log, 2026-10-29).

Question it answers: does a first-time GPU kernel load start failing with
CUDA_ERROR_NOT_PERMITTED once a CUDA process has been alive long enough, outside vLLM?
I04 (and DeepGEMM issue #443, also GB10) failed in DeepGEMM's load_kernel() at
cuKernelGetFunction, on the first load of a cached kernel 29-49 h after boot.

What it does, in the serving image (same CUDA runtime and driver path as vLLM):
  - creates the CUDA primary context through torch, as vLLM does;
  - every TICK seconds runs a small matmul, so the context stays in use;
  - every INTERVAL seconds loads each cubin in CUBINS as a NEW library with DeepGEMM's
    exact call sequence (csrc/jit/handle.hpp, pin 8b1392b, load_kernel l. 141-154):
    cuLibraryLoadFromFile -> cuLibraryGetKernelCount -> cuLibraryEnumerateKernels ->
    cuKernelGetFunction. Libraries are never unloaded, as in DeepGEMM;
  - writes one JSON line per load round to OUT (results per call, process uptime,
    number of libraries loaded so far, MemAvailable). It never stops on an error, so
    the log also shows whether a failure persists or recovers.
The kernels are only loaded, never launched: launching needs the real tensor arguments,
and the call that failed in I04 is the load.

Run it through tools/dg-canary.sh (start / status / stop), not directly.
Environment: CANARY_OUT (jsonl path), CANARY_INTERVAL (s, default 600),
CANARY_TICK (s, default 60), CANARY_CUBINS (colon-separated cubin paths).
"""
import ctypes, json, os, time

import torch

OUT = os.environ.get("CANARY_OUT", "/out/canary.jsonl")
INTERVAL = int(os.environ.get("CANARY_INTERVAL", "600"))
TICK = int(os.environ.get("CANARY_TICK", "60"))
CUBINS = [p for p in os.environ.get("CANARY_CUBINS", "").split(":") if p]

cuda = ctypes.CDLL("libcuda.so.1")
_name = ctypes.c_char_p()


def err(code):
    """CUresult -> 'CUDA_ERROR_...' (0 -> 'ok')."""
    if code == 0:
        return "ok"
    cuda.cuGetErrorName(code, ctypes.byref(_name))
    return f"{_name.value.decode() if _name.value else '?'} ({code})"


def load(path):
    """One DeepGEMM-style load of `path`; returns {call: result} up to the first error."""
    lib, n, kern, func = ctypes.c_void_p(), ctypes.c_uint(), ctypes.c_void_p(), ctypes.c_void_p()
    steps = [
        ("cuLibraryLoadFromFile", lambda: cuda.cuLibraryLoadFromFile(
            ctypes.byref(lib), path.encode(), None, None, 0, None, None, 0)),
        ("cuLibraryGetKernelCount", lambda: cuda.cuLibraryGetKernelCount(ctypes.byref(n), lib)),
        ("cuLibraryEnumerateKernels", lambda: cuda.cuLibraryEnumerateKernels(ctypes.byref(kern), 1, lib)),
        ("cuKernelGetFunction", lambda: cuda.cuKernelGetFunction(ctypes.byref(func), kern)),
    ]
    res = {}
    for name, call in steps:
        res[name] = err(call())
        if res[name] != "ok":
            break
        if name == "cuLibraryGetKernelCount" and n.value != 1:   # DeepGEMM asserts exactly 1
            res[name] = f"kernel count {n.value}, expected 1"
            break
    return res


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return round(int(line.split()[1]) / 1048576, 1)


def main():
    assert CUBINS, "CANARY_CUBINS is empty"
    a = torch.randn(256, 256, device="cuda")       # creates the primary context, as vLLM does
    ctx = ctypes.c_void_p()
    t0, loaded, next_load = time.monotonic(), 0, 0.0
    with open(OUT, "a") as log:
        log.write(json.dumps({"event": "start", "time": time.strftime("%FT%T%z"), "pid": os.getpid(),
                              "interval_s": INTERVAL, "tick_s": TICK, "cubins": CUBINS,
                              "torch": torch.__version__, "cuda": torch.version.cuda}) + "\n")
        log.flush()
        while True:
            (a @ a).sum().item()                       # keep the context in use
            now = time.monotonic()
            if now >= next_load:
                cuda.cuCtxGetCurrent(ctypes.byref(ctx))
                results = {os.path.basename(os.path.dirname(p)) or p: load(p) for p in CUBINS}
                loaded += sum(r.get("cuKernelGetFunction") == "ok" for r in results.values())
                ok = all(r.get("cuKernelGetFunction") == "ok" for r in results.values())
                log.write(json.dumps({"event": "load", "time": time.strftime("%FT%T%z"),
                                      "uptime_h": round((now - t0) / 3600, 3), "ok": ok,
                                      "ctx_current": bool(ctx.value), "libraries_loaded": loaded,
                                      "mem_available_gib": mem_available_gib(),
                                      "results": results}) + "\n")
                log.flush()
                next_load = now + INTERVAL
            time.sleep(TICK)


if __name__ == "__main__":
    main()
