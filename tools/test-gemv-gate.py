#!/usr/bin/env python3
"""CPU test of the compressor GEMV patch gate (patches/dsv4-attention.py).

Checks the dispatch rule of `_dsv4_mm_fp32` without importing vLLM: the helper and its
constants are extracted from the patch file with `ast` and run against plain torch.
  - M = 1 and N > 512: chunked into ceil(N / 512) torch.mm calls, result equal to one mm
  - M = 2..8 (DSpark verify/draft widths) and prefill-sized M: exactly one torch.mm call
  - N <= 512 at any M: exactly one torch.mm call
Speed is not tested here; that is tools/gemv-bench.py on the GPU. CPU torch has no
bf16 `torch.mm(..., out_dtype=float32)` overload, so the test emulates it (upcast, fp32 mm);
that checks the dispatch and the chunk/concat arithmetic, not the CUDA kernel.

Run in the serving image without GPU or network (needs only torch):
  docker run --rm --network none -v /srv/dsv4/run:/r:ro --entrypoint python3 \
    local/vllm-openai:46638857-fi0618 /r/tools/test-gemv-gate.py /r/patches/dsv4-attention.py
"""
import ast
import sys

import torch

PATCH = sys.argv[1] if len(sys.argv) > 1 else "/srv/dsv4/run/patches/dsv4-attention.py"
WANTED = {"_DSV4_GEMV_CHUNK", "_DSV4_GEMV_MAX_M", "_dsv4_mm_fp32"}


def load_helper(path):
    """Return a namespace holding the patch's GEMV constants and helper."""
    tree = ast.parse(open(path).read(), path)
    nodes = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
        or (isinstance(n, ast.Assign) and {t.id for t in n.targets} & WANTED)
    ]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    missing = WANTED - ns.keys()
    assert not missing, f"not found in {path}: {missing}"
    return ns


_REAL_MM = torch.mm


def mm_fp32_out(a, b, out_dtype=None):
    """CPU stand-in for the CUDA-only bf16 torch.mm(..., out_dtype=torch.float32)."""
    assert out_dtype == torch.float32, "the patch must request fp32 output"
    return _REAL_MM(a.float(), b.float())


def count_mm(fn, *args):
    """Call fn with torch.mm replaced by the stand-in; count the calls it makes."""
    calls = 0
    real = torch.mm

    def wrapped(*a, **k):
        nonlocal calls
        calls += 1
        return mm_fp32_out(*a, **k)

    torch.mm = wrapped
    try:
        out = fn(*args)
    finally:
        torch.mm = real
    return out, calls


def main():
    ns = load_helper(PATCH)
    mm, chunk, max_m = ns["_dsv4_mm_fp32"], ns["_DSV4_GEMV_CHUNK"], ns["_DSV4_GEMV_MAX_M"]
    print(f"{PATCH}: chunk={chunk} max_m={max_m}")
    torch.manual_seed(0)
    failures = 0
    for n in (512, 1024, 2048):
        w = torch.randn(n, 4096, dtype=torch.bfloat16)
        for m in (1, 2, 4, 6, 8, 4096):
            x = torch.randn(m, 4096, dtype=torch.bfloat16)
            out, calls = count_mm(mm, x, w)
            ref = mm_fp32_out(x, w.T, out_dtype=torch.float32)
            # The required rule, independent of the file's constants: chunk only at M = 1.
            want = -(-n // chunk) if (m == 1 and n > chunk) else 1
            rel = ((out - ref).abs().max() / ref.abs().max()).item()
            ok = calls == want and out.dtype == torch.float32 and out.shape == ref.shape and rel < 1e-5
            failures += not ok
            print(f"N={n:5d} M={m:5d} mm_calls={calls} expected={want} rel_diff={rel:.1e} "
                  f"{'ok' if ok else 'FAIL'}")
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
