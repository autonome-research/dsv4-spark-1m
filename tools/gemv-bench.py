#!/usr/bin/env python3
"""Isolated BF16 GEMV microbenchmark for the DSV4 compressor patch (log E07.8).

Times torch.mm(x, W.T, out_dtype=float32), the call DeepSeek-V4's compressor uses, against
the same product issued in 512-row chunks (patches/dsv4-attention.py), for K = 4096 and
N = 512 / 1024 / 2048 at M = 1, 2, 4, 8. Weights rotate over enough copies to defeat the L2
cache (decode reads each weight once per step), and each case is timed as a CUDA graph
replay. Prints the environment, raw per-case medians and the max relative difference.

Run inside the serving image (needs about 1 GB of GPU memory, GPU otherwise idle):
  docker run --rm --gpus all -v /srv/dsv4/run/tools:/t:ro --entrypoint python3 \
    local/vllm-openai:46638857-fi0618 /t/gemv-bench.py
"""
import statistics

import torch

K, CHUNK, COPIES, REPS, ROUNDS = 4096, 512, 40, 20, 7


def full(x, w):
    return torch.mm(x, w.T, out_dtype=torch.float32)


def chunked(x, w):
    return torch.cat(
        [torch.mm(x, w[r : r + CHUNK].T, out_dtype=torch.float32)
         for r in range(0, w.shape[0], CHUNK)],
        dim=1,
    )


def time_us(fn, x, ws):
    """Median µs per call over ROUNDS graph replays of REPS × COPIES calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w in ws[:3]:
            fn(x, w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            for w in ws:
                fn(x, w)
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(ROUNDS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / (REPS * COPIES))
    return statistics.median(out)


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  {p.name}  "
          f"sm_{p.major}{p.minor}  L2 {p.L2_cache_size // 2**20} MiB")
    print(f"K={K} chunk={CHUNK} copies={COPIES} reps={REPS} rounds={ROUNDS} (CUDA graph)")
    print("N\tM\tfull_us\tfull_GBs\tchunk_us\tchunk_GBs\tspeedup\tmax_rel_diff")
    torch.manual_seed(0)
    for n in (512, 1024, 2048):
        ws = [torch.randn(n, K, device="cuda", dtype=torch.bfloat16) for _ in range(COPIES)]
        mb = n * K * 2 / 1e6
        for m in (1, 2, 4, 8):
            x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
            tf = time_us(full, x, ws)
            tc = time_us(chunked, x, ws) if n > CHUNK else float("nan")
            ref, got = full(x, ws[0]), chunked(x, ws[0])
            diff = ((ref - got).abs().max() / ref.abs().max()).item()
            print(f"{n}\t{m}\t{tf:.1f}\t{mb / tf * 1e3:.0f}\t{tc:.1f}\t"
                  f"{mb / tc * 1e3:.0f}\t{tf / tc:.2f}\t{diff:.1e}")
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
