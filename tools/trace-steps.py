#!/usr/bin/env python3
"""Per-step GPU time breakdown from one vLLM torch-profiler trace (B1, log 2026-11-01).

Steps are the engine-step annotations `execute_context_X(n)_generation_Y(m)`. Each GPU
kernel/memcpy/memset is attributed to the CPU scope that launched it, via the launch's
correlation id: the innermost `gpu_model_runner: <scope>` annotation (forward, sample, draft,
...; present only with VLLM_CUSTOM_SCOPES_FOR_PROFILING=1, set by DSV4_PROFILE=1 since
2026-11-01), else "unscoped". Kernels are also grouped into coarse classes by name.

Usage:  trace-steps.py <trace.json.gz> [--json out.json]
Steps come from the CPU-side annotation (one per step); each kernel belongs to the step whose
CPU span contains its launch. Prints, per step kind (context/generation token counts), medians
of: step-to-step time (CPU step starts), GPU wall (first to last activity of the step), busy
(union of activity over all streams), idle, and kernel time per scope and per kernel class
(summed over streams, so it can exceed busy).
"""
import bisect, collections, gzip, json, re, statistics, sys

CLASSES = [  # first match wins; order matters
    ("nccl", r"nccl|allreduce|all_reduce|allgather|all_gather|reduce_scatter|cross_device"),
    ("moe", r"moe|expert|fused_experts|grouped|fp4|nvfp4|topk_softmax|silu_and_mul"),
    ("indexer", r"indexer|top_?k|topk|fp8_mqa_logits|paged_mqa_logits"),
    ("attention", r"mla|flash|attn|attention|sparse|rope|rotary"),
    ("gemm_gemv", r"gemm|gemv|cutlass|nvjet|sm90|sm100|sm120|sm121|einsum|matmul|cublas"),
    ("sampling", r"sampl|softmax|argmax|rejection|multinomial|gumbel"),
    ("norm_elementwise", r"norm|elementwise|vectorized|reduce|copy|cat|index|scatter|gather|fill"),
    ("memcpy_memset", r"^memcpy|^memset|Memcpy|Memset"),
]
CLS_RE = [(n, re.compile(p, re.I)) for n, p in CLASSES]
STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")


def klass(name):
    for n, r in CLS_RE:
        if r.search(name):
            return n
    return "other"


def union_len(iv):
    tot, end = 0.0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s; end = e
        elif e > end:
            tot += e - end; end = e
    return tot


def main():
    path = sys.argv[1]
    ev = json.load(gzip.open(path) if path.endswith(".gz") else open(path))["traceEvents"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    launch_ts = {}  # correlation -> (pid, tid, ts) of the CPU launch call
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launch_ts[e["args"]["correlation"]] = (e["pid"], e["tid"], e["ts"])
    scopes = collections.defaultdict(list)  # (pid, tid) -> [(ts, end, name)]
    for e in ev:
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("gpu_model_runner: "):
            scopes[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"], e["name"][18:]))
    for v in scopes.values():
        v.sort()
    starts = {k: [s for s, _, _ in v] for k, v in scopes.items()}

    def scope_of(corr):
        if corr not in launch_ts:
            return "unscoped"
        pid, tid, ts = launch_ts[corr]
        v = scopes.get((pid, tid))
        if not v:
            return "unscoped"
        best = None  # innermost = latest start that still contains ts
        for s, e, n in v[:bisect.bisect_right(starts[(pid, tid)], ts)][::-1][:50]:
            if s <= ts <= e:
                best = n; break
        return best or "unscoped"

    steps = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
                   if e.get("cat") == "user_annotation" and STEP_RE.match(e.get("name", "")))
    if not steps:
        sys.exit("no engine-step annotations found")
    sstart = [s for s, _, _ in steps]
    per_step = collections.defaultdict(list)
    for k in gpu:  # a kernel belongs to the step whose CPU span contains its launch call
        c = launch_ts.get(k.get("args", {}).get("correlation"))
        if c is None:
            continue
        i = bisect.bisect_right(sstart, c[2]) - 1
        if i >= 0 and c[2] <= steps[i][1]:
            per_step[i].append(k)
    rows = []
    for i, (s, e, name) in enumerate(steps):
        ks = per_step.get(i)
        if not ks:
            continue
        iv = [(k["ts"], k["ts"] + k["dur"]) for k in ks]
        wall = max(b for _, b in iv) - min(a for a, _ in iv)
        busy = union_len(iv)
        by_scope, by_cls = collections.Counter(), collections.Counter()
        for k in ks:
            by_scope[scope_of(k.get("args", {}).get("correlation"))] += k["dur"]
            by_cls[klass(k.get("name", ""))] += k["dur"]
        gap = (steps[i + 1][0] - s) / 1e3 if i + 1 < len(steps) else None
        m = STEP_RE.match(name)
        rows.append({"step": i, "kind": f"ctx{m.group(2)}_gen{m.group(4)}", "wall_ms": wall / 1e3,
                     "busy_ms": busy / 1e3, "idle_ms": (wall - busy) / 1e3, "kernels": len(ks),
                     "step_to_step_ms": gap,
                     "scope_ms": {k: v / 1e3 for k, v in by_scope.items()},
                     "class_ms": {k: v / 1e3 for k, v in by_cls.items()}})
    kinds = collections.defaultdict(list)
    for r in rows:
        kinds[r["kind"]].append(r)
    med = lambda xs: statistics.median(xs) if xs else 0.0
    print(f"trace: {path}\nsteps: {len(rows)}")
    for kind, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        print(f"\n== {kind}: {len(rs)} steps (medians, ms)")
        gaps = [r["step_to_step_ms"] for r in rs if r["step_to_step_ms"] is not None]
        print(f"step-to-step {med(gaps):.2f}  ", end="")
        print(f"wall {med([r['wall_ms'] for r in rs]):.2f}  busy {med([r['busy_ms'] for r in rs]):.2f}  "
              f"idle {med([r['idle_ms'] for r in rs]):.2f}  kernels {med([r['kernels'] for r in rs]):.0f}")
        for field in ("scope_ms", "class_ms"):
            keys = sorted({k for r in rs for k in r[field]}, key=lambda k: -med([r[field].get(k, 0) for r in rs]))
            print(f"  {field[:-3]}: " + ", ".join(f"{k} {med([r[field].get(k, 0) for r in rs]):.2f}" for k in keys))
    if "--json" in sys.argv:
        json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
