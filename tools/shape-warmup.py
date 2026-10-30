#!/usr/bin/env python3
"""Per-boot shape warm-up: load every per-token-count GPU kernel while the server is young.

Why (optimisation log, I04 investigation 2026-10-29): for steps of 1-32 tokens DeepGEMM's
fp8_einsum (attention o_proj) compiles the exact token count into the kernel, and
tf32_hc_prenorm_gemm is specialised by a split count that also depends on the token count.
vLLM's boot warm-up covers neither. Each such kernel is loaded on its first use in the
process, and on 2026-10-28 such a first load (a 17-token prompt, 2 days after boot) failed
with CUDA_ERROR_NOT_PERMITTED and killed the engine (I04; same as DeepGEMM issue #443).
Loading them all right after boot removes that trigger. It does not explain the driver error.

What it does: sends one /v1/completions request per prompt length in LENGTHS, with a
token-ID prompt (no chat template, no BOS added) so the prefill step is exactly that many
tokens, and max_tokens 1 so there is no decode step. With DSpark on (detected from the
container args) each request generates 8 tokens instead, so the draft and 6-token verify
kernels load too. Random token IDs, so prompts never hit the prefix cache. Each response
must report prompt_tokens == length. MemAvailable and GPU memory on both nodes are
recorded before and after (memory.json).
Stops at the first failed request (exit 1). Before and after, it lists the DeepGEMM JIT
cache on both nodes, so kernels compiled during the warm-up are recorded.

Usage:  shape-warmup.py <label>
Output: data/WARMUP/<timestamp>-<label>/ requests.jsonl, new-kernels.txt, summary.txt
Run right after every boot, before handing the server over (docs/operations.md).
Requires: no request running (checked from /metrics), so it never competes with a user.
"""
import json, os, random, subprocess, sys, time, urllib.request

BASE, MODEL = "http://127.0.0.1:8888", "deepseek-v4-flash-0731"
DG_CACHE = "/srv/dsv4/runtime-cache/46638857-fi0618-sm121/vllm/deep_gemm/cache"
PEER = "gx10"
VOCAB_LO, VOCAB_HI = 1000, 120000   # ordinary tokens; vocab is 129,280, specials are at the ends
# 1-64: every swap-AB einsum count (1-32) and the small hc_prenorm splits, with margin.
# Larger sizes up to the 8,192-token prefill chunk, for the hc_prenorm split variants.
LENGTHS = list(range(1, 65)) + [80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512,
                                640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
                                3584, 4096, 5120, 6144, 7168, 8192]


def post(path, payload, timeout=300):
    r = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def running_requests():
    text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines()
               if l.startswith("vllm:num_requests_running{"))


def dg_cache(remote):
    """Set of kernel directory names in the DeepGEMM JIT cache on one node."""
    cmd = ["ls", DG_CACHE]
    if remote:
        cmd = ["timeout", "20", "ssh", "-o", "BatchMode=yes", PEER] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True)
    return set(out.stdout.split()) if out.returncode == 0 else None


def spec_on():
    """True if the serving container runs speculative decoding (DSpark)."""
    out = subprocess.run(["docker", "inspect", "dsv4-leader", "--format", "{{join .Args \" \"}}"],
                         capture_output=True, text=True)
    return "--speculative-config" in out.stdout


def memory():
    """MemAvailable (GiB) and GPU memory used by compute processes (MiB) on both nodes."""
    cmd = ("awk '/MemAvailable/{printf \"%.1f \", $2/1048576}' /proc/meminfo; "
           "nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits "
           "| awk '{s += $1} END {print s}'")
    res = {}
    for node, pre in (("leader", []), ("worker", ["timeout", "20", "ssh", "-o", "BatchMode=yes", PEER])):
        out = subprocess.run(pre + ["bash", "-c", cmd] if not pre else pre + [cmd],
                             capture_output=True, text=True).stdout.split()
        res[node] = {"mem_available_gib": float(out[0]), "gpu_used_mib": int(out[1])} if len(out) == 2 else None
    return res


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: shape-warmup.py <label>")
    out = f"/srv/dsv4/run/data/WARMUP/{time.strftime('%Y%m%dT%H%M%S')}-{label}"
    os.makedirs(out)
    if running_requests() > 0:
        sys.exit("a request is running; nothing sent (retry when the server is idle)")
    # With DSpark, the draft and the 6-token verify steps run only while decoding, so
    # each request then generates 8 tokens (a draft round plus a verify step at least).
    max_tokens = 8 if spec_on() else 1
    mem_before = memory()
    before = {"leader": dg_cache(False), "worker": dg_cache(True)}
    rng = random.Random()
    t_all, failed = time.time(), None
    with open(f"{out}/requests.jsonl", "w") as log:
        for n in LENGTHS:
            ids = [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]
            t0, rec = time.time(), {"length": n}
            try:
                r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": max_tokens,
                                             "temperature": 0, "ignore_eos": True})
                rec["prompt_tokens"] = r["usage"]["prompt_tokens"]
                rec["ok"] = rec["prompt_tokens"] == n
            except Exception as e:   # HTTP 500 = engine error; the server may now be down
                rec["ok"], rec["error"] = False, repr(e)[:300]
            rec["seconds"] = round(time.time() - t0, 2)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if not rec["ok"]:
                failed = rec
                break
    after = {"leader": dg_cache(False), "worker": dg_cache(True)}
    with open(f"{out}/new-kernels.txt", "w") as f:
        for node in ("leader", "worker"):
            if before[node] is None or after[node] is None:
                f.write(f"{node}: cache not readable\n")
                continue
            for k in sorted(after[node] - before[node]):
                f.write(f"{node} {k}\n")
    new = sum(1 for _ in open(f"{out}/new-kernels.txt"))
    mem_after = memory()
    json.dump({"before": mem_before, "after": mem_after, "max_tokens": max_tokens},
              open(f"{out}/memory.json", "w"), indent=1)
    mem = "; ".join(f"{n} MemAvailable {mem_before[n]['mem_available_gib']} -> {mem_after[n]['mem_available_gib']} GiB, "
                    f"GPU {mem_before[n]['gpu_used_mib']} -> {mem_after[n]['gpu_used_mib']} MiB"
                    for n in ("leader", "worker") if mem_before[n] and mem_after[n])
    summary = (f"{'FAILED at length ' + str(failed['length']) + ': ' + failed.get('error', 'prompt_tokens mismatch') if failed else 'ok'}; "
               f"{len(LENGTHS) if not failed else LENGTHS.index(failed['length'])} of {len(LENGTHS)} lengths done "
               f"in {time.time() - t_all:.0f} s, max_tokens {max_tokens}; new DeepGEMM kernels: {new} (new-kernels.txt); "
               f"{mem}; output {out}")
    open(f"{out}/summary.txt", "w").write(summary + "\n")
    print(summary)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
