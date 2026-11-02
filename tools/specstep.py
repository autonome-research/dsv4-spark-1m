"""Per prompt: decode tok/s, tokens per step, ms per step and per-position acceptance (accepted drafts at
position i per step) from vLLM's spec-decode counters. Temp 0, 512 tokens, natbench prompts (log 2026-11-01)."""
import json, sys, time, urllib.request
B = "http://127.0.0.1:8888"
P = {"tcp": "Explain in detail how the TCP three-way handshake and congestion control work.",
     "code": "Write a Python function that parses a CSV file and computes per-column statistics, with comments.",
     "story": "Tell a long story about a lighthouse keeper who finds a mysterious map."}
def ctr():
    t = urllib.request.urlopen(B + "/metrics", timeout=10).read().decode()
    g = lambda k: sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(k + "{"))
    pos = [sum(float(l.split()[-1]) for l in t.splitlines()
               if l.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{") and f'position="{i}"' in l)
           for i in range(10)]
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total"), pos
for name, p in P.items():
    d0, a0, p0 = ctr()
    body = {"model": "deepseek-v4-flash-0731", "messages": [{"role": "user", "content": p}], "max_tokens": 512,
            "temperature": 0.0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}), timeout=600)
    t0 = time.time(); first = None; n = None; text = []
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("choices"):
            first = first or time.time(); text.append(d["choices"][0].get("delta", {}).get("content") or "")
        if d.get("usage"): n = d["usage"]["completion_tokens"]
    t1 = time.time(); d1, a1, p1 = ctr(); steps = d1 - d0
    acc_pos = [round((b - a) / steps, 3) for a, b in zip(p0, p1) if steps and b > 0]
    print(json.dumps({"prompt": name, "tok_s": round((n - 1) / (t1 - first), 2), "steps": steps,
                      "tokens_per_step": round(n / steps, 3) if steps else None,
                      "ms_per_step": round((t1 - first) * 1000 / steps, 2) if steps else None,
                      "accept_rate_per_pos": acc_pos, "text_head": "".join(text)[:80]}))
