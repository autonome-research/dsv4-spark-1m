"""Decode speed on natural prompts: streaming chat completion, 512 tokens, ignore_eos; prints one JSON line per prompt."""
import json, sys, time, urllib.request
PROMPTS = ["Explain in detail how the TCP three-way handshake and congestion control work.",
           "Write a Python function that parses a CSV file and computes per-column statistics, with comments.",
           "Tell a long story about a lighthouse keeper who finds a mysterious map."]
temp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
for p in PROMPTS:
    body = {"model": "deepseek-v4-flash-0731", "messages": [{"role": "user", "content": p}], "max_tokens": 512,
            "temperature": temp, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
    t0 = time.time(); first = None; n = None
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("choices") and first is None: first = time.time()
        if d.get("usage"): n = d["usage"]["completion_tokens"]
    t1 = time.time()
    print(json.dumps({"prompt": p[:40], "temperature": temp, "tokens": n, "ttft_s": round(first - t0, 2),
                      "decode_tok_s": round((n - 1) / (t1 - first), 2)}))
