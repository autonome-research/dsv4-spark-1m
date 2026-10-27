#!/usr/bin/env python3
"""Inference probe alongside the F01 soak (2026-10-23).
Every INTERVAL s: one short chat completion (fixed prompt, 128 new tokens) against the leader API.
Appends one JSON line per probe to data/F01/probe-<run>.jsonl. Read-only: never touches the server.
Usage: infer-probe.py <run_id> <hours> [interval_s]"""
import json, sys, time, urllib.request
from datetime import datetime

run_id, hours = sys.argv[1], float(sys.argv[2])
interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
out = f"/srv/dsv4/run/data/F01/probe-{run_id}.jsonl"
url = "http://127.0.0.1:8888/v1/chat/completions"
body = json.dumps({
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "Write a short paragraph about how rivers shape valleys."}],
    "max_tokens": 128, "temperature": 0, "ignore_eos": True,
}).encode()

end = time.time() + hours * 3600
while time.time() < end:
    rec = {"ts": datetime.now().strftime("%F %T")}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
        dt = time.time() - t0
        n = resp.get("usage", {}).get("completion_tokens", 0)
        rec.update(ok=n > 0, completion_tokens=n, seconds=round(dt, 2), tok_s=round(n / dt, 2) if dt else None)
    except Exception as e:  # record and keep probing; the soak monitor judges server health
        rec.update(ok=False, seconds=round(time.time() - t0, 2), error=repr(e)[:300])
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(max(0, interval - (time.time() - t0)))
