#!/usr/bin/env bash
# B1 (log 2026-11-01): profile DSpark decode steps on a server started with DSV4_PROFILE=1.
# For each prompt (code, story; temperature 0, 512 tokens, the natbench prompts) it runs
# /start_profile, one streaming request, /stop_profile. vLLM's profiler skips 10 engine
# steps (the prefill and the first decode steps) and records the next 20, so each trace holds
# 20 decode steps. The traces of both ranks are copied to data/B1/<ts>-<label>/ and the
# leader's is summarised by tools/trace-steps.py (draft vs verify vs sample time per step).
# Short requests only (prompt < 100 tokens). Decode speeds printed here are measured with the
# profiler on and are not comparable to natbench numbers.
# Usage: tools/b1-profile.sh <label>
set -uo pipefail
LABEL=${1:?usage: b1-profile.sh <label>}
BASE=http://127.0.0.1:8888
HERE=/srv/dsv4/run
OUT=$HERE/data/B1/$(date +%Y%m%dT%H%M%S)-$LABEL
LPROF=/srv/dsv4/diagnostics/$(hostname)/prof
WPROF=/srv/dsv4/diagnostics/gx10-6d0c/prof
die() { echo "b1-profile: $*" >&2; exit 1; }

args=$(docker inspect dsv4-leader --format '{{join .Args " "}}' 2>/dev/null) || die "no dsv4-leader container"
[[ $args == *--profiler-config* ]] || die "server not started with DSV4_PROFILE=1"
[[ $args == *--speculative-config* ]] || echo "b1-profile: note: DSpark is off on this server"
curl -sf -m 5 $BASE/health >/dev/null || die "server not healthy"
running=$(curl -s -m 10 $BASE/metrics | awk '/^vllm:num_requests_running\{/ {s += $NF} END {print s + 0}')
[[ $running == 0 ]] || die "a request is running; retry when idle"
mkdir -p "$OUT" || die "cannot create $OUT"
echo "$args" >"$OUT/server-args.txt"

newest() { ls -t "$1"/*.pt.trace.json.gz 2>/dev/null | head -1; }
wnewest() { ssh -o BatchMode=yes gx10 "ls -t $WPROF/*.pt.trace.json.gz 2>/dev/null | head -1"; }

for name in code story; do
  case $name in
    code) prompt="Write a Python function that parses a CSV file and computes per-column statistics, with comments." ;;
    story) prompt="Tell a long story about a lighthouse keeper who finds a mysterious map." ;;
  esac
  lbefore=$(newest "$LPROF"); wbefore=$(wnewest)
  curl -sf -m 30 -X POST $BASE/start_profile >/dev/null || die "start_profile failed"
  python3 -I - "$prompt" >"$OUT/$name-request.json" <<'EOF' || echo "b1-profile: request failed ($name)"
import json, sys, time, urllib.request
body = {"model": "deepseek-v4-flash-0731", "messages": [{"role": "user", "content": sys.argv[1]}],
        "max_tokens": 512, "temperature": 0.0, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True}}
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
print(json.dumps({"tokens": n, "ttft_s": round(first - t0, 2), "decode_tok_s_profiled": round((n - 1) / (t1 - first), 2)}))
EOF
  curl -sf -m 300 -X POST $BASE/stop_profile >/dev/null || echo "b1-profile: stop_profile failed ($name)"
  # the trace is written asynchronously; wait up to 3 min for a new file on each rank
  for _ in $(seq 36); do
    l=$(newest "$LPROF"); w=$(wnewest)
    [[ -n $l && $l != "$lbefore" && -n $w && $w != "$wbefore" ]] && break
    sleep 5
  done
  [[ -n $l && $l != "$lbefore" ]] || { echo "b1-profile: no new leader trace ($name)"; continue; }
  sleep 10   # let the writer finish the file
  cp "$l" "$OUT/$name-rank0.pt.trace.json.gz"
  [[ -n $w && $w != "$wbefore" ]] && scp -q "gx10:$w" "$OUT/$name-rank1.pt.trace.json.gz"
  echo "== $name: $(cat "$OUT/$name-request.json")" | tee -a "$OUT/summary.txt"
  python3 -I "$HERE/tools/trace-steps.py" "$OUT/$name-rank0.pt.trace.json.gz" --json "$OUT/$name-rank0-steps.json" \
    | tee -a "$OUT/summary.txt"
done
echo "output: $OUT"
