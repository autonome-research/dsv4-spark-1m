#!/usr/bin/env bash
# Per-boot 4K decode check: one discarded warm-up, then 4K x3, each through
# run-guarded.sh (ctxbench.py, 256 new tokens). Run once on every boot of a restart
# series so each configuration has n=3 within its boot and boots can be compared
# (optimisation log, restart tests 2026-10-25).
#
# Usage:  boot-4k.sh <label>
# Output: data/RESTART/<timestamp>-<label>/ with config.txt (run id, args, mounts),
#         <name>.result / <name>.mon per request, and summary.txt (median, min, max).
# Read-only towards the server.
set -uo pipefail
T=$(dirname "$(readlink -f "$0")")
OUT=/srv/dsv4/run/data/RESTART/$(date +%Y%m%dT%H%M%S)-${1:?label}
mkdir -p "$OUT"
[[ $(curl -s -o /dev/null -m 10 -w '%{http_code}' localhost:8888/health) == 200 ]] || { echo "API not healthy; nothing run" >&2; exit 1; }
{
  echo "label: $1"
  echo "run_id: $(docker inspect dsv4-leader --format '{{index .Config.Labels "dsv4.t2.run"}}')"
  echo "leader_started: $(docker inspect dsv4-leader --format '{{.State.StartedAt}}')"
  echo "args: $(docker inspect dsv4-leader --format '{{join .Args " "}}')"
  echo "env: $(docker inspect dsv4-leader --format '{{range .Config.Env}}{{.}} {{end}}' | grep -oE '(NCCL_DEBUG|TORCH_NCCL_DESYNC_DEBUG)=[^ ]*' | tr '\n' ' ')"
  echo "mounts:"; docker inspect dsv4-leader --format '{{range .Mounts}}  {{.Source}}{{"\n"}}{{end}}'
} >"$OUT/config.txt"
for name in warmup 4k-1 4k-2 4k-3; do
  "$T/run-guarded.sh" 4096 "$OUT/$name"
  grep -q 'WATCHDOG ABORT' "$OUT/$name.mon" && { echo "WATCHDOG ABORT on $name" >&2; exit 2; }
  echo "$name: $(tail -n 1 "$OUT/$name.result")"
done
python3 -I - "$OUT" <<'EOPY' | tee "$OUT/summary.txt"
import json, pathlib, statistics, sys
out = pathlib.Path(sys.argv[1])
r = [json.loads((out / f"4k-{i}.result").read_text().strip().splitlines()[-1]) for i in (1, 2, 3)]
d = [x["decode_tok_s"] for x in r]; t = [x["ttft_s"] for x in r]
print(f"decode tok/s median {statistics.median(d):.2f} (min {min(d):.2f}, max {max(d):.2f}); TTFT median {statistics.median(t):.1f} s")
EOPY
