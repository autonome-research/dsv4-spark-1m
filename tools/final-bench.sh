#!/usr/bin/env bash
# Final benchmark (finishing plan, 2026-10-24): 4K x3, 64K x2, 256K x1, 1M x1, one request at a time,
# 256 new tokens, ignore_eos, thinking off. Same method as the scoreboard (ctxbench.py), and every
# request goes through run-guarded.sh (client killed if MemAvailable < 6 GiB or PSI > 40 for 30 s).
# Read-only towards the server: it never restarts, stops or reconfigures anything.
# Usage: final-bench.sh [label]     Output: /srv/dsv4/run/data/FINAL/<timestamp>[-label]/
# Takes about 30 min, most of it the 1M prefill (~14 min).
set -uo pipefail
T=$(dirname "$(readlink -f "$0")")
OUT=/srv/dsv4/run/data/FINAL/$(date +%Y%m%dT%H%M%S)${1:+-$1}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/bench.log"; }

[[ $(curl -s -o /dev/null -m 10 -w '%{http_code}' localhost:8888/health) == 200 ]] || { log "API not healthy; nothing run"; exit 1; }
if pgrep -f 'tools/soak[.]sh|infer-probe[.]py' >/dev/null; then
  log "WARNING: soak/probe still running; their probes will overlap with these requests"
fi

# Record exactly what is being measured.
{
  echo "run_id: $(docker inspect dsv4-leader --format '{{index .Config.Labels "dsv4.t2.run"}}' 2>/dev/null)"
  echo "image: $(docker inspect dsv4-leader --format '{{.Config.Image}}' 2>/dev/null)"
  echo "leader_started: $(docker inspect dsv4-leader --format '{{.State.StartedAt}}' 2>/dev/null)"
  echo "args: $(docker inspect dsv4-leader --format '{{join .Args " "}}' 2>/dev/null)"
  echo "mounts:"; docker inspect dsv4-leader --format '{{range .Mounts}}  {{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' 2>/dev/null
  echo "patch md5:"; md5sum /srv/dsv4/run/patches/*.py /srv/dsv4/run/patches/*.so 2>/dev/null | sed 's/^/  /'
} >"$OUT/config.txt"
log "start; config in $OUT/config.txt"

run() {  # run <ctx> <name>
  log "request $2 (target $1)"
  "$T/run-guarded.sh" "$1" "$OUT/$2"
  if grep -q 'WATCHDOG ABORT' "$OUT/$2.mon"; then log "WATCHDOG ABORT on $2; stopping the benchmark"; exit 2; fi
  if ! grep -q '"decode_tok_s"' "$OUT/$2.result"; then log "no result for $2: $(tail -c 300 "$OUT/$2.result")"; exit 3; fi
  log "$2: $(cat "$OUT/$2.result")"
}

run 4096 warmup            # discarded: first request after idle
run 4096 4k-1; run 4096 4k-2; run 4096 4k-3
run 65536 64k-1; run 65536 64k-2
run 262144 256k-1
run 1040000 1m-1           # below max_model_len (1,048,576) with room for the 256 new tokens

python3 -I - "$OUT" <<'EOPY' | tee -a "$OUT/bench.log"
import json, pathlib, statistics, sys
out = pathlib.Path(sys.argv[1]); rows = {}
for f in sorted(out.glob("*.result")):
    if f.stem == "warmup": continue
    r = json.loads(f.read_text().strip().splitlines()[-1]); rows.setdefault(f.stem.split("-")[0], []).append(r)
print(f"{'ctx':>5} {'n':>2} {'prompt_tok':>10} {'TTFT_s':>8} {'prefill_tok/s':>13} {'decode_tok/s':>12}  (median over n)")
for k in ("4k", "64k", "256k", "1m"):
    rs = rows.get(k, [])
    if not rs: continue
    med = lambda key: statistics.median(x[key] for x in rs)
    print(f"{k:>5} {len(rs):>2} {int(med('prompt_tokens')):>10} {med('ttft_s'):>8.1f} {int(med('prefill_tok_s')):>13} {med('decode_tok_s'):>12.2f}")
json.dump(rows, open(out / "summary.json", "w"), indent=1)
EOPY
log "done"
