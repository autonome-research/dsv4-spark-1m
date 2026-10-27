#!/usr/bin/env bash
# E16 serving A/B (optimisation log E16, 2026-10-24): run the same deterministic prompts
# (e12b_client.py: hidden vault code, greedy, 64 tokens) against whatever indexer mode is
# running, so boots with DSV4_IDX_MODE=split / rowshard / off can be compared.
# 64K x2, 256K x1, 1M x1 (target 1,036,288, the E12b/E15 1M size; reference TTFT 813.7 s
# E12b split, 819.2 s E15 split + FI4732). Each request runs under the same memory watchdog
# as run-guarded.sh. Read-only towards the server.
# Usage: e16-ab.sh [contexts...]      default: 65536 65536 262144 1036288
# Output: /srv/dsv4/run/data/E16/<timestamp>-<mode>/
set -uo pipefail
T=$(dirname "$(readlink -f "$0")")
[[ $(curl -s -o /dev/null -m 10 -w '%{http_code}' localhost:8888/health) == 200 ]] || { echo "API not healthy; nothing run" >&2; exit 1; }

# Work out the running mode from the leader's mounts and args, and check the worker matches.
mounts=$(docker inspect dsv4-leader --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}')
args=$(docker inspect dsv4-leader --format '{{join .Args " "}}')
if grep -q dsv4-rowshard-sparse-attn-indexer.py <<<"$mounts"; then MODE=rowshard
elif grep -q dsv4-sparse-attn-indexer.py <<<"$mounts"; then MODE=split
else MODE=off; fi
wmounts=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 gx10 "docker inspect dsv4-worker --format '{{range .Mounts}}{{.Source}}{{\"\n\"}}{{end}}'" | grep -oE 'dsv4-(rowshard-)?[a-z-]*indexer[a-z-]*\.py|dsv4-rowshard-config-attention\.py' | sort | tr '\n' ' ')
lmounts=$(grep -oE 'dsv4-(rowshard-)?[a-z-]*indexer[a-z-]*\.py|dsv4-rowshard-config-attention\.py' <<<"$mounts" | sort | tr '\n' ' ')
[[ "$wmounts" == "$lmounts" ]] || { echo "leader/worker indexer mounts differ: [$lmounts] vs [$wmounts]; not running" >&2; exit 1; }

OUT=/srv/dsv4/run/data/E16/$(date +%Y%m%dT%H%M%S)-$MODE
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/ab.log"; }
{ echo "mode: $MODE"; echo "mounts: $lmounts"; echo "attention-config: $(grep -oE -- '--attention-config [^ ]+ [^ ]+' <<<"$args" || echo none)"
  echo "run_id: $(docker inspect dsv4-leader --format '{{index .Config.Labels "dsv4.t2.run"}}')"; } >"$OUT/config.txt"
log "mode=$MODE; config in $OUT/config.txt"

CTXS=("$@"); (( ${#CTXS[@]} )) || CTXS=(65536 65536 262144 1036288)
i=0
for ctx in "${CTXS[@]}"; do
  i=$((i + 1)); name="$i-$ctx"
  log "request $name"
  python3 -I "$T/e12b_client.py" "$ctx" "$MODE" >"$OUT/$name.json" 2>"$OUT/$name.err" & B=$!
  hi=0
  while kill -0 $B 2>/dev/null; do   # same rules as run-guarded.sh
    la=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); lp=$(awk '/some/{split($2,a,"=");print int(a[2])}' /proc/pressure/memory)
    set -- $(timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 gx10 "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo; awk '/some/{split(\$2,a,\"=\");print int(a[2])}' /proc/pressure/memory" | tr '\n' ' ')
    wa=${1:-99}; wp=${2:-0}
    echo "$(date +%T) L ${la}G psi $lp | W ${wa}G psi $wp" >>"$OUT/$name.mon"
    if (( lp > 40 || wp > 40 )); then hi=$((hi + 1)); else hi=0; fi
    if (( la < 6 || wa < 6 || hi >= 10 )); then log "WATCHDOG ABORT on $name (la=$la wa=$wa hi=$hi)"; kill $B; wait $B 2>/dev/null; exit 2; fi
    sleep 3
  done
  wait $B || { log "client failed on $name: $(tail -c 300 "$OUT/$name.err")"; exit 3; }
  log "$name: $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ("prompt_tokens","ttft_s","prefill_tok_s","code_found")})' "$OUT/$name.json")"
done
log "done"
