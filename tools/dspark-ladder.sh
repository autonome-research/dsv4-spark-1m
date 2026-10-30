#!/usr/bin/env bash
# DSpark experiment ladder (docs/dspark-experiment.md, Stage 1). Read-only towards the server.
#
# Runs one rung at a time against whatever configuration is serving, and after every
# request checks the stop rules of the experiment:
#   1. both ranks ran the same eager NCCL collectives (multiset of op/count/dtype from the
#      NCCL COLL logs; collectives inside CUDA graphs are not logged, E07.1);
#   2. the answer is right (e12b_client hidden code) and the request did not fail;
#   3. MemAvailable stays >= 10 GiB on both nodes (watchdog aborts the request below 6 GiB,
#      as e16-ab.sh does);
#   4. with DSpark on, draft acceptance >= 40 % (lower means a draft-loader defect).
# The first broken rule stops the ladder with exit code 4 and the reason in ladder.log; the
# server is left as it is for diagnosis (py-spy, logs), then the operator rolls back.
#
# Usage: dspark-ladder.sh <label> [rung...]
#   rungs: check (no request: only the rank and metrics checks; use it first, as a baseline),
#          tiny (10 short greedy completions), 4k (ctxbench 4096 x 512, x3), acc (the 52 fixed
#          prompts of gemv-acc.py, for the DSpark-on vs -off comparison), or a context size in
#          tokens for e12b_client (65536, 262144, 660000, 1036288).
#   default rungs: tiny 4k acc 65536
# Output: /srv/dsv4/run/data/DSPARK/<timestamp>-<label>/
#   config.txt (serving config), ladder.log, <n>-<rung>.json (client stdout), .err, .mon
#   (MemAvailable every 3 s), metrics-<n>.txt (spec-decode counters after the rung)
set -uo pipefail
T=$(dirname "$(readlink -f "$0")")
API=http://localhost:8888
PEER=gx10
LDIAG=/srv/dsv4/diagnostics/$(hostname)
WDIAG=/srv/dsv4/diagnostics/gx10-6d0c
MIN_AVAIL_GIB=10        # stop rule 3 (plan); the e16-ab.sh watchdog floor of 6 GiB still applies
MIN_ACCEPT=0.40         # stop rule 4

LABEL=${1:?usage: dspark-ladder.sh <label> [rung...]}; shift
RUNGS=("$@"); (( ${#RUNGS[@]} )) || RUNGS=(tiny 4k acc 65536)
[[ $(curl -s -o /dev/null -m 10 -w '%{http_code}' "$API/health") == 200 ]] || { echo "API not healthy; nothing run" >&2; exit 1; }
wssh() { timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" "$@"; }

OUT=/srv/dsv4/run/data/DSPARK/$(date +%Y%m%dT%H%M%S)-$LABEL
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/ladder.log"; }
stop() { log "STOP: $*"; log "ladder stopped; server left running for diagnosis (docs/dspark-experiment.md, Rollback)"; exit 4; }

# --- What is serving --------------------------------------------------------------------
args=$(docker inspect dsv4-leader --format '{{join .Args " "}}')
started=$(docker inspect dsv4-leader --format '{{.State.StartedAt}}')
SPEC=off; grep -q -- '--speculative-config' <<<"$args" && SPEC=on
{ echo "label: $LABEL"
  echo "run_id: $(docker inspect dsv4-leader --format '{{index .Config.Labels "dsv4.t2.run"}}')"
  echo "leader_started: $started"
  echo "spec: $SPEC $(grep -oE -- "--speculative-config [^ ]+" <<<"$args")"
  echo "flags: $(grep -oE -- '--(no-)?(async-scheduling|enable-prefix-caching)' <<<"$args" | tr '\n' ' ')"
  echo "mounts: $(docker inspect dsv4-leader --format '{{range .Mounts}}{{.Source}} {{end}}' | grep -oE 'dsv4-[a-z0-9-]+\.(py|so)' | tr '\n' ' ')"
} >"$OUT/config.txt"
log "label=$LABEL spec=$SPEC rungs=${RUNGS[*]}; config in $OUT/config.txt"

# --- Stop rule 1: same eager collectives on both ranks -------------------------------------
# The rank processes write nccl.<host>.<pid>.log; this run's file is the newest one. Each
# line "NCCL INFO <Op>: opCount <n> ... count <c> datatype <d>" is one collective call.
LNCCL=$(ls -t "$LDIAG"/nccl.*.log 2>/dev/null | head -1)
WNCCL=$(wssh "ls -t $WDIAG/nccl.*.log 2>/dev/null | head -1")
[[ -n $LNCCL && -n $WNCCL ]] || { log "no NCCL COLL logs found (needs DSV4_DIAG=1)"; exit 1; }
[[ $(date -r "$LNCCL" +%s) -ge $(date -d "$started" +%s) ]] || { log "newest leader NCCL log predates this run: $LNCCL"; exit 1; }
echo "nccl_logs: $LNCCL gx10:$WNCCL" >>"$OUT/config.txt"
SIG='/NCCL INFO [A-Za-z]+: opCount/ { for (i = 1; i < NF; i++) { if ($i == "count") c = $(i + 1); if ($i == "datatype") d = $(i + 1) }
     op = $0; sub(/.*NCCL INFO /, "", op); sub(/:.*/, "", op); n[op " count=" c " dtype=" d]++ }
     END { for (k in n) print n[k], k }'
nccl_check() {   # $1 = step name; equal multisets, retried while the logs settle
  local try l w
  for try in 1 2 3 4 5; do
    sleep 2
    l=$(awk "$SIG" "$LNCCL" | sort); w=$(wssh "awk '$SIG' '$WNCCL' | sort")
    [[ -n $l && $l == "$w" ]] && { log "$1: NCCL ranks agree ($(awk '{s += $1} END {print s}' <<<"$l") eager collectives)"; return 0; }
  done
  diff <(echo "$l") <(echo "$w") >"$OUT/nccl-mismatch-$1.diff"
  stop "$1: NCCL collectives differ between ranks (leader < > worker in nccl-mismatch-$1.diff)"
}

# --- Stop rule 4: acceptance from the server's counters ------------------------------------
metrics() { curl -s -m 10 "$API/metrics" | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)(_total)?\{' ; }
sum_metric() { awk -v m="$1" '$1 ~ "^"m"(_total)?[{]" {s += $NF} END {printf "%d", s}' <<<"$2"; }
accept_check() {   # $1 = step name, $2/$3 = metrics before/after
  [[ $SPEC == on ]] || return 0
  local dt at nd
  dt=$(( $(sum_metric vllm:spec_decode_num_draft_tokens "$3") - $(sum_metric vllm:spec_decode_num_draft_tokens "$2") ))
  at=$(( $(sum_metric vllm:spec_decode_num_accepted_tokens "$3") - $(sum_metric vllm:spec_decode_num_accepted_tokens "$2") ))
  nd=$(( $(sum_metric vllm:spec_decode_num_drafts "$3") - $(sum_metric vllm:spec_decode_num_drafts "$2") ))
  (( dt > 0 )) || { log "$1: no drafts counted"; return 0; }
  log "$1: acceptance $(awk -v a="$at" -v d="$dt" 'BEGIN {printf "%.3f", a / d}') ($at/$dt draft tokens), mean tokens per step $(awk -v a="$at" -v n="$nd" 'BEGIN {printf "%.2f", 1 + a / n}')"
  awk -v a="$at" -v d="$dt" -v m="$MIN_ACCEPT" 'BEGIN {exit !(a / d < m)}' && stop "$1: acceptance below $MIN_ACCEPT"
  return 0
}

# --- Stop rule 3 and the watchdog: run one client under memory monitoring ------------------
run_watched() {   # $1 = step name, rest = command; stdout -> <step>.json
  local name=$1; shift
  "$@" >"$OUT/$name.json" 2>"$OUT/$name.err" & local B=$! low=99 la wa
  while kill -0 $B 2>/dev/null; do
    la=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    wa=$(wssh "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo"); wa=${wa:-99}
    echo "$(date +%T) L ${la}G | W ${wa}G" >>"$OUT/$name.mon"
    (( la < low )) && low=$la; (( wa < low )) && low=$wa
    if (( la < 6 || wa < 6 )); then kill $B; wait $B 2>/dev/null; stop "$name: watchdog, MemAvailable L ${la}G W ${wa}G"; fi
    sleep 3
  done
  wait $B || stop "$name: client failed: $(tail -c 300 "$OUT/$name.err")"
  (( low < MIN_AVAIL_GIB )) && stop "$name: MemAvailable fell to ${low} GiB (< $MIN_AVAIL_GIB)"
  log "$name: lowest MemAvailable ${low} GiB"
}

tiny() {   # 10 short greedy completions; the first DSpark request is where Velvet's hang was
  local i
  for i in $(seq 1 10); do
    curl -s -m 300 "$API/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash-0731","prompt":"The capital of France is","max_tokens":32,"temperature":0,"seed":0}' |
      python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"text": d["choices"][0]["text"], "usage": d["usage"]}))' || return 1
  done
}

# Memory after each rung, both nodes: where DSpark's extra ~5 GiB per node goes and whether
# buffers stay reserved after large requests (optimisation log, 2026-10-29 18:10).
memline() {
  echo "MemAvailable $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB, GPU $(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits | awk '{s += $1} END {print s}') MiB"
}
log "start: memory now: $(memline) | worker $(wssh "$(declare -f memline); memline")"

n=0
for rung in "${RUNGS[@]}"; do
  n=$((n + 1)); name="$n-$rung"
  before=$(metrics)
  log "rung $name"
  case "$rung" in
    check) ;;
    tiny) run_watched "$name" tiny ;;
    4k)   run_watched "$name" bash -c "for i in 1 2 3; do python3 -I '$T/ctxbench.py' 4096 512 || exit 1; done" ;;
    acc)  run_watched "$name" python3 -I "$T/gemv-acc.py" run "dspark-$SPEC-$LABEL" 1 ;;
    [0-9]*)
      run_watched "$name" python3 -I "$T/e12b_client.py" "$rung" "$LABEL"
      python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["code_found"] else 1)' "$OUT/$name.json" ||
        stop "$name: hidden code not found (wrong answer)"
      log "$name: $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ("prompt_tokens","ttft_s","decode_tok_s","code_found")})' "$OUT/$name.json")" ;;
    *) log "unknown rung $rung"; exit 2 ;;
  esac
  after=$(metrics); printf '%s\n' "$after" >"$OUT/metrics-$n.txt"
  [[ $rung == 4k ]] && log "$name: decode tok/s $(python3 -I -c 'import json,sys; print([json.loads(l)["decode_tok_s"] for l in open(sys.argv[1])])' "$OUT/$name.json")"
  [[ $rung == tiny ]] && log "$name: outputs $(python3 -I -c 'import json,sys; print(sorted({json.loads(l)["text"] for l in open(sys.argv[1])}))' "$OUT/$name.json")"
  accept_check "$name" "$before" "$after"
  nccl_check "$name"
  log "$name: memory now: $(memline) | worker $(wssh "$(declare -f memline); memline")"
done
log "done: all rungs passed"
