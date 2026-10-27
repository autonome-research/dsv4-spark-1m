#!/usr/bin/env bash
# Worker-side leader watch for a serving DSV4 run (optimisation log I01, 2026-10-23).
# Mirror of dsv4-peerwatch.sh. When the leader went down on 2026-10-22, nothing on
# the worker noticed: its vLLM kept 93.6 GiB and retried the dead TCPStore for 16 h.
# Every INTERVAL s this probes the leader's vLLM /health over CX7.
#   - worker container not running / absent    -> exit quietly (the run is over)
#   - leader /health failing MISSES times in a row -> latch the worker ALERT and
#     stop the worker container by its exact ID
# The worker has no ssh path to the leader, so the probe is HTTP only.
# DSV4_LEADERWATCH_DRYRUN=1 logs the stop it would do instead of doing it.
#
# Usage (on the worker): dsv4-leaderwatch.sh --run-id ID --worker-cid <64-hex>
#   Copied to the worker by launch-dsv4.sh up and started by start-dsv4-default.sh.
#   Log: ~/.cache/dsv4-t2/runs/<run-id>/leaderwatch.log (worker)
#   Env: DSV4_LEADERWATCH_INTERVAL (10 s), DSV4_LEADERWATCH_MISSES (18), DSV4_T2_LEADER.
set -uo pipefail
LEADER=${DSV4_T2_LEADER:-192.168.200.10}
PORT=${DSV4_T2_LEADER_PORT:-8888}
STATE=${DSV4_T2_STATE_DIR:-$HOME/.cache/dsv4-t2}
INTERVAL=${DSV4_LEADERWATCH_INTERVAL:-10}
MISSES=${DSV4_LEADERWATCH_MISSES:-18}
RUN_ID= WORKER_ID=
die() { echo "dsv4-leaderwatch: $*" >&2; exit 1; }
while (($#)); do
  case $1 in
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --worker-cid) WORKER_ID=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die 'invalid run ID'
[[ $WORKER_ID =~ ^[0-9a-f]{64}$ ]] || die 'worker container ID must be an exact 64-character ID'
[[ $INTERVAL =~ ^[0-9]+$ && $MISSES =~ ^[0-9]+$ && $INTERVAL -ge 1 && $MISSES -ge 1 ]] || die 'bad interval/misses'
LOG="$STATE/runs/$RUN_ID/leaderwatch.log"
mkdir -p "$STATE/runs/$RUN_ID" || die 'cannot create run directory'
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

stop_worker() {
  log "ACTION: $1"
  if [[ ${DSV4_LEADERWATCH_DRYRUN:-0} == 1 ]]; then
    log "dry run: would latch the worker ALERT and stop container $WORKER_ID"
  else
    python3 -B "$STATE/guard.py" latch --run-id "$RUN_ID" --role worker --state-dir "$STATE" \
      --reason "leaderwatch: $1" >>"$LOG" 2>&1
    log "guard latch exit $?"
    timeout --kill-after=2 130 docker stop -t 120 "$WORKER_ID" >>"$LOG" 2>&1
    log "docker stop exit $?"
  fi
  exit 0
}

log "start: run=$RUN_ID leader=$LEADER:$PORT interval=${INTERVAL}s misses=$MISSES dryrun=${DSV4_LEADERWATCH_DRYRUN:-0}"
miss=0
while :; do
  [[ $(timeout --kill-after=2 10 docker inspect --type container "$WORKER_ID" --format '{{.State.Running}}' 2>/dev/null) == true ]] ||
    { log 'exit: worker container is not running'; exit 0; }
  if code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://$LEADER:$PORT/health") && [[ $code == 200 ]]; then
    (( miss > 0 )) && log "leader healthy again after $miss miss(es)"
    miss=0
  else
    miss=$((miss + 1))
    log "leader /health failed ($miss/$MISSES): http=${code:-none}"
    (( miss >= MISSES )) && stop_worker "leader /health failed $miss times in a row"
  fi
  sleep "$INTERVAL"
done
