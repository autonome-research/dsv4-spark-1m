#!/usr/bin/env bash
# Leader-side worker watch for a serving DSV4 run (optimisation log 2026-10-19).
# vLLM's MultiprocWorkerMonitor only watches the leader's local worker processes,
# and the guards only latch ALERTs, so a lost worker left the leader blocked at
# its next collective with nobody told. Every INTERVAL s this checks, by exact
# container ID over the CX7 ssh path, that the worker container is still running.
#   - worker container not running / absent  -> coordinated abort at once
#   - worker unreachable MISSES times in a row -> coordinated abort
# The abort is dsv4-abort.sh (latches the ALERT on both nodes, exact-ID stops).
# Exits quietly when the run ends: leader container gone or another active run.
# DSV4_PEERWATCH_DRYRUN=1 logs the abort it would run instead of running it.
#
# Usage: dsv4-peerwatch.sh --run-id ID --leader-cid <64-hex> --worker-cid <64-hex>
#   Started detached by start-dsv4-default.sh once the run is serving.
#   Log: <state>/runs/<run-id>/peerwatch.log
#   Env: DSV4_PEERWATCH_INTERVAL (10 s), DSV4_PEERWATCH_MISSES (6), DSV4_T2_PEER.
set -uo pipefail
HERE=${DSV4_T2_HERE:-/srv/dsv4/run}
PEER=${DSV4_T2_PEER:-192.168.200.11}
STATE=${DSV4_T2_STATE_DIR:-$HOME/.cache/dsv4-t2}
INTERVAL=${DSV4_PEERWATCH_INTERVAL:-10}
MISSES=${DSV4_PEERWATCH_MISSES:-6}
RUN_ID= LEADER_ID= WORKER_ID=
die() { echo "dsv4-peerwatch: $*" >&2; exit 1; }
while (($#)); do
  case $1 in
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --leader-cid) LEADER_ID=${2:?}; shift 2 ;;
    --worker-cid) WORKER_ID=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die 'invalid run ID'
[[ $LEADER_ID =~ ^[0-9a-f]{64}$ && $WORKER_ID =~ ^[0-9a-f]{64}$ ]] || die 'container IDs must be exact 64-character IDs'
[[ $INTERVAL =~ ^[0-9]+$ && $MISSES =~ ^[0-9]+$ && $INTERVAL -ge 1 && $MISSES -ge 1 ]] || die 'bad interval/misses'
LOG="$STATE/runs/$RUN_ID/peerwatch.log"
mkdir -p "$STATE/runs/$RUN_ID" || die 'cannot create run directory'
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

abort_run() {
  log "ACTION: $1"
  if [[ ${DSV4_PEERWATCH_DRYRUN:-0} == 1 ]]; then
    log "dry run: would run dsv4-abort.sh --run-id $RUN_ID --reason 'peerwatch: $1'"
  else
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason "peerwatch: $1" >>"$LOG" 2>&1
    log "dsv4-abort.sh exit $?"
  fi
  exit 0
}

log "start: run=$RUN_ID peer=$PEER interval=${INTERVAL}s misses=$MISSES dryrun=${DSV4_PEERWATCH_DRYRUN:-0}"
miss=0; last=ok
while :; do
  # The run is over (normal down, abort, or a newer run): stop watching.
  [[ $(head -n 1 "$STATE/active-run" 2>/dev/null) == "$RUN_ID" ]] || { log 'exit: run is no longer the active run'; exit 0; }
  [[ $(timeout --kill-after=2 10 docker inspect --type container "$LEADER_ID" --format '{{.State.Running}}' 2>/dev/null) == true ]] ||
    { log 'exit: leader container is not running'; exit 0; }
  out=$(timeout --kill-after=2 15 ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
        "docker inspect --type container '$WORKER_ID' --format '{{.State.Running}} {{.State.Status}} {{.State.ExitCode}}'" 2>&1)
  rc=$?
  if (( rc == 0 )) && [[ $out == "true running "* ]]; then
    [[ $last != ok ]] && log "worker reachable and running again after $miss miss(es)"
    miss=0; last=ok
  elif (( rc == 0 )); then
    abort_run "worker container not running ($out)"
  elif (( rc == 1 )) && grep -qiE 'no such container|not found' <<<"$out"; then
    abort_run 'worker container absent'
  else
    miss=$((miss + 1)); last=miss
    log "worker unreachable ($miss/$MISSES): rc=$rc ${out//$'\n'/ }"
    (( miss >= MISSES )) && abort_run "worker unreachable for $miss checks (rc=$rc)"
  fi
  sleep "$INTERVAL"
done
