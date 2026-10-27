#!/usr/bin/env bash
# Exact-ID, coordinated DSV4 abort. It never removes containers or kills by name.
#
# Usage: dsv4-abort.sh [--run-id ID] [--reason TEXT]   (default: the active run)
# Latches the ALERT on both nodes, cancels launchers that wrote an owned-process
# descriptor, inspects dsv4-leader / dsv4-worker, and stops only a container whose
# role and run labels match, by exact ID. The worker inspect is retried up to 3 times.
# Record: <state>/abort/<run-id>/ (inspect output and what was stopped).
# Called by launch-dsv4.sh down, the start script on failure, and peerwatch.
# Env: DSV4_T2_HERE, DSV4_T2_PEER, DSV4_T2_STATE_DIR, DSV4_T2_SSH_TIMEOUT,
#      DSV4_T2_STOP_TIMEOUT, DSV4_T2_OWNED_WAIT.
set -uo pipefail

HERE=${DSV4_T2_HERE:-/srv/dsv4/run}
PEER=${DSV4_T2_PEER:-192.168.200.11}
STATE=${DSV4_T2_STATE_DIR:-$HOME/.cache/dsv4-t2}
RUN_ID=${DSV4_T2_RUN_ID:-}
REASON=${DSV4_T2_ABORT_REASON:-operator abort}
die() { echo "dsv4-abort: $*" >&2; exit 1; }
while (($#)); do
  case $1 in
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --reason) REASON=${2:?}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
SSH_TIMEOUT=${DSV4_T2_SSH_TIMEOUT:-30}
STOP_TIMEOUT=${DSV4_T2_STOP_TIMEOUT:-130}
# cancel-owned now waits for the owned launcher to actually exit, so its callers
# need a budget larger than that wait; an expired wrapper is itself a failure.
OWNED_WAIT=${DSV4_T2_OWNED_WAIT:-20}
OWNED_TIMEOUT=$(( OWNED_WAIT + 8 ))
valid_run() { [[ ${1:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; }
if [[ -z $RUN_ID && -r $STATE/active-run ]]; then RUN_ID=$(head -n 1 "$STATE/active-run"); fi
valid_run "$RUN_ID" || die 'no valid active run ID; refusing name-only teardown'

OUT="$STATE/abort/$RUN_ID"
umask 077
mkdir -p "$OUT" "$STATE/runs/$RUN_ID"
chmod 700 "$STATE" "$OUT" "$STATE/runs/$RUN_ID" 2>/dev/null || true
printf '%s\n' "$RUN_ID" >"$OUT/active-run"
printf '%s\n' "$REASON" >"$OUT/reason"

# Latch locally first; failure means local admission cannot be known cancelled.
python3 -B "$HERE/dsv4-t2-guard.py" latch --run-id "$RUN_ID" --role leader --state-dir "$STATE" --reason "$REASON" || die 'could not latch local ALERT'
PEER_LATCH_RC=0
# The reason is free text (peerwatch includes command output), so quote it for the remote shell.
printf -v Q_REASON '%q' "$REASON"
if ! timeout --kill-after=2 "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
    "python3 -B - latch --run-id '$RUN_ID' --role worker --state-dir ~/.cache/dsv4-t2 --reason $Q_REASON" \
    <"$HERE/dsv4-t2-guard.py" >/dev/null 2>&1; then
  PEER_LATCH_RC=1
  echo 'dsv4-abort: worker ALERT latch failed; continuing exact-ID inspection/stop with uncertainty' >&2
fi

# Cancel only launchers that wrote an exact PID/starttime descriptor. Missing means
# no owned launcher; mismatches are failures, never a name/pattern signal. Each call
# blocks until that rank's launcher is confirmed gone, so the inspection below cannot
# race a launcher that is still inside docker run.
OWNED_L_RC=0; OWNED_W_RC=0
timeout --kill-after=2 "$OWNED_TIMEOUT" python3 -B "$HERE/dsv4-t2-guard.py" cancel-owned --run-id "$RUN_ID" --role leader --state-dir "$STATE" --wait "$OWNED_WAIT" || OWNED_L_RC=$?
timeout --kill-after=2 "$OWNED_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
  "python3 -B ~/.cache/dsv4-t2/guard.py cancel-owned --run-id '$RUN_ID' --role worker --state-dir ~/.cache/dsv4-t2 --wait '$OWNED_WAIT'" \
  >/dev/null 2>&1 || OWNED_W_RC=$?
if (( OWNED_L_RC != 0 || OWNED_W_RC != 0 )); then
  echo 'dsv4-abort: a launcher exit was not confirmed; container absence is not knowable on that rank' >&2
fi

inspect_local() {
  timeout --kill-after=2 "$SSH_TIMEOUT" docker inspect --type container dsv4-leader \
    --format '{{.Id}} {{.Name}} {{index .Config.Labels "dsv4.t2.role"}} {{index .Config.Labels "dsv4.t2.run"}}' 2>&1
}
inspect_remote() {
  timeout --kill-after=2 "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
    "timeout --kill-after=2 '$SSH_TIMEOUT' docker inspect --type container dsv4-worker --format '{{.Id}} {{.Name}} {{index .Config.Labels \"dsv4.t2.role\"}} {{index .Config.Labels \"dsv4.t2.run\"}}'" 2>&1
}
LEADER_RECORD=$(inspect_local); LEADER_INSPECT_RC=$?
# Retry a failed or timed-out remote inspect (2026-10-24: one 30 s timeout left the worker running).
for attempt in 1 2 3; do
  WORKER_RECORD=$(inspect_remote); WORKER_INSPECT_RC=$?
  (( WORKER_INSPECT_RC == 0 || WORKER_INSPECT_RC == 1 )) && break
  echo "dsv4-abort: worker inspect attempt $attempt failed (rc=$WORKER_INSPECT_RC)" >&2
  (( attempt < 3 )) && sleep 2
done
printf '%s\n' "$LEADER_RECORD" >"$OUT/leader.inspect"
printf '%s\n' "$WORKER_RECORD" >"$OUT/worker.inspect"
printf '%s\n' "$LEADER_INSPECT_RC" >"$OUT/leader.inspect.rc"
printf '%s\n' "$WORKER_INSPECT_RC" >"$OUT/worker.inspect.rc"

# Only Docker's normal not-found exit (1) is absence. SSH/timeouts, malformed
# output, and name/role/run mismatches are unknown and are never stopped.
inspect_kind() {
  local rc=$1 record=$2 expected_name=$3 expected_role=$4 id name role run extra
  if (( rc != 0 )); then
    if (( rc == 1 )) && grep -qiE 'no such container|not found' <<<"$record"; then printf 'absent\n'; else printf 'unknown\n'; fi
    return
  fi
  [[ -n $record && $record != *$'\n'* ]] || { printf 'unknown\n'; return; }
  read -r id name role run extra <<<"$record"
  if [[ -z ${extra:-} && $id =~ ^[0-9a-f]{64}$ && ${name#/} == "$expected_name" && $role == "$expected_role" && $run == "$RUN_ID" ]]; then
    printf 'eligible %s\n' "$id"
  else
    printf 'unknown\n'
  fi
}
read -r LEADER_KIND LEADER_ID <<<"$(inspect_kind "$LEADER_INSPECT_RC" "$LEADER_RECORD" dsv4-leader leader)"
read -r WORKER_KIND WORKER_ID <<<"$(inspect_kind "$WORKER_INSPECT_RC" "$WORKER_RECORD" dsv4-worker worker)"
[[ $LEADER_KIND == eligible ]] && printf '%s\n' "$LEADER_ID" >"$OUT/leader.cid"
[[ $WORKER_KIND == eligible ]] && printf '%s\n' "$WORKER_ID" >"$OUT/worker.cid"

stop_local() {
  [[ $LEADER_KIND == eligible ]] || return 0
  timeout --kill-after=2 "$STOP_TIMEOUT" docker stop -t 120 "$LEADER_ID" >/dev/null 2>&1
}
stop_remote() {
  [[ $WORKER_KIND == eligible ]] || return 0
  timeout --kill-after=2 "$STOP_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
    "timeout --kill-after=2 '$STOP_TIMEOUT' docker stop -t 120 '$WORKER_ID'" >/dev/null 2>&1
}
STOP_L_RC=0; STOP_W_RC=0
stop_local & STOP_L_PID=$!
stop_remote & STOP_W_PID=$!
wait "$STOP_L_PID" || STOP_L_RC=$?
wait "$STOP_W_PID" || STOP_W_RC=$?

post_local() {
  local out rc
  [[ $LEADER_KIND == eligible ]] || { printf '%s\n' "$LEADER_KIND"; return; }
  out=$(timeout --kill-after=2 "$SSH_TIMEOUT" docker inspect --type container "$LEADER_ID" --format '{{.State.Status}} {{.State.Running}}' 2>&1); rc=$?
  if (( rc == 0 )); then
    [[ $out == 'exited false' || $out == 'created false' || $out == 'dead false' ]] && printf 'stopped\n' || printf 'failed\n'
  elif (( rc == 1 )) && grep -qiE 'no such container|not found' <<<"$out"; then
    printf 'absent\n'
  else
    printf 'unknown\n'
  fi
}
post_remote() {
  local out rc
  [[ $WORKER_KIND == eligible ]] || { printf '%s\n' "$WORKER_KIND"; return; }
  out=$(timeout --kill-after=2 "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
    "timeout --kill-after=2 '$SSH_TIMEOUT' docker inspect --type container '$WORKER_ID' --format '{{.State.Status}} {{.State.Running}}'" 2>&1); rc=$?
  if (( rc == 0 )); then
    [[ $out == 'exited false' || $out == 'created false' || $out == 'dead false' ]] && printf 'stopped\n' || printf 'failed\n'
  elif (( rc == 1 )) && grep -qiE 'no such container|not found' <<<"$out"; then
    printf 'absent\n'
  else
    printf 'unknown\n'
  fi
}
LEADER_STATUS=$(post_local)
WORKER_STATUS=$(post_remote)
(( STOP_L_RC == 0 )) || LEADER_STATUS=failed
(( STOP_W_RC == 0 )) || WORKER_STATUS=failed
# A launcher whose exit was never confirmed can still create a container after this
# inspection, so 'absent' would be a claim this abort cannot support. Report the
# uncertainty instead; re-running the abort once the launcher is gone resolves it.
(( OWNED_L_RC == 0 )) || [[ $LEADER_STATUS != absent ]] || LEADER_STATUS=unknown
(( OWNED_W_RC == 0 )) || [[ $WORKER_STATUS != absent ]] || WORKER_STATUS=unknown
printf '%s\n' "$LEADER_STATUS" >"$OUT/leader.status"
printf '%s\n' "$WORKER_STATUS" >"$OUT/worker.status"
printf 'dsv4-abort: leader=%s worker=%s\n' "$LEADER_STATUS" "$WORKER_STATUS"

RESULT=0
(( PEER_LATCH_RC == 0 && OWNED_L_RC == 0 && OWNED_W_RC == 0 )) || RESULT=1
[[ $LEADER_STATUS == stopped || $LEADER_STATUS == absent ]] || RESULT=1
[[ $WORKER_STATUS == stopped || $WORKER_STATUS == absent ]] || RESULT=1
exit "$RESULT"
