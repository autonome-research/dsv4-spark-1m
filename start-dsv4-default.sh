#!/usr/bin/env bash
# Attended, single-attempt DSV4 T2 startup. Readiness is an API GET only.
#
# Usage: start-dsv4-default.sh          (prints its settings, then waits silently for
#                                        the phrase "START DSV4 T2")
# Settings come from DSV4_* environment variables; see README.md "Configuration".
#
# Sequence, every step fail-closed (any failure runs a coordinated abort):
#   1. operator confirmation, then the admission lock
#   2. guards and allocator observers on both nodes, checked fresh
#   3. dsv4-cluster.service isolation check; preflight inventory of DSV4/vLLM
#      containers, processes and GPU users, stopped only by exact ID and only
#      after the operator answers y
#   4. guards and leader watches from earlier runs stopped
#   5. CX7 fabric check; stopped dsv4 containers from earlier runs parked (renamed)
#   6. gb10-reclaim on both nodes, MemFree >= DSV4_MIN_FREE_GIB
#   7. boot-compaction helper (tools/boot-compact.sh) started on both nodes
#   8. launch-dsv4.sh up (worker, then leader); exact container IDs recorded
#   9. readiness (GET /v1/models) with a detached post-load reclaim, guards to
#      serving mode, peerwatch and leaderwatch started
# Runbook: docs/operations.md. Change history: reports/DSV4 optimisation log.md.
set -uo pipefail

HERE=/srv/dsv4/run
PEER=192.168.200.11
PORT=8888
GUARD="$HERE/dsv4-t2-guard.py"
STATE="${DSV4_T2_STATE_DIR:-$HOME/.cache/dsv4-t2}"
READINESS_TIMEOUT=${DSV4_READINESS_TIMEOUT:-900}
SSH_TIMEOUT=${DSV4_T2_SSH_TIMEOUT:-30}
GPU_TIMEOUT=${DSV4_T2_GPU_TIMEOUT:-10}
RUN_ID="dsv4-t2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOCK_FILE="${DSV4_T2_LOCK_FILE:-$STATE/admission.lock}"
LOCK_FD=
ACTIVE=0
ABORTED=0
GUARD_PID=
OBSERVER=${DSV4_OBSERVER:-/srv/dsv4/run/allocator-observer.sh}
OBS_DIR=
OBS_LEADER_PID=
OBS_STARTED=0

# T2 defaults are deliberately not environment-overridable.
export DSV4_STACK=upstream
export DSV4_IMAGE=local/vllm-openai:46638857-fi0618
export DSV4_GPU_MEMORY_UTILIZATION=0.72
SPEC_MODE=off

fail() { echo "DSV4 T2: $*" >&2; exit 1; }
# Diagnostics mode (NCCL INFO logs, torch NCCL desync/timing/dump-on-timeout). On by default;
# DSV4_DIAG=0 turns it off. Overhead not measurable at 4K (optimisation log R01.3).
DIAG_MODE="${DSV4_DIAG:-1}"
[[ "$DIAG_MODE" == 0 || "$DIAG_MODE" == 1 ]] || fail "DSV4_DIAG must be 0 or 1 (got: $DIAG_MODE)"
# Indexer mode (E12 / E16): split (default), rowshard or off; passed to both nodes by launch-dsv4.sh.
[[ -z "${DSV4_IDX_MODE:-}" || "$DSV4_IDX_MODE" =~ ^(split|rowshard|off)$ ]] || fail "DSV4_IDX_MODE must be split, rowshard or off (got: $DSV4_IDX_MODE)"
# Prefill chunk size (optimisation log E10). DSV4_MAX_BATCH overrides the 8192 default.
MAX_BATCH="${DSV4_MAX_BATCH:-8192}"
[[ "$MAX_BATCH" =~ ^(4096|8192|12288|16384)$ ]] || fail "DSV4_MAX_BATCH must be 4096, 8192, 12288 or 16384 (got: $MAX_BATCH)"
# The KV cache needed for 1M grows with the chunk size (E10: 8192 needs 10.06 GiB).
# DSV4_KV_CACHE_BYTES overrides; sizes without a measurement must be set explicitly.
if [[ -z "${DSV4_KV_CACHE_BYTES:-}" ]]; then
  case "$MAX_BATCH" in
    4096) DSV4_KV_CACHE_BYTES=9663676416 ;;
    8192) DSV4_KV_CACHE_BYTES=11274289152 ;;
    *) fail "DSV4_MAX_BATCH=$MAX_BATCH has no measured KV size; set DSV4_KV_CACHE_BYTES too" ;;
  esac
fi
[[ "$DSV4_KV_CACHE_BYTES" =~ ^[0-9]+$ ]] || fail "DSV4_KV_CACHE_BYTES must be a byte count (got: $DSV4_KV_CACHE_BYTES)"
export DSV4_KV_CACHE_BYTES
echo "DSV4 T2: prefill chunk $MAX_BATCH, KV cache $DSV4_KV_CACHE_BYTES bytes, indexer ${DSV4_IDX_MODE:-split (default)}, diagnostics $DIAG_MODE"
timed() { timeout --kill-after=2 "$1" "${@:2}"; }
ssh_timed() { timed "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" "$@"; }

read_confirmation() {
  [[ -r /dev/tty ]] || fail 'confirmation terminal is unavailable'
  local answer
  IFS= read -r answer </dev/tty || fail 'confirmation was not read'
  [[ $answer == 'START DSV4 T2' ]] || fail 'exact confirmation "START DSV4 T2" was not supplied'
}

latch_abort() {
  (( ACTIVE && !ABORTED )) || return 0
  ABORTED=1
  "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'startup guard/readiness failure' || true
}
stop_observers() {
  (( OBS_STARTED )) || return 0
  kill -TERM "$OBS_LEADER_PID" 2>/dev/null || true
  ssh_timed "p=\$(cat ~/.cache/dsv4-t2/observer.pid 2>/dev/null) || exit 0; kill -TERM \"\$p\" 2>/dev/null || true" >/dev/null 2>&1 || true
}
stop_guards() {
  # A failed run must not leave its guards behind: a stale guard blocks the next
  # attempt's checks. The [.] and the trailing space keep the pattern from matching
  # the remote shell that carries it, or a run ID that merely starts with this one.
  [[ -n $GUARD_PID ]] && kill -TERM "$GUARD_PID" 2>/dev/null
  ssh_timed "pkill -TERM -f 'guard[.]py run --run-id $RUN_ID ' || true" >/dev/null 2>&1 || true
}
cleanup_abort() {
  local rc=$?
  latch_abort
  (( ACTIVE )) && stop_observers
  (( ABORTED )) && stop_guards
  trap - EXIT
  exit "$rc"
}
on_signal() {
  latch_abort
  stop_observers
  stop_guards
  trap - EXIT
  exit 130
}
trap cleanup_abort EXIT
trap on_signal INT TERM HUP

start_observers() {
  local deadline
  [[ -x $OBSERVER ]] || return 1
  OBS_DIR="$STATE/runs/$RUN_ID/observer"
  mkdir -p "$OBS_DIR" && chmod 700 "$OBS_DIR" || return 1
  : >"$OBS_DIR/data.txt"; : >"$OBS_DIR/kernel.jsonl"
  chmod 600 "$OBS_DIR/data.txt" "$OBS_DIR/kernel.jsonl"
  # 60 s under the observer's 3300 s maximum: the worker checks this leader-clock
  # deadline against its own clock, and any lag at exactly 3300 aborts the launch.
  deadline=$(( $(date +%s) + 3240 ))
  "$OBSERVER" --data "$OBS_DIR/data.txt" --journal "$OBS_DIR/kernel.jsonl" --deadline-epoch "$deadline" >>"$OBS_DIR/data.txt" 2>"$OBS_DIR/observer.err" &
  OBS_LEADER_PID=$!
  sleep .1
  kill -0 "$OBS_LEADER_PID" 2>/dev/null || return 1
  OBS_STARTED=1
  timed "$SSH_TIMEOUT" scp -q "$OBSERVER" "$PEER:.cache/dsv4-t2/allocator-observer.sh" || return 1
  ssh_timed "set -e; d=~/.cache/dsv4-t2; chmod 700 \"\$d\"; : >\"\$d/observer.data\"; : >\"\$d/observer.kernel.jsonl\"; chmod 600 \"\$d/observer.data\" \"\$d/observer.kernel.jsonl\"; nohup \"\$d/allocator-observer.sh\" --data \"\$d/observer.data\" --journal \"\$d/observer.kernel.jsonl\" --deadline-epoch $deadline </dev/null >>\"\$d/observer.data\" 2>\"\$d/observer.err\" & echo \$! >\"\$d/observer.pid\"; sleep .1; kill -0 \$(cat \"\$d/observer.pid\")" || return 1
}
check_observers() {
  (( OBS_STARTED )) && kill -0 "$OBS_LEADER_PID" 2>/dev/null &&
    ssh_timed 'p=$(cat ~/.cache/dsv4-t2/observer.pid 2>/dev/null) || exit 1; kill -0 "$p"'
}

read_confirmation
mkdir -p "$STATE" && chmod 700 "$STATE"
exec {LOCK_FD}>"$LOCK_FILE" || fail 'cannot open admission lock'
flock -n "$LOCK_FD" || fail 'another attended T2 admission is active'

# An existing sticky ALERT is rejected by admit; only a human clear command can
# unblock a node.  The descriptor is the only admission created for launch-dsv4 up.
python3 -B "$GUARD" admit --run-id "$RUN_ID" --role leader --state-dir "$STATE" || fail 'local attended admission could not be created'
ACTIVE=1
ssh_timed "mkdir -p ~/.cache/dsv4-t2 && chmod 700 ~/.cache/dsv4-t2" || fail 'worker guard directory is unavailable'
timed "$SSH_TIMEOUT" scp -q "$GUARD" "$PEER:.cache/dsv4-t2/guard.py" || fail 'worker guard copy failed'
ssh_timed "chmod 700 ~/.cache/dsv4-t2/guard.py && python3 -B ~/.cache/dsv4-t2/guard.py admit --run-id '$RUN_ID' --role worker --state-dir ~/.cache/dsv4-t2" || fail 'worker attended admission could not be created'

python3 -B "$GUARD" run --run-id "$RUN_ID" --role leader --state-dir "$STATE" --interval 2 >/dev/null 2>"$STATE/leader.guard.err" {LOCK_FD}>&- &
GUARD_PID=$!
ssh_timed "nohup python3 -B ~/.cache/dsv4-t2/guard.py run --run-id '$RUN_ID' --role worker --state-dir ~/.cache/dsv4-t2 --interval 2 </dev/null >~/.cache/dsv4-t2/worker.guard.log 2>~/.cache/dsv4-t2/worker.guard.err & echo guard-started" || fail 'worker guard did not start'

check_local_guard() { python3 -B "$GUARD" admission --run-id "$RUN_ID" --role leader --state-dir "$STATE" --max-age 6; }
check_worker_guard() { ssh_timed "python3 -B ~/.cache/dsv4-t2/guard.py admission --run-id '$RUN_ID' --role worker --state-dir ~/.cache/dsv4-t2 --max-age 6"; }
wait_for_fresh_guards() {
  local n
  for n in 1 2 3 4 5; do
    check_local_guard && check_worker_guard && return 0
    sleep 1
  done
  return 1
}
wait_for_fresh_guards || fail 'both node guards are missing, stale, malformed, exited, or alerted'
start_observers || fail 'passive allocator observer failed to start on both nodes'
check_observers || fail 'passive allocator observer is not healthy on both nodes'

check_services() {
  local unit="$HOME/.config/systemd/user/dsv4-cluster.service" out
  [[ -L $unit && $(readlink "$unit") == /dev/null ]] || { echo 'leader service is not explicitly masked' >&2; return 1; }
  out=$(timed 10 systemctl --user show dsv4-cluster.service --property=Id --property=LoadState --property=UnitFileState --property=ActiveState) || return 1
  grep -qx 'Id=dsv4-cluster.service' <<<"$out" || return 1
  grep -qx 'LoadState=masked' <<<"$out" || return 1
  grep -qx 'UnitFileState=masked' <<<"$out" || return 1
  grep -qx 'ActiveState=inactive' <<<"$out" || return 1
  ssh_timed "set -e; test ! -e ~/.config/systemd/user/dsv4-cluster.service; out=\$(systemctl --user show dsv4-cluster.service --property=Id --property=LoadState --property=ActiveState) || exit 1; printf '%s\\n' \"\$out\" | grep -qx 'Id=dsv4-cluster.service'; printf '%s\\n' \"\$out\" | grep -qx 'LoadState=not-found'; printf '%s\\n' \"\$out\" | grep -qx 'ActiveState=inactive'"
}
# --- reclaimable blocker preflight -------------------------------------------
# One bounded probe set per node per attempt: DSV4/vLLM containers, matching
# processes, and a single GPU ownership query.  A probe failure latches and ends
# the run with no further probe, preserving the one-query-per-node rule.  Nothing
# is ever signalled unless the attending operator answers y at the terminal, and
# only the exact container IDs and PIDs collected here are touched -- never a
# name pattern, never a global sweep, and containers are stopped, never removed.
BLOCK_FILE="$STATE/runs/$RUN_ID/blockers.txt"
INVENTORY=$(cat <<"SNIP"
# dsv4-preflight-self-marker: this snippet is passed as a command line, so its own
# text is visible in ps.  Every process carrying it -- this shell and the timeout
# or ssh wrapping it -- therefore contains the marker above and is excluded below.
# Without that, the literal pattern words in this script matched the process scan
# and the preflight listed itself as a blocker on an idle machine.  The class
# words are also bracketed so a regression cannot resurrect the self-match.
set -uo pipefail
self_re="dsv4-t2/guard[.]py|allocator-observer[.]sh|dsv4-t2-guard[.]py|dsv4-preflight-self-marker"
proc_re="[v]llm|[p]ython.*serve"
rows=$(timeout --kill-after=2 15 docker ps --no-trunc --format "{{.ID}}|{{.Names}}|{{.Image}}") || exit 2
printf "%s\n" "$rows" | while IFS="|" read -r cid cname cimage; do
  [ -n "${cid:-}" ] || continue
  printf "%s %s" "${cname:-}" "${cimage:-}" | grep -qiE "dsv4|[v]llm|[d]eepseek" || continue
  printf "container|%s|%s from image %s\n" "$cid" "${cname:-?}" "${cimage:-?}"
done
ps -eo pid=,user=,args= | while read -r xpid xuser xargs; do
  [ "$xpid" = "$$" ] && continue
  [ "$xpid" = "$PPID" ] && continue
  printf "%s" "${xargs:-}" | grep -qE "$proc_re" || continue
  printf "%s" "${xargs:-}" | grep -qE "$self_re" && continue
  printf "proc|%s|user %s: %s\n" "$xpid" "$xuser" "$xargs"
done
gpu=$(timeout --kill-after=2 10 nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>&1) || exit 3
printf "%s\n" "$gpu" | while IFS="," read -r gpid gname gmem; do
  gpid=$(printf "%s" "${gpid:-}" | tr -dc "0-9")
  [ -n "$gpid" ] || continue
  gowner=$(ps -o user= -p "$gpid" 2>/dev/null | tr -d " ")
  printf "gpu|%s|user %s: %s using%s\n" "$gpid" "${gowner:-unknown}" "${gname:-?}" "${gmem:-?}"
done
exit 0
SNIP
)

probe_node() {
  local out rc
  if [[ $1 == leader ]]; then out=$(timed 40 bash -c "$INVENTORY"); rc=$?
  else out=$(ssh_timed "$INVENTORY"); rc=$?; fi
  (( rc == 0 )) || return "$rc"
  [[ -n ${out//[[:space:]]/} ]] && printf '%s\n' "$out" | sed -e '/^[[:space:]]*$/d' -e "s/^/$1|/"
  return 0
}
inventory_blockers() {
  local node out rc reason
  mkdir -p "$STATE/runs/$RUN_ID" && chmod 700 "$STATE/runs/$RUN_ID" || return 2
  : >"$BLOCK_FILE" && chmod 600 "$BLOCK_FILE" || return 2
  for node in leader worker; do
    out=$(probe_node "$node"); rc=$?
    if (( rc != 0 )); then
      reason="$node workload inventory failed"
      (( rc == 3 )) && reason="$node GPU query failed"
      python3 -B "$GUARD" latch --run-id "$RUN_ID" --role leader --state-dir "$STATE" --reason "$reason" || true
      echo "DSV4 T2: $reason (rc=$rc); no retry was attempted" >&2
      return 2
    fi
    [[ -n $out ]] && printf '%s\n' "$out" >>"$BLOCK_FILE"
  done
  [[ -s $BLOCK_FILE ]] && return 1
  return 0
}

present_blockers() {
  local node kind id detail
  {
    echo
    echo '=============================================================='
    echo ' DSV4 T2 PREFLIGHT: these workloads must be gone to serve'
    echo '=============================================================='
    while IFS='|' read -r node kind id detail; do
      case $kind in
        container) printf '  %-6s container %s\n             %s\n' "$node" "${id:0:12}" "$detail" ;;
        proc)      printf '  %-6s process   pid %s\n             %s\n' "$node" "$id" "$detail" ;;
        gpu)       printf '  %-6s GPU owner pid %s\n             %s\n' "$node" "$id" "$detail" ;;
      esac
    done <"$BLOCK_FILE"
    echo
    echo '  Answering y will docker stop (graceful, -t 120) each exact container'
    echo '  ID above, then SIGTERM and 10s later SIGKILL each exact PID above.'
    echo '  THIS DESTROYS THAT WORK.  This is a shared machine: check the owners'
    echo '  listed above before answering.  Containers are stopped, never removed,'
    echo '  and nothing outside this list is touched.'
    echo
    printf 'Terminate the workloads listed above and continue? [y/N] '
  } >/dev/tty 2>&1
}
read_yes_no() {
  [[ -r /dev/tty && -w /dev/tty ]] || return 1
  local answer
  IFS= read -r answer </dev/tty || return 1
  printf '\n' >/dev/tty 2>&1 || true
  [[ $answer == y || $answer == Y || $answer == yes || $answer == YES ]]
}

reclaim_blockers() {
  local node kind id detail sig signalled
  while IFS='|' read -r node kind id detail; do
    [[ $kind == container && $id =~ ^[0-9a-f]{64}$ ]] || continue
    echo "DSV4 T2: stopping $node container ${id:0:12} (graceful, not removed)"
    if [[ $node == leader ]]; then
      timed 130 docker stop -t 120 "$id" >/dev/null 2>&1 ||
        echo "DSV4 T2: $node container ${id:0:12} did not stop" >&2
    else
      ssh_timed "timeout --kill-after=2 130 docker stop -t 120 '$id'" >/dev/null 2>&1 ||
        echo "DSV4 T2: $node container ${id:0:12} did not stop" >&2
    fi
  done <"$BLOCK_FILE"
  for sig in TERM KILL; do
    signalled=0
    while IFS='|' read -r node kind id detail; do
      [[ $kind == proc || $kind == gpu ]] || continue
      [[ $id =~ ^[0-9]+$ ]] && (( id > 1 )) || continue
      if [[ $node == leader ]]; then
        [[ $id == "$$" || $id == "$PPID" || $id == "${GUARD_PID:-0}" || $id == "${OBS_LEADER_PID:-0}" ]] && continue
        kill -0 "$id" 2>/dev/null || continue
        echo "DSV4 T2: SIG$sig -> leader pid $id"
        kill -"$sig" "$id" 2>/dev/null ||
          echo "DSV4 T2: could not signal leader pid $id; it may belong to another user" >&2
      else
        echo "DSV4 T2: SIG$sig -> worker pid $id"
        ssh_timed "kill -0 '$id' 2>/dev/null || exit 0; kill -$sig '$id' 2>/dev/null || exit 9" >/dev/null 2>&1 ||
          echo "DSV4 T2: could not signal worker pid $id; it may belong to another user" >&2
      fi
      signalled=1
    done <"$BLOCK_FILE"
    (( signalled )) || break
    [[ $sig == TERM ]] && sleep 10
  done
}

check_services || fail 'role-aware dsv4-cluster.service isolation failed'

# At most two probe rounds: the second only after the operator authorised
# reclamation.  The unchanged fail-closed inventory is the final arbiter.
preflight_clear=0
for preflight_attempt in 1 2; do
  inventory_blockers; inventory_rc=$?
  (( inventory_rc == 0 )) && { preflight_clear=1; break; }
  (( inventory_rc == 2 )) && break
  (( preflight_attempt == 1 )) || break
  present_blockers
  read_yes_no || fail 'a competing model/process workload is present and reclamation was declined'
  reclaim_blockers
done
(( preflight_clear )) || fail 'a competing model/process workload is present or inventory failed'

# --- retire watchers left by earlier runs ------------------------------------
# Optimisation log 2026-10-24 19:31: the 10-23 run's leader guard outlived its
# containers (stopped by this preflight) by 9 h, and when it sat in D-state this
# run's guard latched an ALERT.  The inventory above has just shown that no DSV4
# container runs on either node, so a guard or leader watch carrying another run
# id watches nothing.  Only those exact PIDs get TERM, the same signal "down" uses.
STALE_RE='guard[.]py run --run-id |dsv4-leaderwatch[.]sh --run-id '
list_stale_local() { pgrep -af "$STALE_RE" | grep -vF -- "--run-id $RUN_ID " || true; }
list_stale_worker() { ssh_timed "pgrep -af '$STALE_RE' | grep -vF -- '--run-id $RUN_ID ' || true"; }
retire_stale_watchers() {
  local l w n
  l=$(list_stale_local); w=$(list_stale_worker) || return 1
  [[ -z $l$w ]] && return 0
  [[ -n $l ]] && printf 'DSV4 T2: stopping stale leader watcher %s\n' "${l:0:160}"
  [[ -n $w ]] && printf 'DSV4 T2: stopping stale worker watcher %s\n' "${w:0:160}"
  [[ -n $l ]] && kill -TERM $(awk '{print $1}' <<<"$l") 2>/dev/null
  [[ -n $w ]] && ssh_timed "kill -TERM $(awk '{print $1}' <<<"$w" | tr '\n' ' ') 2>/dev/null; true" >/dev/null
  for n in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    l=$(list_stale_local); w=$(list_stale_worker) || return 1
    [[ -z $l$w ]] && return 0
  done
  printf 'DSV4 T2: still running after TERM:\n%s\n%s\n' "$l" "$w" >&2
  return 1
}
retire_stale_watchers || fail 'stale guards/leader watches from an earlier run could not be stopped'

fabric_ready() {
  timed 10 docker info >/dev/null 2>&1 &&
    timed 10 ping -c 1 -W 1 192.168.200.11 >/dev/null 2>&1 &&
    timed 10 ping -c 1 -W 1 192.168.201.11 >/dev/null 2>&1 &&
    ssh_timed 'docker info >/dev/null 2>&1 && ping -c 1 -W 1 192.168.200.10 >/dev/null 2>&1 && ping -c 1 -W 1 192.168.201.10 >/dev/null 2>&1'
}
wait_for_fabric() {
  local n stable=0
  for n in $(seq 1 120); do
    check_local_guard && check_worker_guard && check_observers || return 1
    if fabric_ready; then
      stable=$((stable + 1)); (( stable >= 6 )) && return 0
    else
      stable=0
    fi
    sleep 5
  done
  return 1
}
wait_for_fabric || fail 'bounded fabric/guard readiness failed'

# dsv4-abort stops containers but never removes them, so a stopped container from
# an earlier run keeps the fixed name and node-launch refuses to start. Rename
# (never remove) exited ones so their logs stay available; a running one fails.
park_stale() {
  local cmd='n=$1; st=$(docker inspect --type container "$n" --format "{{.State.Running}} {{index .Config.Labels \"dsv4.t2.run\"}}" 2>/dev/null) || exit 0
    set -- $st; [ "$1" = false ] || { echo "running container $n"; exit 1; }
    docker rename "$n" "$n-${2:-old}-parked" >/dev/null && echo "parked $n as $n-${2:-old}-parked"'
  timed 20 bash -c "$cmd" _ dsv4-leader || return 1
  ssh_timed "bash -c '$(printf %s "$cmd" | sed "s/'/'\\\\''/g")' _ dsv4-worker" || return 1
}
park_stale || fail 'a dsv4 container is still running, or a stale one could not be parked'

# Step-1 hardening: the up-front weight reservation (~78 GiB/rank) must come from
# truly free memory. Measured 2026-10-12: with 59 GiB of page cache left on the
# leader the reservation went through direct reclaim+compaction (PSI some 6.5)
# and the guard correctly aborted. Drop caches and compact on both nodes now,
# then require MemFree (not MemAvailable) headroom before any container starts.
MIN_FREE_GIB=${DSV4_MIN_FREE_GIB:-100}
timed 120 sudo -n /usr/local/sbin/gb10-reclaim >/dev/null 2>&1 || fail 'leader pre-launch reclaim failed (passwordless gb10-reclaim missing?)'
ssh_timed 'timeout 110 sudo -n /usr/local/sbin/gb10-reclaim' >/dev/null 2>&1 || fail 'worker pre-launch reclaim failed (passwordless gb10-reclaim missing?)'
lf=$(awk '/^MemFree/{print int($2/1048576)}' /proc/meminfo)
wf=$(ssh_timed "awk '/^MemFree/{print int(\$2/1048576)}' /proc/meminfo")
echo "DSV4 T2: pre-launch MemFree leader=${lf}GiB worker=${wf:-?}GiB (need ${MIN_FREE_GIB})"
[[ $lf -ge $MIN_FREE_GIB && ${wf:-0} -ge $MIN_FREE_GIB ]] || fail 'not enough truly free memory after reclaim; reboot the short node'
check_local_guard && check_worker_guard || fail 'guards went stale during pre-launch reclaim'

# Boot-time compaction helper (optimisation log, I02 follow-up / R01.2): on each
# node, runs gb10-reclaim when memory PSI some avg10 stays above 20 for 6 s during
# boot, i.e. before the guard's boot rule (> 40 for 30 s) can fire. The guard is
# unchanged. The helper exits on serving, ALERT or container stop; it is optional,
# so a failure to start only warns. DSV4_BOOT_COMPACT=0 disables it.
if [[ ${DSV4_BOOT_COMPACT:-1} == 1 ]]; then
  setsid nohup "$HERE/tools/boot-compact.sh" --run-id "$RUN_ID" --role leader --state-dir "$STATE" \
    </dev/null >/dev/null 2>>"$STATE/runs/$RUN_ID/boot-compact.err" &
  if timed "$SSH_TIMEOUT" scp -q "$HERE/tools/boot-compact.sh" "$PEER:dsv4-boot-compact.sh" &&
     ssh_timed "chmod +x ~/dsv4-boot-compact.sh && setsid nohup ~/dsv4-boot-compact.sh --run-id '$RUN_ID' --role worker --state-dir ~/.cache/dsv4-t2 </dev/null >/dev/null 2>>~/.cache/dsv4-t2/runs/'$RUN_ID'/boot-compact.err & echo started" >/dev/null 2>&1; then
    echo "DSV4 T2: boot compaction helper started on both nodes; log runs/$RUN_ID/<role>/boot-compact.log"
  else
    echo "DSV4 T2: WARNING: boot compaction helper did not start on the worker" >&2
  fi
fi

# The normal entrypoint is the sole one-attempt launcher.  launch-dsv4 up repeats
# the admission checks and node-launch refuses a direct call.
if ! DSV4_T2_RUN_ID="$RUN_ID" DSV4_T2_STATE_DIR="$STATE" "$HERE/launch-dsv4.sh" up "" "$MAX_BATCH" 1 "$SPEC_MODE" "$DIAG_MODE" 1048576; then
  fail 'coordinated launch failed'
fi

cid_local="$STATE/runs/$RUN_ID/cids/leader.cid"
cid_worker="$STATE/runs/$RUN_ID/cids/worker.cid"
[[ -r $cid_local && -r $cid_worker ]] || fail 'both protected container IDs were not recorded'
LEADER_ID=$(head -n 1 "$cid_local"); WORKER_ID=$(head -n 1 "$cid_worker")
[[ $LEADER_ID =~ ^[0-9a-f]{64}$ && $WORKER_ID =~ ^[0-9a-f]{64}$ ]] || fail 'container IDs are not exact 64-character IDs'
exact_running() {
  [[ $(timed 10 docker inspect --type container "$LEADER_ID" --format '{{.State.Running}}' 2>/dev/null) == true ]] &&
    [[ $(ssh_timed "docker inspect --type container '$WORKER_ID' --format '{{.State.Running}}'") == true ]]
}
wait_for_api() {
  local elapsed=0
  while (( elapsed < READINESS_TIMEOUT )); do
    check_local_guard && check_worker_guard && check_observers || return 3
    exact_running || return 2
    if timed 10 curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  return 1
}
# Step-1 hardening: streaming weights fragments unified memory, and the KV/graph
# allocations that follow then stall in direct compaction (the wedge path).
# Reclaim+compact both nodes once loading finishes. Needs a NOPASSWD rule for
# /usr/local/sbin/gb10-reclaim; without it this is skipped with a warning.
( for _ in $(seq 1 120); do
    [[ $(timed 10 docker inspect --type container "$LEADER_ID" --format '{{.State.Running}}' 2>/dev/null) == true ]] || exit 0
    if timed 10 docker logs "$LEADER_ID" 2>&1 | grep -q "Loading weights took"; then
      sleep 5
      if timed 60 sudo -n /usr/local/sbin/gb10-reclaim >/dev/null 2>&1 &&
         ssh_timed 'timeout 60 sudo -n /usr/local/sbin/gb10-reclaim' >/dev/null 2>&1; then
        echo "[$(date +%T)] post-load reclaim done on both nodes" >>"$STATE/runs/$RUN_ID/postload.log"
      else
        echo "[$(date +%T)] post-load reclaim SKIPPED (no passwordless sudo)" >>"$STATE/runs/$RUN_ID/postload.log"
      fi
      exit 0
    fi
    sleep 10
  done ) </dev/null >/dev/null 2>&1 &
wait_for_api; readiness_rc=$?
if (( readiness_rc != 0 )); then
  fail "API readiness failed (rc=$readiness_rc); coordinated abort requested"
fi
# Leave boot mode: from here the guards apply the strict serving rules again.
touch "$STATE/runs/$RUN_ID/leader/serving" || fail 'could not switch leader guard to serving mode'
ssh_timed "touch ~/.cache/dsv4-t2/runs/'$RUN_ID'/worker/serving" || fail 'could not switch worker guard to serving mode'
printf '%s\n' 'API boot ready; guards switched to serving rules; no inference validation performed.'
# Leader-side worker watch (optimisation log 2026-10-19): nothing else notices a
# lost worker while serving. Detached so it outlives this script; ends with the run.
if [[ ${DSV4_PEERWATCH:-1} == 1 ]]; then
  DSV4_T2_STATE_DIR="$STATE" setsid nohup "$HERE/dsv4-peerwatch.sh" --run-id "$RUN_ID" \
    --leader-cid "$LEADER_ID" --worker-cid "$WORKER_ID" </dev/null >/dev/null 2>>"$STATE/runs/$RUN_ID/peerwatch.err" &
  printf '%s\n' "$!" >"$STATE/runs/$RUN_ID/peerwatch.pid"
  echo "DSV4 T2: worker watch started (pid $!); log $STATE/runs/$RUN_ID/peerwatch.log"
fi
# Worker-side leader watch (optimisation log I01): stops the worker's vLLM when the
# leader's /health stays down, so a lost leader no longer leaves an orphan holding GPU memory.
if [[ ${DSV4_LEADERWATCH:-1} == 1 ]]; then
  ssh_timed "setsid nohup ~/dsv4-leaderwatch.sh --run-id '$RUN_ID' --worker-cid '$WORKER_ID' </dev/null >/dev/null 2>>~/.cache/dsv4-t2/runs/'$RUN_ID'/leaderwatch.err & echo started" >/dev/null 2>&1 &&
    echo "DSV4 T2: leader watch started on the worker; log ~/.cache/dsv4-t2/runs/$RUN_ID/leaderwatch.log (worker)" ||
    echo "DSV4 T2: WARNING: leader watch did not start on the worker" >&2
fi
ACTIVE=0
exit 0
