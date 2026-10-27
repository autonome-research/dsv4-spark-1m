#!/usr/bin/env bash
# Passive, bounded allocator observation only. No model, GPU, or host mutation.
#
# Usage: allocator-observer.sh --data FILE --journal FILE --deadline-epoch EPOCH
# Started on both nodes by start-dsv4-default.sh for the boot window; exits at the
# deadline. Appends periodic samples (memory, PSI, vmstat compaction/reclaim counters,
# GPU state via nvidia-smi, tolerating a timed-out call) to FILE and new kernel-log
# lines to the journal file (data.txt and kernel.jsonl under <state>/runs/<run-id>/observer/).
# Test hooks: OBSERVER_PROC_ROOT, OBSERVER_NVIDIA_SMI, OBSERVER_JOURNALCTL.
set -Eeuo pipefail

usage() {
  printf 'usage: %s --data FILE --journal FILE --deadline-epoch EPOCH\n' "$0" >&2
}
die() {
  printf 'allocator-observer: %s\n' "$*" >&2
  exit 1
}

DATA= JOURNAL= DEADLINE= PROC_ROOT=${OBSERVER_PROC_ROOT:-/proc}
NVIDIA_SMI=${OBSERVER_NVIDIA_SMI:-nvidia-smi}
JOURNALCTL=${OBSERVER_JOURNALCTL:-journalctl}

while (($#)); do
  case $1 in
    --data) DATA=${2:?}; shift 2 ;;
    --journal) JOURNAL=${2:?}; shift 2 ;;
    --deadline-epoch) DEADLINE=${2:?}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n $DATA && -n $JOURNAL && -n $DEADLINE ]] || { usage; exit 2; }
[[ $DEADLINE =~ ^[0-9]+$ ]] || die 'deadline must be a positive epoch integer'
now=$(date +%s)
(( DEADLINE > now && DEADLINE <= now + 3300 )) || die 'deadline must be in the future and no more than 3300 seconds away'
[[ -d $PROC_ROOT && ! -L $PROC_ROOT ]] || die "proc root is not a directory: $PROC_ROOT"

# The caller creates a fresh 0700 run directory; refuse to write into a less-private one.
data_dir=${DATA%/*}; journal_dir=${JOURNAL%/*}
[[ $data_dir != "$DATA" && $data_dir == "$journal_dir" ]] || die 'data and journal must share a directory'
[[ -d $data_dir && ! -L $data_dir && $(stat -c %a -- "$data_dir") == 700 ]] || die 'output directory is not private mode 0700'
for file in "$DATA" "$JOURNAL"; do
  [[ ! -L $file ]] || die "refusing symlink: $file"
  [[ -e $file && -f $file ]] || die "output file does not exist: $file"
  [[ -O $file ]] || die "output file is not user-owned: $file"
  chmod 600 -- "$file"
done

journal_pid=
cleanup_done=0
stop_journal() {
  local expected=0 rc i
  [[ -n ${journal_pid:-} ]] || return 0
  if kill -0 "$journal_pid" 2>/dev/null; then
    kill -TERM "$journal_pid" 2>/dev/null || return 1
    expected=1
    for ((i=0; i<50; i++)); do
      if ! kill -0 "$journal_pid" 2>/dev/null; then break; fi
      sleep 0.1 || return 1
    done
    if kill -0 "$journal_pid" 2>/dev/null; then
      kill -KILL "$journal_pid" 2>/dev/null || return 1
      for ((i=0; i<20; i++)); do
        if ! kill -0 "$journal_pid" 2>/dev/null; then break; fi
        sleep 0.1 || return 1
      done
    fi
  fi
  if wait "$journal_pid"; then rc=0; else rc=$?; fi
  journal_pid=
  if (( expected )); then
    return 0
  fi
  return 1
}
cleanup() {
  local status=$? rc=0
  (( cleanup_done )) && exit "$status"
  cleanup_done=1
  trap - EXIT TERM INT HUP
  stop_journal || rc=1
  (( rc == 0 )) || status=1
  exit "$status"
}
on_signal() {
  local signal=$1
  case $signal in TERM) exit 143 ;; INT) exit 130 ;; HUP) exit 129 ;; esac
}
trap cleanup EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

# A follower that exits is a fatal observer failure; it must never silently lose kernel events.
"$JOURNALCTL" -k -f -n 0 -o json >"$JOURNAL" &
journal_pid=$!
sleep 0.1
if ! kill -0 "$journal_pid" 2>/dev/null; then
  if wait "$journal_pid"; then journal_rc=0; else journal_rc=$?; fi
  journal_pid=
  die "journal JSON follower exited ($journal_rc)"
fi

sample=0
while (( $(date +%s) < DEADLINE )); do
  if ! kill -0 "$journal_pid" 2>/dev/null; then
    if wait "$journal_pid"; then journal_rc=0; else journal_rc=$?; fi
    journal_pid=
    die "journal JSON follower exited ($journal_rc)"
  fi
  sample=$((sample + 1))
  utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
  uptime=$(awk '{print $1}' "$PROC_ROOT/uptime")
  [[ $uptime =~ ^[0-9]+(\.[0-9]+)?$ ]] || die 'invalid /proc/uptime'
  printf 'sample=%d utc=%s uptime_seconds=%s\n' "$sample" "$utc" "$uptime"
  awk '
    BEGIN { want["MemAvailable:"]; want["MemFree:"]; want["Cached:"]; want["SwapCached:"]; want["Shmem:"]; want["Mlocked:"]; want["Unevictable:"] }
    $1 in want { print "meminfo " $1 " " $2 " " $3; got[$1]=1 }
    END { for (key in want) if (!got[key]) exit 1 }
  ' "$PROC_ROOT/meminfo" || die 'incomplete /proc/meminfo sample'
  awk '
    $1 == "some" || $1 == "full" { print "psi " $0; got[$1]=1 }
    END { exit !(got["some"] && got["full"]) }
  ' "$PROC_ROOT/pressure/memory" || die 'incomplete memory PSI sample'
  awk '
    $1 == "pswpin" || $1 == "pswpout" || $1 ~ /^allocstall_/ || $1 == "pgscan_direct" || $1 == "pgsteal_direct" || $1 ~ /^compact_/ { print "vmstat " $0; got[$1]=1 }
    END { exit !(got["pswpin"] && got["pswpout"] && got["pgscan_direct"] && got["pgsteal_direct"]) }
  ' "$PROC_ROOT/vmstat" || die 'incomplete /proc/vmstat sample'
  printf 'buddyinfo\n'
  cat "$PROC_ROOT/buddyinfo"
  if (( sample % 5 == 0 )); then
    printf 'nvidia-smi process allocations\n'
    timeout --signal=TERM --kill-after=2 8 "$NVIDIA_SMI" \
      --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || printf 'nvidia-smi sample failed (rc=%s)\n' "$?"
    printf 'nvidia-smi thermal/power\n'
    timeout --signal=TERM --kill-after=2 8 "$NVIDIA_SMI" \
      --query-gpu=temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader || printf 'nvidia-smi sample failed (rc=%s)\n' "$?"
  fi
  printf '\n'
  sleep 1
 done
