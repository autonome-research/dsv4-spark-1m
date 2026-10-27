#!/usr/bin/env bash
# Boot-time "compact first, abort second" helper (optimisation log, I02 follow-up).
#
# During a T2 boot the KV-cache reservation sometimes stalls in direct compaction
# (I02 class: leader memory PSI 70-90 % right after "GPU KV cache size"). The guard
# aborts when PSI some avg10 stays above 40 for 30 s. In I02 an explicit
# drop_caches + compact_memory released the stall within 10 s, so this helper runs
# that same step (gb10-reclaim) as soon as pressure builds, well before the guard's
# 30 s window. The guard and its rules are unchanged: if compaction does not help,
# the boot is still aborted exactly as before.
#
# Usage:  boot-compact.sh --run-id ID --role leader|worker --state-dir DIR
# Runs on one node; the start script launches one per node. It exits when the run
# is marked serving, when the run's ALERT is latched (abort in progress; the helper
# then stays out of the way), when the node's dsv4 container stops, or after
# DSV4_BOOT_COMPACT_MAX_S (default 1800 s). Log: <state>/runs/<id>/<role>/boot-compact.log
#
# Tunables (env): DSV4_BOOT_COMPACT_PSI (20), DSV4_BOOT_COMPACT_FOR_S (6),
#   DSV4_BOOT_COMPACT_MAX_RECLAIMS (3), DSV4_BOOT_COMPACT_GAP_S (20).
# Test hooks (env): BOOT_COMPACT_PSI_FILE, BOOT_COMPACT_RECLAIM, BOOT_COMPACT_CONTAINER_CMD.
set -uo pipefail

RUN_ID= ROLE= STATE=
while (( $# )); do
  case $1 in
    --run-id) RUN_ID=$2; shift 2 ;;
    --role) ROLE=$2; shift 2 ;;
    --state-dir) STATE=$2; shift 2 ;;
    *) echo "boot-compact: unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || { echo "boot-compact: bad --run-id" >&2; exit 2; }
[[ $ROLE == leader || $ROLE == worker ]] || { echo "boot-compact: --role must be leader or worker" >&2; exit 2; }
[[ -d $STATE/runs/$RUN_ID ]] || { echo "boot-compact: no run directory $STATE/runs/$RUN_ID" >&2; exit 2; }

TRIG=${DSV4_BOOT_COMPACT_PSI:-20}
FOR_S=${DSV4_BOOT_COMPACT_FOR_S:-6}
MAX_RECLAIMS=${DSV4_BOOT_COMPACT_MAX_RECLAIMS:-3}
GAP_S=${DSV4_BOOT_COMPACT_GAP_S:-20}
MAX_S=${DSV4_BOOT_COMPACT_MAX_S:-1800}
PSI_FILE=${BOOT_COMPACT_PSI_FILE:-/proc/pressure/memory}
RECLAIM=${BOOT_COMPACT_RECLAIM:-"timeout 60 sudo -n /usr/local/sbin/gb10-reclaim"}
CONTAINER_CMD=${BOOT_COMPACT_CONTAINER_CMD:-"docker ps -q --filter name=^dsv4-$ROLE\$"}

RUN_DIR=$STATE/runs/$RUN_ID
mkdir -p "$RUN_DIR/$ROLE"
LOG=$RUN_DIR/$ROLE/boot-compact.log
exec 9>"$RUN_DIR/$ROLE/boot-compact.lock"
flock -n 9 || { echo "boot-compact: already running for $RUN_ID/$ROLE" >&2; exit 0; }

log() { echo "$(date +%T) $*" >>"$LOG"; }
psi() { awk '/^some/{split($2, a, "="); print a[2]}' "$PSI_FILE"; }
# Free blocks of order >= 9 in the Normal zone(s): what the stalled allocations need.
high_order() { awk '$4 == "Normal" { s = 0; for (i = 14; i <= NF; i++) s += $i; t += s } END { print t + 0 }' /proc/buddyinfo; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }

log "start run=$RUN_ID role=$ROLE trigger=psi_some>${TRIG} for ${FOR_S}s max_reclaims=$MAX_RECLAIMS gap=${GAP_S}s"
start=$SECONDS high_since= reclaims=0 last_reclaim=-100000 seen_container=0 peak=0
while (( SECONDS - start < MAX_S )); do
  [[ -e $RUN_DIR/$ROLE/serving ]] && { log "serving; exit (reclaims=$reclaims, peak psi=$peak)"; exit 0; }
  [[ -e $RUN_DIR/ALERT || -e $STATE/ALERT ]] && { log "ALERT latched; abort path owns the node; exit (reclaims=$reclaims, peak psi=$peak)"; exit 0; }
  if [[ -n $(eval "$CONTAINER_CMD" 2>/dev/null) ]]; then
    seen_container=1
  elif (( seen_container )); then
    log "container stopped; exit (reclaims=$reclaims, peak psi=$peak)"; exit 0
  fi
  p=$(psi) || p=0
  gt "$p" "$peak" && peak=$p
  if gt "$p" "$TRIG"; then
    [[ -n $high_since ]] || { high_since=$SECONDS; log "psi_some=$p above $TRIG; high-order blocks=$(high_order)"; }
    if (( SECONDS - high_since >= FOR_S && reclaims < MAX_RECLAIMS && SECONDS - last_reclaim >= GAP_S )); then
      reclaims=$((reclaims + 1)); last_reclaim=$SECONDS
      log "reclaim #$reclaims: psi_some=$p high-order blocks=$(high_order)"
      t0=$SECONDS
      if out=$(eval "$RECLAIM" 2>&1); then rc=0; else rc=$?; fi
      log "reclaim #$reclaims done rc=$rc in $((SECONDS - t0))s: ${out//$'\n'/ }; psi_some=$(psi) high-order blocks=$(high_order)"
    fi
  elif [[ -n $high_since ]]; then
    log "psi_some=$p back below $TRIG after $((SECONDS - high_since))s"; high_since=
  fi
  sleep 2
done
log "max runtime ${MAX_S}s reached; exit (reclaims=$reclaims, peak psi=$peak)"
