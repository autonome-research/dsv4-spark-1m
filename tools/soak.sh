#!/usr/bin/env bash
# 24 h soak monitor for a serving DSV4 run (finishing plan, 2026-10-23).
# Every INTERVAL s: API /health, containers, ALERTs on both nodes, memory PSI/MemAvailable,
# D-state, watchdog processes, guards left from other runs, new hung-task/mt7925/NVRM/Xid
# kernel lines. The leader WiFi BSSID is logged for reference only (not an anomaly).
# Appends one line per sample to data/F01/soak-<run_id>.log; exits 1 on the first anomaly.
#
# Usage: soak.sh <run_id> [hours]      default 24 h; SOAK_INTERVAL (s, default 300);
#        SOAK_WIFI_CON = leader WiFi profile whose pinned BSSID is logged (optional)
set -uo pipefail
RUN_ID=${1:?run id}; HOURS=${2:-24}; INTERVAL=${SOAK_INTERVAL:-300}
OUT=/srv/dsv4/run/data/F01/soak-$RUN_ID.log
BSSID=n/a
[[ -n ${SOAK_WIFI_CON:-} ]] && BSSID=$(nmcli -g 802-11-wireless.bssid connection show "$SOAK_WIFI_CON" 2>/dev/null | tail -1)
since=$(date '+%F %T'); end=$((SECONDS + HOURS*3600))
say() { echo "$(date '+%F %T') $*" | tee -a "$OUT"; }
bad() { say "ANOMALY: $*"; exit 1; }
say "start run=$RUN_ID hours=$HOURS interval=${INTERVAL}s pinned_bssid=$BSSID"
while (( SECONDS < end )); do
  h=$(curl -s -o /dev/null -m 10 -w '%{http_code}' localhost:8888/health)
  lc=$(docker inspect dsv4-leader --format '{{.State.Running}}' 2>/dev/null)
  w=$(ssh -o BatchMode=yes -o ConnectTimeout=5 gx10 "docker inspect dsv4-worker --format '{{.State.Running}}' 2>/dev/null; ls ~/.cache/dsv4-t2/ALERT ~/.cache/dsv4-t2/runs/$RUN_ID/ALERT 2>/dev/null | wc -l; awk 'NR==1{split(\$2,a,\"=\");print a[2]}' /proc/pressure/memory; awk '/MemAvailable/{printf \"%.0f\n\",\$2/1048576}' /proc/meminfo; pgrep -fc 'dsv4-leaderwatch[.]sh'" 2>/dev/null | tr '\n' ' ')
  read -r wc wal wpsi wav wlw <<<"$w"
  la=$(ls ~/.cache/dsv4-t2/ALERT ~/.cache/dsv4-t2/runs/$RUN_ID/ALERT 2>/dev/null | wc -l)
  lpsi=$(awk 'NR==1{split($2,a,"=");print a[2]}' /proc/pressure/memory)
  lav=$(awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo)
  # D-state only counts if the same PID is still in D 15 s and 45 s later (brief I/O waits are
  # normal). Log 2026-10-25 09:47: a 15 s D was gone by the time it was printed, so the names
  # are now captured from the samples themselves, and a short one is logged, not fatal.
  dsamp() { ps -eo pid=,stat=,comm=,wchan:32= | awk '$2 ~ /^D/ && $3 != "fstrim" {print $1, $3, $4}' | sort; }
  d1=$(dsamp); d=0; dnote=""
  if [[ -n $d1 ]]; then
    sleep 15; d2=$(comm -12 <(echo "$d1" | awk '{print $1}') <(dsamp | awk '{print $1}'))
    if [[ -n $d2 ]]; then
      dnote=$(grep -wF -f <(echo "$d2") <<<"$d1" | tr '\n' ';')
      sleep 30; d=$(comm -12 <(echo "$d2") <(dsamp | awk '{print $1}') | grep -c . || true)
      (( d == 0 )) && say "note: D-state 15-45 s, cleared: $dnote"
    fi
  fi
  pw=$(pgrep -fc "dsv4-peerwatch[.]sh --run-id $RUN_ID")
  # Guards from other runs (log 2026-10-24 19:31: a leaked 10-23 guard tripped this run's ALERT).
  sg=$(pgrep -af 'guard[.]py run --run-id ' | grep -vcF -- "--run-id $RUN_ID " || true)
  cur=$(nmcli -t -f ACTIVE,BSSID dev wifi 2>/dev/null | awk -F: '/^yes/{print toupper(substr($0,5))}' | tr -d '\\')
  k=$(journalctl -k --since "$since" --no-pager 2>/dev/null | grep -cE "blocked for more than|mt7925|NVRM|Xid")
  say "health=$h leader=$lc worker=$wc alerts=$la/$wal psi=$lpsi/$wpsi availG=$lav/$wav D=$d peerwatch=$pw leaderwatch=$wlw bssid=$cur kernel_lines=$k stale_guards=$sg"
  [[ $h == 200 ]] || bad "API /health=$h"
  [[ $lc == true && $wc == true ]] || bad "container not running (leader=$lc worker=$wc)"
  [[ $la == 0 && $wal == 0 ]] || bad "ALERT present (leader=$la worker=$wal)"
  awk "BEGIN{exit !($lpsi>10 || $wpsi>10)}" && bad "memory PSI high ($lpsi/$wpsi)"
  (( lav >= 6 && wav >= 6 )) || bad "MemAvailable low ($lav/$wav GiB)"
  (( d == 0 )) || bad "D-state >45 s: $dnote"
  (( sg == 0 )) || bad "guard from another run still running: $(pgrep -af 'guard[.]py run --run-id ' | grep -vF -- "--run-id $RUN_ID " | cut -c1-120 | tr '\n' ';')"
  (( pw >= 1 && wlw >= 1 )) || bad "watchdog missing (peerwatch=$pw leaderwatch=$wlw)"
  (( k == 0 )) || bad "kernel: $(journalctl -k --since "$since" --no-pager | grep -E 'blocked for more than|mt7925|NVRM|Xid' | tail -3 | tr '\n' ';')"
  sleep "$INTERVAL"
done
say "COMPLETE: ${HOURS} h with no anomaly"
