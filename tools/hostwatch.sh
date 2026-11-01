#!/usr/bin/env bash
# Always-on peer host watch (optimisation log I05/I06, 2026-11-01).
# Runs on both nodes, independent of the model. Every INTERVAL s it opens a TCP connection
# to the peer's sshd over both CX7 links. A host that is powered off or hung fails this;
# a stopped model does not. After MISSES failures in a row it records the peer as DOWN,
# and when the peer answers again it records UP with the outage length. Each transition
# also runs NOTIFY_CMD if one is set (empty until a notification channel is chosen).
#
# Usage: hostwatch.sh <peer-name> <ip> [<ip> ...]
# Log:   $HOME/dsv4-hostwatch.log (one line per transition, plus start and a daily heartbeat)
# Env:   HOSTWATCH_INTERVAL (10 s), HOSTWATCH_MISSES (3), HOSTWATCH_NOTIFY_CMD (receives the
#        message as $1). One instance per host (flock). Started at boot by the user crontab.
set -u
PEER=${1:?usage: hostwatch.sh <peer-name> <ip> [<ip> ...]}; shift
(($#)) || { echo "hostwatch: no peer IPs given" >&2; exit 1; }
IPS=("$@")
INTERVAL=${HOSTWATCH_INTERVAL:-10}; MISSES=${HOSTWATCH_MISSES:-3}
NOTIFY=${HOSTWATCH_NOTIFY_CMD:-}
LOG=$HOME/dsv4-hostwatch.log
exec 9>"$HOME/.dsv4-hostwatch.lock"; flock -n 9 || { echo "hostwatch already running" >&2; exit 1; }

log() { printf '%s %s %s\n' "$(date '+%F %T %Z')" "$(hostname -s)" "$*" >>"$LOG"; }
event() {
  log "$1"
  [[ -n $NOTIFY ]] && { timeout 30 bash -c "$NOTIFY \"\$1\"" _ "$(hostname -s): $1" >>"$LOG" 2>&1 || log "notify failed (rc=$?)"; }
}
probe() {   # 0 if any peer IP accepts a TCP connection on port 22
  local ip
  for ip in "${IPS[@]}"; do timeout 3 bash -c "</dev/tcp/$ip/22" 2>/dev/null && return 0; done
  return 1
}

log "start: peer=$PEER ips=${IPS[*]} interval=${INTERVAL}s misses=$MISSES notify=$([[ -n $NOTIFY ]] && echo set || echo none)"
state=unknown miss=0 first_miss=0 down_since=0 beat=$(date +%F)
while :; do
  if probe; then
    if [[ $state == down ]]; then
      event "UP: $PEER answers again after $(( ($(date +%s) - down_since) / 60 )) min down"
    elif [[ $state == unknown ]]; then
      log "peer $PEER reachable"
    fi
    state=up miss=0
  else
    miss=$((miss + 1)); (( miss == 1 )) && first_miss=$(date +%s)
    if [[ $state != down ]] && (( miss >= MISSES )); then
      down_since=$first_miss
      event "DOWN: $PEER unreachable on ${IPS[*]} since $(date -d "@$down_since" '+%T') ($miss probes)"
      state=down
    fi
  fi
  [[ $(date +%F) != "$beat" ]] && { beat=$(date +%F); log "heartbeat: peer $PEER $state"; }
  sleep "$INTERVAL"
done
