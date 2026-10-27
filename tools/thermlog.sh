#!/usr/bin/env bash
# Thermal / clock log while DSV4 serves (Velvet's D8: keep watching thermal throttling).
# Every INTERVAL s (default 30) appends one nvidia-smi sample to DIR/<host>-<YYYYMMDD>.csv:
#   time, graphics clock, SM clock, power, temperature, utilisation, clocks_event_reasons.active
# Reason bits (NVML): 0x4 SW power cap, 0x20 SW thermal slowdown, 0x40 HW thermal slowdown, 0x80 HW power brake.
# A stalled nvidia-smi is logged and skipped. Stops when killed. One instance per host (flock).
set -u
INTERVAL=${THERMLOG_INTERVAL:-30}; DIR=${THERMLOG_DIR:-/srv/dsv4/run/data/thermal}
mkdir -p "$DIR"; exec 9>"$DIR/.lock-$(hostname -s)"; flock -n 9 || { echo "thermlog already running on $(hostname -s)" >&2; exit 1; }
while :; do
  f="$DIR/$(hostname -s)-$(date +%Y%m%d).csv"
  out=$(timeout --kill-after=2 10 nvidia-smi --query-gpu=timestamp,clocks.gr,clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader 2>&1) \
    || out="$(date '+%Y/%m/%d %T'), nvidia-smi failed (rc=$?)"
  printf '%s\n' "$out" >> "$f"
  sleep "$INTERVAL"
done
