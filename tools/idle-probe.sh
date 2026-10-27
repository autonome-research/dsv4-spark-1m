#!/usr/bin/env bash
# Idle-latency probe (optimisation log E08/E09): after each idle gap, send one tiny
# chat request and log its HTTP status and end-to-end latency.
#
# Usage: idle-probe.sh <logfile> <gap_minutes>...
#   e.g. idle-probe.sh data/E09/idle.log 15 30 60   (waits 15 min, probes, waits 30, ...)
# Appends "PROBE start" / "PROBE done gap=… http=… latency=…s" lines to <logfile>.
set -u
OUT=${1:?logfile}; shift
(( $# )) || { echo "usage: idle-probe.sh <logfile> <gap_minutes>..." >&2; exit 2; }
for gap in "$@"; do
  sleep $((gap*60))
  t0=$(date +%s.%N); echo "$(date +%T) PROBE start after ${gap}min idle" >> "$OUT"
  c=$(curl -s -m 600 -o /dev/null -w "%{http_code}" localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Say OK."}],"max_tokens":4,"chat_template_kwargs":{"thinking":false}}')
  echo "$(date +%T) PROBE done gap=${gap}min http=$c latency=$(echo "$(date +%s.%N)-$t0"|bc)s" >> "$OUT"
done
