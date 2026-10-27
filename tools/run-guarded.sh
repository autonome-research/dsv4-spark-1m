#!/usr/bin/env bash
# Run one ctxbench.py request under a memory watchdog covering both nodes.
#
# Usage: run-guarded.sh <context_tokens> <output_prefix>
#   <prefix>.result  ctxbench.py output (last line is the JSON result)
#   <prefix>.mon     one line every 3 s: MemAvailable and memory PSI (some avg10), leader | worker
#
# The client is killed, which makes vLLM abort the request, when either node has
# MemAvailable < 6 GiB, or memory PSI > 40 for 10 consecutive samples (~30 s).
# The kill is logged as "WATCHDOG ABORT" in <prefix>.mon. If the worker cannot be read,
# its values are logged empty and only the leader limits apply for that sample.
# Read-only towards the server.
D=$(dirname "$0"); CTX=${1:?context_tokens}; OUT=${2:?output_prefix}
python3 -I "$D/ctxbench.py" "$CTX" 256 > "$OUT.result" 2>&1 & B=$!
hi=0
while kill -0 $B 2>/dev/null; do
  la=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); lp=$(awk '/some/{split($2,a,"=");print int(a[2])}' /proc/pressure/memory)
  w=$(timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 gx10 "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo; awk '/some/{split(\$2,a,\"=\");print int(a[2])}' /proc/pressure/memory" | tr '\n' ' '); set -- $w; wa=$1; wp=$2
  echo "$(date +%T) L avail ${la}G psi $lp | W avail ${wa}G psi $wp" >> "$OUT.mon"
  if [ "$lp" -gt 40 ] || [ "${wp:-0}" -gt 40 ]; then hi=$((hi+1)); else hi=0; fi
  if [ "$la" -lt 6 ] || [ "${wa:-99}" -lt 6 ] || [ $hi -ge 10 ]; then echo "$(date +%T) WATCHDOG ABORT la=$la wa=$wa hi=$hi" >> "$OUT.mon"; kill $B; break; fi
  sleep 3
done
wait $B 2>/dev/null; echo done >> "$OUT.mon"
