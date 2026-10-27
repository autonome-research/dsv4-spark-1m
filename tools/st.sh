#!/usr/bin/env bash
# One-screen status of a DSV4 start or run: MemAvailable and containers on both
# nodes, GPU utilisation and power, a latched ALERT, the post-load reclaim log, and
# the last key lines of the leader log (errors, KV cache, graph capture, startup).
#
# Usage: st.sh [log_lines]      default 12
date +%T
la=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo)
wa=$(timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 gx10 "awk '/MemAvailable/{printf \"%.1f\",\$2/1048576}' /proc/meminfo; echo -n ' '; docker ps --format '{{.Names}}:{{.Status}}' | grep dsv4")
echo "leader avail ${la}G $(docker ps --format '{{.Names}}:{{.Status}}' | grep dsv4) | worker avail $wa"
echo "gpu L: $(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader)  W: $(timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=5 gx10 nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader)"
[ -s ~/.cache/dsv4-t2/ALERT ] && echo "ALERT: $(cat ~/.cache/dsv4-t2/ALERT)"
R=$(cat ~/.cache/dsv4-t2/active-run 2>/dev/null); [ -n "$R" ] && cat ~/.cache/dsv4-t2/runs/"$R"/postload.log 2>/dev/null
docker logs dsv4-leader 2>&1 | grep -vE "it/s\]|s/it\]" | grep -E "ERROR|Error|Traceback|WARN|took|KV cache|concurrency|Capturing|startup complete|Uvicorn|Loading|compil|Avail" | tail -${1:-12} | cut -c1-200
docker logs dsv4-leader 2>&1 | tail -1 | tr '\r' '\n' | tail -1 | cut -c1-200
