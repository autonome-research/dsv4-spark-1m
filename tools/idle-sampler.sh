#!/usr/bin/env bash
# 1 Hz host sampler used while investigating boot stalls (E03, I02).
# Each second: MemFree/MemAvailable/Cached (GiB), memory and IO PSI, deltas of
# pgmajfault, direct/kswapd page scans and compaction scans, GPU utilisation/clock/power
# (nvidia-smi with a 3 s timeout, "hang" if it stalls), the target process's state and
# wchan, its thread-state histogram, and RoCE receive bytes.
#
# Usage: idle-sampler.sh <pid> <outfile>     runs until <pid> exits
set -u
P=${1:?pid}; OUT=${2:?outfile}
rd(){ awk -v k="$1" '$1==k{print $2}' /proc/vmstat; }
rx(){ cat /sys/class/infiniband/*/ports/1/counters/port_rcv_data 2>/dev/null | awk '{s+=$1}END{print s+0}'; }
pm=$(rd pgmajfault); ps_=$(rd pgscan_direct); pk=$(rd pgscan_kswapd); cm=$(rd compact_migrate_scanned); r=$(rx)
while [ -d "/proc/$P" ]; do
  sleep 1
  m=$(awk '/^MemFree/{f=$2}/^MemAvailable/{a=$2}/^Cached/{c=$2}END{printf "%.1f %.1f %.1f",f/1048576,a/1048576,c/1048576}' /proc/meminfo)
  psi=$(awk -F'[ =]' '/some/{print $3}' /proc/pressure/memory)/$(awk -F'[ =]' '/some/{print $3}' /proc/pressure/io)
  npm=$(rd pgmajfault); nps=$(rd pgscan_direct); npk=$(rd pgscan_kswapd); ncm=$(rd compact_migrate_scanned); nr=$(rx)
  g=$(timeout 3 nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader,nounits 2>/dev/null) && g=${g// /} || g=hang
  st=$(awk '{print $3}' /proc/$P/stat 2>/dev/null)/$(cat /proc/$P/wchan 2>/dev/null)
  thr=$(ls /proc/$P/task 2>/dev/null | while read t; do awk '{print $3}' /proc/$P/task/$t/stat 2>/dev/null; done | sort | uniq -c | tr -s ' \n' ' ')
  echo "$(date +%T) free/avail/cache $m psi(m/io) $psi majflt $((npm-pm)) scan_d $((nps-ps_)) scan_k $((npk-pk)) compact $((ncm-cm)) gpu $g rx $(((nr-r)*4/1048576))MB proc $st thr[$thr]"
  pm=$npm; ps_=$nps; pk=$npk; cm=$ncm; r=$nr
done >> "$OUT"
