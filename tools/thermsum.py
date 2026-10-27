#!/usr/bin/env python3
"""Summarise thermlog.sh CSVs (optimisation log E13).

Usage: thermsum.py <csv>...

Per file: samples, busy samples (utilisation >= 50 %), median and minimum busy
graphics clock, maximum temperature, and the share of busy samples with each
slowdown bit set (SW power cap, SW/HW thermal slowdown, HW power brake)."""
import statistics, sys
BITS = {0x4: "sw_power_cap", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        f = [x.strip() for x in line.split(",")]
        if len(f) != 7 or not f[1].endswith("MHz"): continue
        rows.append((int(f[1].split()[0]), float(f[4]), int(f[5].split()[0]), int(f[6], 16)))
    busy = [r for r in rows if r[2] >= 50]
    out = {"file": path, "samples": len(rows), "busy": len(busy)}
    if busy:
        out.update(busy_clk_median=statistics.median(r[0] for r in busy), busy_clk_min=min(r[0] for r in busy),
                   temp_max=max(r[1] for r in rows))
        for b, n in BITS.items():
            out[n + "_pct"] = round(100 * sum(1 for r in busy if r[3] & b) / len(busy), 1)
    print(out)
