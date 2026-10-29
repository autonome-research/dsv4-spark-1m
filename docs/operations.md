# Operations

Runbook for the two-node DSV4 deployment. Commands run on the leader
(`spark-61dd`) unless marked **worker**. The worker is `gx10` in the leader's SSH
config and `192.168.200.11` over the CX7 link.

- [Ground rules](#ground-rules)
- [Health check](#health-check)
- [Start and stop](#start-and-stop)
- [Alerts](#alerts)
- [Recovery](#recovery)
- [Monitoring and soak tests](#monitoring-and-soak-tests)
- [Changing the configuration](#changing-the-configuration)
- [Where things are](#where-things-are)

---

## Ground rules

- **Do not press a power button or reboot a node while the model is serving.** A
  node that looks dead over WiFi or Tailscale is usually still serving over CX7.
  Check over the direct cable first (`ssh gx10` from the leader). The I03 outage
  was a power-button press on a healthy worker.
- **One GPU workload per node.** The model uses almost all of the unified memory;
  another GPU job can push a node into memory pressure and wedge it.
- **`/srv/dsv4/live/` is the reference configuration and is never edited.** This
  repository (`/srv/dsv4/run/`) is the working copy.
- **Alerts are cleared by a person**, after the cause is understood.
- Changes to the guard, abort or other safety scripts, and anything that needs
  root, are reviewed by the operator before they are applied. Each is recorded in the
  optimisation log and committed; node setup that needs root is in `setup/`.

---

## Health check

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8888/health   # expect 200
/srv/dsv4/run/launch-dsv4.sh status
ls ~/.cache/dsv4-t2/ALERT 2>/dev/null; ssh gx10 'ls ~/.cache/dsv4-t2/ALERT 2>/dev/null'   # expect no output
```

`tools/st.sh` prints a one-screen summary (memory on both nodes, containers, GPU
utilisation, alert, recent key log lines).

---

## Start and stop

### Start

```bash
/srv/dsv4/run/start-dsv4-default.sh
```

The script prints its settings line and then waits, without a prompt, for the exact
phrase `START DSV4 T2`. If the preflight finds DSV4/vLLM containers, processes or GPU
users on either node, it lists them by exact ID and asks `y`/`n`. Answering `y`
stops each listed container (`docker stop -t 120`; never removed) and signals each
listed PID (SIGTERM, then SIGKILL 10 s later), which destroys that work: check the
owners first, since the machines are shared. Guards and leader watches left by
earlier runs are stopped automatically.

Typical output ends with:

```
DSV4 T2: worker watch started (pid …); log ~/.cache/dsv4-t2/runs/<run-id>/peerwatch.log
DSV4 T2: leader watch started on the worker; log ~/.cache/dsv4-t2/runs/<run-id>/leaderwatch.log (worker)
```

**After every successful start**, before handing the server over, run the shape warm-up
and then the 4K check:

```bash
/srv/dsv4/run/tools/shape-warmup.py <label>
```

```bash
/srv/dsv4/run/tools/boot-4k.sh <label>
```

The warm-up (about 2 min, 92 short requests) loads every GPU kernel that is specialised
to a token count while the process is young. vLLM's own warm-up does not cover these, and
a first-time load two days after boot crashed the engine on 2026-10-28 (I04, optimisation
log; DeepGEMM issue #443). It must print `ok`. A `FAILED` line means the engine may be
down: read `docker logs dsv4-leader`.

If the start fails, the launcher runs a coordinated abort and stops its own
guards. Read the last `fail` message; it names the failed check.

| Message | Meaning | Action |
|---|---|---|
| `another attended T2 admission is active` | another start holds the admission lock | wait for it, or confirm it is gone |
| `sticky ALERT exists; clear it explicitly first`, then `local attended admission could not be created` | an `ALERT` is still latched, usually from `down` | clear both nodes ([Clear](#clear)) |
| `both node guards are missing, stale, malformed, exited, or alerted` | a guard did not start or did not report in time | read `~/.cache/dsv4-t2/leader.guard.err` (worker: `~/.cache/dsv4-t2/worker.guard.err`) |
| `not enough truly free memory after reclaim` | `MemFree < 100 GiB` on a node | look for another GPU process; reboot that node only if nothing else is using memory |
| `leader/worker pre-launch reclaim failed` | `gb10-reclaim` or its sudoers rule missing on that node | install `setup/gb10-reclaim` and run `sudo bash setup/node-prep.sh` on that node |
| `indexer mismatch: worker […] leader […]`, then `coordinated launch failed` | the ranks mounted different indexer files | check `DSV4_IDX_MODE` and that `patches/` was copied to the worker |
| `API readiness failed (rc=1)` | `/v1/models` did not answer within `DSV4_READINESS_TIMEOUT` (900 s) | read `docker logs dsv4-leader` and `ssh gx10 docker logs dsv4-worker` |
| `API readiness failed (rc=2)` | a container stopped during boot | same logs; usually an engine error |
| `API readiness failed (rc=3)` | a guard alerted or went stale, or an observer died, during boot | read the `ALERT` reason ([Alerts](#alerts)); `boot:psi…` is the boot stall |

### Stop

```bash
/srv/dsv4/run/launch-dsv4.sh down
sudo -n /usr/local/sbin/gb10-reclaim
ssh gx10 sudo -n /usr/local/sbin/gb10-reclaim
```

`down` stops peerwatch first (a deliberate stop must not look like a lost worker),
then runs `dsv4-abort.sh` (exact-ID stop of both containers), then the run's guards
and observers. Leaderwatch exits by itself once the worker container has stopped.
It prints what was left, for example `guards/observers left for <run>: leader=0 worker=0`.

The abort **latches an alert on both nodes** (reason `launch-dsv4 down`), so
[clear both](#clear) before the next start.

---

## Alerts

A guard writes `~/.cache/dsv4-t2/ALERT` and `~/.cache/dsv4-t2/runs/<run-id>/ALERT`
when it sees distress. **Alerts never stop the model by themselves**; they block
the next start until cleared, and the soak monitor stops on them.

### Investigate

```bash
cat ~/.cache/dsv4-t2/ALERT                                   # "<unix time> <reason>"
cat ~/.cache/dsv4-t2/runs/<run-id>/leader/snapshot.json      # metrics at the moment of the alert
journalctl -k --since "<alert time - 5 min>" --until "<alert time + 5 min>"
```

| Reason | Usual cause |
|---|---|
| `MemAvailable<6GiB` | a request or another process is exhausting unified memory |
| `boot:psi_some>40_for_30s`, `boot:D-state>60s` | boot mode only: sustained memory pressure or a long kernel block during start. Usually the KV-allocation stall (I02 class, roughly 1 boot in 3): clear and start again, and see [Boot stall](#boot-stall-at-kv-allocation-i02-class) |
| `D-state>10s` | a process blocked in the kernel; `snapshot.json` lists `[pid, starttime]`. Check what the PID is: in the past it was `fstrim` (now ignored), a guard left over from an earlier run, and the daily `systemd-tmpfiles-clean` (2026-10-27; the PID's start time matched the service start in `journalctl`). While serving, the model keeps running; clear the alert before the next start |
| `residual_growth>8GiB/10s+psi_stall` | serving mode only: memory outside the model grew by more than 8 GiB in 10 s while memory PSI was above 1 |

### Clear

After the cause is understood:

```bash
python3 /srv/dsv4/run/dsv4-t2-guard.py clear --confirm "CLEAR T2 ALERTS"          # leader
ssh gx10 'python3 ~/dsv4-t2-guard.py clear --confirm "CLEAR T2 ALERTS"'           # worker
```

---

## Recovery

### Worker container still running after a failed start or abort

The abort stops only containers it can identify. Its remote `docker inspect` of the
worker is retried up to 3 times; if all attempts fail, the worker record is
`unknown` and the worker container is left running (about 90 GiB). The abort record
is in `~/.cache/dsv4-t2/abort/<run-id>/`. The next start's preflight lists the
container and stops it after you answer `y`. To stop it by hand, get its exact ID first:

```bash
ssh gx10 'docker ps --no-trunc --filter name=dsv4-worker'
ssh gx10 'docker stop -t 120 <exact id>' && ssh gx10 sudo -n /usr/local/sbin/gb10-reclaim
```

### Boot stall at KV allocation (I02 class)

Symptoms: right after `GPU KV cache size` in the leader log, memory PSI rises to
70–95 % on one node and `nvidia-smi` may hang. Most stalls clear by themselves in
1–2 minutes, but the guard aborts after 30 s above 40 %; one (I02) lasted 11 minutes
until a reclaim with compaction released it within 10 s.

During boot, `tools/boot-compact.sh` runs that reclaim automatically when PSI stays
above 20 for 6 s (at most 3 times, 20 s apart). After a stalled boot, read its log on
both nodes; it records PSI and the free high-order block count before and after
each reclaim:

```bash
cat ~/.cache/dsv4-t2/runs/<run-id>/leader/boot-compact.log
ssh gx10 'cat ~/.cache/dsv4-t2/runs/<run-id>/worker/boot-compact.log'
```

If a node is still stuck after the abort, reclaim it by hand:

```bash
sudo -n /usr/local/sbin/gb10-reclaim          # on the stuck node
```

Root cause not established; record the time, PSI and the helper log in the
optimisation log.

### Node unreachable over WiFi or Tailscale

The CX7 link is separate from WiFi. From the leader:

```bash
ssh gx10 'uptime; nmcli -t -f GENERAL.STATE dev show wlP9s9; docker ps'
```

The worker runs `wifi-keepalive.timer` (every 5 min) to bring WiFi back after a
failed handshake. On the leader, WiFi roams between two routers with the same
SSID; serving is unaffected.

### Leader down while the worker keeps running

`dsv4-leaderwatch.sh` on the worker stops the worker container after 18 failed
`/health` probes 10 s apart, and latches the worker alert. Its log is
`~/.cache/dsv4-t2/runs/<run-id>/leaderwatch.log` on the worker.

---

## Monitoring and soak tests

Long-running monitors are started detached so they survive the operator's session:

```bash
cd /srv/dsv4/run
R=$(head -n 1 ~/.cache/dsv4-t2/active-run)
setsid nohup tools/soak.sh "$R" 24 </dev/null >/dev/null 2>data/F01/soak-$R.err &
setsid nohup python3 -I tools/infer-probe.py "$R" 24 1800 </dev/null >/dev/null 2>data/F01/probe-$R.err &
```

| Tool | Output | Notes |
|---|---|---|
| `tools/soak.sh <run> [hours]` | `data/F01/soak-<run>.log` | One line every 5 min; exits on the first `ANOMALY` |
| `tools/infer-probe.py <run> <hours> [interval]` | `data/F01/probe-<run>.jsonl` | One 128-token greedy request every 30 min |
| `tools/thermlog.sh` | `data/thermal/<host>-<date>.csv` | GPU clocks, power, temperature, throttle reasons every 30 s |
| `tools/thermsum.py <csv>...` | stdout | Busy-clock median, max temperature, % throttled |

The thermal logger does not survive a reboot. Start it after the server is serving
(it polls `nvidia-smi`, so keep it out of the boot window):

```bash
setsid nohup /srv/dsv4/run/tools/thermlog.sh </dev/null >/dev/null 2>>/srv/dsv4/run/data/thermal/thermlog.err &
scp -q /srv/dsv4/run/tools/thermlog.sh gx10:dsv4-thermlog.sh
ssh gx10 'mkdir -p ~/dsv4-thermal && THERMLOG_DIR=$HOME/dsv4-thermal setsid nohup ~/dsv4-thermlog.sh </dev/null >/dev/null 2>>$HOME/dsv4-thermal/thermlog.err &'
```

`soak.sh` treats as an anomaly: `/health` ≠ 200, a container not running, any
`ALERT`, memory PSI > 10, `MemAvailable` < 6 GiB, a process in D-state for 45 s, a
missing peerwatch or leaderwatch, a guard from another run, or a new kernel
hung-task, mt7925, NVRM or Xid line.

---

## Changing the configuration

1. Change one setting at a time, through its environment variable where one exists
   (see the README's configuration table).
2. Record the run ID, the setting and the measurement in the optimisation log,
   tagged **[measured]**, **[inferred]** or **[hypothesis]**.
3. Commit:

   ```bash
   tools/commit-snapshot.sh "E17: <what changed>"
   ```

Launcher edits keep the existing style: fail-closed checks, exact IDs, bounded
timeouts on every remote call, and a comment that names the log entry that
motivated the change. Patch files are never edited in place; a change is a new file
with a new switch value ([patches.md](patches.md)).

The DSpark experiment has its own procedure, switches, stop rules and rollback:
[dspark-experiment.md](dspark-experiment.md).

---

## Where things are

| Path | Contents |
|---|---|
| `~/.cache/dsv4-t2/active-run` | ID of the current run |
| `~/.cache/dsv4-t2/runs/<run-id>/` | Guard state (`leader/`), `ALERT`, `cids/`, `observer/`, peerwatch log, post-load reclaim log, `leader/boot-compact.log`, blockers |
| `~/.cache/dsv4-t2/runs/<run-id>/` (worker) | Worker guard state, `leaderwatch.log`, `worker/boot-compact.log` |
| `~/.cache/dsv4-t2/abort/<run-id>/` | Abort record: inspected containers and what was stopped |
| `data/<experiment>/` | Benchmark output per experiment (untracked) |
| `data/FINAL/<timestamp>/` | `final-bench.sh` output: `config.txt`, per-request results, `summary.json` |
| `/srv/dsv4/diagnostics/<host>/` | NCCL logs and profiler traces |
| `/srv/dsv4/runtime-cache/46638857-fi0618-sm121/` | Persisted TileLang/Triton caches (removes first-request JIT delay) |
| `data/thermal/` | Thermal logs (`<host>-<date>.csv`; worker files copied from `gx10:~/dsv4-thermal/`) |
| `~/reports/` | Canonical optimisation log and final report |
