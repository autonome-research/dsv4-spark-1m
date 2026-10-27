#!/usr/bin/env python3
"""Passive, fail-closed DSV4 T2 host guard (one instance per node and run).

The guard never stops or signals the model. While sampling it only reads /proc
(meminfo, pressure/memory, vmstat, process states) and writes its own state files;
on distress it latches a sticky ALERT that blocks the next start until a person
clears it.

Subcommands (all take --run-id, --role leader|worker, --state-dir unless noted):
  run           sample every --interval s; boot rules until runs/<id>/<role>/serving
                exists, serving rules afterwards (see BOOT_* and alert_reasons)
  check         is this run's guard alive, fresh and not alerted
  admission     check the attended admission descriptor and a fresh guard
  admit         create the admission descriptor (refused while an ALERT exists)
  latch         latch an ALERT with --reason (used by dsv4-abort.sh)
  cancel-owned  stop a launcher that registered an exact pid/starttime, then wait
  clear         remove all ALERTs; needs --confirm "CLEAR T2 ALERTS"

State (default ~/.cache/dsv4-t2, or DSV4_T2_STATE_DIR via the callers): ALERT,
admission.json, active-run, runs/<id>/ALERT, runs/<id>/<role>/{current,snapshot}.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

GB = 1024 ** 3
# /proc/pressure/memory "some avg10": percent of the last 10s in which at least one
# task stalled on memory.  A healthy DSV4 boot holds this at 0.00 throughout; it is
# the corroboration the residual-growth gate requires.  Independent of the
# MemAvailable floor and the D-state check, which remain unconditional.
PSI_STALL_AVG10 = 1.0
# Boot-phase rules (2026-10-12, approved by operator). Every guarded boot so far
# has shown a short reclaim/compaction stall when the KV cache is allocated
# (PSI some avg10 8-33, brief D-state). Until the start script marks the run
# serving (runs/<id>/<role>/serving), only a SUSTAINED stall aborts: PSI some
# avg10 above BOOT_PSI_AVG10 continuously for BOOT_PSI_SECONDS, or any task in
# D-state longer than BOOT_DSTATE_SECONDS. The MemAvailable floor is unchanged.
BOOT_PSI_AVG10 = 40.0
BOOT_PSI_SECONDS = 30.0
BOOT_DSTATE_SECONDS = 60.0
# Weekly fstrim legitimately sits in D-state for minutes (optimisation log, 2026-10-21).
# Matched on the exact comm, never a substring.
D_STATE_IGNORE = frozenset({"fstrim"})
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ROLES = {"leader", "worker"}
VM_KEYS = (
    "pswpin", "pswpout", "pgscan_direct", "pgsteal_direct",
    "allocstall_dma", "allocstall_dma32", "allocstall_normal",
    "allocstall_movable", "compact_migrate_scanned", "compact_free_scanned",
    "compact_stall", "compact_success", "compact_fail",
)


def die(message: str) -> int:
    print(f"dsv4-t2-guard: {message}", file=sys.stderr)
    return 1


def valid_run(run_id: str) -> bool:
    return bool(RUN_RE.fullmatch(run_id))


def root_path(value: str | None) -> Path:
    return Path(value or os.environ.get("DSV4_T2_STATE_DIR", str(Path.home() / ".cache/dsv4-t2")))


def checked_args(args: argparse.Namespace) -> tuple[Path, str, str]:
    if not valid_run(args.run_id):
        raise ValueError("invalid run ID")
    if args.role not in ROLES:
        raise ValueError("role must be leader or worker")
    root = root_path(args.state_dir)
    return root, args.run_id, args.role


def atomic_write(path: Path, data: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp, flags, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def touch_alert(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text[:4096] + "\n")


def proc_stat(proc: Path, pid: int) -> tuple[str, str]:
    text = (proc / str(pid) / "stat").read_text(encoding="ascii")
    close = text.rfind(")")
    if close < 0:
        raise ValueError("malformed /proc/PID/stat")
    fields = text[close + 2 :].split()
    if len(fields) < 20:
        raise ValueError("short /proc/PID/stat")
    return fields[0], fields[19]  # state is field 3; starttime is field 22


def proc_comm(proc: Path, pid: int) -> str:
    text = (proc / str(pid) / "stat").read_text(encoding="ascii", errors="replace")
    return text[text.find("(") + 1 : text.rfind(")")]


def self_starttime(proc: Path) -> str:
    return proc_stat(proc, os.getpid())[1]


def read_metrics(proc: Path) -> tuple[dict, set[tuple[int, str]]]:
    mem: dict[str, int] = {}
    mem_keys = {"MemTotal", "MemAvailable", "MemFree", "Active", "Inactive", "Slab", "PageTables", "Unevictable"}
    for line in (proc / "meminfo").read_text(encoding="ascii").splitlines():
        key, _sep, rest = line.partition(":")
        if key in mem_keys:
            fields = rest.split()
            if not fields or not fields[0].isdigit():
                raise ValueError(f"invalid {key}")
            # meminfo reports these values in KiB.
            mem[key] = int(fields[0]) * 1024
    if set(mem) != mem_keys:
        raise ValueError("required meminfo fields missing")
    vmstat: dict[str, int] = {}
    for line in (proc / "vmstat").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in VM_KEYS and fields[1].isdigit():
            vmstat[fields[0]] = int(fields[1])
    for required in ("pswpin", "pswpout"):
        if required not in vmstat:
            raise ValueError(f"{required} missing from vmstat")
    dstate: set[tuple[int, str]] = set()
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            state, starttime = proc_stat(proc, pid)
        except FileNotFoundError:
            # A process can disappear between listdir and stat; that is not a D-state sample.
            continue
        except (PermissionError, ValueError) as exc:
            # A persistent unreadable/malformed PID record makes this health observer uncertain.
            raise ValueError(f"cannot collect PID {pid}: {exc}") from exc
        if state == "D":
            try:
                if proc_comm(proc, pid) in D_STATE_IGNORE:
                    continue
            except FileNotFoundError:
                continue
            dstate.add((pid, starttime))
    # PSI is the kernel's own stall accounting and is what separates a legitimate
    # allocation from distress.  An absent or malformed file is fail-closed: without
    # it the residual gate cannot be evaluated as specified, so the guard dies rather
    # than silently degrading to growth-only (which aborts every healthy startup).
    psi: dict[str, float] = {}
    for line in (proc / "pressure" / "memory").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or fields[0] not in {"some", "full"}:
            continue
        for field in fields[1:]:
            key, _sep, value = field.partition("=")
            if key == "avg10":
                try:
                    parsed = float(value)
                except ValueError as exc:
                    raise ValueError(f"invalid psi {fields[0]} avg10") from exc
                if not math.isfinite(parsed):
                    raise ValueError(f"non-finite psi {fields[0]} avg10")
                psi[fields[0]] = parsed
    if set(psi) != {"some", "full"}:
        raise ValueError("required /proc/pressure/memory avg10 fields missing")
    # Host residual is a heuristic, not attribution; MemAvailable is a separate floor.
    residual = (mem["MemTotal"] - mem["MemFree"] - mem["Active"] - mem["Inactive"]
                - mem["Slab"] - mem["PageTables"] - mem["Unevictable"])
    return {
        "mem_total": mem["MemTotal"],
        "mem_available": mem["MemAvailable"],
        "residual": residual,
        "psi_some_avg10": psi["some"],
        "psi_full_avg10": psi["full"],
        "vmstat": vmstat,
        "d_state": [[pid, start] for pid, start in sorted(dstate)],
    }, dstate


def alert_reasons(metrics: dict, residual_window: list[tuple[float, int]], dstate_started: dict[tuple[int, str], float], mono: float, psi_threshold: float = PSI_STALL_AVG10, boot: bool = False, psi_high_for: float = 0.0) -> list[str]:
    reasons: list[str] = []
    if metrics["mem_available"] < 6 * GB:
        reasons.append("MemAvailable<6GiB")
    if boot:
        if psi_high_for >= BOOT_PSI_SECONDS:
            reasons.append(f"boot:psi_some>{BOOT_PSI_AVG10:g}_for_{BOOT_PSI_SECONDS:g}s")
        if any(mono - started > BOOT_DSTATE_SECONDS for started in dstate_started.values()):
            reasons.append(f"boot:D-state>{BOOT_DSTATE_SECONDS:g}s")
        return reasons
    old = next(((stamp, value) for stamp, value in residual_window if mono - stamp >= 10.0), None)
    if old and metrics["residual"] - old[1] > 8 * GB:
        # Growth alone is not distress.  Measured 2026-09-28: a healthy boot reserves
        # the CUDA/KV pool in one step -- 82 GiB inside 16s, >=10x this threshold --
        # with psi_some_avg10 at exactly 0.00, MemAvailable never below 35 GiB and
        # allocstall_normal never nonzero.  Growth-only aborted every startup at
        # ~110s.  A wedge stalls; a load does not, so require corroboration.
        stall = metrics.get("psi_some_avg10", 0.0)
        if stall > psi_threshold:
            reasons.append("residual_growth>8GiB/10s+psi_stall")
    if any(mono - started > 10.0 for started in dstate_started.values()):
        reasons.append("D-state>10s")
    return reasons


def state_paths(root: Path, run_id: str, role: str) -> tuple[Path, Path, Path]:
    directory = root / "runs" / run_id / role
    return directory, directory / "current.json", directory / "snapshot.json"


def has_alert(root: Path, run_id: str | None = None) -> bool:
    if (root / "ALERT").exists():
        return True
    runs = root / "runs"
    if runs.is_dir() and any(path.name == "ALERT" for path in runs.glob("*/ALERT")):
        return True
    return run_id is not None and (root / "runs" / run_id / "ALERT").exists()


def latch(root: Path, run_id: str, reason: str) -> None:
    stamp = f"{time.time():.3f} {reason}"[:4096]
    touch_alert(root / "ALERT", stamp)
    touch_alert(root / "runs" / run_id / "ALERT", stamp)


def sample(args: argparse.Namespace) -> int:
    try:
        root, run_id, role = checked_args(args)
        proc = Path(args.proc_root)
        if not proc.is_dir() or proc.is_symlink():
            return die("proc root is not a directory")
        if has_alert(root, run_id):
            return die("sticky ALERT exists; human clear is required")
        directory, current, snapshot = state_paths(root, run_id, role)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        guard_start = self_starttime(proc)
    except (OSError, ValueError) as exc:
        return die(str(exc))

    previous_d: dict[tuple[int, str], float] = {}
    residual_window: list[tuple[float, int]] = []
    psi_high_since: float | None = None
    alerted = False
    samples = 0
    while args.samples <= 0 or samples < args.samples:
        mono = time.monotonic()
        try:
            metrics, dstate = read_metrics(proc)
            residual_window.append((mono, metrics["residual"]))
            residual_window = [(stamp, value) for stamp, value in residual_window if mono - stamp <= 12.0]
            dstate_started = {key: previous_d.get(key, mono) for key in dstate}
            previous_d = {key: dstate_started[key] for key in dstate}
            boot = not (directory / "serving").exists()
            if metrics["psi_some_avg10"] > BOOT_PSI_AVG10:
                psi_high_since = mono if psi_high_since is None else psi_high_since
            else:
                psi_high_since = None
            psi_high_for = 0.0 if psi_high_since is None else mono - psi_high_since
            reasons = alert_reasons(metrics, residual_window, dstate_started, mono, boot=boot, psi_high_for=psi_high_for)
            status = "alert" if alerted or reasons else "ok"
            payload = {
                "run_id": run_id, "role": role, "status": status,
                "guard_pid": os.getpid(), "guard_starttime": guard_start,
                "heartbeat_monotonic": mono, "heartbeat_unix": time.time(),
                "metrics": metrics, "mode": "boot" if boot else "serving",
            }
            if reasons and not alerted:
                reason = ";".join(reasons)
                payload["alert_reason"] = reason
                atomic_write(snapshot, json.dumps({**payload, "alerted_at_monotonic": mono}, separators=(",", ":"))[:65536] + "\n") if not snapshot.exists() else None
                latch(root, run_id, reason)
                alerted = True
            atomic_write(current, json.dumps(payload, separators=(",", ":"))[:65536] + "\n")
            samples += 1
        except (OSError, ValueError) as exc:
            return die(f"sampling failed: {exc}")
        if args.samples > 0 and samples >= args.samples:
            break
        time.sleep(max(0.1, args.interval))
    return 0


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("state is not an object")
    return value


def check_guard(args: argparse.Namespace) -> int:
    try:
        root, run_id, role = checked_args(args)
        if has_alert(root, run_id):
            return die("sticky ALERT exists")
        _directory, current, _snapshot = state_paths(root, run_id, role)
        data = load_json(current)
        if data.get("run_id") != run_id or data.get("role") != role or data.get("status") != "ok":
            return die("guard state is not current and healthy")
        pid = data.get("guard_pid")
        start = data.get("guard_starttime")
        heartbeat = data.get("heartbeat_monotonic")
        if (not isinstance(pid, int) or pid <= 0 or not isinstance(start, str)
                or isinstance(heartbeat, bool) or not isinstance(heartbeat, (int, float))
                or not math.isfinite(heartbeat)):
            return die("guard identity or heartbeat is malformed")
        state, actual_start = proc_stat(Path(args.proc_root), pid)
        if state == "Z" or actual_start != start:
            return die("guard PID/starttime is not alive and unchanged")
        if time.monotonic() - heartbeat > args.max_age or time.monotonic() < heartbeat - 1:
            return die("guard heartbeat is stale")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return die(f"guard check failed: {exc}")


def check_admission(args: argparse.Namespace) -> int:
    try:
        root, run_id, role = checked_args(args)
        descriptor = load_json(root / "admission.json")
        if descriptor.get("run_id") != run_id or descriptor.get("status") != "active":
            return die("attended admission is missing or for another run")
        if descriptor.get("roles") != ["leader", "worker"]:
            return die("admission roles are malformed")
        created = descriptor.get("created_unix")
        if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
            return die("admission timestamp is malformed")
        if (root / "CANCELLED").exists() or has_alert(root, run_id):
            return die("admission is cancelled or alerted")
        return check_guard(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return die(f"admission check failed: {exc}")


def admit(args: argparse.Namespace) -> int:
    try:
        root, run_id, _role = checked_args(args)
        if has_alert(root, run_id):
            return die("sticky ALERT exists; clear it explicitly first")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {"run_id": run_id, "roles": ["leader", "worker"], "status": "active", "created_unix": time.time()}
        atomic_write(root / "admission.json", json.dumps(payload, separators=(",", ":")) + "\n")
        atomic_write(root / "active-run", run_id + "\n")
        return 0
    except (OSError, ValueError) as exc:
        return die(str(exc))


def cancel_owned(args: argparse.Namespace) -> int:
    try:
        root, run_id, role = checked_args(args)
        path = root / "owned-process.json"
        if not path.exists():
            return 0
        data = load_json(path)
        pid, start = data.get("pid"), data.get("starttime")
        if data.get("run_id") != run_id or data.get("role") != role or not isinstance(pid, int) or pid <= 0 or not isinstance(start, str):
            return die("owned process descriptor is malformed or belongs to another run")
        try:
            _state, actual_start = proc_stat(Path(args.proc_root), pid)
        except FileNotFoundError:
            return 0  # The owned launcher exited between descriptor read and collection.
        if actual_start != start:
            return die("owned process identity changed")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return 0
        # Signalling is not cancellation. The launcher defers its handler until an
        # in-flight bounded `docker run` returns, so returning here let a caller
        # read "cancelled" while a container was still being created. Wait for the
        # owned process to actually leave, bounded, and fail closed if it does not:
        # an unconfirmed exit must never be reported as a completed cancellation.
        # Never escalate to SIGKILL, and never touch anything but this exact PID.
        deadline = time.monotonic() + max(0.0, args.wait)
        while True:
            try:
                state, current_start = proc_stat(Path(args.proc_root), pid)
            except FileNotFoundError:
                return 0
            except (PermissionError, ValueError) as exc:
                return die(f"cannot confirm owned process exit: {exc}")
            if current_start != start:
                return 0  # The PID was reused; the owned launcher is gone.
            if state == "Z":
                return 0  # Exited, awaiting reap by its own parent.
            if time.monotonic() >= deadline:
                return die(
                    f"owned launcher PID {pid} did not exit within {max(0.0, args.wait):.1f}s; "
                    "it may still create a container, so cancellation is NOT confirmed"
                )
            time.sleep(0.1)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return die(f"owned process cancellation failed: {exc}")


def clear_alerts(args: argparse.Namespace) -> int:
    if args.confirm != "CLEAR T2 ALERTS":
        return die('explicit confirmation required: "CLEAR T2 ALERTS"')
    root = root_path(args.state_dir)
    try:
        (root / "ALERT").unlink(missing_ok=True)
        (root / "CANCELLED").unlink(missing_ok=True)
        for path in (root / "runs").glob("*/ALERT"):
            path.unlink(missing_ok=True)
        return 0
    except OSError as exc:
        return die(str(exc))


def latch_command(args: argparse.Namespace) -> int:
    try:
        root, run_id, _role = checked_args(args)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_write(root / "CANCELLED", run_id + "\n")
        latch(root, run_id, args.reason)
        return 0
    except (OSError, ValueError) as exc:
        return die(str(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-id", required=True)
        p.add_argument("--role", required=True, choices=sorted(ROLES))
        p.add_argument("--state-dir")
    p = sub.add_parser("run"); common(p); p.add_argument("--proc-root", default="/proc"); p.add_argument("--interval", type=float, default=2.0); p.add_argument("--samples", type=int, default=0); p.set_defaults(func=sample)
    p = sub.add_parser("check"); common(p); p.add_argument("--proc-root", default="/proc"); p.add_argument("--max-age", type=float, default=6.0); p.set_defaults(func=check_guard)
    p = sub.add_parser("admission"); common(p); p.add_argument("--proc-root", default="/proc"); p.add_argument("--max-age", type=float, default=6.0); p.set_defaults(func=check_admission)
    p = sub.add_parser("admit"); common(p); p.set_defaults(func=admit)
    p = sub.add_parser("latch"); common(p); p.add_argument("--reason", required=True); p.set_defaults(func=latch_command)
    p = sub.add_parser("cancel-owned"); common(p); p.add_argument("--proc-root", default="/proc"); p.add_argument("--wait", type=float, default=20.0); p.set_defaults(func=cancel_owned)
    p = sub.add_parser("clear"); p.add_argument("--state-dir"); p.add_argument("--confirm", required=True); p.set_defaults(func=clear_alerts)
    args = parser.parse_args()
    try:
        return args.func(args)
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
