# Data

Raw measurement output for the experiments in
[`reports/DSV4 optimisation log.md`](../reports/DSV4%20optimisation%20log.md). Only this
index is tracked by git; the data itself (about 63 MB, mostly profiler traces) stays on
the leader at `/srv/dsv4/run/data/`. Experiments E01–E06 predate this directory; their
numbers are recorded in the log only.

Files are never edited after a run. A re-measurement gets a new file or a new
timestamped directory.

## File formats

| Suffix | Written by | Contents |
|---|---|---|
| `*.result` | `tools/run-guarded.sh` → `tools/ctxbench.py` | client output; the **last line** is the JSON result (`prompt_tokens`, `ttft_s`, `prefill_tok_s`, `completion_tokens`, `decode_tok_s`) |
| `*.mon` | `tools/run-guarded.sh`, `tools/e16-ab.sh` | one line every 3 s: MemAvailable and memory PSI on leader and worker; `WATCHDOG ABORT` if the request was killed |
| `*.jsonl` | ctxbench / e12b runs, `tools/infer-probe.py` | one JSON object per request |
| `*.json` | `tools/e12b_client.py`, `tools/gemv-acc.py` | one request with its output tokens and logprobs, or a GEMV accuracy run |
| `config.txt` | `final-bench.sh`, `boot-4k.sh`, `e16-ab.sh` | run ID, container args, NCCL env, mounted files (md5 in FINAL) |
| `summary.txt`, `summary.json`, `summary.tsv` | the same tools | median / min / max of the run |
| `*.csv` (thermal) | `tools/thermlog.sh` | `time, graphics clock, SM clock, power, temperature, utilisation, clocks_event_reasons` |
| `*-trace.json.gz` | vLLM torch profiler | Chrome traces, one per rank |

## Directories

| Directory | Log entry | What it holds |
|---|---|---|
| `E07/` | E07.0–E07.8 | 4K/1M decode baselines, clock and HCA changes, async scheduling, GEMV patch, GPU utilisation |
| `E08/` | E08, INV-1 | Idle-latency probes and the engine crash log (WiFi control plane) |
| `E09/` | E09 | Idle probes after pinning `VLLM_HOST_IP` to CX7 |
| `E10/` | E10 | 8192-token prefill chunks: 4K, 64K, 1M and memory log |
| `E11/` | E11 | 256K profile before the split indexer (traces, profiler summary) |
| `E12/` | E12a, E12b | Split-indexer exactness checks and the 1M same-prompt A/B |
| `E13/` | E13 | Leader/worker clock and thermal comparison at 256K |
| `E14/` | E14 | 256K profile with the split indexer |
| `E15/` | E15 | FlashInfer #4732: offline test logs (`test-orig/aot/fix-*.log`) and serving check |
| `E16/<ts>-rowshard/` | E16 | Upstream row-sharding serving A/B (64K ×2, 256K, 1M) |
| `F01/` | F01, F02 | Soak monitor logs (`soak-<run>*.log`) and inference probes (`probe-<run>*.jsonl`) |
| `FINAL/<ts>-f02-boot/` | Final benchmark | 4K ×3, 64K ×2, 256K, 1M on the F02 boot, with `config.txt` and md5s |
| `GEMV/` | F02, R01.1 | GEMV patch accuracy runs, patch on (2 runs) and off |
| `RESTART/<ts>-<label>/` | R01.0–R01.6 | Per-boot 4K check (warm-up + 3) for each boot of the restart series |
| `thermal/` | E13, R01.7 | `<host>-<date>.csv` for both nodes (worker files copied from `gx10:~/dsv4-thermal/`); no data 2026-10-23 → 2026-10-26 16:51 |
| `investigation/` | INV-1 | Network investigation write-ups (`network.md`, `timeline.md`, `history.md`) |

Other evidence lives outside this directory: guard state, alerts, peerwatch and
boot-compaction logs in `~/.cache/dsv4-t2/runs/<run-id>/` (each node), abort records in
`~/.cache/dsv4-t2/abort/<run-id>/`, and NCCL logs and profiler output in
`/srv/dsv4/diagnostics/<host>/`.
