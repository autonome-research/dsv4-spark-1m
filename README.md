# dsv4-spark-serving

Serve **DeepSeek-V4-Flash-0731 (NVFP4)** with a **1M-token context** on two
**NVIDIA DGX Spark (GB10)** nodes, using vLLM with tensor parallelism across a
ConnectX-7 RoCE link.

This repository holds the launcher, the vLLM and FlashInfer patches, the host
safety watchers and the benchmark tools used to bring the model up stably and
to improve its decode and prefill speed.

| | |
|---|---|
| **Status** | Serving; handed over 2026-10-26. 24 h stability soak (F02) passed; final benchmark done |
| **Model** | `deepseek-v4-flash-0731`, NVFP4, 73.85 GiB per rank |
| **Context** | 1,048,576 tokens (`--max-model-len`), one request at a time |
| **Hardware** | 2 × DGX Spark GB10 (128 GB unified memory each), CX7 200G direct cable |
| **Software** | vLLM `0.26.1rc1.dev1219+g46638857f`, FlashInfer `0.6.18.dev20260819`, image `local/vllm-openai:46638857-fi0618` |
| **API** | OpenAI-compatible, `http://spark-61dd:8888/v1` |

---

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [Using the API](#using-the-api)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Safety model](#safety-model)
- [Repository layout](#repository-layout)
- [Benchmarking](#benchmarking)
- [Known issues and limitations](#known-issues-and-limitations)
- [Upstream status](#upstream-status)
- [Reusing this work](#reusing-this-work)
- [Further documentation](#further-documentation)
- [Acknowledgements](#acknowledgements)
- [Licence](#licence)

---

## Results

Single request, thinking off, speculative decoding (DSpark) off. Baseline is the
first configuration that served 1M (E06, 2026-10-12). Experiment IDs (E06, E10,
...) refer to entries in the [optimisation log](reports/DSV4%20optimisation%20log.md).

| Metric | Baseline (E06) | Final (2026-10-25) | Change |
|---|---|---|---|
| Decode, 4K context | 22.79 tok/s¹ | 27.20 tok/s (n=3, 27.12–27.32) | +19 % |
| Decode, 64K context | — | 26.61 tok/s (n=2) | |
| Decode, 1M context | 21.08 tok/s | 25.12 tok/s | +19 % |
| Prefill, 64K (TTFT) | 46.1 s (E07.6) | 29.2 s | −37 % |
| Prefill, 256K (TTFT) | 143.9 s (E13) | 137.3 s | −5 % |
| Prefill, 1M (TTFT) | 1,113.3 s (18.5 min) | 866.0 s (14.4 min) | −22 % |
| First request after 60 min idle | engine crash after ~30 min (E08) | 0.16 s (E09) | fixed |

Final numbers: `tools/final-bench.sh` on the configuration in this repository, after
26 h of uptime (`data/FINAL/20261025T135749-f02-boot/`, `ctxbench.py`, 256 output
tokens). Baselines use the same client.

¹ E06, 256 output tokens, the same method as the final number. The first 512-token
measurement (E07.0, 23.56 tok/s) gives +15.4 %; the final report quotes both.

With the deterministic retrieval client (`tools/e12b_client.py`, 1,036,267 tokens)
the current stack does 1M in 819.2–820.3 s (E15, E16); before the FlashInfer fix it
did 813.7 s (E12b). The gap to 866.0 s is attributed to the different client and
prompt [inferred, E16]; the prompt-dependent cost has not been isolated.

Typical time to first token: about 2 s at 4K, 30 s at 64K, 2.3 min at 256K and
14.4 min at 1M.

The decode target from the project brief, 29–33 tok/s, is **not yet met**. See
[Known issues](#known-issues-and-limitations).

What produced the gains (details in [docs/patches.md](docs/patches.md)):

| Change | Main effect |
|---|---|
| Async scheduling (`--async-scheduling`) | Decode +7.6 % at 4K; removes the host gap between steps |
| Prefill chunk 8192 (was 4096) | 64K prefill +55 %, 1M prefill +4.9 % |
| Split Lightning Indexer across TP ranks | 1M TTFT −19.5 % on an identical prompt |
| Chunked compressor GEMV | Decode +1.5–3.3 % at 4K; task accuracy unchanged (R01.1) |
| FlashInfer #4732 backport | Removes a named-barrier hang in the SM120/121 sparse MLA prefill kernel |
| `VLLM_HOST_IP` pinned to the CX7 link | vLLM control plane no longer crosses WiFi; fixes post-idle crashes |
| Fixed KV cache size (`--kv-cache-memory-bytes`) | Same KV capacity on every boot (1.04 × 1M tokens) |

---

## Quick start

All commands run on the **leader** (`spark-61dd`) as the service account. The worker
(`gx10-6d0c`) is driven over SSH; nothing needs to be started there by hand.

### Prerequisites

- Both nodes reachable over the CX7 link (`192.168.200.10` ↔ `192.168.200.11`) and
  passwordless SSH from the leader to the worker (`gx10`).
- The image `local/vllm-openai:46638857-fi0618` and the model weights present on
  both nodes.
- `/usr/local/sbin/gb10-reclaim` (drop caches + compact memory) on both nodes, runnable
  with passwordless `sudo -n`. Install [`setup/gb10-reclaim`](setup/gb10-reclaim) there
  (`sudo install -m 755 setup/gb10-reclaim /usr/local/sbin/`); [`setup/node-prep.sh`](setup/node-prep.sh)
  adds the sudoers rule.
- No other GPU workload on either node. The model uses almost all of the memory.

### Start

```bash
/srv/dsv4/run/start-dsv4-default.sh
```

The launcher is **attended**. It prints the settings it will use, for example

```
DSV4 T2: prefill chunk 8192, KV cache 11274289152 bytes, indexer split (default), diagnostics 1
```

and then waits, without a prompt, for the exact phrase `START DSV4 T2`. If the
preflight finds other DSV4/vLLM containers, processes or GPU users on either node, it
lists them and asks `y`/`n`; `y` stops those containers and signals those processes
(SIGTERM, then SIGKILL after 10 s), which destroys their work. A start takes about
5 minutes to reach serving.

Settings can be changed per start with environment variables
(see [Configuration](#configuration)):

```bash
DSV4_IDX_MODE=rowshard /srv/dsv4/run/start-dsv4-default.sh
```

### Check

```bash
curl -s localhost:8888/health -o /dev/null -w '%{http_code}\n'   # 200 when serving
/srv/dsv4/run/launch-dsv4.sh status                               # containers, GPUs, key boot lines
```

### Stop

```bash
/srv/dsv4/run/launch-dsv4.sh down
sudo -n /usr/local/sbin/gb10-reclaim                  # leader
ssh gx10 sudo -n /usr/local/sbin/gb10-reclaim         # worker
```

`down` stops peerwatch, then both containers through the coordinated abort, then
the run's guards and observers; leaderwatch exits by itself once the worker container
has stopped. The reclaim returns the GPU's unified memory to the page allocator.

The abort **latches an alert on both nodes** (reason `launch-dsv4 down`). Clear both
before the next start, or the start is refused with `sticky ALERT exists`:

```bash
python3 /srv/dsv4/run/dsv4-t2-guard.py clear --confirm "CLEAR T2 ALERTS"
ssh gx10 'python3 ~/dsv4-t2-guard.py clear --confirm "CLEAR T2 ALERTS"'
```

For alerts, recovery and day-to-day operation see
[docs/operations.md](docs/operations.md).

---

## Using the API

The server speaks the OpenAI Chat Completions API. No API key is required.

```bash
curl http://spark-61dd:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "deepseek-v4-flash-0731",
        "messages": [{"role": "user", "content": "Hello"}]
      }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://spark-61dd:8888/v1", api_key="unused")
reply = client.chat.completions.create(
    model="deepseek-v4-flash-0731",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"thinking": False}},  # thinking is on by default
)
print(reply.choices[0].message.content)
```

Notes:

- One request runs at a time; a second request queues until the first finishes.
- Prefix caching is on, so repeating a long prompt prefix skips its prefill.
- The host name resolves over Tailscale; the leader's Tailscale IP works the same way.

---

## Configuration

Environment variables read by the launch chain (`start-dsv4-default.sh` →
`launch-dsv4.sh` → `node-launch.sh`). Model settings are passed to both ranks; the
watcher, helper and gate settings apply on the leader. Defaults are the measured,
adopted configuration.

| Variable | Default | Values | Effect |
|---|---|---|---|
| `DSV4_MAX_BATCH` | `8192` | `4096`, `8192`, `12288`, `16384` | Prefill chunk (`--max-num-batched-tokens`). 12288/16384 need `DSV4_KV_CACHE_BYTES` |
| `DSV4_KV_CACHE_BYTES` | derived from chunk | bytes | KV cache size. 8192 → `11274289152` (1,094,398 tokens) |
| `DSV4_IDX_MODE` | `split` | `split`, `rowshard`, `off` | Lightning Indexer prefill: our key split, upstream row sharding (E16), or stock |
| `DSV4_ROWSHARD_MIN_ROWS` | `8192` | integer | `indexer_tp_shard_min_rows` for `rowshard` mode |
| `DSV4_PATCH` | `1` | `0`, `1` | Chunked compressor GEMV patch |
| `DSV4_FI4732` | `1` | `0`, `1` | FlashInfer #4732 sparse-MLA module |
| `DSV4_ASYNC` | on when DSpark is off | `0`, `1` | `--async-scheduling` |
| `DSV4_DIAG` | `1` | `0`, `1` | NCCL/Torch diagnostic logging (`NCCL_DEBUG=INFO`, desync debug) |
| `DSV4_PREFIX_CACHE` | `1` | `0`, `1` | `--enable-prefix-caching` |
| `DSV4_PROFILE` | `0` | `0`, `1` | Torch profiler config |
| `DSV4_PEERWATCH` | `1` | `0`, `1` | Leader-side worker watch |
| `DSV4_LEADERWATCH` | `1` | `0`, `1` | Worker-side leader watch |
| `DSV4_BOOT_COMPACT` | `1` | `0`, `1` | Boot-time compaction helper (`tools/boot-compact.sh`) on both nodes: `gb10-reclaim` when memory PSI some avg10 > 20 for 6 s during boot; at most 3 reclaims, 20 s apart. Tunables `DSV4_BOOT_COMPACT_{PSI,FOR_S,MAX_RECLAIMS,GAP_S,MAX_S}` |
| `DSV4_MIN_FREE_GIB` | `100` | GiB | Minimum `MemFree` per node before launch |
| `DSV4_READINESS_TIMEOUT` | `900` | seconds | Time allowed to reach a healthy API |

Fixed in the launcher: image `local/vllm-openai:46638857-fi0618`, TP=2, KV dtype
`fp8_ds_mla`, attention backend `FLASHINFER_MLA_SPARSE_DSV4`, MoE backend
`deep_gemm`, `--gpu-memory-utilization 0.72`, `--no-enable-flashinfer-autotune`,
max model length 1,048,576.

---

## How it works

```
                 Tailscale / LAN  :8888 (OpenAI API)
                          │
   ┌──────────────────────┴───────────┐        ┌──────────────────────────────────┐
   │ spark-61dd  (leader, TP rank 0)  │        │ gx10-6d0c  (worker, TP rank 1)   │
   │                                  │  CX7   │                                  │
   │  dsv4-leader   vLLM API + rank 0 │◄──────►│  dsv4-worker   vLLM rank 1       │
   │                                  │ RoCE   │                                  │
   │  dsv4-t2-guard.py   (host guard) │ NCCL + │  guard.py           (host guard) │
   │  allocator-observer.sh           │ vLLM   │  allocator-observer.sh           │
   │  dsv4-peerwatch.sh → watches W   │ ctrl   │  dsv4-leaderwatch.sh → watches L │
   └──────────────────────────────────┘        └──────────────────────────────────┘
        192.168.200.10 / 201.10                     192.168.200.11 / 201.11
```

**Launch chain.** `start-dsv4-default.sh` (attended, fail-closed) →
`launch-dsv4.sh up` (copies scripts and patches to the worker, starts the worker
then the leader, checks both ranks run the same indexer mode) → `node-launch.sh`
on each node (builds the `docker run` command locally, so JSON arguments are never
re-quoted by a remote shell).

**Start sequence**, each step fail-closed:

1. `START DSV4 T2` confirmation, then the admission lock.
2. Guards and allocator observers start on both nodes and must report fresh.
3. `dsv4-cluster.service` isolation check, then a preflight inventory of DSV4/vLLM
   containers, processes and GPU users on both nodes; any found are listed and
   stopped only after the operator answers `y`.
4. Guards and leader watches left by earlier runs are stopped.
5. CX7 fabric check; stopped containers from earlier runs are renamed (parked).
6. `gb10-reclaim` on both nodes, then `MemFree ≥ 100 GiB` on each.
7. The boot-compaction helper starts on both nodes.
8. Worker, then leader container start; exact container IDs are recorded and both
   ranks must report the same indexer files.
9. Wait for `/v1/models`; reclaim again after the weight load; guards switch from
   boot rules to serving rules; peerwatch and leaderwatch start.

**Networking.** NCCL and vLLM's engine-to-worker control plane both use the CX7
link (`VLLM_HOST_IP` is pinned to the CX7 address). WiFi carries only the API and
SSH, so a WiFi roam or drop does not affect inference. HCA names are discovered at
launch by [`cx7-nccl-env.sh`](cx7-nccl-env.sh), not hard-coded.

**Patches** are bind-mounted into the stock image at container start rather than
baked into a new image, so each one can be switched off per start and the image
stays identical to the one that was validated. See [docs/patches.md](docs/patches.md).

---

## Safety model

The two nodes share unified CPU/GPU memory, and running out of it can wedge a
node. Every safety component is passive by default and acts only on exact IDs.

| Component | Runs on | What it does |
|---|---|---|
| `dsv4-t2-guard.py` | both | Samples `/proc` every 2 s. Latches an `ALERT` file on `MemAvailable` < 6 GiB, a process in D-state > 10 s, or > 8 GiB of memory growth in 10 s together with memory pressure. During boot it uses relaxed rules (sustained PSI, longer D-state) so the KV allocation stall does not abort a healthy start. Never kills anything |
| `allocator-observer.sh` | both | Bounded, read-only allocator and kernel-log recording during boot |
| `dsv4-peerwatch.sh` | leader | If the worker container stops, or 6 consecutive checks 10 s apart fail to reach the worker (about 1–2.5 min), runs a coordinated abort |
| `dsv4-leaderwatch.sh` | worker | If the leader's `/health` fails 18 × 10 s, latches the worker alert and stops the worker container |
| `dsv4-abort.sh` | leader | Coordinated abort: latches alerts on both nodes, then stops only containers whose name, role and run labels match, by exact ID. Never removes containers; never stops a container it cannot identify |
| `tools/boot-compact.sh` | both | During boot only: runs `gb10-reclaim` when memory pressure builds, before the guard's boot rule can fire. Exits on serving, alert or container stop |
| `tools/run-guarded.sh` | leader | Benchmark client watchdog: kills the client (vLLM then aborts the request) if `MemAvailable` < 6 GiB or PSI > 40 for 30 s on either node |

A latched alert blocks the next start until a person clears it; see
[docs/operations.md](docs/operations.md#alerts).

---

## Repository layout

```
.
├── start-dsv4-default.sh      Attended entry point (start)
├── launch-dsv4.sh             Two-node orchestration: up | down | status | logs | wlogs
├── node-launch.sh             Per-node docker run; patch and indexer-mode selection
├── cx7-nccl-env.sh            NCCL/UCX environment for the CX7 fabric (discovered HCAs)
├── dsv4-t2-guard.py           Host guard (passive, fail-closed)
├── allocator-observer.sh      Boot-time allocator/kernel observer
├── dsv4-peerwatch.sh          Leader-side worker watch
├── dsv4-leaderwatch.sh        Worker-side leader watch
├── dsv4-abort.sh              Coordinated exact-ID abort
├── patches/                   Bind-mounted vLLM/FlashInfer patches (see docs/patches.md)
│   ├── dsv4-attention.py                      chunked compressor GEMV
│   ├── dsv4-sparse-attn-indexer.py            split indexer (default)
│   ├── dsv4-rowshard-*.py                     upstream row sharding port (E16)
│   ├── dsv4-fi4732-sparse_mla_sm120.so        rebuilt FlashInfer module
│   ├── *.orig.py                              unmodified upstream files for diffing
│   ├── fi4732/                                FlashInfer #4732 backport sources and tests
│   └── e16-upstream-rowshard/                 vLLM #54951 + #61647 port, tests, bundle
├── tools/                     Benchmarks, soak, monitoring and the boot-compaction helper
├── setup/                     One-time root setup per node (sudoers rule, worker WiFi keepalive)
├── reports/                   Optimisation log and final report (copies; see below)
├── docs/                      Operations and patch documentation
├── LICENSE, NOTICE            Apache-2.0
└── data/                      Benchmark and soak output (untracked; index in data/README.md)
```

`reports/` holds copies of the canonical reports in `~/reports/`.
`tools/commit-snapshot.sh "message"` refreshes the copies and commits.

---

## Benchmarking

All benchmark tools are read-only towards the server and run one request at a time.

```bash
tools/final-bench.sh [label]          # 4K ×3, 64K ×2, 256K, 1M; ~30 min → data/FINAL/<ts>/
tools/run-guarded.sh 65536 data/r64k  # one ctxbench request under the memory watchdog
tools/boot-4k.sh <label>              # per-boot 4K check (warm-up + 3) → data/RESTART/<ts>-<label>/
tools/e16-ab.sh                       # indexer-mode A/B (split / rowshard / off) → data/E16/
tools/gemv-acc.py run patch-on 2      # GEMV patch accuracy; then: gemv-acc.py compare a.json b.json
```

Method (`tools/ctxbench.py`): a unique filler prefix per request defeats the prefix
cache; TTFT is the prefill time; decode rate is `(completion_tokens − 1) /
(t_last − t_first)` over 256 output tokens with `ignore_eos`, sampled at temperature
0.6. Short-output clients (`e12b_client.py`, 64 tokens, greedy) are used for
retrieval and exactness checks; their decode rates are **not** comparable with
`ctxbench.py`. Cross-boot differences below about 2 % are within the boot-to-boot
spread and should not be attributed to a change.

---

## Known issues and limitations

- **Decode is below target**: 27.2 tok/s at 4K against 29–33 tok/s. Profiling puts
  weight-streaming GEMM/GEMV at ~72 % of the step and NCCL at 5–8 %; the roofline is
  about 43 tok/s. Without speculative decoding (DSpark, not started) the remaining
  candidates are small and unproven.
- **Boot stall at KV allocation (I02 class)**: in roughly 1 boot in 3, memory
  fragmentation stalls the KV-cache allocation and the boot guard aborts the start;
  a retry has always succeeded. Since 2026-10-26 `tools/boot-compact.sh` compacts
  memory during boot as soon as pressure builds (R01.4); it has not yet met a stall,
  so its effect is unproven. Root cause not established.
- **Boot-to-boot speed spread**: an unchanged configuration gives 26.8–27.3 tok/s
  at 4K on different boots (1.8 %); within a boot the spread is ≤ 0.2 tok/s.
- **Leader runs hotter** than the worker (+8 °C, SW thermal slowdown in 9 % of busy
  samples). Airflow has not been checked physically.
- **WiFi**: the leader's BSSID pin does not stop roaming between the two routers.
  Serving is unaffected (CX7), but SSH/Tailscale access can drop. A wired
  management network is the recommended fix.
- **Cross-boot nondeterminism**: greedy outputs are identical within a boot but can
  differ between boots with an unchanged configuration (E12b).
- **Single request**: the configuration is sized for one 1M request; concurrency
  above 1 is not tested.

---

## Upstream status

| Item | Status |
|---|---|
| FlashInfer #4732 | Merged upstream; backported here because the image predates it |
| Split indexer | Not proposed: vLLM #54951 + #61647 (indexer TP row sharding) address the same cost. Ported here as `DSV4_IDX_MODE=rowshard` with `indexer_tp_shard_min_rows` 8192. Serving A/B (E16): correct at 1M, no speed difference from our split (820.3 vs 819.2 s). Split stays the default; GB10 data can go to the upstream PR |
| Chunked compressor GEMV | Candidate: +1.5–3.3 % decode with unchanged accuracy (R01.1). Needs an SM12x gate before a proposal |

## Reusing this work

Everything here is Apache-2.0 (see [Licence](#licence)), so the patches and scripts may
be used, modified and redistributed, including commercially. In practice they are tied
to the setup they were measured on:

- **vLLM version.** Each patch is a whole replacement file, bind-mounted over the
  matching file in `local/vllm-openai:46638857-fi0618` (vLLM commit `46638857fdbb`).
  On another vLLM version they will likely fail or misbehave unless ported; diff
  against the `*.orig.py` files to see the actual change.
- **Model.** All three patches touch DeepSeek-V4 code paths (compressor, Lightning
  Indexer, sparse MLA).
- **Hardware.** Measured on GB10 (DGX Spark, SM121). The FlashInfer `.so` is compiled
  for SM12x and this image's FlashInfer; it will not load elsewhere.

| Patch | Who can use it |
|---|---|
| FlashInfer #4732 backport | Anyone on SM120/SM121 (Blackwell desktop or GB10) with a FlashInfer older than the #4732 merge: it fixes a real prefill hang. With a newer FlashInfer it is not needed |
| Split indexer | DeepSeek-V4 at long context with TP ≥ 2, where the replicated indexer was about 46 % of 1M prefill [inferred]. On newer vLLM, prefer the upstream row sharding (#54951 + #61647), which measured the same speed (E16) |
| Chunked compressor GEMV | GB10 and possibly similar parts; +1.5–3.3 % decode. On other GPUs cuBLAS may already be efficient and the gain may be zero. Needs an architecture gate before going upstream |

**Running the whole stack on another pair of DGX Sparks.** The repository works as a
recipe: launcher, safety guards, patches and benchmarks. Adapt:

- hostnames and CX7 addresses (`spark-61dd`, `gx10-6d0c`, the SSH alias `gx10`,
  `192.168.200.x`/`201.x`) in the top-level scripts, `cx7-nccl-env.sh`, `tools/` and
  `setup/` (`grep -rn -E "gx10|spark-61dd|192\.168\.20[01]"` lists them);
- the Hugging Face cache path (`DSV4_HF_CACHE`, used by `node-launch.sh`);
- the one-time node setup in `setup/` (`gb10-reclaim`, sudoers rule);
- the image: build or obtain vLLM `46638857fdbb` with FlashInfer `0.6.18.dev20260819`,
  or port the patches to your version.

Then follow [Quick start](#quick-start) and check results against
[Results](#results) with `tools/final-bench.sh`.

---

## Further documentation

- [docs/operations.md](docs/operations.md): runbook for starting, stopping, alerts, recovery and monitoring
- [docs/patches.md](docs/patches.md): what each patch changes, why, and how it was validated
- [reports/DSV4 final report.md](reports/DSV4%20final%20report.md): goals, results, incidents and open items (v1.0, 2026-10-26)
- [reports/DSV4 optimisation log.md](reports/DSV4%20optimisation%20log.md): the full, evidence-tagged experiment log

Evidence in the reports is tagged **[measured]**, **[inferred]** or **[hypothesis]**;
corrections are appended rather than rewritten.

## Acknowledgements

Built on [vLLM](https://github.com/vllm-project/vllm) and
[FlashInfer](https://github.com/flashinfer-ai/flashinfer). The baseline launcher is
a modified copy of the operator's configuration in `/srv/dsv4/live/` (not part of
this repository). Files under `patches/`
derived from vLLM and FlashInfer keep their upstream licences (Apache-2.0).

## Licence

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
