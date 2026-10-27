# DSV4 optimisation log

Engineering log for serving DeepSeek-V4-Flash-0731 (NVFP4) on two GB10 nodes. This is a running record of what we changed, why, how we measured it, and what happened. It is append-only. Corrections are added as new dated notes and never silently edited away.

- **Owner:** the user account. Entries are written by Claude from measurements taken in the session.
- **Background research:** `reports/DSV4 on two GB10 optimisation.md`, which holds the hypotheses and estimates; this log holds what was measured.
- **Started:** 2026-10-12.

## How to read this log

| Field | Meaning |
|---|---|
| Status | `RUNNING`, `SUCCESS`, `FAILED`, `PARTIAL`, `INCONCLUSIVE`, `SUPERSEDED` |
| Evidence tags | **[measured]** comes from a log, benchmark or counter. **[inferred]** follows from measured facts but was not directly observed. **[hypothesis]** is not tested yet. |
| Times | Local time is UTC-7. Container logs use UTC, and each entry says which one it uses. |

Each experiment records the question, its first-principles reasoning, the change, the method, the result, a verdict with analysis, and the next step. Decisions with lasting effect also go in the decision record at the end.

## Goal and success criteria

| # | Criterion | Target | Status |
|---|---|---|---|
| G1 | One request at 1M context on 2×GB10, TP=2 | completes with no OOM, wedge or crash | **Met** 2026-10-12 (E06) |
| G2 | Boot is repeatable without manual recovery | clean boot with guards on | **Met** (E05) |
| G3 | Decode speed at long context | near the bandwidth ceiling; the report estimates 29–33 tok/s with DSpark off | **Open**: 26.93 tok/s at 4K (E07.8); 26.29 at 64K and 24.76 at 1M (E07.6, before the patch); was 23.56 / 21.08 (E07.0 / E06) |
| G4 | Prefill speed | improve 1M TTFT (currently 18.5 min) | **Improving**: 1M TTFT 13.6 min (1,274 tok/s) with the split indexer (E12b, −19.5 % against the same-client baseline of 1,010.3 s); 16.8 min / 64K 29.8 s with 8192 chunks (E10); was 17.7 min / 46.1 s (E07.6) |
| G5 | Latency after idle | first request after idle ≈ warm latency | **Met** (E09): 0.16–0.27 s after 30–60 min idle. A live WiFi roam has not been observed yet |

## Scoreboard: best known numbers

Config C1 (see below), DSpark off. Measured 2026-10-12, single request, 256 output tokens, `ignore_eos`, thinking off. Tool: `scratchpad/ctxbench.py`.

| Context | Prompt tokens | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|---|
| 4K | 3,827 | 1.8 | 2,163 | 22.79 |
| 32K | 32,711 | 16.2 | 2,018 | 22.25 |
| 128K (warm) | 131,627 | 69.9 | 1,884 | 21.74 |
| 512K | 526,689 | 396.1 | 1,330 | 21.62 |
| 1M | 1,045,933 | 1,113.3 | 939 | 21.08 |

Memory headroom during the 512K and 1M runs [measured]: leader MemAvailable about 8–9 GiB, worker about 16 GiB, memory PSI 0. The guard's floor is 6 GiB.

## Environment (fixed unless an entry says otherwise)

| Item | Value |
|---|---|
| Nodes | spark-61dd (leader, 192.168.200.10) and gx10-6d0c (worker, 192.168.200.11). GB10, 121.7 GiB unified memory each |
| OS / driver | kernel 6.17.0-1032-nvidia, driver 610.57.04, CUDA UMD 13.3 |
| Interconnect | ConnectX-7 RoCE, two devices (rocep1s0f1, roceP2p1s0f1). NCCL over IB verbs, GDR 0 |
| Image | `local/vllm-openai:46638857-fi0618`: vLLM 0.26.1rc1.dev1219+g46638857f, FlashInfer 0.6.18.dev20260819 |
| Model | deepseek-v4-flash-0731, NVFP4 weights, 73.85 GiB per rank after load [measured] |

### Config C1: current known-good

- Launch scripts are in `/srv/dsv4/run/`, a modified copy of the operator's `/srv/dsv4/live/`. The launcher is `start-dsv4-default.sh`.
- Parallelism and kernels: TP=2 across both nodes, KV dtype `fp8_ds_mla`, attention `FLASHINFER_MLA_SPARSE_DSV4`, MoE `deep_gemm`, DSpark off.
- KV cache: `--kv-cache-memory-bytes 9663676416` (9 GiB). This gives 1,747,385 KV tokens, or 1.67× of 1M [measured].
- Flags: `--enable-prefix-caching --no-async-scheduling --enable-chunked-prefill --no-enable-flashinfer-autotune`.
- Environment variables: `UCX_MEM_MMAP_HOOK_MODE=none`, `UCX_RCACHE_MAX_UNRELEASED=1024`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
- Host settings: `vm.compaction_proactiveness=20` (kernel default) and `nvidia-smi -lgc 0,2100` on both nodes.
- Diagnostics left on: `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET,COLL`, which writes one log line per collective, plus `TORCH_NCCL_DESYNC_DEBUG=1`. E07 asks whether these cost decode speed.
- Runtime compile cache: `/srv/dsv4/runtime-cache/46638857-fi0618-sm121`. It started empty on 2026-10-12.

---

## Experiment log

### E01: Run the operator's launcher from a second account (2026-10-12) · SUCCESS

- **Question:** Can the attended T2 launch framework run from the user account without changing the operator's live files?
- **Change:** Copied the scripts to `/srv/dsv4/run/`. HERE, the guard, observer, cache and diagnostics paths now point at that copy or at `$HOME`. scp targets `"$PEER:"`. The worker IPs were added to the `gx10` ssh Host entry. Scripts were made executable. The weights mount stays the operator's HF cache, mounted by the docker daemon.
- **Result:** The launcher reached the admission, guard and launch stages [measured].
- **Failures fixed along the way** [measured]:
  1. Raw-IP ssh to the worker failed with a publickey error.
  2. The observer script was not executable.
  3. "another attended T2 admission is active": a leftover guard process had inherited the flock file descriptor. The guard now starts with `{LOCK_FD}>&-`, and `stop_guards()` is called on abort.
  4. The container name clashed with an exited container. `park_stale()` now renames exited `dsv4-leader`/`dsv4-worker` containers to `*-<run>-parked`.

### E02: Guard alert at weight reservation, caused by page cache (2026-10-12) · SUCCESS

- **Observation** [measured]: The guard aborted the boot with `residual_growth>8GiB/10s+psi_stall` while each rank reserved about 78 GiB for weights. About 59 GiB of page cache was resident when the launch began.
- **First principles:** On GB10, GPU allocations come out of the same physical memory as page cache. A large allocation while free memory is full of cache forces direct reclaim and compaction inside the allocation path, which shows up as PSI and D-state.
- **Change:** Before launch, the script runs `sudo -n gb10-reclaim` (drop_caches plus compact_memory) on both nodes. It then requires MemFree ≥ `DSV4_MIN_FREE_GIB=100` on both, and reclaims again after "Loading weights took".
- **Result:** The weight-reservation alert did not recur on later boots [measured]. The weight load itself took 150.3 s [measured, 2026-10-12 18:12 UTC boot].

### E03: `vm.compaction_proactiveness=0` (2026-10-12) · FAILED, no benefit shown

- **Reason for trying it:** The research report recommended 0, so that background compaction would not compete with the model.
- **Result** [measured]:
  - With 0 set on both nodes, the boot still aborted at KV allocation. The 18:13 UTC alert was `D-state>10s` on the leader with PSI 33.3 and MemAvailable 21 GiB. The D-state processes included the operator's `nvtop` (pid 394774).
  - The setting was restored to 20 at 18:17 UTC.
  - The next boot, at 19:31 UTC, still alerted at KV allocation: `residual_growth+psi_stall`, leader PSI 7.7, worker PSI 8.3.
- **Verdict:** Neither value prevented the KV-allocation stall, so 0 had no benefit. 20 is the kernel default, and we keep it because there is no evidence for changing it.
- **Correction (2026-10-12, local evening):** An earlier chat summary said 0 "made the stalls worse". The measurements above do not support that claim. They show only that 0 did not help.
- **Pausing `nvtop`:** the operator's `nvtop` was paused with SIGSTOP on both nodes because it appeared in D-state. The stall continued after the pause [measured]. The pause is still in place, and `sudo pkill -CONT nvtop` resumes it.

### E04: KV cache size pinned instead of profiled (2026-10-12) · SUCCESS

- **First principles:** Profiling sizes KV from whatever memory is free at that moment, so the size varies between boots. Earlier boots gave 3.08×–3.29× of 1M, up to 3,450,683 tokens [measured, 03:46 UTC]. That uses memory we don't need for single-request 1M and removes headroom.
- **Change:** `--kv-cache-memory-bytes 9663676416` (9 GiB).
- **Result:** 1,747,385 tokens, which is 1.67× of 1M and the same on every boot [measured]. MemAvailable during the 1M request stayed at 8–9 GiB on the leader and 16 GiB on the worker [measured].

### E05: Boot-phase guard rules (2026-10-12) · SUCCESS. User approved: "Loosen guard at boot"

- **Observation** [measured]: Every boot shows a brief direct-reclaim stall during KV allocation, and it clears by itself. The strict serving-mode rules (residual growth over 8 GiB in 10 s with PSI above 1.0, or D-state over 10 s) aborted the boot every time.
- **Change** in `dsv4-t2-guard.py`. The strict original is kept as `.strict`.
  - Boot mode lasts until the launcher touches `runs/<id>/<role>/serving` after API readiness.
  - In boot mode, the guard alerts only when PSI some avg10 stays above 40 for 30 s, or when a process is in D-state for more than 60 s.
  - MemAvailable < 6 GiB alerts in both modes. Serving mode keeps the strict rules.
  - The rules were unit-tested against the readings that had alerted earlier.
- **Result** [measured]:
  - Run `dsv4-t2-20261012T195705Z-779454` was admitted at 19:57:05 UTC, and the API was ready at 20:02:46 UTC (5 min 41 s, including attended steps).
  - The KV-allocation stall peaked at leader PSI ≈ 9 and worker PSI ≈ 29 and cleared in about 40 s.
- **Consequence:** The server is protected less during boot. Serving-mode protection is unchanged.

### E06: Context sweep to 1M (2026-10-12) · SUCCESS. G1 met

- **Method:**
  - `ctxbench.py` builds a prompt of random filler words with a unique UUID prefix, which defeats the prefix cache. The filler length is calibrated with `/tokenize`.
  - Requests are streamed, generate 256 tokens with `ignore_eos`, and have thinking off.
  - Runs at 512K and 1M were wrapped in `run-guarded.sh`. It kills the client if MemAvailable drops below 6 GiB or PSI stays above 40 for 10 samples on either node.
- **Result:** See the scoreboard.
- **First-request effects** [measured]:
  - The first 4K request had TTFT 2.8 s; the repeat had 1.8 s.
  - The first 128K request had TTFT 164.9 s; the repeat had 69.9 s.
  - During the first 128K request, the leader log shows a TileLang JIT compile (20:07:04 UTC) and a Triton JIT warning (`BuildPrefillChunkMetadataKernel`, 20:10:59 UTC).
- **Analysis:**
  - Decode is nearly flat: −7.5% from 4K to 1M. That suggests per-token cost is dominated by weights and communication rather than by attention over the KV cache [inferred].
  - Prefill falls from about 2,000 tok/s at 4K–128K to 939 tok/s at 1M. Attention cost grows with context, which is consistent with this [inferred].
  - vLLM's logged "Avg prompt throughput" (13k–104k tok/s) is a logging artifact. Use client-side TTFT instead.

### E06a: Serving-mode guard alert during the first 128K request · INCONCLUSIVE, root cause not proven

- **Observation** [measured]: At 20:07 UTC the leader guard latched `D-state>10s` with PSI 66.4 and MemAvailable 11 GiB. The D-state process was pid 792621.
- **What the process was:**
  - `observer.err` records `timeout ... nvidia-smi --query-compute-apps` being killed. The wrapper's pid was 792620, adjacent to 792621.
  - So the D-state process was very likely the observer's `nvidia-smi` [inferred], not a vLLM process.
  - The observer stopped recording at that point.
- **Interpretation:** Memory pressure rose during the first long prefill and coincided with JIT compiles. `nvidia-smi` blocked behind it. No request failed, and the 128K repeat was clean [measured]. Which of JIT, prefill working set or `nvidia-smi` caused the PSI spike is not determined.
- **Current state:** The alert is still latched. It is passive, so the server has no automatic protection right now.

### E07: Decode gap, 23.6 vs 29–33 tok/s · RUNNING

- **Fact:** Decode is about 25–30% below the bandwidth-based estimate in the research report: about 25 ms of memory time at 205 GB/s, giving 30–35 ms per step, or 29–33 tok/s.
- **First-principles model:** Time per token = weight and KV reads ÷ bandwidth + per-layer cross-node collectives (latency-bound at batch 1) + host gaps between steps + kernel inefficiency (kernels not reaching bandwidth).

**E07.0: Baseline on C1 + E09** (2026-10-13, run 161723Z) [measured]

- **Method:** `ctxbench.py 4096 512`, run three times back-to-back on a warm server. Data is in `/srv/dsv4/run/data/E07/baseline.jsonl`.

| Run | Decode (tok/s) | TTFT (s) |
|---|---|---|
| 1 | 23.56 | 1.8 |
| 2 | 23.55 | 1.7 |
| 3 | 23.64 | 1.7 |
| **Median** | **23.56 (42.4 ms per token)** | |

- **Repeatability:** the spread is ±0.2%, so changes of about 1% are detectable.
- **Longer outputs:** a 1,200-token decode at 1K context gave 23.71 tok/s. The 10-12 figure of 22.79 used 256 tokens, which includes more ramp-up.
- **One-off first TTFT:** one 752-token prompt had a 7.8 s TTFT because of first-use JIT compiles: Triton `_gumbel_sample_kernel` and TileLang `mhc_pre_big_fuse_broadcast_with_norm_tilelang` [measured, log warnings]. This adds to E06a's list of shapes that warmup does not cover.

**E07.1: Is NCCL debug logging the cost? NO (refuted without a restart)** [measured]

- The NCCL debug log grew by 108 KiB, about one line, per generated token over 512 tokens.
- The one logged collective per step is `AllGather count 64640`, the vocabulary-sharded logits gather (129,280 ÷ 2).
- Every other collective runs inside CUDA graphs, where NCCL does not log.
- One `fprintf` per 42 ms step costs microseconds, so the expected effect is under 0.1%. This candidate is dropped.

**E07.2: GPU activity during decode** [measured, nvidia-smi at 200 ms intervals, 1,200-token decode]

| Node | Mean utilisation | Mean power |
|---|---|---|
| Leader | 90.7% | 20.7 W |
| Worker | 86.1% | 18.7 W |

- **Interpretation** [inferred]:
  - "Utilisation" counts any kernel being resident, including an NCCL kernel spin-waiting. During the E08 stall, the GPU read 96% at 12 W while doing no useful work.
  - At about 20 W, much of the "busy" time is probably waiting, either in collectives or on the peer, rather than streaming memory. This is consistent with communication and synchronisation latency being a large share of the 42.4 ms.
  - It is not proof. A per-kernel timeline is needed.
- **Configuration found running:**
  - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, auto-enabled.
  - Graphs captured as PIECEWISE plus FULL at size 1.
  - `--no-async-scheduling`.
  - `NCCL_DEBUG=INFO` and `TORCH_NCCL_ENABLE_TIMING=1`.

**E07.3: Next step, profile one restart instead of guessing** · PREPARED

- **Change:**
  - `DSV4_PROFILE=1` adds `--profiler-config` (torch profiler, no stacks, frontend ignored, skip 10 iterations, record 20).
  - The profiler is inactive until `POST /start_profile`, so serving behaviour is unchanged.
  - Traces go to `/srv/dsv4/diagnostics/<host>/prof/` on each node.
  - The config was validated against the image's `ProfilerConfig`. Backups are `*.bak-20261013-prof`.
- **Plan:**
  1. Re-run the baseline to confirm the profiler config costs nothing.
  2. Profile a 4K decode.
  3. Break the 42.4 ms step into GPU compute, NCCL time, and idle gaps between kernels or steps, on both ranks.
  4. Pick the next change from the largest component.
- **Candidate changes, ranked by what the profile shows:**
  - **Communication** [hypothesis]: `NCCL_PROTO=LL` or `LL128`, or `NCCL_ALGO` for small all-reduces.
  - **Host gaps** [hypothesis]: async scheduling, or FULL graphs without breakable graphs. The operator measured breakable=0 as a prefill regression, so it would be checked for both decode and prefill.
  - **Kernels** [hypothesis]: MoE backend or kernel choice.

### E08: Slow first request after idle · ROOT CAUSE FOUND. The server crashed during the test; a fix is ready but not verified (E09)

- **First symptom** [measured, 2026-10-13 00:00–00:02 UTC]:
  - After about 3.5 h idle, a 2-token reply took more than 100 s. The first curl timed out at 60 s, and the retry took 46 s.
  - The EngineCore logged `shm_broadcast.py:801 No available shared memory broadcast block found` every 10 s.
  - The next two 100-token requests took 6.0 s and 4.3 s.
- **What the log message means:** It comes from `acquire_read`, so the engine was waiting for the workers' result for that step [inferred from source]. NCCL logs show no warnings.
- **Method:**
  - A 1 Hz sampler ran on each node: memory, PSI, faults, reclaim and compaction, GPU util/clock/power, TP-worker state, and RoCE bytes.
  - A probe sent one 4-token request after 15, 30, 60 and 120 min of idle. The warm baseline was the same request back-to-back.

**Probe results** [measured]:

| Idle | Result | Warm baseline |
|---|---|---|
| 15 min | 3.60 s, HTTP 200 | 0.25 s and 0.42 s |
| ~29.5 min (baseline requests at ~17:29:30 reset the clock) | **HTTP 500 after 300.0 s, engine died** | n/a |
| 60 min and 120 min | not run, because the server was down | n/a |

**What the samplers showed during the 30-minute probe** (17:59:03–18:04:03 local) [measured]:

- **Leader:**
  - GPU util 96% at about 12–13 W, with the SM clock at 2093 MHz. Low power at high util looks like a spin-wait, not compute [inferred].
  - The TP0 main thread and one helper thread were running (R).
  - RoCE receive was 0 MB/s.
- **Worker:**
  - GPU util 0%, and every TP-worker thread was sleeping.
  - The main thread was in `poll_schedule_timeout`, which means it was waiting for its next message.
  - RoCE receive was 0 MB/s.
- **Both nodes:** PSI 0, no reclaim or compaction, and a handful of major faults (11 at most in one sample).
- **Conclusion:** The worker never received the step. The leader sat waiting for its peer [inferred].

**Where the message was stuck** [measured, `ss -tin` and socket-to-process mapping inside both containers]:

- The engine-to-worker broadcast is a ZMQ connection. The EngineCore (pid 127) listens on `10.0.0.239:44029`, and the worker connects from `10.0.0.106:48510`. **Both addresses are WiFi (`wlP9s9`) addresses, not CX7.** NCCL, Gloo and the TCPStore all use the CX7 subnets (192.168.200.x and 192.168.201.x).
- On the leader during the stall, that socket showed `notsent:1390`, `backoff:9`, `pmtu:68`, `mss:36` and `lastack` about 1,977 s earlier. It had retransmitted 93,312 bytes over its life.
- The worker had not received anything on it for about 1,977 s.

**Why the WiFi path failed** [measured, NetworkManager journal]:

- From 16:19 local onward, `wlP9s9` switched DHCP leases between two networks, 10.0.0.x and 192.168.86.x, roughly every 5 minutes. Correction (2026-10-13, INV-1): the roaming dates back to at least Aug 7, and during both stalls only the leader changed subnet. Leader leases: 16:19, 16:24, 16:45, 16:51, 16:56, 17:01, 17:33, 17:38, 17:39, 17:44, 17:50, 17:55, 17:57.
- vLLM bound the connection at boot (12:57), when both nodes were on 10.0.0.x. Whenever either node was on the other network, packets could not get through. TCP then backed off exponentially, with RTO roughly doubling up to 120 s.
- **Mechanism:**
  - 15-min probe: the connection happened to be working, apart from a short retry.
  - 3.5 h stall: about 100 s, until the next retransmit landed while both nodes were on 10.0.0.x.
  - 30-min probe: longer than `VLLM` execute-model's 300 s limit, so the EngineCore raised a fatal `TimeoutError` (`multiproc_executor.py:433`) and the leader container exited [measured: exit 0, EngineCore traceback].

**Root cause:**
- `vllm.utils.network_utils.get_ip()` returns `VLLM_HOST_IP` if it is set. Otherwise it returns the source address of the default route.
- Our launcher did not set `VLLM_HOST_IP`, so the engine-to-worker control plane (`multiproc_executor.py:152`, `shm_broadcast.py:524`) used the WiFi interface, which is the default route.
- WiFi roaming then made that link drop out intermittently.

**Ruled out for this symptom** [measured]: memory pressure or paging, GPU clock drop, NCCL/RoCE problems, CPU starvation.

**Rule:** Every inter-node path must be pinned to CX7. A single unpinned path follows the default route.

**Evidence:** `/srv/dsv4/run/data/E08/`. It holds `idle-leader.log`, `idle-worker.log` and `dsv4-leader-crash.log`.

### INV-1: Why the deployment keeps failing (2026-10-13) · COMPLETE

- **Scope:** Three read-only investigations covered the network and WiFi, every launch attempt from 10-10 to 10-13, and the operator's history from August to October. The full write-ups are in `/srv/dsv4/run/data/investigation/` (`network.md`, `timeline.md`, `history.md`). Key claims were spot-checked independently, as noted below.

**Finding 1: one recurring root cause behind most serving hangs.**

- vLLM's engine-to-worker control queues use `get_ip()`. That is the default-route address, which is WiFi `wlP9s9`. No launcher, ours or the operator's, ever set `VLLM_HOST_IP`.
  - [measured, spot-checked] 19 saved leader logs from Aug 15–31 show `mq_connect_ip=10.0.0.239`.
  - [measured, `network.md`] The leader's broadcast queue, the worker's response queue and the cross-node broadcaster are all bound this way.
- **Why the WiFi drops out** [measured, `network.md`]:
  - Two routers broadcast the same SSID, "<home WiFi SSID>": an ISP gateway (10.0.0.x) and a mesh router behind it (192.168.86.x). Their signal levels are nearly equal (-60 vs -61 dBm).
  - wpa_supplicant roams between them within one profile, and each roam is followed by a new DHCP lease on the other subnet. There are no disconnect or beacon-loss events.
  - The journal shows this back to Aug 7 on the leader and Aug 26 on the worker, with 236 switches on Oct 9 alone. A stay on the Google network lasts about 318 s, which is longer than vLLM's 300 s execute-model timeout.
  - The two nodes roam independently. During both E08 stalls, only the leader changed subnet; the worker stayed on 10.0.0.106. This corrects E08's "both nodes".
- **The operator's earlier hangs on another model S10 and S12 (Oct 2–3)** [measured, spot-checked]:
  - The leader roamed to 192.168.86.215 at 22:08:45 and 22:40:30 local. The containers exited at 22:13:47 and 22:45:33, +302 s each time (300 s timeout plus teardown), and both before WiFi returned.
  - S12's recorded symptoms match E08 exactly: leader GPU spinning, worker GPU at 0%, workers waiting in `shm_broadcast.dequeue`.
  - The earlier "TP rank collective mismatch / prefix cache" explanation is not supported. **Confidence: high.**
- **The operator's post-reboot stalls, Aug 21/25/26** [inferred, `history.md`]: Autostart launched while the leader was on 192.168.86.x. The "ping check and settle delay" fix worked because WiFi had returned by then. **Confidence: medium.** Aug 25 also had a real RoCE GID problem.
- **Still open** [`history.md`]: the DSpark first-request hang, the 660K `sample_tokens` timeout, and the Aug 15 and 151K `execute_model` timeouts.
  - Each shows the same signature: rank 0 at 96%, rank 1 idle, a 300 s timeout.
  - But the leader's WiFi was stable at those times, and the worker's journal starts only on Aug 26, so a worker-side roam cannot be checked. The 660K event also had severe memory pressure.
  - The claim that DSpark causes rank divergence rests on one run in each mode and is not proven.
- **Fix:** E09 (`VLLM_HOST_IP` set to the CX-7 IP). It matches vLLM's multi-node guidance ([troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/)). Raising `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` would only delay the failure.

**Finding 2: failures that are not WiFi.**

- These are memory, driver or kernel problems: driver memory retained after exit (S1, S14); compaction and reclaim stalls (E02, E05); `ibv_reg_mr` on kernel 7.0; the 09-23 D-state hangs and freezes; the Aug 13 OOM kill.
- **96% ownerless GPU:** this is the state left after one rank exits, with the peer spinning in an all-reduce. Reclaim plus a small CUDA job clears it, as seen on 10-12. It is a symptom, not a GPU bug.
- **Our 10-12 launch failures** were all launcher or guard issues (E01, E02, E05) [measured, `timeline.md`]. WiFi did not change during them [measured, `network.md`]. The launcher's ssh always uses CX-7 (192.168.200.11).

**Finding 3: new issues found along the way** [measured, `timeline.md`, spot-checked where noted]:

1. **Orphaned worker** [spot-checked]: since the leader died at 01:04 UTC, `dsv4-worker` has sat at 96% GPU, about 155% CPU and about 106 GB used. Its log grows with "Broken pipe" lines (143 MB). The leader GPU also reads 96% with no process. Both 779454 guards are still running.
2. **Weekly `fstrim`** on Mondays around 00:55–01:23 PDT puts processes into D-state over 10 s. It latched guard alerts on 10-12. In serving mode this would latch a passive alert. It is a false positive to handle in the guard.
3. **Worker driver allocation failures:** `NV_ERR_NO_MEMORY` near the end of the worker's weight load, rising 1 → 3 → 7 across boots. About 1.6 GB of swap-out happened then. Only the worker has swap (16G, swappiness 60).
4. **Not yet explained:**
   - The 3.6 s probe at 00:29 while the network was stable.
   - The leader loads weights about 55 s slower than the worker in every run.
   - The leader ignores SIGTERM for the full 120 s on every abort.
   - The memory-pressure spike during the first 128K request (E06a).

### E09: Pin vLLM's control plane to CX7 with `VLLM_HOST_IP` · SUCCESS (structural); a live WiFi roam has not been observed yet

- **Change:**
  - `node-launch.sh` sets `VLLM_HOST_IP` to the IPv4 address of `$NCCL_SOCKET_IFNAME` (`enP2p1s0f1np1`). That is 192.168.201.10 on the leader and 192.168.201.11 on the worker.
  - The launch fails if the address cannot be resolved.
  - A backup of the old script is at `node-launch.sh.bak-20261012-hostip`.
- **Cleanup before relaunch** (2026-10-13):
  - Ran `launch-dsv4.sh down`, stopped the old guards and cleared the alerts.
  - Ran `gb10-reclaim` on both nodes.
  - Ran a small CUDA job on each node, which cleared the ownerless 96% GPU state to 0%.
- **Run:** `dsv4-t2-20261013T161723Z-1368174`.
  - Weights loaded in 153.3 s, and the KV cache was 1,747,385 tokens.
  - The API came up with no guard alert.
- **Pass criterion 1, no vLLM socket on WiFi: PASS** [measured]:
  - Both containers have `VLLM_HOST_IP` set: .201.10 and .201.11.
  - The boot log shows `mq_connect_ip=192.168.201.10`.
  - The EngineCore listens on and connects via 192.168.201.10.
  - A socket-to-process scan inside both containers found 0 sockets on 10.0.0.x or 192.168.86.x.
- **Pass criterion 2, idle probes: PASS** [measured]:

| Idle | Latency | HTTP |
|---|---|---|
| warm (just after boot) | 0.26 s | 200 |
| 30 min | 0.27 s | 200 |
| 60 min | 0.16 s | 200 |

  - The pre-fix results were 3.6 s at 15 min, and 300 s followed by the engine dying at 30 min (E08).
  - No `shm_broadcast` or timeout lines appeared in the leader log during the window.
- **Pass criterion 3, survives a WiFi roam: NOT YET OBSERVED.**
  - The worker re-leased 10.0.0.106 8 times in the first 30 min, but stayed on the same subnet.
  - Neither node changed subnet during the 60-min window.
  - The original trigger, the leader moving to 192.168.86.x, did not happen during the test.
  - The fix removes WiFi from vLLM's path entirely (criterion 1), so a roam should have no effect [inferred]. It still needs to be confirmed by a live event.
- **Evidence:** `/srv/dsv4/run/data/E09/`.

---

## Decision record

| ID | Date | Decision | Why | Consequence | Status |
|---|---|---|---|---|---|
| D1 | 2026-10-12 | Run from a modified copy (`/srv/dsv4/run`), never edit `/srv/dsv4/live` | The operator's live setup must stay intact | We maintain two script trees | Accepted |
| D2 | 2026-10-12 | Reclaim and gate MemFree ≥ 100 GiB before launch | E02 | Launch takes about 1 min longer | Accepted |
| D3 | 2026-10-12 | Pin KV at 9 GiB | E04 | Fixed 1.67× of 1M; no headroom for concurrency above 1 at 1M | Accepted |
| D4 | 2026-10-12 | Relax guard rules during boot only | E05 (user approved) | Less protection during boot | Accepted |
| D5 | 2026-10-12 | Keep `compaction_proactiveness=20` | E03 | none | Accepted |
| D6 | 2026-10-12 | Pin `VLLM_HOST_IP` to the CX7 address on each node | E08: WiFi roaming stalled and then killed the engine | The server no longer depends on WiFi | Accepted, checked by E09 |
| D7 | 2026-10-16 | Async scheduling on by default when DSpark is off | E07.6: +7.6% decode at 4K, +17.5% at 1M vs E06; 1M and idle probes passed | Untested with DSpark on, so it stays off there; one 16 h run of evidence so far | Accepted |
| D8 | 2026-10-20 | GPUs stay uncapped (no `-lgc 0,2100`) | The operator's reply 2026-10-20 (decision 1a), on condition that thermal throttling is watched | Thermal throttling on the leader (E13) must be monitored | Accepted |
| D9 | 2026-10-20 | Bind-mounted vLLM patches (`attention.py`, `sparse_attn_indexer.py`) may be the default | The operator's reply 2026-10-20 (decision 3a); the operator also asks that they be proposed to vLLM main, after reading vLLM's CONTRIBUTING.md | `apply-idxsplit-default.sh` may be applied | Accepted |
| D10 | 2026-10-20 | DSpark may go ahead | The operator's reply 2026-10-20 (decision 2a). The operator says the rank-divergence deadlock and large-context prefill are still open, and that the WiFi-roaming hangs are probably not the main DSpark problem; the operator suspects TP=2 and GB10 compatibility. First review open/closed vLLM PRs that tried to fix this, and why they were not merged | No window named; plan for a possible reboot of both nodes | Accepted, not started |

## Open items and housekeeping

*Updated 2026-10-21. The earlier list (run 1368174 running, observer not restarted) was out of date: that run ended long ago and the observer was fixed on 2026-10-19.*
- **Server is running:** run `dsv4-t2-20261021T005449Z-775571` (since 2026-10-20 17:59 local) with 8192 chunks, the split indexer, the GEMV patch, the FlashInfer #4732 fix (E15) and peerwatch.
- **Staged, not applied:** `scratchpad/apply-fi4732-default.sh` (makes E15 the default).
- **Upstream:** replace our split indexer with vLLM #54951 + #61647 (`indexer_tp_shard_min_rows` 8192) and A/B it; contribute GB10/TP=2 threshold data instead of our patch. GEMV patch: needs a same-boot benchmark and an accuracy eval before any upstream proposal.
- **DSpark (D10):** survey done; first one-change experiment not started; needs a window agreed with the operator.
- **Next project** (another model) stability work: queued by the operator, not started.
- **Known false alerts:** the weekly `fstrim.timer` trips the guard's D-state check; not fixed.
- **Thermal:** the leader runs hotter and slows under load (E13); the operator asked us to keep watching it (D8). No clock logging during serving yet.
- The operator's `nvtop` was paused on both nodes on 2026-10-13. Current state not re-checked.

## Artifacts

| Path | What it is |
|---|---|
| `/srv/dsv4/run/` | Launcher and guard in use. `dsv4-t2-guard.py.strict` is the original guard |
| `~/.cache/dsv4-t2/runs/<run-id>/` | Guard state, alerts and post-load reclaim log |
| `/srv/dsv4/diagnostics/<host>/nccl.*.log` | NCCL INIT, NET and COLL logs |
| `/srv/dsv4/run/tools/ctxbench.py`, `run-guarded.sh`, `st.sh` | Benchmark and watchdog |
| `/srv/dsv4/run/tools/idle-sampler.sh`, `idle-probe.sh`; data in `/srv/dsv4/run/data/E08/` | E08 instruments and data |

Tools are in `/srv/dsv4/run/tools/`, and experiment data is in `/srv/dsv4/run/data/<exp>/`.


#### E07.4 — Baseline after the Oct 13 reboot and recabling [measured] (2026-10-15)
- **Setup:** run dsv4-t2-20261015T235436Z-108343, with config C1 plus `DSV4_PROFILE=1` (profiler configured but not started). The NCCL HCAs are now f0 (rocep1s0f0, roceP2p1s0f0). The GPU clock lock is **absent**: the reboot cleared `nvidia-smi -lgc 0,2100`, and nothing re-applied it. Ready in 270 s (page cache warm).
- **Result:** `ctxbench.py 4096 512` ×3 after one warm-up gave 24.76, 24.80 and 24.08 tok/s, median **24.76 tok/s (40.4 ms/token), +5.1% vs E07.0**. Prefill rose from 2079–2205 to 2288–2301 tok/s. Data: `data/E07/baseline-f0-prof.jsonl`.
- **Clocks during decode:** leader 2483 MHz at 91% util and about 37 W; worker 2548 MHz at 89% util and about 36 W. In E07.0 both were 2093 MHz at about 20 W.
- **Analysis:** three variables changed (clock cap, NIC port, profiler config), so the gain cannot be attributed with certainty. The clock is the likely cause [inferred]: a different port on the same NIC should not change latency, and an idle profiler config costs nothing. If so, a 19–22% SM clock rise bought only 5% decode. That means roughly a quarter of the step scales with SM clock, and the rest is bound by memory, communication or synchronisation [inferred].
- **Correction to E07.2:** the "about 20 W at 90% util" reading was taken under the 2100 MHz cap, so it is weaker evidence of waiting than stated.
- **Open:** why was `-lgc 0,2100` set (stability, power, or the operator's choice)? Keep the clocks uncapped or restore the cap? An A/B test with the cap restored, everything else the same, would isolate the clock effect.


#### E07.5 — Decode step profile, 4K context [measured unless tagged] (2026-10-15)
- **Method:** with run 20261015T235436Z (uncapped clocks, f0 HCAs), `POST /start_profile`, then `ctxbench.py 4096 128`, then `POST /stop_profile`. The torch profiler captured 20 decode iterations (`execute_context_0(0)_generation_1(1)`) on both ranks. Analysis covers the 18 complete inner steps per rank. Scripts are `stepbreak.py`, `excl.py` and `gaps.py` in the scratchpad. Traces are in `/srv/dsv4/diagnostics/<host>/prof/`.
- **Step:** median 41.6 ms on both ranks (CPU annotation to annotation), 2128 GPU kernels per step. GPU busy (union of kernel intervals) is 38.7 ms on the leader and 37.2 ms on the worker. Kernel time summed across streams is 47.2 ms, so about 8.5 ms of the work runs concurrently on side streams.
- **Exclusive time per step** (time shared between concurrent kernels is split evenly):

| Class | Leader ms | % | Worker ms | % |
|---|---|---|---|---|
| FP8 dense GEMMs (DeepGEMM M=1: attention projections, shared expert) | 14.92 | 35.8 | 14.89 | 35.9 |
| MoE routed experts (DeepGEMM grouped FP4) | 8.22 | 19.7 | 8.16 | 19.7 |
| BF16 cuBLAS GEMV (compressors, router gate, indexer) | 4.13 | 9.9 | 4.13 | 9.9 |
| LM head (BF16 GEMV, grid 8080) | 2.81 | 6.7 | 2.80 | 6.7 |
| NCCL (87 AllReduce bf16 RING_LL + 1 AllGather per step) | 3.15 | 7.6 | 1.98 | 4.8 |
| Attention (sparse MLA decode, top-k, indexer) | 1.89 | 4.6 | 1.85 | 4.5 |
| Other (elementwise, quantisation, hyper-connection) | 3.28 | 7.9 | 3.25 | 7.8 |
| GPU idle | 3.06 | 7.4 | 4.18 | 10.1 |

- **Idle placement:** 0.2–0.3 ms from step start to the first kernel; about 2.1–2.5 ms of small bubbles between kernels (only 0.6–1.0 ms of it in gaps over 50 µs); 0.8–1.6 ms from the last kernel to the next step (host-side step overhead with `--no-async-scheduling`).
- **NCCL:** the AllReduce median is 16–18 µs per call (p90 35–37 µs). The leader spends about 1.2 ms more per step in NCCL than the worker, consistent with waiting on the worker at sync points [inferred].
- **Hypothesis E07.2 refuted.** Decode is not mostly waiting on communication: NCCL is 5–8% of the step and the GPU is busy 90–93%. The step is dominated by weight-streaming GEMM/GEMV kernels (about 72%).
- **Roofline estimate [inferred, assumptions stated]:** weight bytes read per token per rank, assuming TP-sharded attention, shared-expert and LM-head weights, EP-split experts (on average 3 of 6 active experts per rank) and replicated BF16 compressor/gate weights. FP8 dense about 2.93 GB, routed experts about 1.73 GB, BF16 GEMVs about 0.71 GB, LM head 0.53 GB: about 6.0 GB in total. At 273 GB/s (spec) that is about 22 ms; at a practical about 230 GB/s, about 26 ms. Current 41.6 ms ≈ 62% of the practical roofline.
  - Implied bandwidth per class: LM head about 188 GB/s, routed experts about 205 GB/s, FP8 dense about 160–200 GB/s, BF16 GEMVs about 75 GB/s (worst; the router gate GEMV is 41 µs for 2 MB with 16 CTAs on a 48-SM GPU).
- **Where the remaining time goes, beyond the about 26 ms floor (about 15 ms):** idle 3–4 ms; NCCL 2–3 ms; "other" small kernels 3.3 ms; attention 1.9 ms; BF16 GEMV inefficiency about 2–3 ms; FP8 dense below roofline about 2–3 ms. No single item is above 10% of the step.
- **Candidate next changes, ranked by expected gain per unit effort:**
  1. Async scheduling (removes host overhead between steps, est. 0.8–1.6 ms, about 2–4%). Config flag only; the research report recommends it.
  2. Clock policy: keep uncapped (+5.1% measured in E07.4, not isolated) or restore `-lgc 0,2100`. Needs the reason for the cap.
  3. NCCL tuning for 87 small AllReduces (≤1 ms ceiling).
  4. Kernel work: BF16 GEMV path (router gate, compressors) and DeepGEMM M=1 efficiency. Code changes, larger effort.


#### E07.6 — Async scheduling A/B (DONE: +7.6%, 2026-10-16)
- **Hypothesis:** with `--no-async-scheduling`, the host's per-step work (scheduling, sampling output handling) runs while the GPU sits idle. E07.5 measured 0.8–1.6 ms from the last kernel to the next step. Async scheduling overlaps it with the GPU. Expected gain 2–4% decode; the change could also fail or be unstable with this multi-node mp executor and breakable graphs.
- **Change:** `DSV4_ASYNC=1` swaps `--no-async-scheduling` for `--async-scheduling` on both nodes (node-launch.sh, plus passthrough in launch-dsv4.sh; backups `*.bak-20261015-async`). The flag parses in this image. Default behaviour is unchanged.
- **Protocol:** same as E07.4 (warm-up, then `ctxbench.py 4096 512` ×3, median), plus a prefill check and a 1M-context sanity request. Compare against E07.4 (24.76 tok/s), with clocks and HCAs unchanged.
- **Result (2026-10-16, run started 00:22 UTC) [measured]:** `async_scheduling: True` confirmed in the engine config; clocks still uncapped (2411 MHz idle) and HCAs still f0.
  - `ctxbench.py 4096 512` ×3 after warm-up: 26.64, 26.48 and 26.67 tok/s. **Median 26.64 tok/s (37.5 ms/token), +7.6% vs E07.4 (24.76)**, +13.1% vs E07.0. Spread ±0.4%.
  - 64K context (65,826 prompt tokens): decode 26.29 tok/s, prefill 1428 tok/s (TTFT 46.1 s). No earlier 64K figure was measured under the same config, so there is nothing to compare against.
  - 4K prefill: 2192–2238 tok/s vs 2288–2301 in E07.4 (−2 to −4%). The change is small; whether it is caused by async scheduling or is noise is not established.
  - Correctness spot check (temperature 0): "17*23 = 391" and the first 8 primes are correct.
- **Analysis:** the gain (2.9 ms/token) exceeds the 0.8–1.6 ms host gap measured after each step in E07.5. Async scheduling therefore also hides other per-step host work, probably the step-start gap and leader/worker sync skew [inferred, not profiled]. A profile with async on would confirm it.
- **Idle stability with async on [measured]:** 4-token probe after 30 min idle → HTTP 200 in 0.25 s (18:01 PDT); after a further 60 min → HTTP 200 in 0.25 s (19:01 PDT). Data: `data/E07/async-idle.log`. Same result as E09 with async off. No WiFi roam was checked in this window.
- **1M check, first attempt:** HTTP 400. `ctxbench.py 1048576` built a 1,054,520-token prompt, over max_model_len 1,048,576. ctxbench extrapolates tokens per word from a 4,000-word sample, which gives about ±1% error at 1M; on 2026-10-12 the same target gave 1,045,933. This was a tool artifact, not caused by async. Re-run at target 1,030,000.
- **1M check, re-run [measured] (19:01–19:19 PDT):** 1,035,911 prompt tokens, under `run-guarded.sh`; the watchdog did not fire.
  - TTFT 1059.6 s (17.7 min), prefill 978 tok/s, decode **24.76 tok/s**, 256 tokens. Compared with E06 (1,045,933 tokens, async off, clock cap on, f1 HCAs): TTFT 1113.3 s, prefill 939 tok/s, decode 21.08 tok/s. That is decode **+17.5%** and prefill +4%. The prompt was 1% shorter, which accounts for part of the prefill difference.
  - Minimum MemAvailable during the run: leader 11 GiB, worker 16 GiB; memory PSI 0 throughout (E06: about 8–9 and 16 GiB).
  - The background job reported exit 1 only because `grep -c WATCHDOG` matched nothing.
- **Decision:** keep async on as the new baseline (D7). The 1M and idle checks passed. **Default changed 2026-10-16 at the user's request:** async is ON by default when DSpark is off and OFF by default when DSpark is on. `DSV4_ASYNC=0/1` overrides either default. Backups `*.bak-20261016-asyncdefault`. The selection logic was tested for all six spec/env combinations.
- *Correction 2026-10-16:* an earlier line here said the 1M and idle checks were still pending. They were done and passed (see above).

#### E07.7 — Decode step profile with async on, 4K context [measured unless tagged] (2026-10-16)
- **Method:** same as E07.5, with the run started 2026-10-16 19:10 PDT (container log 10-17 02:10 UTC) (async on by default, `DSV4_PROFILE=1`, clocks uncapped, f0 HCAs). Warm-up `ctxbench.py 4096 64` (26.3 tok/s), then `/start_profile`, `ctxbench.py 4096 128`, `/stop_profile`. Traces: leader `dp0_pp0_tp0_..._rank0.1792203234116962183.pt.trace.json.gz`, worker `..._rank1.1792203234093644506.pt.trace.json.gz` in `/srv/dsv4/diagnostics/<host>/prof/`.
- **The request measured during profiling ran at 16.54 tok/s** (`data/E07/async-prof.jsonl`). That reflects profiler overhead on a short 128-token request; it is not a regression. The step durations below are measured from inside the trace.
- **Step:** median **38.1 ms** leader, 38.2 ms worker (E07.5: 41.6 ms). 38.1 ms ≈ 26.2 tok/s, consistent with the 26.64 tok/s measured without the profiler in E07.6.
- **Exclusive time per step:**

| Class | Leader ms | % | Worker ms | % | E07.5 leader ms |
|---|---|---|---|---|---|
| FP8 dense GEMMs | 14.55 | 38.2 | 14.34 | 37.5 | 14.92 |
| MoE routed experts | 8.39 | 22.0 | 8.08 | 21.1 | 8.22 |
| BF16 cuBLAS GEMV | 4.17 | 10.9 | 4.14 | 10.8 | 4.13 |
| LM head | 2.83 | 7.4 | 2.79 | 7.3 | 2.81 |
| NCCL | 1.64 | 4.3 | 2.47 | 6.5 | 3.15 |
| Attention | 1.96 | 5.1 | 1.88 | 4.9 | 1.89 |
| Other | 3.39 | 8.9 | 3.31 | 8.6 | 3.28 |
| **GPU idle** | **1.15** | **3.0** | **1.13** | **2.9** | **3.06** (worker 4.18) |

- **Idle placement:** step start to first kernel 334 µs leader, 43 µs worker; small mid-step gaps 0.85–0.97 ms (0.3–0.5 ms in gaps over 50 µs); **tail gap (last kernel to next step) 0 µs on both ranks** (E07.5: 0.8–1.6 ms).
- **NCCL:** 86–87 AllReduce bf16 RING_LL per step, median 15.4 µs leader / 22.8 µs worker. In this run the worker, not the leader, spends more time in NCCL (2.5 vs 1.6 ms), so the waiting side has swapped compared with E07.5 [inferred: whichever rank finishes its compute first waits at the AllReduce].
- **Analysis:** this confirms the E07.6 inference. Async removed the tail gap entirely and cut GPU idle from 3–4 ms to about 1.1 ms per step; compute kernel times are unchanged within ±3%. The GPU is now busy about 97% of the step, so **further host/scheduling work has at most about 1 ms (3%) left to gain.** The remaining gap to the about 26 ms roofline estimate (E07.5) is about 12 ms, and it now lies almost entirely inside kernels: FP8 dense about 14.5 ms (largest), routed MoE 8.2–8.4, BF16 GEMV 4.1 (worst bandwidth efficiency), "other" 3.4, NCCL 1.6–2.5, attention 1.9. 38.1 ms ≈ 68% of the practical roofline [inferred, same assumptions as E07.5].
- **Next candidates, updated:** (1) clock policy (pending the operator); (2) BF16 GEMV path, est. ≤2–3 ms if brought to about 200 GB/s; (3) DeepGEMM M=1 FP8 dense efficiency, est. ≤2–3 ms; (4) fusing "other" small kernels; (5) NCCL small-message tuning, ≤1 ms. Items 2–4 are kernel or code changes [hypothesis for all gain estimates].

### 2026-10-14 — Blocker: CX7 links have no IPv4 after reboot [measured]
- Both nodes did a clean shutdown at about 22:26–22:27 on Oct 13. The worker came back at 22:29 and the leader at 23:06. At 22:26:53 the f1 ports went `Link down`.
- After the reboot, both nodes have carrier on the **f0** ports (enp1s0f0np0, enP2p1s0f0np0). The f1 ports show NO-CARRIER. The cables were therefore moved from f1 to f0 on both nodes.
- The static IPs 192.168.200.x and 201.x are configured in netplan for the f1 ports only, so the f0 ports have only link-local IPv6. Pings to the worker's 192.168.200.11 and 201.11 fail. The worker is reachable over tailscale (<tailscale-ip>).
- `cx7-nccl-env.sh` discovers the up ports, so NCCL would pick f0. `node-launch.sh` would stop with "no IPv4 on <ifname>". No launch is possible until the f0 ports have IPs or the cables are back on f1.
- E07.3 (profiling restart) is blocked on this. The sticky `launch-dsv4 down` ALERT is still set on both nodes.

### 2026-10-15 — INV-2: why the CX7 link lost its IPs [measured, from journals on both nodes]
| Time (PDT) | Node | Event |
|---|---|---|
| Oct 13 18:42 | leader | The operator session opened |
| 18:56:30–33 | worker | system powered off |
| 18:56:34 | leader | both f1 links `Link down` |
| 18:56:41 | leader | `Power key pressed short` → power off |
| 19:11 / 19:16 | leader / worker | boot; f1 links up at 19:16 (cables on f1) |
| 22:21–22:26 | both | new the operator sessions |
| 22:26:49 | worker | `Power key pressed short` → power off |
| 22:26:53 | leader | f1 links down |
| 22:27:01 | leader | `Power key pressed short` → power off |
| 22:29 | worker | boot; no CX7 link |
| 23:05:57 | worker | **f0** links up (cables now on port 0) |
| 23:06 | leader | boot with f0 up, f1 NO-CARRIER |
| Oct 15 00:22:52 | worker | `Power key pressed short` → power off; boot 00:23 |

- **Cause:** someone physically present pressed the power buttons and re-cabled the nodes. They were not crashes; every shutdown is a clean `Power key pressed` poweroff. Between 22:29 and 23:06 the QSFP cables went into port 0 on both nodes. The operator had sessions open each time. Who acted is [inferred], not proven.
- **Why the IPs are gone:** the static IPs are defined only in the netplan profiles for the f1 ports. The f0 ports are held by NetworkManager "Wired connection" DHCP profiles that stay stuck in `activating`, because there is no DHCP server on a point-to-point link.
- **Links are healthy:** IPv6 link-local ping works on both planes (enp1s0f0 ↔ enp1s0f0 at 0.7–1.4 ms, enP2p1s0f0 ↔ enP2p1s0f0 at 1.3 ms). The pairing is the same as before; only the addresses are missing.
- The operator's own `cx7-nccl-env.sh` notes that the cable has moved between f0 and f1 before. This setup has no persistent IPv4 config for f0.

### 2026-10-15 — INV-2 follow-up: first IP fix applied to the wrong node [measured]
- The worker's nmcli commands were also run on the **leader**. Leader result: Wired connection 1 (enP2p1s0f0np0) = 192.168.201.11 (should be .10), and Wired connection 3 (bound to **enP7s7**, the 10GbE port, no carrier) = 192.168.200.11. The worker still has no CX7 IPv4.
- Effect: the leader answers for both worker IPs itself. A ping to 200.11 or 201.11 returned in 0.04 ms, and ssh to 200.11 returned the leader's own host key ("REMOTE HOST IDENTIFICATION HAS CHANGED"). This must be fixed before any launch, or NCCL and the control plane would connect to the leader itself.
- **Resolved 2026-10-15 16:45 [measured].** Static IPv4 is set via NM profiles on the f0 ports: leader "Wired connection 2" (enp1s0f0np0, 200.10) and "Wired connection 1" (enP2p1s0f0np0, 201.10); worker "Wired connection 3" (enp1s0f0np0, 200.11) and "Wired connection 1" (enP2p1s0f0np0, 201.11). The leader's enP7s7 profile is back to auto and down.
  - Ping round trip is 0.6–0.8 ms on both planes. Host key at 200.11 = worker key (TtglmiVe…), and `ssh gx10` works again.
  - ibdev2netdev shows rocep1s0f0 and roceP2p1s0f0 Up on both nodes. cx7-nccl-env.sh discovers NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0, SOCKET_IFNAME=enP2p1s0f0np0 and GID index 5. So VLLM_HOST_IP is unchanged at 192.168.201.x.
  - These profiles persist across reboots. If the cables go back to f1, the old netplan f1 profiles apply again, and the f0 profiles sit idle without carrier.
  - The relaunch has a new variable: the HCAs are now the f0 ports (until Oct 13 they were f1). The E07.3 baseline re-run will show whether decode speed changed.

### 2026-10-15 — Relaunch friction after `launch-dsv4.sh down` [measured]
- After `down`, the next `start-dsv4-default.sh` failed with "another attended T2 admission is active". The old run's guard (`dsv4-t2-guard.py run`) and `allocator-observer.sh`, plus its `journalctl -k -f` child, kept running on both nodes. They inherited the `admission.lock` file descriptor, so they held the flock. Killing them by PID freed the lock.
- Memory again held about 90 GiB per node after `down` (MemFree 30–33 GiB). `gb10-reclaim` plus a tiny CUDA job restored it to 122–124 GiB.
- Open item: `down` should stop the run's guard and observer processes. Today a manual cleanup is needed between runs.

### 2026-10-16 — Run 20261016T0022Z ended: worker rebooted by the operator [measured]
- 10:35:44 PDT: worker journal shows `sudo[170592]: <operator> : ... COMMAND=/usr/sbin/reboot`. The worker's vLLM logged "Parent process exited" at 17:35:45 UTC and its container exited 137 (not OOM). The worker was back up at 10:36.
- The leader did not detect the loss. Its API stayed up and requests hung: the E07.7 profile request at 10:44 PDT got no tokens, with leader GPU 0% at 13 W. The client was killed at 10:52. The leader needs `launch-dsv4.sh down`.
- After the reboot, the worker's CX7 IPv4 came back automatically (NM profiles from 2026-10-15 persist). Leader↔worker ping is 0.3 ms. The clock cap is absent on the worker as well.
- **E07.7 (profile with async on):** not captured; needs a fresh run.
  - *Update 2026-10-16:* captured on the next run; see E07.7.
- The run was stable for 16 h, including the 1M request and both idle probes, until the external reboot.

### 2026-10-16 — Why async scheduling was off: provenance [measured from the operator's files, read-only]
- `/srv/dsv4/live/launch-dsv4.sh:80-81`: "Async scheduling is intentionally disabled while isolating the EngineCore/shared-memory deadlock." It was an isolation step during debugging, not a finding that async is unsafe.
- The operator's upstream bug report (`live/diagnostics/upstream-bug-report-dspark-rank-divergence.md`) describes the deadlock: first-request TP rank divergence and hang with the rank-0 GPU wedged at 96%. It was seen with **DSpark on** and vLLM 0.25.2. The operator named DSpark as the strongest differentiator. Our E08 found a separate cause of shm-broadcast stalls (WiFi control plane), fixed in E09.
- `live/dsv4_validate.py:110` requires `--no-async-scheduling` for the operator's approved live DSpark config. That validator applies to `live/`, not to our `run/` stack.
- Implication: with DSpark off (our config), async passed 4K, 64K and 1M runs plus the 30/60-min idle probes during one 16 h run (E07.6). With DSpark on, async is untested and touches the exact path the operator was debugging, so it should not be combined with DSpark without a separate test.

### 2026-10-17 — Status check after about 14 h idle [measured]
- At 09:32 PDT both containers had been up 14 h (the run from E07.7, async on, `DSV4_PROFILE=1`). Health 200, no ALERT on either node, worker up 22 h 56 m (no new reboot), leader GPU 0% at 2411 MHz, leader MemAvailable 22 GiB.
- The first request after the idle period (8 tokens, temperature 0) returned the correct "391" in **7.4 s**. The next two took 0.17 s and 0.18 s. The earlier idle probes (30 and 60 min, E07.6) were 0.25 s, so a one-off delay of about 7 s on the first request after a long idle is new. The cause is not established; the E08 slow-first-request pattern after a long idle is a candidate [hypothesis].
- *Follow-up, same day [measured]:* **the cause is a one-time kernel JIT compile, not the idle period.** The E08 hypothesis above is refuted. The leader and worker logs show `jit_monitor` warnings at 16:32:30–36 UTC, exactly during the 7.4 s request: TileLang compiled `mhc_pre_big_fuse_with_norm_tilelang` (about 4 s) and Triton compiled `_count_expert_num_tokens`.
  - Every earlier request in this run had a 3.8K-token or longer prompt. This was the first very short prompt, and these kernels are specialised by shape. Startup warm-up only covered one in-inference compile (`_gumbel_sample_kernel` at 02:13).
  - Reproduced: the prompt "Hi" took 4.2 s and triggered a new compile (`mhc_pre_big_fuse_broadcast_with_norm_tilelang`). Repeats and other short or medium prompts took 0.35–0.44 s. One 26-word prompt took 1.36 s once with no JIT warning logged, then 0.41 s on repeat; that cause is not identified [an unmonitored compile or plan is a hypothesis].
  - The impact is a one-time 1–7 s on the first request of a new shape class after each boot. Decode speed and stability are not affected.
  - Possible fix (not applied): after the API reports ready, the launcher sends a few short warm-up prompts of different lengths, and optionally persists the TileLang/Triton caches across restarts.

#### E07.8 — BF16 GEMV investigation [measured unless tagged] (2026-10-17)
- **Kernel-to-weight map** (from the E07.7 leader trace, the safetensors shapes and `vllm/models/deepseek_v4/attention.py:530-557`). All are cuBLAS `gemvx`, BF16 weights replicated on both ranks (`disable_tp=True`, `ReplicatedLinear`):

| Weight | Shape | MB | Calls/step | Median µs | Effective GB/s | Overlapped with other kernels | Stream |
|---|---|---|---|---|---|---|---|
| Compressor `fused_wkv_wgate`, ratio-4 layers (grid 512) | 2048×4096 | 16.8 | 21 | 112.7 | 149 | 38% | aux 0 |
| Compressor `fused_wkv_wgate`, ratio-128 layers (grid 256) | 1024×4096 | 8.4 | 20 | 80.1 | 105 | 66% | aux 0 |
| Indexer compressor `fused_wkv_wgate` (grid 128) | 512×4096 | 4.2 | 21 | 60.1 | 70 | 100% | aux 2 |
| Indexer `weights_proj` (grid 16) | 64×4096 | 0.5 | 21 | 43.5 | 12 | 100% | aux 1 |
| MoE router gate (grid 64) | 256×4096 | 2.1 | 43 | 10.8 | 195 | 53% | mostly default |

  - The three compressor/indexer GEMMs already run on aux streams in parallel with `fused_wqa_wkv` on the default stream (`execute_in_parallel`). The low effective bandwidth of the two fully overlapped kernels is contention, not their own cost.
  - BF16 weight bytes per step per rank are about 0.68 GB.
- **Correction to E07.5:** E07.5 said "the router gate GEMV is 41 µs for 2 MB with 16 CTAs". That was wrong. The 16-CTA kernel is the indexer `weights_proj` (0.5 MB, fully hidden on an aux stream). The router gate is the 64-CTA kernel at 10.8 µs and 195 GB/s, which is efficient.
- **GB10 bandwidth ceiling, isolated, server idle, in-container torch:** 1 GiB read-reduce **257 GB/s**; 1 GiB copy 233 GB/s (read + write). E07.5 assumed 230 GB/s as the practical ceiling, so the roofline floor is lower: about 6.0 GB per token at 257 GB/s ≈ 23 ms. 38.1 ms ≈ 61% of that [inferred].
- **Isolated cuBLAS BF16 GEMV (M=1, K=4096), weights rotated over 40 copies to defeat L2:** 2048 rows 102–105 µs (160–165 GB/s); 1024 rows 53–58 µs (145–157); 512 rows 18–19 µs (218–228); 256 rows 10 µs (205). Every call form tried (`mm` with fp32 out, which is current; `linear` + float; `mv`; transposed) is within ±5%. **cuBLAS is inefficient at 1024 and 2048 rows.**
- **Splitting into 512-row chunks** (`torch.mm` per chunk, fp32 out, under CUDA graph): 2048 rows 100.7 → **76.4 µs** (220 GB/s); 1024 rows 53.6 → **38.9 µs** (216 GB/s). 256-row chunks are worse than 512.
- **Expected gain if applied to the two compressor GEMMs [hypothesis]:** kernel time saved is 21×24 + 20×15 ≈ 0.8 ms per step. Because these run overlapped with the FP8 `fused_wqa_wkv` and share bandwidth, the wall-time gain is ≤0.8 ms (≤2%, about 26.6 → ≤27.2 tok/s at 4K). The E07.5 estimate of "2–3 ms from the BF16 GEMV path" was too high.
- **Where decode time is now (E07.7 + E07.8):** weight-streaming kernels (FP8 dense, MoE, BF16 GEMV, LM head) total about 29.7 ms for about 5.9 GB ≈ 198 GB/s, which is 77% of the 257 GB/s ceiling. Non-weight work is about 8.4 ms: other 3.4, attention 1.9, NCCL 1.6–2.5, idle 1.1.
- **Patch prepared (2026-10-17), not yet applied:** `/srv/dsv4/run/patches/dsv4-attention.py` is the image's `vllm/models/deepseek_v4/attention.py` (original saved as `dsv4-attention.orig.py`, md5 af742cb3…) with a helper `_dsv4_mm_fp32` that splits the two compressor `fused_wkv_wgate` GEMMs into 512-row chunks when M ≤ 8; otherwise it is unchanged.
  - Verified in the container [measured]: identical shapes and dtype. Decode-shape outputs differ by ≤1.5e-7 relative to the max value (fp32 summation order); M=2 with 2048 rows by 4.4e-6; prefill (M=4096) is bit-exact. Under CUDA graph, 107.2 → 73.6 µs (2048 rows) and 57.2 → 37.6 µs (1024 rows).
  - Wiring into the launchers (bind mount with `DSV4_PATCH=0` to disable, copy to the worker in `launch-dsv4.sh up`) was blocked by the auto-mode permission check. It is staged as `apply-e078-patch.sh` in the session scratchpad for the user to review and run.
- **Note on the first-request JIT delay:** `node-launch.sh` already persists the TileLang and Triton caches (`$RUNTIME_CACHE_ROOT/{tilelang,triton}`). The kernels compiled on 2026-10-17 should therefore load from cache after a restart [hypothesis, not yet verified], so the warm-up-prompt fix may be unnecessary.

### 2026-10-18 — First boot with the E07.8 patch aborted by a stale worker guard [measured]
- The user applied `apply-e078-patch.sh` (backups `*.bak-20261017-patch`), ran `down`, and launched run dsv4-t2-20261018T182606Z-356448 (async on, profiler off). The patched `attention.py` was loaded: log line numbers shifted from `attention.py:798` to `:810`, as the patch predicts.
- The engine booted (API up at 18:30:49 UTC) and received SIGTERM at 18:30:56. Abort reason: "startup guard/readiness failure".
- **Root cause:** the worker's guard from the *previous* run (dsv4-t2-20261017T020853Z-243288, pid 16232, in serving mode) was still running. During the new boot's model load, it wrote the worker's top-level ALERT `residual_growth>8GiB/10s+psi_stall` at 11:30:42 PDT, and the start script then aborted the new run.
- **My error:** before the launch I checked the worker with `pgrep -af "dsv4-t2-guard|allocator-observer"`. On the worker the guard runs as `~/.cache/dsv4-t2/guard.py`, which that pattern does not match, so I reported the worker as clean when it was not. Corrected check: `pgrep -af "guard.py|allocator-observer|journalctl -k -f"` on both nodes.
- **The patch is not implicated.** No exception occurred, and the engine was killed from outside.
- Clean-up 2026-10-18: killed pid 16232 on the worker. No guard, observer or container is left on either node; reclaim done (MemFree leader 117 GiB, worker 119 GiB); admission lock free. This is the same relaunch-friction issue as 2026-10-15: `down` does not stop the guards.

### 2026-10-18 — Second relaunch aborted: worker WiFi down, so clock skew [measured]
- Run dsv4-t2-20261018T185430Z-365790 aborted at 11:54:55 PDT, about 25 s after confirmation and before any container started (reason: "startup guard/readiness failure").
- **Direct cause:** the worker's `allocator-observer.sh` refused to start, with "deadline must be in the future and no more than 3300 seconds away". `start-dsv4-default.sh` computes the deadline as leader `date +%s` + 3300 exactly, and the worker clock was **1.26 s behind** the leader, so the deadline was over 3300 s away on the worker.
- **Why the clock drifted:** the worker has had no WiFi since **2026-10-17 22:31:03 PDT**. A re-association to "<home WiFi SSID>" timed out in the 4-way handshake. NetworkManager treated this as a key problem ("asking for new key"), found no secrets agent, and marked the activation failed (`no-secrets`). After such a failure NM does not auto-retry. The PSK is stored system-wide (`psk-flags=0`), so the key itself is fine.
- Consequences: the worker has no default route and no NTP (`System clock synchronized: no`), and its Tailscale has been offline since about then. The serving path is unaffected (CX7 only, with `VLLM_HOST_IP` pinned per E09).
- **Fix (needs the user; worker sudo needs a password):** `nmcli con up "<home WiFi SSID>"` on the worker, then wait for NTP to sync before relaunching.
- **Launcher weakness [inferred]:** the observer start fails with any worker clock lag above about 0 s, because the deadline is set to exactly the maximum allowed. A margin (for example +3240 s) would make it tolerant of small skew. This is a change to the start script and has not been made.

#### E07.8 result — chunked compressor GEMV patch (2026-10-18) [measured] · SUCCESS, small
- Run dsv4-t2-20261018T192109Z-368186 (async on, profiler off, patch on). This followed the worker WiFi fix (`nmcli con up "<home WiFi SSID>"` by the user, NTP synced, skew now ≤0.3 s) and clearing the alerts. Patch confirmed mounted on both containers (`grep -c _dsv4_mm_fp32` = 3). Clocks uncapped (2489 MHz under load), f0 HCAs.
- `ctxbench.py 4096 512` ×3 after warm-up: 26.93, 26.93 and 26.85 tok/s. **Median 26.93 tok/s (37.13 ms/token) vs 26.64 (37.54 ms) in E07.6: +1.1%, −0.41 ms/token.** The two runs' ranges do not overlap (E07.6: 26.48–26.67; now 26.85–26.93). This compares two separate boots, not an A/B within one boot. The gain is inside the predicted ≤0.8 ms.
- 4K prefill 2227–2234 tok/s (E07.6: 2192–2238), so no change.
- Correctness spot checks (temperature 0): "391" for 17×23; the first 8 primes are correct; a coherent two-sentence Rayleigh-scattering answer.
- **First request after restart:** 0.29 s for the short prompt, then "Hi" 0.37 s. TileLang and Triton still log the in-inference "JIT compilation" for the two kernels from 2026-10-17, but they load from the persisted cache (`$RUNTIME_CACHE_ROOT`), so there is no multi-second delay. The 2026-10-17 hypothesis is confirmed; the warm-up-prompt fix is not needed.
- **Decision:** keep the patch on by default (`DSV4_PATCH=0` disables it). Not yet checked at 1M under the patch. Decode-shape outputs differ from unpatched only by fp32 summation order (≤1.5e-7 relative), and prefill is bit-exact.

### 2026-10-18 — Launcher hardening prepared (not yet applied) [measured where stated]
- **Problem:** relaunches after `down` failed three times: 2026-10-15 (admission lock held), 2026-10-18 18:30 (stale worker guard) and 2026-10-18 18:54 (worker clock 1.26 s behind, observer deadline). `down` (`dsv4-abort.sh`) stops the containers but not the run's guards or allocator observers. The start script sets the observer deadline at exactly the observer's 3300 s maximum, so any worker clock lag aborts the launch.
- **Change staged as `apply-launcher-hardening.sh` in the session scratchpad** (backups `*.bak-20261018-hardening`):
  1. `launch-dsv4.sh down`: after `dsv4-abort.sh`, TERM the run's guard on both nodes (`pkill -f "guard[.]py run --run-id <run> "`), the leader observer (`allocator-observer[.]sh --data <state>/runs/<run>/observer/`) and the worker observer (pid file, verified by `/proc/<pid>/cmdline`). Then print how many are left on each node. The exit code is still abort's. Matching is by exact run ID or verified pid only; the `[.]` stops the pattern matching the remote shell that carries it.
  2. `start-dsv4-default.sh`: observer deadline +3240 s instead of +3300, tolerating up to 60 s of worker clock lag.
- **Verified before applying [measured]:** the script was applied to copies in the scratchpad, where it ran cleanly and passed `bash -n`. On the live run dsv4-t2-20261018T192109Z-368186, `pgrep` with the exact patterns matched only that run's guard and observer on each node (leader guard 368212 and observer 368230; worker guard 124227 and observer 124722, pid file verified). The observer's TERM handler stops its `journalctl -k -f` child.
- It is not applied yet, because a direct launcher edit was previously blocked by the permission check; the user applies it. It takes effect at the next `down` and launch, with no relaunch needed.
- *Update 2026-10-18:* the user applied `apply-launcher-hardening.sh`. Verified: both backups exist, both changes are present, and `bash -n` passes. It takes effect at the next `down` and launch; the first `down` should print `dsv4-down: guards/observers left …: leader=0 worker=0`.

### E10: Prefill chunk size (`--max-num-batched-tokens`) 4096 → 8192 · PLANNED (2026-10-18)
- **Question:** does a larger prefill chunk cut 1M TTFT (G4)?
- **Reasoning [hypothesis]:** prefill throughput falls with depth: 2,230 tok/s at 4K, 1,428 at 64K, 978 at 1M (E07.6). With 4096-token chunks, a 1M prompt takes about 253 engine steps. Each step pays fixed costs: per-step scheduling and metadata, the TP all-reduces, weight streaming (about 6 GB per rank per step), and attention/indexer passes over the KV cache so far. Doubling the chunk halves the number of steps, so the fixed per-step costs and per-step weight reads are paid half as often. Attention FLOPs over the prefix do not change.
  - Expected gain is uncertain. If the 4096-token GEMMs are already compute-bound and attention dominates at depth, the gain will be small.
- **Risk:** activation and workspace memory grows with chunk size. The 1M run (E07.6) reached a minimum MemAvailable of 11 GiB on the leader, and the guard alerts below 6 GiB. Several buffers are sized by `max_num_batched_tokens` (topk indices, compressor scratch).
- **Change staged:** `apply-maxbatch-env.sh` (scratchpad; backup `start-dsv4-default.sh.bak-20261018-maxbatch`) makes the chunk size settable with `DSV4_MAX_BATCH` (allowed: 4096, 8192, 12288, 16384; default 4096, so a launch without it is unchanged). Validation runs before the confirmation prompt, so a typo cannot start guards or set an ALERT. Tested on a copy: applies cleanly, `bash -n` passes, unset → 4096, 8192 accepted, 5000 rejected.
- **Protocol:**
  1. Launch with `DSV4_MAX_BATCH=8192`.
  2. Measure 4K and 64K (`ctxbench.py`), plus 4K decode ×3 to check decode is unchanged (CUDA graphs are sized by `max-cudagraph-capture-size 1`, not the chunk size).
  3. Record MemAvailable during a 64K prefill.
  4. Only if headroom looks safe, run 1M under `run-guarded.sh` and compare with E07.6 (TTFT 1059.6 s, prefill 978 tok/s).
  5. Revert to 4096 if 1M MemAvailable goes below about 8 GiB or no gain is seen.
- *Update 2026-10-18 19:30 PDT:* the user applied `apply-maxbatch-env.sh` (diff as staged). **The first `down` with the hardened launcher [measured]:** printed `dsv4-abort: leader=stopped worker=stopped` and `dsv4-down: guards/observers left for dsv4-t2-20261018T192109Z-368186: leader=0 worker=0`. An independent check found no guard, observer, `journalctl -k -f` or container on either node, and the admission lock free. **No manual process clean-up was needed for the first time.** Reclaim brought MemFree to 117 GiB on the leader and 119 GiB on the worker; worker clock synced.
- **E10 attempt 1 (2026-10-18 20:15–20:19 PDT, run dsv4-t2-20261019T031459Z-423288) [measured] · FAILED at engine init (config):** `max_num_batched_tokens: 8192` was confirmed in the engine args.
  - vLLM raised `ValueError: To serve at least one request with the model's max seq len (1048576), 10.06 GiB KV cache is needed, which is larger than the available KV cache memory (9.0 GiB) … estimated maximum model length is 757504`.
  - With 4096, the same 9 GiB pin held 1,747,385 KV tokens (1.67× of 1M, E04). Doubling the chunk size therefore roughly doubled the KV memory needed for a 1M request. Part of the DSV4 KV/state layout scales with `max_num_batched_tokens` [inferred; the exact tensors were not identified].
  - The abort cleaned up properly: no guard, observer or container left on either node, and the admission lock free.
  - **Next:** relaunch with `DSV4_KV_CACHE_BYTES=11274289152` (10.5 GiB, about 0.44 GiB above the reported need). The pinned KV allocation grows by 1.5 GiB per node, so the 1M MemAvailable floor (11 GiB leader in E07.6) is expected to drop by at least that much [inferred]. The memory check during 64K comes first.
- **E10 attempt 2 (2026-10-18 about 20:22–20:26 PDT, run dsv4-t2-20261019T032206Z-430837) [measured] · FAILED, same error:** the engine args showed `'kv_cache_memory_bytes': 9663676416` (9 GiB), not the intended 10.5 GiB, so the same KV check failed.
  - The launcher passes `DSV4_KV_CACHE_BYTES` through to both nodes (node-launch.sh:45; launch-dsv4.sh worker ssh string), so the variable most likely was not in the launch command [inferred; the exact command typed was not visible].
  - Clean-up was automatic again: nothing left on either node, lock free. Reclaim done (MemFree 117 / 119 GiB).
- **E10 attempt 3 (run dsv4-t2-20261019T033602Z-445582, booted 20:40 PDT) [measured] · booted:** the variables were set with `export` on separate lines. The launcher's `/proc/<pid>/environ` showed `DSV4_MAX_BATCH=8192` and `DSV4_KV_CACHE_BYTES=11274289152`; the engine args showed the same.
  - Attempt 3 found that attempt 2's launcher environment also lacked `DSV4_KV_CACHE_BYTES`, which confirms the cause was the command line, not the scripts.
  - `GPU KV cache size: 1,094,398 tokens, max concurrency for 1,048,576: 1.04x`. Idle MemAvailable after boot: leader 20.1 GiB, worker 24.6 GiB.
- **E10 results at 4K and 64K (8192 chunks) [measured]:**

| | E07.6 / E07.8 (4096 chunks) | E10 (8192 chunks) | Change |
|---|---|---|---|
| 4K decode ×3 | 26.85–26.93, median 26.93 (E07.8) | 26.46, 26.80, 26.75; median 26.75 | −0.7%; within about 1% run-to-run spread |
| 4K prefill (single chunk) | 2192–2238 | 2208–2221 | none (expected: 3.8K fits one chunk either way) |
| **64K prefill** (65,822 tokens) | **1,428 tok/s, TTFT 46.1 s** | **2,210 tok/s, TTFT 29.8 s** | **+55%, −16.3 s** |
| 64K decode | 26.29 | 26.34 | none |
| Min MemAvailable during the tests (1 s sampling) | — | leader 17.2 GiB, worker 21.7 GiB | — |

  - **Analysis [inferred]:** the gain is far above the per-step-weight-read share a simple model predicts. With 4096-token chunks, prefill throughput fell from about 2,200 tok/s (one chunk) to 1,428 at 64K (16 chunks). With 8192-token chunks, 64K runs at the same 2,210 tok/s as a single chunk. So per-chunk overhead, not attention FLOPs, dominated multi-chunk prefill at 64K. Its source is not identified; candidates are per-chunk indexer/compressor passes over the prefix and eager-mode (breakable-graph) overhead [hypothesis]. A larger chunk (16384) may gain more, at a further KV-memory cost.
  - Data: `/srv/dsv4/run/data/E10/` (`b8192-4k.jsonl`, `b8192-64k.jsonl`, `mem-8192.log`). The 1M run under `run-guarded.sh` is in progress.
- **E10 result at 1M (8192 chunks, 2026-10-18 20:43–21:00 PDT, `run-guarded.sh 1030000`) [measured]:** 1,035,910 prompt tokens.
  - **TTFT 1010.1 s (16.8 min), prefill 1,026 tok/s, decode 25.07 tok/s**, 256 tokens. Compared with E07.6 (4096 chunks, same prompt size 1,035,911): TTFT 1059.6 s and prefill 978 tok/s. That is **prefill +4.9% (−49.5 s)**; decode +1.3% (24.76 → 25.07, which includes the E07.8 GEMV patch).
  - The watchdog did not fire. Sampled every 3 s, **min MemAvailable was leader 15 GiB and worker 21 GiB, with memory PSI 0 on both throughout.** E07.6 at 4096 had leader 11 GiB, so the larger KV pin did not reduce the floor. Why the floor is higher is not established.
- **Analysis [inferred]:** the chunk-size gain is large at mid-range context and small at 1M.
  - Per-chunk time at 64K: 4096 chunks average 2.9 s per chunk (46.1 s / 16); 8192 chunks average 3.7 s per chunk (29.8 s / 8). Doubling the chunk cost only about 28% more per chunk, so a large share of each chunk's time does not scale with the chunk's own token count. The pattern fits work proportional to the prefix length, paid once per chunk (for example, per-chunk passes over the compressed or indexer cache) [hypothesis]. With n tokens and chunk c, that term grows as n²/c, so doubling c halves it.
  - At 1M only about 5% was saved. The chunk-independent work, sparse attention and indexer scoring per token over the whole prefix, dominates there.
  - Implication: 16384-token chunks would likely give a further but smaller gain at 1M and a larger one at 64K–256K, at more KV memory (needs a measured requirement first).
- **Verdict: SUCCESS.** 8192 chunks give +55% prefill at 64K and +5% at 1M, with no decode loss and a safe memory floor. **Recommended as the new default** (`DSV4_MAX_BATCH=8192` with `DSV4_KV_CACHE_BYTES=11274289152`). That needs a launcher default change; not made yet.

### 2026-10-19 — E10 adoption: 8192 chunks as the launcher default (applied)
- Script `scratchpad/apply-default-8192.sh` (backup `start-dsv4-default.sh.bak-20261019-default8192`) changes `start-dsv4-default.sh`: `DSV4_MAX_BATCH` defaults to 8192; when `DSV4_KV_CACHE_BYTES` is unset it is set from the chunk size (4096 → 9663676416, 8192 → 11274289152) and exported to `launch-dsv4.sh`/`node-launch.sh`; 12288/16384 refuse to start without an explicit KV size (none measured). The chosen values are printed at launch.
- [measured] Tested on a copy of the script: unset → 8192 / 11274289152; 4096 → 4096 / 9663676416; 16384 without KV → refused; invalid value → refused; 16384 with explicit KV → accepted.
- [measured] Applied by the user 2026-10-19 10:56 PDT (`syntax OK`, diff as intended). Takes effect at the next `START DSV4 T2`; the running server already uses these values.

### 2026-10-19 — Overnight ALERTs on run dsv4-t2-20261019T033602Z-445582: weekly fstrim (false alarm)
- [measured] Leader ALERT `D-state>10s` at 00:58:50 PDT; `fstrim.service` started 00:58:40 and finished 01:00:26 (1.3 TiB trimmed on /).
- [measured] Worker ALERT `D-state>10s` at 00:31:43 PDT; `fstrim.service` started 00:31:33 and finished 00:32:16.
- [measured] Server unaffected: both containers up 14 h, `/health` on :8888 returns 200 at 11:0x PDT.
- [inferred] Same known false-alert class (the guard counts fstrim's D-state). Open fix: exclude fstrim from the D-state check or move the fstrim timer. Alerts still need the human clear.

## E11 — Where long-context prefill time goes (2026-10-19)
**Setup.** Run dsv4-t2-20261019T181018Z-505899: 8192 chunks, KV 11274289152, async on, GEMV patch on, `DSV4_PROFILE=1` (torch profiler, delay 10 / max 20 iterations, leader rank 0 only). `POST /start_profile`, then `tools/run-guarded.sh 262144`. Data: `data/E11/` (`p256k.result`, `p256k.mon`, `leader-trace.json.gz`, `profiler_out_0.txt`); analysis `scratchpad/e11.py`.

**Request.** [measured] 263,520 prompt tokens, TTFT 162.8 s (1,619 tok/s, with profiler capturing 20 of the steps), decode 26.0 tok/s. Lowest MemAvailable leader 14 GiB / worker 19 GiB, PSI ≤ 1, watchdog did not fire, no ALERT.

**Per-chunk breakdown.** [measured] 20 steps captured (18:15:47–18:17:17 UTC); step 0 partial, steps 1–18 are full 8192-token chunks. GPU kernel time ≈ step wall time (GPU never idle in prefill). [inferred] From timing, steps 1–18 are about chunks 11–28 (prefix about 90K–230K tokens).

| kernel class (ms per chunk) | step 1 | step 18 | change per chunk |
|---|---|---|---|
| indexer logits (`deep_gemm::sm120_fp8_mqa_logits`) | 361 | 1143 | +46 |
| indexer top-k (`topKPerRowPrefill<512>`) | 60 | 285 | +13 (increments grow 5 → 20) |
| sparse MLA prefill (`sparse_mla_prefill_mg_dual_kernel`) | 570 | 936 | +21.5 |
| MoE FP4 GEMMs + gather/scatter/silu | 1592 | 1605 | 0 |
| dense GEMM / all-reduce / mHC / zero_kv / other | 1407 | 1414 | 0 |
| **chunk total** | **3957** | **5347** | **+81** |

[measured] Totals over the 20 steps (93.4 s GPU): MoE FP4 GEMM 29.3 %, sparse MLA 16.3 %, indexer logits 16.2 %, other 8.6 %, mHC 8.2 %, all-reduce 6.4 %, dense GEMM 5.6 %, MoE other 4.8 %, indexer top-k 3.3 %, zero_kv 1.2 %.

**Model.** [inferred] Chunk time ≈ 3.09 s fixed + 9.9 ms per 1K tokens of prefix (fit to steps 1–18). Cross-check: predicts about 1,040 s for 1M against 1,010 s measured (E10), +3 %; predicts about 143 s for 256K against 162.8 s measured, −12 % (cause not established: chunk-index estimate or profiler effect). At 1M the model puts about 39 % of prefill in the fixed part (MoE etc.) and about 61 % in the prefix-dependent part: indexer logits ≈ 36 %, sparse MLA ≈ 17 %, indexer top-k ≈ 10 % (top-k grows faster than linear, so likely more).

**Code fact.** [measured, from `dsv4-attention.orig.py` lines 792, 804, 811] The Lightning Indexer uses all `index_n_heads` (64) heads with `ReplicatedLinear` `wq_b` and `weights_proj`, i.e. both TP ranks compute the full indexer logits and top-k for every chunk. [hypothesis, not tested] Splitting the indexer's prefix (KV) range between the two GPUs and merging the two top-k lists (512 candidates each, a small exchange) would roughly halve indexer logits + top-k per rank, up to about 20–25 % of 1M TTFT by the model. Needs a check of the worker trace to confirm the worker does the same work, and a code change in vLLM's indexer path.

**Verdict.** At long context, prefill is dominated by the indexer (logits + top-k), not by MoE or communication. The MoE and other fixed costs (about 3.1 s per 8192 chunk, ≈ 2,650 tok/s) bound the short-context prefill rate.

### E11b — Worker trace from the same run (2026-10-19)
- [measured] The worker recorded its own trace during E11 (`data/E11/worker-trace.json.gz`, rank 1, same 20 steps, same step wall times as the leader).
- [measured] The worker runs the same full indexer: top-k 59 → 287 ms per chunk (leader 60 → 285), logits 350 → 989 ms (leader 361 → 1143). This confirms from the trace that both GPUs compute the whole indexer for every chunk, so it is duplicated work.
- [measured] The worker's logits kernel is faster than the leader's by up to 13 % (step 18: 989 vs 1143 ms), and the worker's all-reduce time grows 340 → 509 ms while the leader's stays about 300 ms. [inferred] The worker waits at the all-reduce for the slower leader, so the leader is the critical path. [hypothesis] The cause is clock or power behaviour under load on the leader; idle clocks are equal on both (2411 MHz, max 3003). Not measured under load.
- [inferred] For the split-indexer idea in E11, the gain per rank is about half of logits + top-k, on the leader's critical path: about 0.7 s of 5.35 s per chunk at about 230K context, and about 20–25 % of 1M TTFT by the E11 model (still untested).

## E12 — Split-indexer design (2026-10-19, design only, nothing run)
**Facts from code and config.** [measured]
- Model config: 43 layers; `compress_ratios` gives 21 layers at ratio 4 (these have the Lightning Indexer), 20 at 128 and 5 at 0. `index_topk` 512, `index_n_heads` 64, `index_head_dim` 128, hidden 4096.
- E11 trace: 62 indexer logits and 62 top-k kernels per 8192 chunk (about 3 sub-chunks × 21 layers), and 88 all-reduces per chunk taking 300 ms in total. [inferred] At 64 MB each (8192 × 4096 bf16) that is about 19 GB/s effective.
- The prefill loop in `vllm/model_executor/layers/sparse_attn_indexer.py` (lines 449–530): per sub-chunk it gathers the indexer K cache (`cp_gather_indexer_k_quant_cache`), runs `fp8_fp4_mqa_logits(q, k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False)`, then `top_k_per_row_prefill`, then `_merge_dcp_topk_global`.
- vLLM already has an exact cross-rank top-k merge (`_merge_dcp_topk_global`, for decode context parallel, DCP). Its docstring argument: any token in the global top-K is also in its own rank's local top-K, so exchanging the local candidates is exact. But DSV4's model code (`vllm/models/deepseek_v4/`) has no DCP support, so turning DCP on is not an option without porting all DSV4 caches and attention.

**Design.** [hypothesis, not tested]
- Patch `sparse_attn_indexer.py` (bind-mounted like the GEMV patch, gated by a new `DSV4_IDX_SPLIT`, default off until validated). Prefill path only; decode unchanged.
- With TP = 2 and the indexer replicated, each sub-chunk's keys are split at a point s: rank 0 scores keys [0, s), rank 1 scores [s, L). K is sliced and ks/ke are clamped and offset per rank, so each rank's logits and top-k cover about half the keys. s is chosen to balance work (about half of the mean row end).
- Each rank: local top-512 → global ids (rank 1 adds s) → gather the scores → TP all-gather of (score, id) → top-512 of the 1,024 candidates.
- Exchange size [inferred]: 32 MB per indexer layer per 8192 chunk (fp32 score + int32 id), about 21 × 32 MB ≈ 670 MB per chunk, about 35 ms at 19 GB/s. Saving [inferred] about 0.7 s per chunk at about 230K context, growing linearly with context.

**Open points to check in the offline test.**
1. Do `top_k_per_row_prefill` indices count from the row start or from 0?
2. Does downstream sparse MLA need a particular index order?
3. Tie behaviour.
4. That time falls by about half with a half-size K slice.

**Offline test plan (E12a).** A standalone script in the same image on one GPU, at real shapes (about 2,730 rows per sub-chunk; keys 57K for 230K context and 256K for 1M), with random FP8 data:
- Compare the per-row top-512 sets of the full path against split + merge (expect identical, apart from exact ties).
- Time the full path against each half.
- Gate: each half ≤ 55 % of the full time, and sets identical.

Then E12b is a serving test: a temperature-0 output comparison with the patch on and off, plus 64K / 256K / 1M TTFT.

### E12a — Offline split-indexer test (2026-10-19): PASS
**Setup.** Server down. Throwaway container from the same image on the leader GPU. Script `scratchpad/e12a/e12a_idxsplit.py` (runner `run-e12a.sh`), output `data/E12/e12a.jsonl`. The shapes match the real indexer: FP8 path (the run logs "Using FP8 indexer cache"), 64 heads × 128, 2,731 rows per sub-chunk, keys = context / 4, a single request (ks = 0). Random data. Median of 10 timed runs.

**Findings.** [measured]
- Top-k ids count from each row's `ks`, not from 0 (checked with ks = 1000). The current top-k output is not sorted.
- The split point must be aligned: an unaligned K slice start (s = 8362) fails in DeepGEMM with `CUDA_ERROR_INVALID_VALUE`; multiples of 64 work. [inferred] TMA needs the fp32 K-scale slice 16-byte aligned (8362 × 4 B is not). Rule: s multiple of 64.

| context | keys | full logits + top-k | half 0 | half 1 | merge (torch) | worst half / full | rows identical | tie-only rows |
|---|---|---|---|---|---|---|---|---|
| 64K | 17,066 | 5.35 ms | 2.75 | 2.89 | 2.91 | 0.54 | 2731 / 2731 | 0 |
| 230K | 58,026 | 20.61 ms | 9.49 | 9.65 | 3.09 | 0.47 | 2730 / 2731 | 1 |
| 1M | 258,730 | 96.32 ms | 48.45 | 48.36 | 3.14 | 0.50 | 2730 / 2731 | 1 |

"Tie-only" means the two selections differ only by keys with exactly equal scores. The current kernel's choice between equal scores is arbitrary too.

**Gate** (identical apart from ties, and each half ≤ 0.55 × full): passed at all three lengths.

**Net effect.** [inferred]
- Per indexer sub-chunk, the saving is full − half − merge − exchange. The exchange is about 11 MB, about 0.6 ms at 19 GB/s.
  - 64K: about −0.4 ms, a small loss. The merge costs more than it saves.
  - 230K: about +7 ms.
  - 1M: about +44 ms.
- So the split should switch on only above a key-count threshold (about 32K keys ≈ 128K context, to be tuned).
- Over a full 1M prefill (62 sub-chunks per 8192 chunk, 128 chunks, average context 512K by linear interpolation): about 160 s saved of 1,010 s, about −16 % TTFT (≈ 14.2 min).
- A faster merge (the torch gather + cat + topk now takes 3 ms) would raise this towards −18 %.
- These are estimates from offline timings; the serving test (E12b) decides.

### E12b — Split-indexer patch written and checked offline (2026-10-19)
- Patch `patches/dsv4-sparse-attn-indexer.py` (md5 ffcdc4a0…), generated from the image's `vllm/model_executor/layers/sparse_attn_indexer.py` (saved as `patches/dsv4-sparse-attn-indexer.orig.py`, md5 51d96e3a…).
- It adds `_dsv4_split_ok` / `_dsv4_split_logits_topk` and one branch in the prefill loop. The split is used only when the sub-chunk has ≥ 32,768 keys, the FP8 cache is in use, DCP = 1, TP = 2, not XPU, and no CUDA graph is being captured.
- Split point s = n/2 rounded down to a multiple of 64. Merge: TP all-gather of (score, id) pairs, then torch.topk of the 1,024 candidates.
- Launcher wiring staged as `scratchpad/apply-e12b-idxsplit.sh` (gate `DSV4_IDX_SPLIT`, default 0). Dry-run on copies: syntax OK, diff as intended. Not yet applied.
- [measured] Offline check of the patched function itself (`scratchpad/e12b/check_patch.py`, `data/E12/e12b-precheck.jsonl`), both ranks simulated with a stand-in all-gather, against the original logits + top-k:
  - single request at 230K: 2730 / 2731 rows identical, 1 tie-only;
  - two sequences in one chunk (nonzero ks, rows with no keys on one rank): 2731 / 2731;
  - rows with fewer than 512 keys: 512 / 512.
  - No invalid sentinels.
- My error during the check: the first short-row case set `ke` up to 40,212 with only 40,000 keys. The original path read past the end, the patch clamped, so 212 rows "differed". The test was fixed (n = 40,448), not the patch.
- [measured] The empty-row case returns -1 from `top_k_per_row_prefill`.
- Serving test client `tools/e12b_client.py`: deterministic prompt per context length with a hidden 8-digit code at 37 % depth, temperature 0, 64 tokens, logprobs recorded.

### E12b baseline — split indexer OFF (2026-10-19, run dsv4-t2-20261019T215340Z-549769)
[measured] `tools/e12b_client.py`, greedy, 64 tokens; data `data/E12/e12b-base.jsonl`. Patch not mounted (0 sparse_attn mounts, no "split indexer" log line).

| context | prompt tokens | TTFT | prefill tok/s | decode tok/s | code found |
|---|---|---|---|---|---|
| 64K | 65,873 | 36.8 s | 1,792 | 30.58 | yes |
| 256K | 261,619 | 141.8 s | 1,845 | 27.38 | yes |

[inferred] The 64K run was the first request after boot. Its TTFT is above E10's 29.8 s, probably first-request JIT (E07 status entry); not verified. The 64-token decode rates are over short outputs and are not comparable with E07/E10 decode numbers.
- [measured] Warm 64K re-run (target 65537, different prompt, same server): 65,933 tokens, TTFT 39.4 s (1,671 tok/s), code found. So the 36.8 s was not first-request JIT; my JIT guess above is wrong. 64K TTFT with this client (37–39 s) is above E10's 29.8 s with `ctxbench.py`; the cause has not been looked into. The on/off comparison uses this client only.

### E12b — split indexer ON, first results (2026-10-19)
- [measured] First boot with `DSV4_IDX_SPLIT=1` (run dsv4-t2-20261019T220356Z-560283) was aborted by the start script at 15:08:29 PDT ("startup guard/readiness failure"), about 20 s after the post-load reclaim.
  - Patch mounted on both nodes; no Python errors in the leader log.
  - The leader allocator observer logged an `nvidia-smi` call killed after its 8 s timeout at that time.
  - [hypothesis] The observer check failed (`wait_for_api` rc = 3). The rc was not captured, so this is unconfirmed.
  - [inferred] Not the patch: it only runs during prefill, and no request had been sent.
  - Retry (run dsv4-t2-20261019T222156Z-568783) became ready in 260 s with the patch mounted on both nodes.
- [measured] "DSV4 split indexer active" logged once on the leader and once on the worker; no errors. Data in `data/E12/e12b-split.jsonl`.

| context | TTFT base (boot 549769) | TTFT split (boot 568783) | code found | tokens identical before first difference |
|---|---|---|---|---|
| 64K (65536) | 36.8 s | 29.7 s | yes | 3 of 55 |
| 64K (65537) | 39.4 s | 29.5 s | yes | 3 of 60 |
| 256K | 141.8 s | 134.0 s (−5.5 %) | yes | 3 of 52 |

- [inferred] **At 64K the split is not active** (about 16.5K keys per indexer sub-chunk, below the 32,768 threshold), yet TTFT differs by 19–25 % and the text diverges at token 3 (newline vs double newline, both low-margin whitespace choices).
  - So between boots there is both a TTFT difference not caused by the patch (the baseline boot was slow; 29.5–29.7 s matches E10's 29.8 s) and token-level nondeterminism. Cause unknown.
  - Consequences: the cross-boot token comparison cannot validate exactness (the offline checks E12a / E12b-precheck remain the exactness evidence), and the 256K −5.5 % is confounded by the slow baseline boot.
- 1M run (split on) in progress; comparison against E10's 1,010.1 s (same chunk / KV settings, split off).

### E12b — 1M with the split indexer ON (2026-10-19, run dsv4-t2-20261019T222156Z-568783)
- Two of my client errors first: target 1,048,576 and then 1,036,288 both gave HTTP 400. The prompt came out at 1,058,675 tokens, over `max_model_len`, because the filler's word counter grows longer with depth and the per-word estimate overshoots near 1M. Fixed in `tools/e12b_client.py`: it re-scales only when over target by more than 1 %, so the 64K / 256K prompts are unchanged (their overshoot ≤ 0.5 %).
- [measured] 1,036,267 prompt tokens: **TTFT 813.7 s (1,274 tok/s)**, code found, answer coherent. Lowest MemAvailable leader 16 GiB / worker 21 GiB, PSI 0, no watchdog, no ALERT. Data `data/E12/e12b-split-1M.json` / `.mon`.
- [measured] Against E10 (1,035,910 tokens, 8192 chunks, same KV, split off): 1,010.1 s → 813.7 s = **−19.4 %** (−196 s). E11 / E12a model predicted about −16 %.
- [inferred] Caveats:
  - The baseline is from a different client (`ctxbench.py`) and a different boot.
  - Today's baseline boot showed a 19–25 % slowdown at 64K that was not caused by the patch, but this split boot's 64K (29.5–29.7 s) matches E10's (29.8 s). So the comparison boot does not look slow.
  - A same-client 1M baseline with the split off would make this conclusive.

### 2026-10-19 — Boot aborts traced to the allocator observer's nvidia-smi timeout
- [measured] Boot dsv4-t2-20261020T014345Z-600020 (split OFF, for the confirmation run) aborted with "startup guard/readiness failure" at 18:48:56 PDT, 12 s after the post-load reclaim. This is the same pattern as boot 560283 (split ON), so the patch did not cause either.
- [measured] Across the last 8 boot directories: the two boots that aborted after load (560283, 600020) each have exactly one `nvidia-smi … Killed` line in `observer/observer.err`. The three boots that reached serving (505899, 549769, 568783) have none. 438368 and 430837 failed earlier for other, logged reasons.
- [measured, from code] `allocator-observer.sh` runs `set -Eeuo pipefail` and calls `nvidia-smi` (8 s timeout) every 5th sample. A timed-out call ends the script; `check_observers` in `start-dsv4-default.sh` then fails, `wait_for_api` returns 3, and the start script aborts.
- [inferred] `nvidia-smi` stalls for more than 8 s while the GPU is busy right after weight loading (KV / graph setup). The observer's death, not memory distress, ends the boot. The guards' memory rules never fired in either boot.
- Fix staged: `scratchpad/apply-observer-smi-fix.sh` (backup `allocator-observer.sh.bak-20261019-smi`) appends `|| printf 'nvidia-smi sample failed (rc=%s)\n' "$?"` to both nvidia-smi calls, so a timeout is recorded in the data and sampling continues. Tested: under `set -Eeuo pipefail` a timed-out command now prints rc 124 and the script continues. The start script copies the observer to the worker at each start, so one edit covers both nodes. Not yet applied.
- [measured] Observer fix applied by the user (2 `sample failed` handlers in `allocator-observer.sh`). The next boot, dsv4-t2-20261020T015949Z-607768 (split OFF), logged one `nvidia-smi sample failed` in the observer data, the observer stayed alive, and the boot reached ready in 160 s. This confirms the cause and the fix.

### E12b — Confirmation: same-client 1M with the split OFF (2026-10-19, run 607768) — SUCCESS
- [measured] Same client, same prompt (1,036,267 tokens, the same code 39623663): TTFT **1,010.3 s** (1,026 tok/s), code found. Lowest MemAvailable leader 16 GiB / worker 21 GiB, PSI ≤ 2, no watchdog, no ALERT. Data `data/E12/e12b-base-1M.json` / `.mon`.
- [measured] Split ON (813.7 s) vs OFF (1,010.3 s) on the identical prompt: **−19.5 % (−197 s)**. The OFF result also matches E10's 1,010.1 s with the other client, so the earlier caveat is resolved.
- **Verdict: SUCCESS.** The split indexer cuts 1M TTFT from 16.8 to 13.6 min. Exactness rests on the offline checks (E12a, E12b-precheck). Both 1M answers found the code. Recommended as the default.

## E13 — Leader vs worker GPU clocks under load (2026-10-19)
**Setup.** Run 607768 (split OFF). `nvidia-smi --query-gpu=… -lms 1000` on both nodes during one 263,915-token prompt (`tools/e12b_client.py 262145`): TTFT 143.9 s, 1,834 tok/s, code found. Data `data/E13/clk-leader.csv`, `clk-worker.csv`, `p256k.json`. Statistics are over samples with util ≥ 90 % (leader 146, worker 145).

| | leader | worker |
|---|---|---|
| graphics clock, median (min–max) | 2,418 MHz (2,268–2,476) | 2,483 MHz (2,444–2,522) |
| power, median | 74.2 W | 70.5 W |
| temperature, median (max) | 80 °C (88) | 72 °C (84) |
| temperature, first → last third | 72 → 82 °C | 63.5 → 78 °C |
| idle temperature before the run | 53 °C | 47 °C |
| throttle reasons (samples) | none 115, SW power cap 17, **SW thermal slowdown 13**, HW slowdown + HW thermal slowdown 1 | none 129, SW power cap 16 |

[measured] The leader runs 8 °C hotter, draws 3.7 W more at a 2.6 % lower median clock, and hits thermal slowdown (`0x20`, once `0x48`). The worker never does.

[inferred] The 2.6 % clock gap cannot by itself explain E11b's 13–15 % gap in the indexer logits kernel. In E11/E11b the gap is kernel-specific: logits 1143 vs 989 ms (+15.6 %), sparse MLA 936 vs 884 ms (+5.9 %), MoE FP4 GEMM 1380 vs 1388 ms (≈ 0). [hypothesis] The logits kernel writes a large fp32 [rows × keys] matrix (about 0.6 GB per call at 230K), so it may be memory-bandwidth sensitive, and the leader's unified memory is shared with the API server / EngineCore and guard processes. Not tested. A standalone kernel benchmark on each node with the server down would separate hardware from workload.

[inferred] The leader's higher idle and load temperatures point to cooling or airflow differences between the two boxes; worth a physical check.

### E13b — Standalone benchmark on both nodes, server down (2026-10-19)
**Setup.** Identical container and script on both nodes at the same time (`scratchpad/e13/e13_bench.py`, `run-e13b.sh`). Data `data/E13/e13b-leader.json`, `e13b-worker.json`.

| [measured] | leader | worker | leader vs worker |
|---|---|---|---|
| 2 GiB copy (read + write) | 246.3 GB/s | 246.7 GB/s | −0.2 % |
| write | 199.5 GB/s | 199.9 GB/s | −0.2 % |
| bf16 GEMM 8192³ | 96.5 TFLOP/s | 98.6 TFLOP/s | −2.1 % |
| indexer logits, 224K shape | 16.76 ms | 16.16 ms | +3.7 % slower |
| indexer logits, 1M shape | 75.33 ms | 72.78 ms | +3.5 % slower |
| top-k, 224K / 1M | 4.85 / 22.98 ms | 4.89 / 22.71 ms | ≈ 0 / +1.2 % |
| logits 1M, 60 s sustained (first → last) | 75.32 → 78.71 ms | 73.29 → 75.82 ms | last +3.8 % |
| end of run (clock, temp, power) | 2,112 MHz, 83 °C, 92.3 W | 2,190 MHz, 78 °C, 92.1 W | −3.6 % clock |

[measured] From the E11 traces: the logits kernel never overlaps other GPU work on either node (overlap fraction 0.000; single stream), and its median duration is 13.34 ms on the leader vs 12.82 ms on the worker (+4 %).

[inferred]
- Memory bandwidth is identical, so the "memory contention" hypothesis in E13 is not supported.
- With nothing else running, the leader is 2–4 % slower, which matches its lower clock at equal power and higher temperature.
- In serving (E11/E11b) the per-chunk logits gap grew over the run: chunk step 1 +3 %, step 9 +6 %, step 18 +15.6 %. Standalone, the gap is about 3.5 % at both the 224K and the 1M shape, so it does not grow with context size. A gap that grows with time under load fits progressive thermal throttling on the leader (E13: 72 → 82 °C, SW thermal slowdown in 9 % of busy samples). Not proven: the standalone sustained test lasted 60 s, while a prefill lasts minutes.
- Practical lever: leader cooling or airflow (physical check), or a clock cap that keeps both GPUs equal and cool (the operator's decision 1).

## 2026-10-19 — Worker-loss detection (staged)
**Finding.** [measured, from code]
- During serving nothing watched the worker. The guards are passive: they only latch ALERTs, and `start-dsv4-default.sh` exits once the API is ready.
- vLLM's `MultiprocWorkerMonitor` (`v1/executor/multiproc_executor.py`) waits only on the sentinels of local worker processes.
- [inferred] A lost worker therefore leaves the leader blocked at its next collective, with no alert. Not reproduced: that would need a real worker failure.

**Fix.** `dsv4-peerwatch.sh` (scratchpad `peerwatch/`, installed by `scratchpad/apply-peerwatch.sh`; backups `*.bak-20261019-peerwatch`).
- Started detached by the start script once the API is ready (`DSV4_PEERWATCH=0` skips it).
- Every 10 s it runs `docker inspect` of the worker container by exact ID over the CX7 ssh path.
  - Worker container not running or absent → `dsv4-abort.sh` at once (latches the ALERT on both nodes, exact-ID stops).
  - Unreachable 6 times in a row (about 60–90 s) → the same abort.
  - It exits by itself when the leader container stops or another run becomes active.
  - It logs to `runs/<id>/peerwatch.log` (only state changes).
- `launch-dsv4.sh down` kills the run's watcher before aborting, so a deliberate down is not reported as a lost worker.

**Test.** [measured] Dry-run mode (`DSV4_PEERWATCH_DRYRUN=1`, interval 1 s, misses 3), temp state dir, against the live run 647825:
- healthy → no action;
- nonexistent worker ID → "worker container absent" → would abort;
- unreachable peer 192.0.2.1 → 3 logged misses (rc 255) → would abort;
- an exited worker container (an old parked one) → "worker container not running (false exited 0)" → would abort;
- nonexistent leader ID → exits quietly;
- another active run → exits quietly.
- The live server stayed healthy (/health 200). The apply script was dry-run on copies: syntax OK, diff as intended. Not yet applied.
- [measured] Applied by the user 2026-10-19 20:39 PDT (`syntax OK`, diff as intended; `dsv4-peerwatch.sh` installed). Takes effect at the next start.

## 2026-10-20 — The operator's reply to the 2026-10-19 message [measured: the operator's text]
- Clock cap: leave uncapped while thermal throttling is not a problem; keep watching it (D8).
- DSpark: the rank-divergence deadlock and prefill at large contexts are still issues. The operator thinks the WiFi-roaming hangs are not the main DSpark problem and suspects TP=2 and GB10 compatibility. Go ahead when ready, after reviewing the PRs against vLLM main that tried to fix this and why they were not merged (D10).
- Patches: may be on by default; also see whether they can be merged into vLLM main so they are not bespoke to one model; read vLLM's CONTRIBUTING.md before opening anything (D9).
- The operator asks that the technical details of each change be understood well enough to explain (needed for upstream PRs).
- Next project queued by the operator: inference stability for another model. Not started.

## E14 — Re-profile at 256K with the split indexer ON (2026-10-20) · COMPLETE
**Setup.** Run dsv4-t2-20261020T225038Z-724557, started by the user with `DSV4_IDX_SPLIT=1 DSV4_PROFILE=1`. [measured] The patched `sparse_attn_indexer.py` is mounted in both containers (md5 ffcdc4a05c6e…); profiler config as E11 (torch, delay 10 / max 20 iterations). Method as E11: `POST /start_profile`, then `tools/run-guarded.sh 262144`.

**Request.** [measured] 263,521 prompt tokens, TTFT 169.7 s (1,553 tok/s, profiler capturing 20 steps), decode 25.9 tok/s. Lowest MemAvailable leader 14 GiB / worker 18 GiB, PSI ≤ 1, watchdog did not fire. Peerwatch ran live for the first time (started 15:55:47, no events). Data: `data/E14/` (`p256k.result`, `p256k.mon`, `leader-trace.json.gz`, `worker-trace.json.gz`, `profiler_out_0.txt`); analysis `scratchpad/e11.py`, `scratchpad/e14_other.py`.

**Step alignment.** [measured] Steps 1–5 match E11 within 1–2 % in every kernel class (indexer logits 365 / 506 ms at steps 1 / 5 against 361 / 505), so the 20 captured steps cover the same chunks as E11. The split engages from step 6 (first sub-chunks with ≥ 32,768 keys); from step 7 indexer logits drop to about 60 % of the step-5 level.

**Per-chunk breakdown, step 18 (about 230K prefix), leader.** [measured]

| kernel class (ms) | E11 split OFF | E14 split ON | change |
|---|---|---|---|
| indexer logits | 1143 | 545 | −598 (−52 %) |
| indexer top-k | 285 | 86 | −199 (−70 %) |
| sparse MLA | 936 | 910 | −26 |
| MoE FP4 GEMM + other | 1605 | 1564 | −41 |
| all-reduce / all-gather (NCCL) | 297 | 323 | +26 |
| other (not classified) | 405 | 545 | +140 |
| mHC, dense GEMM, zero_kv | 692 | 693 | 0 |
| **step wall** | **5347** | **4653** | **−694 (−13 %)** |

- [measured] Kernels per step at step 18: 2,334 (E11) → 6,702 (E14). The extra kernels are the torch merge in the patch: elementwise (46 ms), scatter/gather (29 ms), `radixSortKVInPlace` (10.5 ms, from `torch.topk`), `CatArrayBatchedCopy` (10.4 ms) and smaller ones. Together with the all-gather that is about 166 ms per chunk, i.e. about 2.6 ms per indexer call (63 calls per chunk), consistent with the offline merge time of about 3 ms (E12a).
- [measured] Top-k fell by more than half (−70 %): its cost per row grows faster than linearly with the number of keys (E11), so halving the keys more than halves it.
- [measured] The worker shows the same pattern (step 18: logits 989 → 484, top-k 287 → 82, wall 5347 → 4653). The leader's logits kernel is still slower than the worker's (545 vs 484 ms, +13 %), and the worker's NCCL time is higher (475 vs 323 ms). [inferred] The worker still waits for the leader at the collectives, as in E11b (E13: the leader runs hotter).
- [measured] GPU time over the 20 captured steps: 93.4 s → 86.5 s (−7.4 %). GPU busy ≈ step wall in every step, so the extra launches did not starve the GPU.
- [measured] Step 6 was 5,392 ms with every kernel class (including MoE, 1,757 against about 1,385 ms) about 27 % slower; step 7 onward normal. [hypothesis] A transient GPU slowdown (clock/thermal) or the first allocation of the merge buffers; clocks were not logged in this run.

**TTFT.** [measured] 169.7 s against 162.8 s in E11, although every captured step from 7 on is faster. [inferred] The difference is outside the captured steps (the other ~234 chunks). Not explained; this run did not log clocks, and E13 showed the leader slowing over time. Profiled TTFTs are not used for speed claims: the speed evidence stays E12b without the profiler (256K −5.5 %, 1M −19.5 % same client).

**Where 1M prefill time goes now.** [inferred, E11 model with the measured E14 indexer reduction; not profiled at 1M] Of 813.7 s: fixed per-chunk cost (MoE, mHC, dense GEMM, all-reduce, other) about 3.1 s × 128 chunks ≈ 395 s (≈ 49 %); indexer (logits + top-k, now halved) ≈ 240 s (≈ 30 %); sparse MLA ≈ 175 s (≈ 21 %); torch merge ≈ 0.17 s × about 110 split chunks ≈ 20 s (≈ 2 %). The sum (≈ 830 s) is within 2 % of the measured 813.7 s.

**Verdict.** The split indexer works as designed in serving: at about 230K context it halves indexer logits, cuts top-k by 70 % and saves about 0.7 s per chunk (−13 %), on both ranks. Remaining prefill targets at 1M, largest first: the fixed MoE FP4 cost (about 1.56 s per chunk, the largest single class), the still-dominant indexer, sparse MLA, then the merge (a fused merge kernel would recover at most about 2 % of 1M TTFT).

## 2026-10-20 — Upstream check for our two vLLM patches (research, read-only)
Notes: `reports/research_notes/vLLM upstream patches/notes.md` (every claim tagged with a URL or file:line).
- [measured: GitHub API] vLLM #54951 "[Perf][GLM-5.3-Flash] Shard long-context indexer prefill rows across TP" (zigzagcai) was merged 2026-10-11. It removes the same duplicated indexer work that our split targets, but splits **query rows** across TP ranks instead of keys. Each rank scores its rows against all keys and its final int32 top-k indices are all-gathered (`all_gatherv`), so there is no score merge and no tie question. It works for any TP and is generic to DSA indexer models.
- [measured: GitHub API] #61647 (merged 2026-10-15) gates it on **total** scheduled prefill rows through `AttentionConfig.indexer_tp_shard_min_rows`, default 65,536. [inferred] With our 8,192-token chunks it would not engage by default; it would need `indexer_tp_shard_min_rows` ≤ 8192.
- [from the notes, not re-verified by me] Our image (46638857) predates both. No CUDA key-split PR like ours exists upstream. A key-split variant for MiniMax (#57702) was closed unmerged.
- **Consequence:** our key-split patch would very likely be a duplicate under vLLM's contribution rules (`docs/contributing/README.md`, `AGENTS.md`: duplicate PRs must not be opened; AI-assisted PRs need full human review, disclosure, DCO sign-off and end-to-end benchmarks).
- **Better path** [hypothesis, untested]: use the upstream row sharding (newer image, or a backport of #54951 + #61647) with `indexer_tp_shard_min_rows: 8192`, A/B against our patch at 64K / 256K / 1M. Row sharding has no 166 ms/chunk merge (E14), so it may be equal or faster. Contribute GB10 / TP=2 measurements for the threshold upstream instead of our patch.
- GEMV compressor patch (`dsv4-attention.py`, +1.1 % decode): no upstream duplicate found; a weak case on its own (one platform, cross-boot measurement). It would need a GB10/SM12x gate and a same-boot benchmark plus accuracy eval, or could go as a cuBLAS report instead.

## 2026-10-20 — DSpark upstream survey (research, read-only) and a GB10 prefill-hang fix our image lacks
Notes: `reports/research_notes/DSpark upstream survey/notes.md`. Below, "[measured]" means checked by me in the running container or via the GitHub API; the rest is from the notes.
- [measured] vLLM #51538 "[Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark" was merged 2026-08-15. [from notes] It is in our image (vLLM `0.26.1rc1.dev1219+g46638857f`) but not in Anemll's `dspark-vllm-gx10:0.1.1` (vLLM 0.25.2.dev0). The operator's backport PR to Anemll (#12) is unreviewed; that repo is inactive. [measured] Our image's DSpark code is in `vllm/v1/worker/gpu/spec_decode/dspark/`.
- [from notes] PRs still open that bear on TP rank divergence: #56831 / #62769 (adaptive-verification sync), #49027 (cross-rank agreement before drafting), #50843 (out-of-vocabulary sampled token), #52499 (GB10 spec-decode shapes, waits for FlashInfer #4802), #52292 (autotune hang at multi-node start). There is no general TP consensus guard in vLLM. Clue from the operator's 660K incident: rank 0 alone did one extra logits gather of 6 × 64,640 elements, i.e. one rank took a drafting branch the other did not.
- [measured] FlashInfer #4732 "[bug fix] Issue 3700 sparse mla sm121 hang" was merged 2026-09-01. Issue #3700 is a flaky timeout of `test_sparse_mla_sm120` on Spark. The fix changes the producer/consumer named-barrier handshake in `sparse_mla_sm120/prefill_kernel.cuh` to alternate between two barrier IDs (`bar_sync_alt` / `bar_arrive_alt` in `arch/barrier.cuh`); arriving twice in one phase desynchronised the barrier.
- [measured] Our image's FlashInfer is `0.6.18.dev20260819` and does **not** have this fix: `barrier.cuh` has no `bar_sync_alt`, and `prefill_kernel.cuh` still uses the single-ID `bar_sync_t<1>` / `bar_arrive_t<1>` (lines 157, 522, 785, 826, 1517). That file defines `sparse_mla_prefill_mg_dual_kernel`, the sparse MLA prefill kernel in all our E11 / E14 traces.
- [inferred] Our DSpark-off serving runs the kernel with the known GB10 hang bug. It has not hung in our runs so far, but a hang would show as a GPU stuck busy (the 96 % wedge class) with the other rank waiting at a collective. It is a stability fix (G1) independent of DSpark, and it may also explain part of the operator's "prefill at large context" problem [hypothesis].

## E15 — FlashInfer #4732 backport: sparse MLA SM120/121 prefill hang fix (2026-10-20) · SUCCESS (offline test and serving check pass)
**Facts.** [measured]
- Our image's `sparse_mla_sm120/arch/barrier.cuh` and `prefill_kernel.cuh` are byte-identical to the PR's base (git blobs a06c86d / 1d2b344). The PR diff applies cleanly, and the results equal the PR's result blobs (7aaad22 / 9b0e8ac).
- FlashInfer loads the prebuilt module `flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so` whenever it exists (`jit/core.py` `try_load`), so header changes alone do nothing. The prebuilt `.so` holds sm_120 and sm_121a code. [inferred] It was built from the same 0819 sources, so it lacks the fix too; this cannot be read from the binary.
- The image has nvcc 13.0.88 and ninja. FlashInfer's JIT targets `-gencode=arch=compute_121a,code=sm_121a` on GB10.
- The bug, as read from the source (`patches/fi4732/README.md`): IO warps block on named barrier 1 per tile, while math warps only *arrive* on it and then proceed on the next tile's mbarrier. A delayed IO warp group lets the math warps arrive twice in one barrier phase. The fix alternates barrier IDs 1/5 by tile parity. Barrier ID 5 is otherwise unused in that file (used: 2, 3).

**Files.**
- `patches/fi4732/`: patched and original headers, the PR diff, FlashInfer's test at commit f1e2a5c, and a README.
- Scratchpad `fi4732/build.sh`: builds the module twice in a throwaway container with no GPU and no network (`orig` = image headers, as a control for the build pipeline; `fix` = #4732), and copies `fix` to `patches/dsv4-fi4732-sparse_mla_sm120.so`.
- Scratchpad `fi4732/test.sh`: runs `pytest -k prefill` of FlashInfer's test against three variants (`aot` = prebuilt, as served today; `orig`; `fix`) with a hard timeout, where a timeout counts as a hang. Output goes to `data/E15/`.
- `scratchpad/apply-fi4732.sh`: launcher wiring, `DSV4_FI4732=1` bind-mounts the `.so` read-only on both nodes; default 0. Dry-run on copies: syntax OK, diff as intended. Not applied.

**Plan.**
1. With the server down: build, then test.
2. Gate: `fix` passes the same tests as `aot`, with no new failures and no hang.
3. Then a serving run with `DSV4_FI4732=1`: 64K / 256K / 1M TTFT and the retrieval check, compared with E12b. Not expected to change speed (the same instructions apart from the barrier ID).
4. [hypothesis] The original hang is flaky (issue #3700: an intermittent CI timeout), so a clean A/B cannot prove the fix. It can only show that the fix does no harm. Evidence for the fix is the upstream analysis and merge.

### E15 — Build and offline test (2026-10-20, server down, leader GPU)
**Build.** [measured] Two builds, about 25 s each, both sm_121a only:
- `orig` .so md5 bb964162… (image headers);
- `fix` .so md5 23acbe9f… (#4732 headers), copied to `patches/dsv4-fi4732-sparse_mla_sm120.so`.

SASS check of the prefill kernels (`cuobjdump -sass`; arrive instructions counted):

| module | `BAR.ARV 0x1` | `BAR.ARV 0x5` | note |
|---|---|---|---|
| prebuilt (`aot`) | 120 | 0 | sm_120 + sm_121a copies |
| `orig` | 60 | 0 | |
| `fix` | 60 | 60 | the fix is in the binary |

**Test.** [measured] FlashInfer's `test_sparse_mla_sm120.py` (commit f1e2a5c), `pytest -k prefill` (163 tests), 3 rounds × 3 variants, 900 s hard timeout. Data: `data/E15/summary.tsv`, `test-*.log`.

| variant | round 1 | round 2 | round 3 | hangs |
|---|---|---|---|---|
| `aot` (prebuilt, what serving uses today) | 163 passed, 14 s | 163 passed, 13 s | **HANG** (after 131 passed; stuck in `test_sparse_mla_sm120_prefill_dsv4[True-256-128-1024]`) | 1 / 3 |
| `orig` (our build, image headers) | **HANG** (after 130; `prefill_dsv4[True-256-128-256]`) | **HANG** (after 20; `prefill_dsv3_2[True-256-64]`) | **HANG** (after 20; same test) | 3 / 3 |
| `fix` (#4732) | 163 passed, 12 s | 163 passed, 12 s | 163 passed, 13 s | 0 / 3 |

- Every hang: the GPU stayed at 96 % utilisation with the test process alive and no progress until the timeout killed the container. After each kill the GPU process was gone and the next run started normally. No reboot or reclaim was needed; at the end the GPU was at 0 % with MemAvailable 113 GiB.
- [inferred] The unfixed kernel hangs on GB10 in FlashInfer's own tests, including DSV4 prefill cases, in the prebuilt module we serve with (1 of 3 runs) and more often in our own build of the same source (3 of 3). The difference between the two unfixed builds is most likely code generation or timing (the prebuilt one has different build settings); not investigated. With the fix: 0 hangs in 3 runs and all 163 tests pass, the same results as the prebuilt module when it does not hang.
- [inferred] This is a real stability risk for DSV4 serving today. A hang in serving would show as the leader or worker GPU stuck at 96 % with the other rank waiting at a collective: the "96 % wedge" class seen before. Whether any past serving hang was this bug is not established.
- Limits: 3 runs per variant is a small sample, and "0 of 3" for the fix does not prove the hang is gone. The upstream analysis and merge are the main evidence that the fix is correct.

**Next.** The user applies `scratchpad/apply-fi4732.sh`, then starts with `DSV4_FI4732=1`. Serving check: verify the mount on both nodes (md5 23acbe9f…), then 64K / 256K / 1M TTFT and the retrieval check against E12b. If clean, make it the default (as with the split indexer).

### E15 — Serving check with the fix (2026-10-20, run dsv4-t2-20261021T005449Z-775571) — PASS
**Setup.** [measured] Started by the user with `DSV4_FI4732=1` after applying `apply-fi4732.sh` (backups `*.bak-20261020-fi4732`). `sparse_mla_sm120.so` md5 23acbe9f… in both containers; split indexer on (md5 ffcdc4a0…); profiler off; peerwatch running. Ready at 17:59:37 (boot about 4.8 min). Client `tools/e12b_client.py` with the same targets and prompts as E12b. Data: `data/E15/serve-*.json|.mon`; runner `scratchpad/fi4732/serve-check.sh`.

| context | prompt tokens | TTFT, E12b split (no fix) | TTFT, E15 split + fix | decode tok/s | code found |
|---|---|---|---|---|---|
| 64K | 65,873 | 29.7 s | 29.9 s (+0.7 %) | 30.62 | yes |
| 256K | 261,619 | 134.0 s | 134.9 s (+0.7 %) | 27.31 | yes |
| 1M | 1,036,267 | 813.7 s | 819.2 s (+0.7 %) | 30.49 | yes |

- [measured] Lowest MemAvailable: leader 16 GiB, worker 20 GiB. PSI 0 on both. No watchdog trigger, no ALERT, no peerwatch events.
- [inferred] Speed is unchanged within run-to-run noise: +0.7 % at every size, smaller than the boot-to-boot spread seen in E12b. The fix changes only which barrier ID is used on odd tiles. The answer text differs slightly from E12b in wording but the code is right at every size; cross-boot text is not comparable anyway (E12b).
- **Verdict.** The fix is safe to serve: same speed, same retrieval, and FlashInfer's own tests show it removes a hang that the prebuilt module hits (E15 offline). Recommend making `DSV4_FI4732=1` the default, staged as `apply-fi4732-default.sh` for the user to run.

## 2026-10-21 — Housekeeping: fstrim guard fix, thermal logging, nvtop
- **fstrim false alerts.** [measured] The leader's ALERT snapshot from 2026-10-19 (`runs/…445582/leader/snapshot.json`) lists one D-state PID, 489943. The journal shows `fstrim[489943]` ran 00:58:40–01:00:26. Fix staged as `scratchpad/apply-guard-fstrim.sh`: `D_STATE_IGNORE = {"fstrim"}`, matched on the exact `comm` from `/proc/PID/stat`. Tested on a fake `/proc`: fstrim in D is ignored, while a kworker in D and a process whose comm contains "fstrim)" are still reported. Not applied. Takes effect for guards started after it (the launcher copies the guard to the worker at each start).
- **Thermal logging (D8).** `tools/thermlog.sh` (nvidia-smi every 30 s: clocks, power, temperature, utilisation, `clocks_event_reasons`), one instance per host. Started 2026-10-21 21:48 on the leader (`data/thermal/`) and the worker (`~/dsv4-thermal/`). Summary: `tools/thermsum.py`.
- **nvtop.** [measured] the operator's nvtop (PID 78955, state Sl+) is running on the leader, not paused; none on the worker.
- **Message to the operator** drafted: `reports/Message to the operator 2026-10-21.md`.

## E16 — Backport of upstream indexer row sharding (vLLM #54951 + #61647) · PORTED, not run (2026-10-21)
Work dir: `scratchpad/upsplit/` (`PORT.md` = per-hunk notes, `port-combined.diff`, `tree/` git repo with commits base → port 54951 → port 61647, tests).
- [verified] Our three files (`model_executor/layers/sparse_attn_indexer.py`, `v1/attention/backends/mla/indexer.py`, `config/attention.py`) are byte-identical to upstream commit 46638857fdbb (our image's source).
- [measured] The PR diffs do not apply directly: 5 of 18 hunks fail, and others land only with fuzz 3 and offsets up to −332 lines. Ported by hand; 3 files, +186 / −17.
- Adaptations, each because our tree lacks upstream code:
  - no DSV4.1 `candidate_blocks` path;
  - no PCP+DCP KV all-gather block;
  - no `max_prefill_seq_len` field;
  - an added `pydantic.Field` import.

  The new functions and classes are source-identical to upstream. I reviewed the indexer and gate diffs. Each rank narrows its query rows and causal bounds per sub-chunk, the KV gathers stay unconditional, and one `all_gatherv` of int32 top-k indices runs per indexer call. Row counts come from CPU metadata (identical on both ranks), balanced by indexer cost.
- **Engagement in our setup** [from PORT.md, based on our launcher and the E08 boot log]:
  - requires TP > 1, DCP = 1, no PCP, PyNCCL on, `VLLM_USE_NCCL_SYMM_MEM` off, no batch-invariant mode, and a CUDA-graph mixed mode that is not FULL (ours resolves to FULL_AND_PIECEWISE, so mixed batches are PIECEWISE);
  - total prefill rows ≥ `indexer_tp_shard_min_rows`. That is set with `--attention-config '{"indexer_tp_shard_min_rows": 8192}'`; at the default 65,536 it never engages with our 8,192-token chunks.
- **Tests so far:** `py_compile` OK. A CPU simulation of the row bookkeeping (`tests/sim_row_shard_cpu.py`) matches the unsharded result bit for bit in 300 random cases at TP 2/3/4, and fails on a deliberately broken copy. Upstream's GPU test is ported (one argument removed) but not run.
- **Launcher constraint:** all three files must be mounted on both nodes. A leader/worker mismatch would make the ranks disagree on the collective, and NCCL would hang. It replaces our key-split file at the same path, so the two are mutually exclusive.
- **Next:**
  1. With the server down: upstream's test plus an exact-top-k check at DSV4 shapes in a throwaway container.
  2. Launcher wiring (a mode switch instead of `DSV4_IDX_SPLIT`).
  3. Serving A/B against our key split: 64K / 256K / 1M; the 1M reference is 813.7 s (E12b), 819.2 s with E15.

## I01 — Incident: leader network stack deadlock and orphaned worker (2026-10-22) · ROOT CAUSE FOUND (investigated 2026-10-23)
Run `dsv4-t2-20261021T005449Z-775571` (up since 2026-10-20 17:55 local).
- **No GPU fault.** [measured]
  - Leader: no Xid. The NVRM lines in boot `7c9042ce` stop at 2026-10-20 17:59 (boot-time `NV_ERR_NO_MEMORY`, a known class). None appear on 21–22 Oct.
  - The leader vLLM log has no errors. The last request was 2026-10-21 01:19 UTC. It shut down cleanly at 2026-10-23 00:20:11 UTC (17:20 local), exit 0, not OOM-killed.
  - Worker: the log is clean until 2026-10-23 00:20:12 UTC, one second after the leader powered off.
- **Trigger** [measured]: leader WiFi `wlP9s9` (MediaTek mt7925) had been roaming between 5 GHz APs every ~5 min since 09:31. At 10:25:40 it tried a 6 GHz BSSID (<bssid>, 6775 MHz), and wpa_supplicant logged "Authentication request to the driver failed".
- **Deadlock** [measured: hung-task traces, 10:28:15]:
  - `kworker/u82:0` (cfg80211_wiphy_work) holds the wiphy mutex and waits forever in `ieee80211_destroy_auth_data → … → mt7925_mac_sta_remove → mt7925_roc_abort_sync → cancel_work_sync`.
  - NetworkManager waits for the wiphy mutex in `nl80211_pre_doit`, while holding `rtnl` [inferred from the "likely owned by NetworkManager" lines].
  - Then wpa_supplicant, tailscaled, cups-browsed, rfkill_poll and the Ethernet `rtl8127_esd_task` (blocked in `rtnl_lock`) pile up behind it.
  - The work being cancelled presumably needs the wiphy mutex itself, i.e. a self-deadlock [hypothesis: an mt7925 driver bug, not checked upstream].
- **Guard:** the leader guard alerted `D-state>10s` at 10:25:53 (the stuck PIDs are the kworkers above). Alerting does not abort, so serving continued. vLLM uses CX7 (D6), which is not affected.
- **Recovery:** [measured] logind logged "Power key pressed short" at 17:20:11, i.e. a physical power-button shutdown. The operator logged in over tailscale at 17:21. [inferred] Remote access was dead, since tailscaled was itself stuck on the lock.
- **Orphan worker** [measured, 2026-10-23 09:22]: container `dsv4-worker` is still Up. `VLLM::Worker_TP1` (PID 810281) holds 93.6 GiB and logs TCPStore "Broken pipe" to 192.168.200.10:25000 every second (113k lines). Peerwatch runs on the leader only, so nothing tears the worker down when the leader dies.
- **Exposure:** the leader is back on the same 6 GHz AP (channel 165, 6775 MHz), so this can recur. The worker's mt7925 had a firmware message timeout and reset on 2026-10-21 19:46 (no deadlock).
- **Next:** (1) stop the orphaned worker container and reclaim its memory; (2) keep the leader's WiFi off 6 GHz or roaming, or put management on wired Ethernet; (3) give the worker a leader-loss watchdog; (4) check upstream for an mt7925 `roc_abort_sync` deadlock fix.
- **Cleanup (2026-10-23)** [measured]:
  - Stopped `dsv4-worker` (exit 0).
  - Ran `gb10-reclaim` on both nodes (rc 0). MemFree went to 118.7 GiB on the worker (from 24 GiB free) and 118.7 GiB on the leader; worker GPU 0 %, 5 W.
  - Stopped the stale worker guard (PID 801478).
  - The leader ALERT for run 775571 is still there; a human clears it before the next launch.

## I02 — Boot wedged at KV allocation on the leader, recovered without reboot (2026-10-23) · ROOT CAUSE NOT PROVEN
Run `dsv4-t2-20261023T162854Z-14489`, launched 09:28:55 local on default settings (DSV4_FI4732 was still 0).
- **Timeline** [measured, local time]:
  - 09:33:21 the leader rank reserved 10.5 GiB KV (1,094,398 tokens); the engine made no progress after that.
  - 09:33:24 the post-load `gb10-reclaim` ran (drop_caches).
  - 09:33:58 the guard alerted `boot:psi_some>40_for_30s`. At the alert the leader showed memory PSI some/full 87/87, MemAvailable 29 GiB, and `nvidia-smi` calls in D-state in `os_acquire_rwlock_write` (the GPU driver lock).
  - The coordinated abort stopped the worker cleanly. On the leader, `docker stop -t 120` returned, but `VLLM::Worker_TP0` (PID 18420) stayed in R state using ~10 CPUs, mostly system time, with no signal pending.
  - Memory PSI stayed at 92–95 % for ~11 min; 16 `nvidia-smi` processes queued on the driver lock.
  - 09:43 `docker kill`: the container was reported exited, but PID 18420 kept running in the kernel.
  - `/proc/buddyinfo` (Normal zone) had no free blocks of order ≥ 9 throughout.
- **Recovery** [measured]: at 09:45 `gb10-reclaim` (drop_caches + compact_memory) created high-order blocks (order 9–13: 3206/980/172/20/3). Within 10 s PID 18420 exited (Z, then gone), D-state count went to 0 and PSI fell. A second reclaim on both nodes left MemFree 118.9 GiB each; both GPUs idle at ~4.5 W. **No reboot was needed.**
- **Interpretation** [inferred]: failure class 2 from memory (high-order fragmentation → direct compaction under the GPU driver lock). The driver was retrying high-order allocations that could not succeed until compaction freed contiguous blocks. Why this boot fragmented, when the same config booted cleanly on 19–20 Oct, is not established. Differences [measured]: leader uptime 16 h after a cold power-off, and the thermal logger (D8) plus allocator observer were polling `nvidia-smi` every 30 s; whether that polling matters is [hypothesis].
- **Lesson** [measured, single instance]: for this wedge, an explicit compaction released it where SIGKILL could not. A guard action of "compact first, abort second" in boot mode is a candidate change; not made.

## 2026-10-23 — Finishing items 6–9
- **6. FlashInfer #4732 fix on by default** (applied): `node-launch.sh` and `launch-dsv4.sh` default `DSV4_FI4732` to 1; `DSV4_FI4732=0` disables. The .so (md5 23acbe9f…, same as E15) is copied to the worker on each launch. Backups `*.bak-20261023`.
- **7. Leader WiFi pinned to a 5 GHz BSSID**: needs root; command handed to the user. Not applied yet.
- **8. Worker-side leader watch** (applied): `dsv4-leaderwatch.sh`, copied to the worker on each launch and started after serving. It probes `http://192.168.200.10:8888/health` every 10 s. After 18 consecutive failures (3 min) it latches the worker ALERT and stops the worker container by exact ID. It exits when the worker container stops; `DSV4_LEADERWATCH=0` disables it. Tested on the worker with a throwaway container in dry-run mode: leader down → action after 3 misses (test setting); healthy stand-in endpoint → silent; worker container stopped → quiet exit. Not yet run in a real serving run.
- **9. fstrim guard exception**: an edit to the guard was blocked by the session's safety policy (it modifies the safety monitor). Staged as `staged/apply-guard-fstrim.sh` for the user to run. Applied to a copy, it compiles, and against a fake `/proc` it reports D-state for `kworker/u8:1` and for a comm `x fstrim)`, but not for `fstrim`.
- **First boot with items 6–9 in place** (run `dsv4-t2-20261023T172856Z-26506`, launched 10:28:56 local) [measured]:
  - Serving at 10:35:00 (6 min 4 s); peak leader memory PSI some avg10 2.7, so no stall. The watcher's stall clear-up (reclaim on PSI > 40 for 15 s) never fired.
  - FlashInfer #4732 .so mounted on both nodes.
  - Leaderwatch running on the worker and peerwatch on the leader, both started 10:34:58.
  - Leader WiFi pinned to BSSID <bssid> (5785 MHz).
  - Test request "17*23" answered "391" in 0.16 s.

## F01 — Final-settings check and 24 h soak (2026-10-23, run dsv4-t2-20261023T172856Z-26506)
- **4K decode check** [measured]: after one warm-up, `ctxbench.py 4096 512` ×3 gave 26.98, 26.89 and 26.83 tok/s. Median **26.89 tok/s**, against 26.93 in E07.8 (−0.15 %, inside the run-to-run spread). Prefill 2,239–2,249 tok/s (E07.8: 2,227–2,234). The FlashInfer #4732 fix, the guard change and the watchdogs did not change speed. Data `data/F01/4k-baseline.jsonl`.
- **Soak**: `tools/soak.sh` started 10:41 local; one sample every 5 min for 24 h, written to `data/F01/soak-<run>.log`. It exits on the first anomaly. Checks: API /health; both containers running; no ALERT on either node; memory PSI ≤ 10 and MemAvailable ≥ 6 GiB on both; no D-state except fstrim; peerwatch and leaderwatch alive; leader WiFi BSSID; no new hung-task, mt7925, NVRM or Xid kernel lines.
- **Soak restart, 11:25 local**: the first soak stopped at 11:25:17 on `D=1`, but the follow-up listing found no D-state process. It was a transient I/O wait, not a stall; the samples from 10:40 to 11:25 were otherwise clean. The check now counts a PID only if it is still in D-state 15 s later, and the 24 h window restarts from this point.

### F01 — 2026-10-23 14:52 soak monitor detached [measured]
- The user is leaving and their laptop goes offline, so the soak monitor no longer runs as a Claude-session background task. It now runs detached on the leader: `setsid nohup tools/soak.sh dsv4-t2-20261023T172856Z-26506 21` (PID 107446, own session).
- The samples from 11:25:34 to 14:50:52 were all clean. The new monitor started at 14:52:35 and its first sample was clean. It ends about 11:52 on 2026-10-24, covering 24 h 27 min in total with a gap of about 2 min.
- On an anomaly it writes `ANOMALY:` to `data/F01/soak-<run>.log` and exits. It never stops or changes the server.

### F01 — 2026-10-23 14:55 inference probe added [measured]
- The soak only checks /health, so a separate detached probe now exercises generation: `tools/infer-probe.py <run> 20.9 1800` (PID 108380). Every 30 min it sends one fixed short prompt with 128 new tokens, ignore_eos and temperature 0. Each result is written to `data/F01/probe-<run>.jsonl`.
- The probe is read-only and logs errors without stopping. The rate it reports is end-to-end (including prefill), so it reads slightly below pure decode.
- First probe at 14:55:50: ok, 128 tokens in 5.05 s, 25.33 tok/s end-to-end. This is consistent with the 26.89 tok/s 4K decode baseline.

### I03 — 2026-10-23 19:33 worker power-off; soak F01 ended by an external event
- **[measured]** The worker (gx10, ASUS) journal shows `systemd-logind: Power key pressed short.` → `Powering off...` at 19:33:35 and an orderly systemd power-off at 19:33:46. The next boot was at 19:34:27.
  - The user reports that "it crashed", then they turned it on.
  - Just before the press, the worker looked healthy:
    - soak sample at 19:33:13: health 200, both containers running, PSI 0, D=0, no kernel lines;
    - inference probe at 19:25:50: 25.75 tok/s;
    - the previous boot shows no Xid, hung-task or panic lines;
    - pstore is empty.
  - **[inferred]** The machine had not kernel-crashed. Whatever looked like a crash, the power-off itself came from the short power-button press.
  - **[hypothesis]** The worker's WiFi (wlP9s9) had been disconnected since 10:24:13 (`need-auth -> failed (no-secrets)`), so the worker was unreachable over WiFi/Tailscale all afternoon. Serving was unaffected because it uses CX7. This, or a blank or slept desktop, may have looked like a crash.
- **[measured]** The watchdogs behaved as designed:
  - peerwatch: worker unreachable from 19:34; `ACTION: worker container not running (exited 137)` at 19:34:36; dsv4-abort stopped both sides (exit 0); ALERTs latched on both nodes.
  - soak: `ANOMALY: API /health=000` at 19:38:14, then exit.
  - The inference probe was stopped by hand at 19:38. Its 10 probes from 14:55 to 19:25 were all ok, at 25.33–25.79 tok/s.
- **F01 result:** 8 h 08 min clean (11:25:34–19:33:13, with a 2 min gap at 14:50–14:52). That is not the full 24 h. The interruption was external, not a server fault.
- Cleanup: leader `gb10-reclaim` at 19:4x took MemFree to 118.8 GiB. The worker had a fresh boot with 117 GiB free.
- To resume: a human clears the ALERTs on both nodes and relaunches; then a fresh 24 h soak.

#### I03 addendum — full investigation (2026-10-23 19:5x) [measured unless tagged]
- The user's account is that the model crashed. Every source checked shows the stack healthy up to the button press:
  - **Worker guard heartbeat** at 19:33:34.31, 1.2 s before `Power key pressed short` (19:33:35): status ok, PSI 0/0, no D-state, MemAvailable 24.6 GiB.
  - **Worker vLLM:** 0 error, exception, traceback or timeout lines in the whole run (17:31–19:33). Its first line after warmup is the shutdown, `Parent process exited`, at 19:33:36, 1 s after the press. Exit 137 came from the shutdown SIGKILL; OOMKilled=false.
  - **Leader vLLM:**
    - every request returned 200: 24 POSTs and 118 GETs from localhost, plus 3,226 /health probes from the worker's leaderwatch;
    - there were no external clients, so the model was not in use when the press happened;
    - the last completion was at 19:25:55, then the server was idle;
    - the first non-routine line is the peerwatch-triggered shutdown at 19:34:37.
  - **Worker journal 17:00–19:33:35:** no OOM, kills, segfaults, NVRM/Xid, GPU, display or gnome lines. The only kernel lines are apparmor denials of a snap firmware-notifier at 18:00.
  - **Worker sar:** CPU 99.2 % idle and about 24.4 GiB free memory, flat at 19:10, 19:20 and 19:30.
  - **Leader kernel:** CX7 link down at 19:33:47, which is the worker power-off. Nothing before that.
- **[inferred]** The model and the worker did not crash. The outage was the short power-button press, which systemd-logind handles as a power-off by default.
- **[inferred]** The worker had no internet since 10:24:13 (WiFi `no-secrets`; ubuntupro at 18:51: `Temporary failure in name resolution`), so the operator saw it offline over Tailscale for about 9 h. That is a likely reason it looked crashed. It came back online at 19:34:41 after the reboot.
- The earlier real failures (I01: leader mt7925 WiFi deadlock; I02: boot-time memory wedge) are different mechanisms and are unaffected by this.
- Prevention options (root; for the user to decide):
  1. A wired network cable, or the WiFi secret stored system-wide, so the worker does not drop off Tailscale.
  2. Optionally `HandlePowerKey=ignore` (or `suspend`) in logind.conf on both nodes, so a short press cannot take down a serving run.

#### I03 — worker WiFi fix staged (power-button change declined by the user)
- **[measured]** The worker's WiFi profile already stores the PSK system-wide (psk-flags=0, no user permissions). The 10:24 drop was a 4-way handshake timeout (`CTRL-EVENT-DISCONNECTED reason=15`, 6 GHz at 6775 MHz, <bssid>). NetworkManager treated it as a bad key (`asking for new key` → `no-secrets`) and stopped autoconnecting.
- Staged `staged/apply-worker-wifi-keepalive.sh` (md5 b6ca08b9…, copied to gx10:~/). It installs a root systemd timer that runs every 5 min (first run 5 min after boot) and does `nmcli connection up "<home WiFi SSID>"` only when wlP9s9 is not connected. It also sets autoconnect-retries to 0 (unlimited). It touches WiFi only. The user runs it.
- **[hypothesis]** A risk to watch: reconnect attempts exercise mt7925, the driver behind the I01 deadlock on the leader. The timer only acts while the link is already down. A wired cable remains the stronger fix.

#### I03 — worker WiFi keepalive applied (2026-10-24 ~11:03) [measured]
- The user ran the staged script on gx10. Results: `wifi-keepalive.timer` active, first run 11:03:27 and every 5 min after; autoconnect-retries=0; wlP9s9 connected. There are no wifi-keepalive journal entries yet because the link was up, so the timer has had nothing to do.

### 2026-10-24 — Relaunch after I03; orphaned worker from an interrupted start [measured unless tagged]
- **Run `dsv4-t2-20261024T175931Z-180783`:**
  - The user started it while both ALERTs were still latched, then pressed Ctrl+C in the launcher.
  - It had reached `post-load reclaim done on both nodes` at 11:03:55.
  - The leader container got a vLLM `shutdown triggered` at 11:05:04 and exited 0.
  - The worker container `f59e485cf3c7` kept running, holding 89,852 MiB on the GPU, until the next launch's preflight stopped it at 11:11:29 (the user answered y).
- **[hypothesis]** Ctrl+C sends SIGINT to the whole foreground process group. `on_signal` → `latch_abort` runs `dsv4-abort.sh` in that same group, so the abort is interrupted too: the leader stop completed, but the worker stop (over ssh) did not. Not yet proven.
  - Candidate fix, not applied: ignore INT/TERM inside `on_signal` while the abort runs, or start the abort with `setsid`.
  - Mitigations already in place: leaderwatch is only started after readiness, so it did not cover this case; the next launch's preflight caught the orphan.
- The user cleared the ALERTs on both nodes. Run `dsv4-t2-20261024T181019Z-188432` started at 11:10:19 and is booting.

### F02 — 24 h soak, second attempt (run dsv4-t2-20261024T181019Z-188432) [measured]
- The run was ready by about 11:14 on 2026-10-24. peerwatch (PID 197344) and leaderwatch (worker PID 71066) were alive.
- 4K check: `ctxbench.py 4096 512` gave decode **26.87 tok/s** and prefill 2,153 tok/s with TTFT 1.8 s. This was the first request after boot, with no warm-up. It matches F01's 26.89.
- Soak (`tools/soak.sh <run> 24`, PID 197837) and inference probe (`tools/infer-probe.py <run> 24 1800`, PID 197838) both run detached, with `setsid nohup`. They started at 11:16:36 and run to about 11:16 on 2026-10-25.
- First samples were clean. The first probe returned 128 tokens at 25.33 tok/s end-to-end.
- Output files: `data/F01/soak-<run>.log` and `data/F01/probe-<run>.jsonl`.

#### Correction (2026-10-24, same day) — the orphan was not caused by Ctrl+C [measured]
- The Ctrl+C hypothesis above is **wrong**.
  - `~/.cache/dsv4-t2/abort/dsv4-t2-20261024T175931Z-180783/` shows the abort began at 11:04:01 with reason `startup guard/readiness failure`. That was the launcher refusing because the ALERTs were still latched.
  - It got leader=stopped and worker=`unknown`: `worker.inspect.rc`=137 with empty output.
- Timeline from the worker sshd log: latch at 11:04:01; cancel-owned at 11:04:02; worker `docker inspect` ssh accepted at 11:04:32, then killed by `timeout --kill-after=2 30` at about 11:05:04.
  - With an unknown worker record, dsv4-abort.sh deliberately stops nothing on the worker (it never stops what it cannot identify). That left container f59e485cf3c7 running.
- **Why the remote `docker inspect` took more than 30 s: not proven.**
  - Worker guard at 11:05:37: ok, MemAvailable 24.4 GiB, PSI 0, no D-state.
  - sar at 11:10: load 0.06, 0 blocked.
  - There are no dockerd journal entries.
- Staged fix (not applied): `staged/apply-abort-retry.sh`. It retries the worker inspect up to 3 times when the result is neither success nor Docker's not-found. It leaves the never-stop-unknown rule unchanged.

### 2026-10-24 — Prep work during the F02 soak (nothing touched the running server)
- **`tools/final-bench.sh [label]`** (new): one command for the final benchmark.
  - Sequence: 4K warm-up (discarded), 4K ×3, 64K ×2, 256K ×1, then 1M ×1 at target 1,040,000.
  - Each request runs through `run-guarded.sh` with ctxbench 256 new tokens, the same method as the scoreboard. It stops on a watchdog abort or a missing result.
  - It records run ID, image, vLLM args, mounts and patch md5s in `config.txt`, and writes a median table plus `summary.json` to `data/FINAL/<ts>/`.
  - It is read-only towards the server. The syntax and the summary logic were tested on synthetic results; it has not been run against the server.
- **E16 port lost:** the old scratchpad was wiped by the 2026-10-22 leader reboot, and no copy exists on either node. It is being redone in `scratchpad/upsplit/` from the image's files. Only `docker create`/`cp` is used; no container runs.
- **`staged/apply-abort-retry.sh`:** see the correction above. Applied and diffed on a scratch copy only.

#### Record fix — fstrim guard exception was applied (2026-10-23 10:27) [measured]
- The user ran `staged/apply-guard-fstrim.sh` on 2026-10-23. Evidence:
  - backup `dsv4-t2-guard.py.bak-20261023T102745`;
  - `D_STATE_IGNORE = frozenset({"fstrim"})` at line 32 of the leader guard;
  - the exception is also present in the worker's `~/.cache/dsv4-t2/guard.py` and `~/dsv4-t2-guard.py`.
- Earlier entries left this as "staged".
- Also confirmed from the running leader's environment: NCCL debug logging is still on (`NCCL_DEBUG=INFO`, subsystems INIT,NET,COLL).

### 2026-10-24 — Final report draft
- `reports/DSV4 final report (draft).md` was written from this log. Benchmark, E16 and F02 values are TBD. It was corrected for the fstrim status and for the orphan cause.

### 2026-10-24 11:26 — Live WiFi roam with a subnet change during serving (F02) [measured]
- The leader roamed despite the profile's BSSID pin (`802-11-wireless.bssid` is still <bssid>).
  - Kernel: `disconnect from AP <bssid> for new auth to <bssid>` at 11:26:08 (5745 MHz).
  - The DHCP address changed from 10.0.0.239 to 192.168.86.215, a different subnet. [inferred] The same SSID is served by two different routers.
- No mt7925 or hung-task kernel lines followed, and rtnl was not stuck.
- Serving was unaffected: the soak at 11:26:38 showed health 200, both containers running, no alerts, PSI 0, and the watchdogs alive.
- **This is the first observed live roam with a subnet change while serving.** It confirms E09 (vLLM control plane pinned to CX7) and closes the G5 caveat.
- **[inferred]** The BSSID pin does not stop supplicant-initiated roaming, so the I01 mitigation is weaker than recorded. Leaderwatch still covers the orphan case.

### E16 — Port re-created (2026-10-24) · PORTED, CPU-tested, not run on GPU
Now kept in persistent storage: `/srv/dsv4/run/patches/e16-upstream-rowshard/`. It contains `PORT.md`, `port-combined.diff`, `tree/` (git: base a9298c1 → #54951 23eeda9 → #61647 694f448), `tests/` and `upstream/`. It is no longer in /tmp.
- **[measured]** Base files:
  - the image's vllm is at `/usr/local/lib/python3.12/dist-packages/vllm`, commit 46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3;
  - the image's three files are byte-identical to upstream at that commit;
  - the image's `sparse_attn_indexer.py` is byte-identical to `patches/dsv4-sparse-attn-indexer.orig.py`.
  - The port is on the image original, not on our key-split file.
- **Diff:** +185/−17 across three files: `config/attention.py` +5/−1, `sparse_attn_indexer.py` +107/−13, `indexer.py` +73/−3.
  - Of 19 hunks, 7 were hand-ported. #54951's 9 GLM-only hunks (`models/glm5next`, absent at our base) were skipped.
  - New functions, classes and fields are AST-identical to upstream at 24f632ca. The earlier lost port was +186/−17, and the one-line difference is unexplained.
- **Tests:**
  - `py_compile` OK.
  - CPU simulation driving the real `PrefillRowShard` class: 400/400 match at each of TP 2, 3 and 4. 4 deliberately broken variants all fail.
  - Host has no torch or numpy, so the balancing function was transcribed into plain Python.
  - The upstream GPU test was ported (`tests/test_indexer_tp_row_shard.py`) but not run.
- **Engagement, verified against the code:**
  - CUDA, TP > 1, DCP = 1, no PCP, PyNCCL on, no NCCL symm-mem, no batch-invariant mode, and mixed CUDA-graph mode not FULL. Ours is FULL_AND_PIECEWISE, which is allowed.
  - Total prefill rows in the step must be ≥ max(`indexer_tp_shard_min_rows`, TP).
  - **[inferred]** With 8,192-token batches and min_rows 8192, it engages only on all-prefill steps. That matches our single-request long prefill.
  - The profile run reserves an extra 2 × rows × width × 4 B when the config gate passes.
- **Deployment:**
  - Mount all three ported files on both nodes, with `DSV4_IDX_SPLIT=0`. Our key-split file mounts at the same path, so the two are mutually exclusive.
  - Add `--attention-config '{"indexer_tp_shard_min_rows": 8192}'`.
  - The launcher needs a mode switch for this; not written yet.
- **Next, after F02:** run the upstream GPU test in a throwaway container with the server down, then the serving A/B at 64K, 256K and 1M against our key split (1M reference 819.2 s, E15).

### 2026-10-24 — Launcher switches, E16 test tooling, GEMV accuracy tool (during F02; running server untouched)
Backups of the launchers are `*.bak-20261024T114923` (node-launch.sh, launch-dsv4.sh, start-dsv4-default.sh). The changes take effect at the next start only.

- **Indexer mode switch** (`node-launch.sh`): `DSV4_IDX_MODE=split|rowshard|off`.
  - The default is unchanged: `split`, mounting the same file (md5 ffcdc4a0…).
  - Legacy `DSV4_IDX_SPLIT=0` with no mode means `off`.
  - `rowshard` mounts `patches/dsv4-rowshard-{sparse-attn-indexer,mla-indexer,config-attention}.py` (md5 81456b6a / 98276bb8 / 92d09a11, copied from the E16 tree) and adds `--attention-config '{"indexer_tp_shard_min_rows": N}'`, with N from `DSV4_ROWSHARD_MIN_ROWS` (default 8192).
  - Each node prints `node-launch: indexer mode=… files: <md5s>`. `launch-dsv4.sh` now copies the rowshard files to the worker, passes both variables, and **aborts the start if the leader's and worker's indexer lines differ** (mismatched collectives would hang NCCL).
  - Tested in isolation for all modes, legacy settings and invalid values. A real launch has not been run.
- **Diagnostics switch** (`start-dsv4-default.sh`): `DSV4_DIAG=0|1`, previously hard-coded to 1. The default stays 1 until the overhead is measured (4K A/B planned). The start banner now prints the indexer mode and the diagnostics setting.
- **E16 unit tests [measured]:** upstream's `test_indexer_tp_row_shard.py`, ported, run on the ported files in a throwaway container from the serving image (`--network none`, no GPU): **12 passed** in 14.8 s.
  - The tests are CPU-only, using a fake TP group.
  - The `--memory 4g` cap was ignored by this kernel ("does not support memory limit"); leader MemAvailable stayed about 21 GiB.
  - This replaces the planned "GPU test with the server down".
- **`tools/e16-ab.sh [ctx…]`:**
  - It detects the running indexer mode from the leader's mounts and refuses if the worker's indexer mounts differ.
  - Then it runs `e12b_client.py` at 64K ×2, 256K and 1M (target 1,036,288) under the run-guarded watchdog rules, writing to `data/E16/<ts>-<mode>/`.
  - Mode detection was tested against the running server (split, mounts match). No requests have been sent.
  - **[measured from code]** The ported row-sharding emits no log line when it engages, so engagement must be shown by TTFT or the profiler.
- **`tools/gemv-acc.py`:** run / compare. Fixed prompts (40 arithmetic, 12 facts, 2 open-ended), greedy, 128 tokens, top-5 logprobs.
  - Scores: exact-answer accuracy, token identity, first divergence, and mean |Δlogprob| on the shared prefix.
  - Two passes on one boot give the noise floor. The patch-on run on the current boot started 2026-10-24 about 11:55.

#### GEMV accuracy, patch-on half (run 188432, 2026-10-24 11:53) [measured]
- `gemv-acc.py run patch-on 2` → `data/GEMV/patch-on-20261024T115338.json`.
- Accuracy was 41/52 on both passes: 12/12 facts and 29/40 arithmetic. All 11 misses are multi-digit products answered without thinking; the errors are close-but-wrong numbers.
- **Same-boot noise floor:** the two passes were token-identical on 54/54 prompts, with mean |Δlogprob| 0.00000. Greedy decode is fully deterministic within one boot.
- The run took about 2 min, interleaved with the soak; the soak stayed clean.
- Next: restart with `DSV4_PATCH=0`, run `gemv-acc.py run patch-off 2`, then compare. Because E12b showed cross-boot divergence with an unchanged config, a cross-boot "patch-on again" pass would be needed to separate the patch effect from boot-to-boot noise.

### F02 soak status check, 2026-10-24 19:31
- [measured] soak.sh exited at 12:11:42 with ANOMALY "ALERT present (leader=2 worker=0)", about 55 min into the run. The leader ALERT (global + run) was written at 12:09:10 with reason `D-state>10s` by guard 188460 (current run). The flagged PID was 26636.
- [measured] PID 26636 is the leader guard from the *previous* run dsv4-t2-20261023T172856Z-26506. It started Oct 23 10:30 and is still alive (state S, current.json updated 19:31). Its run's containers were replaced at 11:10, but it never exited. Its own last snapshot (11:03:54) is the residual_growth+psi_stall alert from the refused start.
- [measured] Serving was not affected. Health is 200, both containers have been up for 8 h, and the worker WiFi is connected. infer-probe has run 17/17 probes OK since 11:16. Sixteen ran at about 26 tok/s; one, at 12:16:36, was slow (14.8 s, 8.7 tok/s, cause not identified).
- [inferred] The alert is a false positive from a leaked guard process, not a model or kernel fault. Kernel log 12:00–12:15: no entries. What put the old guard into D-state for >10 s is unknown.
- [hypothesis] The launcher/abort path does not stop the old run's guard when a new run replaces it. This needs a fix (stop guards whose run-id ≠ current run) before handover. Guard edits must be staged for the user.
- Consequence: the F02 soak monitor covered only 55 min. The 24 h evidence so far is the probe series (8 h, 17/17 OK). Restarting the soak needs a human to clear the alert and to stop PID 26636.

### Stale guard prevention, 2026-10-24 ~19:50
- [measured] Root path: start-dsv4-default.sh stops guards only for its *own* run id (stop_guards), and `launch-dsv4.sh down` only for the run being brought down. The 10-23 run was never brought down: the 10-24 preflight stopped its containers directly. The preflight excludes guard processes from the blocker list by design, so the old leader guard survived. The worker's old guard was gone because the worker rebooted on 10-23.
- Staged, not applied: `staged/apply-retire-stale-watchers.sh` (no sudo) inserts `staged/retire-stale-watchers.block` right after the blocker preflight. That is the point where no DSV4 container runs on either node. It TERMs every `guard.py run` / `dsv4-leaderwatch.sh` process on both nodes whose --run-id differs from the new run, waits up to 10 s, and fails the start if any remain. Observers are not touched; they carry a deadline.
- [measured] Tested on a scratch copy with fake processes: old guard and leaderwatch stopped, current-run process kept, no-op when nothing is stale, fail-closed when a process ignores TERM; real guards untouched. The apply script was tested on a copy: inserts 27 lines, idempotent, `bash -n` OK.
- Applied: tools/soak.sh (backup soak.sh.bak-*) now counts guards from other runs (`stale_guards=`) and flags them as an ANOMALY on the first tick. Against the current state it reports 1 (PID 26636).

### F02 soak restarted, 2026-10-24 20:15
- [measured] The user stopped the stale guard 26636, cleared the alerts on both nodes and applied apply-retire-stale-watchers.sh (start-dsv4-default.sh has the block; `bash -n` OK). Running now: only this run's guards (leader 188460, worker 45145), peerwatch 197344 and leaderwatch 71066.
- The earlier F02 files were kept as `soak-<run>.part1.log` (11:16–12:11) and `probe-<run>.part1.jsonl` (17 probes, all OK).
- [measured] soak.sh (PID 444181) and infer-probe.py (PID 444182) were restarted detached for 24 h; they end about 20:15 on 2026-10-25. First tick all clear, stale_guards=0; first probe OK at 25.97 tok/s. One tick showed peerwatch=2; only one peerwatch process exists, so this was most likely a transient child process being counted.

### F02 status, 2026-10-25 ~11:00
- [measured] Soak part 2 ran 2026-10-24 20:15 → 2026-10-25 09:47:32: 162 clean ticks (13.5 h), then `ANOMALY: D-state processes:` with an empty list. One PID was in D for the 15 s recheck but was gone when the message was built, so its name was not recorded. Guard alerts: none on either node. Kernel log 09:40–09:55: only a WiFi reassociation at 09:40:03 to the pinned BSSID. Kept as `soak-<run>.part2.log`.
- [measured] Other findings in part 2:
  - 23 ticks (00:51–09:37) on BSSID <bssid> instead of the pinned <bssid>, with no effect on serving. This is the BSSID pin not holding again.
  - 8 ticks showed peerwatch=2; only one peerwatch process exists.
- [measured] infer-probe: 30/30 OK, 25.91–26.21 tok/s, still running until about 20:15. Containers up 24 h.
- [inferred] The anomaly was a short I/O wait, not a fault. The guard (10 s D-state rule, 2 s sampling) did not alert.
- Change to tools/soak.sh (backup soak.sh.bak-*): D-state is now fatal only if the same PID is in D at 0, 15 and 45 s. Names and wchan come from the samples themselves. A 15–45 s D is logged as `note:`. Tested with stubbed samples (persistent → anomaly, transient → note, none → clear).
- Soak part 3 started at about 11:00 for 9 h, to end with the probe at about 20:00–20:15. F02 coverage: 20:15–09:47 and 11:00–20:00 by soak, the full 24 h by probe.

### Local git repo, 2026-10-25 ~11:20
- [measured] /srv/dsv4/run is now a local git repo (`git init --shared=group`, branch main, no remote). Author: the user account (e-mail redacted) (repo-local config). First commit a6c6e31 "DSV4 serving state at F02 soak": 77 files.
- Not tracked (.gitignore): data/, *.bak-*, __pycache__, patches/e16-upstream-rowshard/image/ (extracted from the image), patches/fi4732/build/ (the runtime .so is tracked). The nested E16 tree repo is kept as `tree.bundle` (verified).
- reports/ holds copies of the two reports. `tools/commit-snapshot.sh "msg"` refreshes them from ~/reports and commits.
- A scan for secrets found only sudoers/NOPASSWD rule text and comments, no credentials. /srv/dsv4/live and the other /srv/dsv4 directories are untouched.

### Documentation pass, 2026-10-25 ~12:00
- New: README.md (overview, results, quick start, API, configuration, architecture, safety model, layout, benchmarking, known issues, upstream status), docs/operations.md (runbook), docs/patches.md (each patch: problem, change, result, validation, md5s).
- Comment- and docstring-only changes, checked by `bash -n` and by an identical non-comment diff (bash) or an identical AST (Python): start-dsv4-default.sh, launch-dsv4.sh, cx7-nccl-env.sh, node-prep.sh, tools/idle-probe.sh, idle-sampler.sh, st.sh, ctxbench.py, thermsum.py.
- [measured] Bug fixed: tools/run-guarded.sh never assigned the worker PSI (`wp`), so its watchdog checked only the worker's MemAvailable, not its memory pressure. Every earlier run-guarded benchmark had that gap; none of them reported high worker PSI by other means. The fix was tested against the live worker (wa=21 wp=0).
- Documented, not changed: node-prep.sh still runs `nvidia-smi -lgc 0,2100`, which contradicts D8 (clocks uncapped).
- Deliberately not touched:
  - patch files, because results identify them by md5;
  - dsv4-t2-guard.py, dsv4-abort.sh and allocator-observer.sh, because they are safety scripts;
  - soak.sh, dsv4-peerwatch.sh and infer-probe.py, because they are running now.

### Documentation policy for the research write-up, 2026-10-25
- User instruction: keep complete, continuous logs and documentation of all work (objectives, changes, tests, results, failures, incidents). A research-style report will be written from them after all work is done.
- Each entry should therefore state:
  - the run ID and data path;
  - n, and whether the comparison is same-boot or cross-boot;
  - any confounds;
  - the evidence tag.
- Raw data in data/ is kept, and each result is committed (tools/commit-snapshot.sh).
- Gaps to close before the write-up:
  1. final benchmark on one frozen configuration (final-bench.sh);
  2. repeated or same-boot A/Bs for the main gains (async, GEMV, clocks, split indexer) to give error bars;
  3. E16 row sharding vs split;
  4. GEMV accuracy, patch off vs on;
  5. NCCL diagnostics overhead (DSV4_DIAG=0 vs 1);
  6. one consolidated, dated hardware/software table.

### Off-site backup and final benchmark start, 2026-10-25 13:57
- [measured] Off-site copy is on the worker at gx10:~/dsv4-backup/:
  - git bundle of the repo, all commits to 932a3c8;
  - reports tarball;
  - data/ tarball (59 MB).
  SHA256SUMS verified on gx10, and the bundle clones back. gx10 disk is at 90 % (91 GB free).
- Final benchmark started on the F02 boot (run dsv4-t2-20261024T181019Z-188432, up since 2026-10-24 11:12): tools/final-bench.sh f02-boot → data/FINAL/20261025T135749-f02-boot/. It now uses the fixed run-guarded.sh, which also checks worker PSI.
  - The soak and probe were still running, so a probe may overlap one request; final-bench logged the warning.
  - Chained after it: gemv-acc.py run patch-on-26h 1, a same-boot pass 26 h after the 2026-10-24 11:52 patch-on baseline (2 passes), to check for drift with uptime.
- [measured] First results: warm-up 27.23 tok/s; 4k-1 27.32 tok/s decode, TTFT 1.7 s, 2,228 tok/s prefill. That is above F01's 26.89; a reason will be considered once all repeats are in.

### Final benchmark on the F02 boot, 2026-10-25 13:57–14:17 [measured]
- **Setup:**
  - run dsv4-t2-20261024T181019Z-188432, up 26 h;
  - default configuration: 8192 chunks, KV 11274289152 bytes, async, split indexer, GEMV patch, FI4732, DSV4_DIAG=1;
  - `tools/final-bench.sh f02-boot` (ctxbench.py, 256 new tokens, ignore_eos, thinking off), each request under run-guarded.sh with the worker-PSI fix;
  - data `data/FINAL/20261025T135749-f02-boot/`, with config.txt recording the args, mounts and md5s.

| ctx | n | prompt tok | TTFT s | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| 4K | 3 | 3,824 | 1.7 (all 3) | 2,228 (2,224–2,231) | **27.20** (27.12–27.32) |
| 64K | 2 | 65,824 | 29.2 / 29.3 | 2,251 | **26.61** (26.57, 26.65) |
| 256K | 1 | 263,518 | 137.3 | 1,919 | 26.15 |
| 1M | 1 | 1,045,929 | **866.0** | 1,208 | 25.12 |

- Warm-up (discarded): 27.23 tok/s. No watchdog trips. Minimum MemAvailable 16 GiB on the leader (during 1M) and 20 GiB on the worker. Memory PSI 0 throughout on both nodes. No soak anomaly during the benchmark.
- Probe overlap: the 14:15:20 infer-probe request arrived during the 1M request. With max-num-seqs 1 it queued, finishing at 14:17:15 (114.9 s, logged as 1.11 tok/s), so it did not run concurrently with any measurement. [inferred] It did not affect the results. The low probe figure is queueing, not a fault.
- **Comparisons with the same client (ctxbench.py via run-guarded.sh), across boots:**
  - 4K decode: 27.20 vs F01 26.89 (+1.2 %) and E07.8 26.93 (+1.0 %).
  - 64K: TTFT 29.2 vs E10 29.8 s; decode 26.61 vs 26.34.
  - 256K: TTFT 137.3 s vs E13 143.9 s (split off) = −4.6 %.
  - 1M:
    - TTFT 866.0 s for 1,045,929 tokens vs E10 1,010.1 s for 1,035,910 tokens (split off) = −14.3 % in time and +17.7 % in prefill rate (1,208 vs 1,026 tok/s);
    - vs E06 1,113.3 s = −22.2 %;
    - decode 25.12 vs E10 25.07.
- **Open discrepancy:** 1M TTFT 866.0 s is 6.4 % above E12b (813.7 s) and 5.7 % above E15 (819.2 s). Those used e12b_client.py, a different prompt and 1,036,267 tokens.
  - [inferred] The 1 % extra tokens explains about 16 s at most (E11 chunk model: about 13 s per extra chunk at 1M prefix), which leaves about 35 s unexplained.
  - [hypothesis] Candidates: the client/prompt difference, leader thermal state after 26 h, or cross-boot variance. A same-boot e12b_client 1M run would separate them.
- Thermal logging (tools/thermlog.sh) is **not running**. The last file is spark-61dd-20261022.csv; it presumably stopped with the I01 leader power-off on 2026-10-22 and was never restarted. So there is no thermal data for F01, F02 or this benchmark. The operator's D8 condition (watch throttling with clocks uncapped) has not been met since 2026-10-22.
- **GEMV drift check:** `gemv-acc.py run patch-on-26h 1` (14:17), compared with the 2026-10-24 11:52 patch-on run on the same boot:
  - accuracy 41/52 in all 3 passes;
  - greedy tokens identical 54/54;
  - mean |Δlogprob| 0.00000.
  [measured] Greedy output is bit-stable within a boot over 26 h of uptime. Data: `data/GEMV/patch-on-26h-20261025T141747.json`.

### F02 status check, 2026-10-25 18:13 [measured]
- Health 200; both containers up 31 h; no ALERT on either node; worker WiFi connected.
- Soak part 3 is clean through 18:12: no anomaly and no D-state notes, peerwatch=1, leaderwatch=1, stale_guards=0, leader on the pinned BSSID.
- infer-probe 44/44 OK since 2026-10-24 20:15; latest 17:45 at 25.98 tok/s.
- No kernel warnings since 14:17.

### F02 result: 24 h+ stability soak, 2026-10-24 11:10 → 2026-10-25 20:15 [measured]
- **Run:** dsv4-t2-20261024T181019Z-188432, default configuration, one boot. At 22:11 on 2026-10-25 the containers had been up 35 h, health was 200 and neither node had an ALERT.
- **Soak monitor** (tools/soak.sh, 5-min ticks), 283 ticks over 23.5 h:

| Part | Window | Ticks | Ended by |
|---|---|---|---|
| 1 | 10-24 11:16–12:11 | 12 | ANOMALY: leader ALERT from a leaked guard of the previous run (monitoring false positive; fix applied) |
| 2 | 10-24 20:15 – 10-25 09:47 | 163 | ANOMALY: one 15 s D-state with an empty list (monitor too strict; rule changed to 45 s) |
| 3 | 10-25 10:56–19:57 | 108 | **COMPLETE: 9 h with no anomaly** |

  - Across all ticks: health 200 and both containers running in every tick; memory PSI ≤ 0.26; minimum MemAvailable 17 GiB leader / 21 GiB worker; no kernel hung-task, mt7925, NVRM or Xid lines; peerwatch and leaderwatch present throughout.
  - Leader BSSID: off the pinned router in 49 of 283 ticks, with no effect on serving.
  - peerwatch counted 2 in 10 ticks, a transient count; one process.
- **Inference probe** (tools/infer-probe.py, 30-min interval, 128 greedy tokens): **66/66 OK** from 10-24 11:16 to 10-25 19:45, with one 29-min gap at the 20:15 restart. Median 26.02 tok/s end-to-end (25.91–26.21 excluding two outliers):
  - 10-24 12:16, 8.67 tok/s, cause unknown;
  - 10-25 14:15, 1.11 tok/s, queued behind the final benchmark's 1M request.
- **Load during the soak:** the final benchmark (4K–1M, 13:57–14:17), the GEMV pass and any the operator traffic (not measured).
- **Verdict:**
  - [measured] The model served without interruption for the whole window (35 h of uptime at the time of writing). No serving fault, memory distress or kernel error was recorded.
  - [inferred] Both monitor stops were monitoring artefacts, not model or host faults. Each has been fixed: stale-watcher retirement in the launcher, and the 45 s D-state rule in soak.sh.
  - Caveat for the write-up: the monitor itself covered 23.5 h of the 32.7 h span and the longest unbroken monitor window is 13.5 h (part 2). The probe series covers the full 24 h.

## R01: restart test series, 2026-10-25 (night)
**Plan, one change per boot; the user starts each boot:**
1. DSV4_PATCH=0, for GEMV accuracy and speed;
2. DSV4_IDX_MODE=rowshard, for E16;
3. DSV4_DIAG=0, for the NCCL diagnostics overhead;
4. defaults.

Every boot also gets `tools/boot-4k.sh <label>` (new: warm-up + 4K ×3 via run-guarded.sh → data/RESTART/<ts>-<label>/). That gives n=3 within each boot and a cross-boot spread for the defaults (F02 boot vs boot 4).

### R01.0: baseline on the F02 boot before the first restart, 2026-10-25 22:3x [measured]
- `boot-4k.sh defaults-f02-boot` on run dsv4-t2-20261024T181019Z-188432 (defaults; env NCCL_DEBUG=INFO, TORCH_NCCL_DESYNC_DEBUG=1).
- Decode median **27.26 tok/s** (27.14–27.31), TTFT 1.7 s; warm-up 27.38. This agrees with the FINAL 4K (27.20, 27.12–27.32) taken 8.5 h earlier on the same boot.

### R01.1 attempt 1: start refused, 2026-10-25 22:40 [measured]
- `launch-dsv4.sh down` stopped both containers (leader=stopped worker=stopped; guards/observers left 0/0). Reclaim left MemAvailable at 117 GiB on the leader and 118 GiB on the worker.
- `DSV4_PATCH=0 start-dsv4-default.sh` was refused: "dsv4-t2-guard: sticky ALERT exists; clear it explicitly first". The ALERT on both nodes reads `launch-dsv4 down`, because down runs dsv4-abort.sh and the abort latches an alert by design. My stop instructions (README, docs/operations.md) had left out the clear step. Corrected: both docs now say to clear both nodes after `down`.
- The operator then pasted old terminal scrollback into the shell. Every line failed as "command not found" or a syntax error. Checked afterwards: no DSV4 container running on either node, no guard, watcher or observer processes, and leader MemFree 118 GiB. Nothing changed.

### R01.1 attempt 2: start failed, 2026-10-25 22:44–22:46 [measured]
- Run dsv4-t2-20261026T054416Z-682750 (DSV4_PATCH=0). Alerts were cleared first. Guards started on both nodes at 22:44:35; the blocker preflight found nothing (blockers.txt empty).
- At 22:45:24 the launcher failed and ran the coordinated abort, reason "startup guard/readiness failure" (the generic reason for any fail). No container had started. The specific fail message is only in the operator's terminal; requested.
- Abort record: leader inspect rc 0, matching the old stopped container ead9f8eb… of run 188432, status unknown; **worker inspect rc 137**, the 30 s timeout, status unknown. That is the same symptom as 2026-10-24 11:04, though harmless here because nothing was running. After the abort, both nodes were clean: no guards, observers or containers.
- [measured] The worker's WiFi roamed at **22:45:16**, 8 s before the failure: <bssid> → <bssid> on 6 GHz, DHCP renewed at 10.0.0.106 by 22:45:17. Worker journal: SSH logins from 192.168.200.10 at 22:44:52, 22:45:10 and 22:45:24–55, none between 22:45:10 and 22:45:24.
- [measured] Afterwards, with no start running, the abort's exact remote inspect command took 0.29–0.49 s (5/5) and worker `docker info` took 0.48 s.
- [hypothesis] A worker-side stall around the WiFi roam (for example a DNS or logind delay in SSH session setup, even though the TCP path is CX7) made a bounded remote call time out. Not shown; the fail message is needed.

### R01.1: boot with the GEMV patch off, 2026-10-26 [measured]
- Attempt 3 (run dsv4-t2-20261026T163651Z-704748, DSV4_PATCH=0) waited 09:36–10:19 at the confirmation prompt. The operator confirmed at about 10:19. The operator was informed of the outage, which had lasted since 2026-10-25 22:39.
- During the start, a watcher (2 s interval) measured SSH from the leader to the worker over CX7 at 0.31–0.47 s on every sample, with the worker WiFi steady on <bssid>. Before the confirmation, 600 samples (09:37–10:00) were also all ~0.3 s. [inferred] The 2026-10-25 22:45 stall was transient; its cause is still unidentified (see attempt 2).
- API healthy at 10:23:55, post-load reclaim done at 10:23:47; launcher completed with peerwatch and leaderwatch running and no ALERT.
- Verified from the container mounts: no `dsv4-attention.py` on either rank, so the GEMV patch is off. Split indexer and FI4732 are mounted on both ranks.
- Running: `boot-4k.sh patch-off`, then `gemv-acc.py run patch-off 2`.

### R01.1 result: GEMV patch off, speed and accuracy [measured]
- **Speed:** `boot-4k.sh patch-off` → data/RESTART/*-patch-off/. Decode median **26.38 tok/s** (26.36, 26.38, 26.40), TTFT 1.7 s; warm-up 26.35.
  - Patch on (F02 boot): FINAL 27.20 (27.12–27.32) and R01.0 27.26 (27.14–27.31).
  - Patch effect: **+3.1 %** (≈ −1.1 ms per token), across boots. That is larger than the +1.1 % of E07.8 (also across boots) and above the E07.8 ceiling estimate of ≤0.8 ms.
  - [inferred] The spread within each boot (≤0.2 tok/s) is much smaller than the gap. Boot-to-boot variance for an unchanged configuration is not yet known; R01.4 (defaults boot) gives a second patch-on boot to bound it.
- **Accuracy:** `gemv-acc.py run patch-off 2` → data/GEMV/patch-off-20261026T102611.json, compared with patch-on-20261024T115338.json (another boot).
  - Task accuracy: 41/52 in both passes with the patch off; 41/52 in all three patch-on passes. No difference.
  - Determinism within the patch-off boot: pass 1 vs pass 2 identical.
  - Patch on vs off: 52/54 prompts token-identical, mean |Δlogprob| 0.0043 (max 0.279). The 2 divergences are at near-ties, and both outputs are correct or equivalent:
    - "most native speakers" at token 29: ' native' p≈0.56 (on) vs ' speakers' p≈0.62 (off);
    - Fibonacci at token 75: ' Fibonacci' vs ' number', p≈0.5–0.6.
  - [inferred] The patch changes no answers. The remaining differences are of the size expected from fp32 summation-order changes at near-tied tokens; cross-boot divergence also happens with an unchanged config (E12b), so they cannot be attributed to the patch alone.
- **Verdict:** keep the patch on (default). It is a measured +3.1 % decode with no accuracy change, which makes it an upstream candidate. A same-boot A/B is still missing, because the patch is a bind mount, so on/off needs a restart.

### R01.2 attempt 1: rowshard boot aborted by the boot PSI rule, 2026-10-26 11:16–11:23 [measured]
- Run dsv4-t2-20261026T181558Z-765584 (DSV4_IDX_MODE=rowshard, otherwise defaults). Weights loaded in 149.8 s; KV cache reserved at 11:20:20 (1,094,398 tokens).
- From 11:20:20 the leader stalled: EngineCore logged "No available shared memory broadcast block" every 10 s, and leader memory PSI some/full avg10 was 74–77 %. The worker stayed at PSI ≤ 0.8. At 11:21:01 the leader guard latched `boot:psi_some>40_for_30s` and the launcher aborted.
- The stall cleared by itself: PSI fell from 11:22:00 and JIT/kernel warm-up started at 11:22:00 inside the container, before the abort's stop completed. Both containers were stopped at about 11:23:06 (leader and worker status "stopped", inspect rc 0 on both); no orphan. MemAvailable on the leader was 29 GiB after the stop.
- **Not attributable to rowshard.** Allocator-observer data for the last 10 boots (max memory PSI some avg10 during boot):

| Run | Settings | Max PSI | Samples > 40 |
|---|---|---|---|
| 181558Z-765584 | rowshard | 77.4 | 26 |
| 163651Z-704748 | patch off | 7.0 | 0 |
| 054416Z-682750 | patch off (failed before load) | 0.0 | 0 |
| 181019Z-188432 | defaults | 4.7 | 0 |
| 172856Z-26506 | defaults | 11.0 | 0 |
| 175931Z-180783 | defaults | 68.6 | 13 |
| 162854Z-14489 (I02) | defaults | 90.3 | 30 |
| 005449Z-775571 | defaults | 6.3 | 0 |
| 225038Z-724557 | defaults | 48.1 | 7 |
| 031419Z-647825 | defaults | 6.7 | 0 |

  [measured] 4/10 boots exceeded PSI 40, three of them with default settings. [inferred] This is the I02-class KV-allocation compaction stall. It is intermittent, so it cannot be blamed on rowshard from one boot. Rowshard does add a transient exchange-buffer reservation in its dummy run, which could make the stall more likely; that is untested [hypothesis].
- **Open (I02 follow-up):**
  - The stall happens despite the pre-launch reclaim and compaction. The fragmentation probably arises during the 74 GiB weight load [hypothesis].
  - The boot rule (PSI > 40 for 30 s) aborts stalls that would clear by themselves in ~100 s, but I02 did not clear in 11 min, so the rule cannot simply be relaxed.
  - Candidate (from I02, still not made): compact first, abort second. Needs a guard change, staged for the operator.
- Next: retry rowshard once.

### R01.2 attempt 2: rowshard stalled again, 2026-10-26 11:40–11:47 [measured]
- Run dsv4-t2-20261026T183932Z-778681 ( DSV4_IDX_MODE=rowshard). KV reserved at 11:43:56. The leader's memory PSI rose to **79.85** (watcher, 2 s samples), and the guard latched `boot:psi_some>40_for_30s` at 11:44:39; abort. The container's JIT warm-up began at 11:45:03, so the stall lasted about 67 s and cleared by itself, as in attempt 1. Both containers stopped (status stopped, inspect rc 0 on both); no guards or observers left; worker unaffected.
- **2/2 rowshard boots stalled, vs 3/8 for other recent boots.** If the base rate were ~3/8, two stalls in a row would happen by chance about 14 % of the time.
  - [inferred] Suggestive but not conclusive that rowshard raises the risk.
  - [hypothesis] Mechanism: upstream's `_reserve_row_shard_exchange_transient` reserves exchange buffers in the dummy run right after the KV reservation, adding high-order allocations while memory is already fragmented.
- **Decision:** E16 serving A/B postponed. It cannot run until the boot stall is handled (I02 follow-up: compact first, abort second; or a pre-KV compaction). No GPU result for rowshard yet; the CPU tests (sim 400/400, upstream 12/12) still stand.

### R01.3: diagnostics off, 2026-10-26 11:51–12:00 [measured]
- Run dsv4-t2-20261026T185118Z-789739, DSV4_DIAG=0, otherwise defaults (GEMV patch, split indexer, FI4732 all mounted). Both containers run with NCCL_DEBUG=WARN and no TORCH_NCCL_DESYNC_DEBUG.
- **No boot stall:** peak leader PSI 14.3 (watcher), healthy at 11:55:44, post-load reclaim 11:55:41, peerwatch and leaderwatch running, no ALERT. This boot is now serving the operator.
- `boot-4k.sh diag-off`: decode median **26.86 tok/s** (26.86, 26.86, 26.96), TTFT 1.7 s.
- **Interpretation:**
  - [inferred] Diagnostics-on boots range 26.89 (F01), 26.93 (E07.8) and 27.20–27.26 (F02 boot). The diag-off boot sits inside or below that range. NCCL INFO logging emits roughly one line per step outside CUDA graphs (E07.1), so a real slowdown from turning it *off* is implausible. **Diagnostics overhead: not measurable (< the ~1.5 % boot-to-boot spread).**
  - **Boot-to-boot variance (patch on, 4K decode), from four boots: 26.86–27.26 tok/s, a spread of ~1.5 %.** Within each boot the spread is ≤ 0.2 tok/s. Any 4K decode difference across boots below ~1.5 % is therefore not attributable.
  - **Correction to R01.1:** the GEMV patch effect is +1.8 % to +3.3 % (patch off 26.38 vs patch-on boots 26.86–27.26), not a point value of +3.1 %. It still exceeds the boot spread and agrees in sign with E07.8 (+1.1 %); the earlier "+3.1 %" compared against the fastest patch-on boot.
- The F02 boot being the fastest (27.2) is unexplained [hypothesis: thermal state, or GPU memory placement per boot].

### R01.4: boot stall fix staged, "compact first, abort second" (2026-10-26 13:10) [measured unless tagged]
- **Problem:** 4/10 recent boots (2/2 rowshard) hit the I02-class KV-allocation stall. The leader's memory PSI was 70–90 % right after the KV reservation. The guard's boot rule (> 40 for 30 s) aborted them, although R01.2's two stalls cleared by themselves in ~67–100 s. I02 itself did not clear in 11 min until an explicit `gb10-reclaim` freed high-order blocks, after which it cleared within 10 s.
- **Change (staged, not applied):**
  - `tools/boot-compact.sh` (new): a per-node helper that runs `sudo -n /usr/local/sbin/gb10-reclaim` (drop_caches + compact_memory) when memory PSI some avg10 stays above 20 for 6 s during boot. It does at most 3 reclaims, at least 20 s apart.
  - It logs PSI and the free high-order (order ≥ 9) block count before and after each reclaim to `runs/<id>/<role>/boot-compact.log`.
  - It exits on serving, on a latched ALERT (it never acts during an abort), on container stop, or after 1800 s.
  - `staged/boot-compact.block` + `staged/apply-boot-compact.sh` insert the launch of the helper on both nodes just before the coordinated launch in `start-dsv4-default.sh`. `DSV4_BOOT_COMPACT=0` disables it. **The guard and its boot rule are unchanged:** if compaction does not help, the boot is aborted exactly as before.
- **Tests** (scratch, fake PSI file and fake reclaim), all as expected:
  - sustained 75 % → reclaim at 6 s, then at +8 and +16 s (test gap 8 s), max 3;
  - a 4 s spike to 30 % → no reclaim;
  - ALERT latched → exit;
  - container gone → exit;
  - a second instance → refuses (flock);
  - a bad run id or role → rc 2;
  - real `/proc` and docker inputs → idle at PSI 0.
  - Apply script on a copy: inserts once, idempotent, `bash -n` OK.
- **Caveat** [measured, single instance]: in I02 the post-load reclaim ran 3 s after the KV reservation and did not prevent the stall. Only the reclaim run later, during the stall, released it. Whether a reclaim during the stall shortens these ~60–100 s stalls is untested [hypothesis]. The helper's log will show it on the next stalled boot.
- **Validation plan:** apply; on the next boots (including the E16 rowshard retry), read `boot-compact.log` on both nodes. Success means a stalled boot reaches serving with the guard's 30 s rule never firing.
- **Applied** by the user at 13:57 (backup start-dsv4-default.sh.bak-20261026T135717); hook at line 402ff; bash -n OK. Not yet exercised by a boot.

### R01.5: first boot with the compaction helper, E16 rowshard retry (2026-10-26 14:00) [measured]
- Run dsv4-t2-20261026T210020Z-848734, DSV4_IDX_MODE=rowshard (4 rowshard/attention patches mounted). Launched ~14:00:20; serving at 14:05:31 (~5 min 10 s). No ALERT.
- **No stall this boot:** peak memory PSI some avg10 was 8.09 on the leader and 20.65 on the worker (for ~2 s at 14:05:24, the post-load reclaim moment). The helper started on both nodes (14:01:11/12), logged the worker spike and correctly did not reclaim (< 6 s), then exited on serving at 14:05:36. **The helper ran cleanly, but its stall path was not exercised;** the fix is still unproven. This is also the first rowshard boot of three that did not stall (now 2/3), which weakens the R01.2 suggestion that rowshard raises the risk.
- 4K check (`boot-4k.sh rowshard`): decode median **26.86** (26.83–26.94), TTFT 1.7 s. Same as the defaults boot R01.3 (26.86); as expected, rowshard does not change short-context decode.
- E16 long-context A/B started: `tools/e16-ab.sh` (64K ×2, 256K, 1M).

### E16 result: rowshard vs split, long context (2026-10-26 14:07–14:24, run dsv4-t2-20261026T210020Z-848734) [measured]
- `tools/e16-ab.sh` (e12b_client, the same deterministic vault-code prompts as E12b/E15). Data `data/E16/20261026T140659-rowshard/`. Reference: E15 (split + FI4732), the same client and prompts.

| Context | TTFT rowshard | TTFT E15 split | Δ | Vault code |
|---|---|---|---|---|
| 64K | 29.1 s | 29.9 s | −2.7 % | found (both) |
| 256K | 134.4 s | 134.9 s | −0.4 % | found (both) |
| 1M (1,036,267 tok) | 820.3 s | 819.2 s | +0.1 % | found (both) |

- The second 64K request was a prefix-cache hit (TTFT 1.0 s), so it is excluded. The client's `decode_tok_s` covers only ~60 tokens and is not comparable to ctxbench decode (it reads 27–33 here and 27–31 in E15), so it is not used. 4K ctxbench decode was the same as defaults (26.86, R01.5).
- **Interpretation** [inferred]: no measurable prefill or decode gain from rowshard at any context. All differences are within the ~1.5 % boot-to-boot spread (R01.3), apart from 64K at −2.7 %, which comes from one request on one boot. Answers are correct at every context. The generated wording differs slightly from E15, as expected from a different reduction order [inferred].
- **Decision:** E16 closed, no gain; split stays the default. The rowshard patches stay in `patches/e16-upstream-rowshard/` for reference. For upstream, a comment can report "works on 2× GB10 at 1M with correct output, no speed change at TP=2".
- Note: the 1M TTFT here (820 s, e12b_client) and the final benchmark's 866 s (ctxbench) use different clients and prompts. The "open gap vs E12b" in the final report is therefore a client/prompt difference, not a regression, at least for split + FI4732 vs E15 [inferred; to be checked against the final benchmark's notes].

### R01.6: back to full defaults; second boot with the compaction helper (2026-10-26 16:04–16:12) [measured]
- 14:47 the rowshard run was stopped by `launch-dsv4.sh down` (both containers exit 0; ALERT latched as designed). The alert clears and restart did not follow until ~16:04, so the **model was offline for the operator from 14:47 to 16:10 (~1 h 23 min)**. Operator delay, no fault.
- Run dsv4-t2-20261026T230446Z-926391, full defaults (DSV4_DIAG=1; dsv4-attention.py (GEMV), dsv4-sparse-attn-indexer.py (split) and the FI4732 .so mounted). Serving at 16:10:03, about 5 min 20 s after launch. Peerwatch and leaderwatch running; no ALERT.
- **No stall again:** peak memory PSI some avg10 was 7.98 on the leader and 26.46 on the worker. The worker spent 4 s above 20 at 16:09:51, around the post-load reclaim, with only 8 free order ≥ 9 blocks. The helper correctly did not act (< 6 s) and exited on serving on both nodes. **The helper is now 2/2 clean runs, but its stall path is still unexercised.**
  - [hypothesis] The worker's brief spike at the post-load point appears on both helper boots (20.7, 26.5). Watch whether the worker ever crosses the 6 s trigger.
- 4K check (`boot-4k.sh defaults-r016`): decode median **26.78** (26.68–26.80), TTFT 1.7 s. This is the lowest patch-on boot so far, 0.3 % below R01.3 and 1.8 % below the F02 boot (27.26). The range of patch-on boots widens to 26.78–27.26 (1.8 %). Same config as F02, so this is boot-to-boot variation [inferred].

### R01.7: three small hardening items (2026-10-26 16:50) [measured]
- **Clock-cap trap removed.** `node-prep.sh` no longer runs `nvidia-smi -lgc 0,2100`; the header now explains why. The adopted config is uncapped (E07.4, +5.1 % decode), and re-running the one-time setup would have re-applied the cap silently until the next reboot. No effect on the running server: clocks read 2411 MHz on both nodes. Backup `node-prep.sh.bak-20261026T*`.
- **Abort retry applied** (`staged/apply-abort-retry.sh`; agreed by the user). `dsv4-abort.sh` now retries the remote worker `docker inspect` up to 3 times when the rc is neither 0 nor 1. This is the 2026-10-24 11:05 case, where one timed-out inspect left the worker running. Eligibility to stop is unchanged. `bash -n` OK; the script runs `set -uo pipefail` without `-e`, so a failed attempt falls through to the next try. Not yet exercised by a real abort.
- **Thermal logger restarted** on both nodes at 16:51 (leader `data/thermal/`, worker `~/dsv4-thermal/`), after serving began, per the I02 caution about `nvidia-smi` polling during boot. First samples: leader 48 °C, 11.4 W; worker 53 °C, 12.5 W; 2411 MHz; no throttle reasons. The gap in thermal data runs from 2026-10-22 to 2026-10-26 16:51 (no data for F01, F02 or the final benchmark). `docs/operations.md` now gives the restart commands.

### Handover documents, 2026-10-26 17:xx
- **`reports/DSV4 final report.md` v1.0** written from this log. It supersedes the draft, which stays in git history (removed from the tree; the original remains in ~/reports).
- **`reports/Message to the operator 2026-10-26 handover.md`** written: what is serving, expected speeds, start/stop/clear, failure handling, changes since 2026-10-21, and four decisions for the operator (DSpark, leader cooling, wired network, upstreaming).
- README updated: results note on the two 1M clients, GEMV +1.5–3.3 %, boot stall and boot spread under known issues, E16 result in the upstream table, `DSV4_BOOT_COMPACT` in the configuration table. `docs/operations.md`: boot-stall handling in the alert table.
- **Correction to R01.3:** with R01.6 (26.78) the patch-on 4K range is 26.78–27.26 (1.8 %), so the GEMV patch effect is **+1.5 % to +3.3 %** (patch off 26.38), not +1.8 % to +3.3 %.
- `tools/commit-snapshot.sh` now copies the log, the final report and the handover note.

### Repository clean-up and documentation review, 2026-10-26 17:00–17:30
Two read-only reviews checked the docs against the code and log, and the code comments against the behaviour. Every finding was verified before it was applied.
- **Tree:**
  - 35 timestamped backups moved to an ignored `.backups/` (not deleted).
  - Already-applied one-off scripts removed from the tree; git history keeps them: `staged/apply-*.sh`, `staged/*.block`, the pre-boot-rules `dsv4-t2-guard.py.strict`, and the generated mutation tests `tests/broken/` (regenerate with `tests/make_broken.py`).
  - New `setup/`: `node-prep.sh` (moved), `worker-wifi-keepalive.sh` (moved from staged/) and a copy of `/usr/local/sbin/gb10-reclaim` (identical on both nodes).
  - `data/README.md` indexes every data directory and file format; it is the only tracked file under data/. Worker thermal CSVs were copied into `data/thermal/`.
- **Research-relevant correction [measured]:** `tools/ctxbench.py`, the scoreboard client for every ctxbench number in this log, samples at **temperature 0.6**; its docstring said "greedy". The docstring is fixed and the behaviour is unchanged, so all results stay comparable. With ignore_eos and a fixed output length, the sampled tokens do not change the per-token work [inferred]. e12b_client, gemv-acc and infer-probe are greedy (temperature 0).
- **Docs fixed:**
  - Start sequence: the confirmation comes before the admission lock, plus the service check, parking, the helper, and readiness via `/v1/models`.
  - The confirmation shows no prompt; preflight `y` also signals PIDs (SIGTERM, then SIGKILL).
  - `down` stops only peerwatch first.
  - Abort semantics and the 3× retry; peerwatch timing (6 checks, about 1–2.5 min).
  - `residual_growth` is a serving-mode rule only; the sticky-ALERT failure message and the readiness rc 1/2/3 are documented.
  - Boot-stall recovery now uses the helper's log.
  - GEMV +1.5–3.3 % and its accuracy result; the E16 result in patches.md and PORT.md; the PORT.md upstream test status (12 passed, 2026-10-24); PORT.md paths and impersonal wording; FlashInfer README validation.
  - Removed the Tailscale IP from the README.
  - The CPU row-shard simulation was re-run with the corrected command: 400/400 at TP 2/3/4, as recorded.
- **Code (behaviour-neutral or bug fixes; `bash -n`/`py_compile` all pass):**
  - `launch-dsv4.sh`: removed an unused DSpark-era docker command block, whose `${!v}` loop could fail `status` without the fabric; `status` re-tested OK.
  - `start-dsv4-default.sh`: the failure-path worker `pkill` pattern now uses `guard[.]py … <run-id> `, matching launch-dsv4 `down`, so it cannot match its own remote shell or a prefix run ID; verified against the live worker guard.
  - `dsv4-abort.sh`:
    - the worker latch quotes the free-text reason with `%q` (a `'` in a peerwatch reason would have broken the worker latch); tested through a real remote shell;
    - the latch uses the worker's own state dir;
    - 2 s between inspect retries;
    - header with usage.
  - `node-launch.sh`: fails clearly if `cx7-nccl-env.sh` is missing or finds no CX7 link (it previously died later on an unbound variable); duplicate `NCCL_DEBUG=INFO` removed (the later one still wins); comments corrected.
  - Tools:
    - ssh timeouts and BatchMode in run-guarded, e16-ab and st.sh;
    - idle-sampler now prints "hang" when nvidia-smi times out, as documented;
    - quoting and argument checks.
  - Headers or usage added to the guard, observer, watchers and abort.
  - `peerwatch` was updated by write-and-rename, so the running instance (run 926391) was not disturbed.
  - `setup/node-prep.sh` now requires root and uses `set -euo pipefail`.
  - `tools/commit-snapshot.sh` adds the Co-Authored-By trailer and copies the handover note.
- **Not changed** (noted for later; none affects results):
  - style differences (`--env` vs `-e`);
  - duplicated prompt code in e12b_client;
  - small guard refactors;
  - hard-coded peer IPs alongside `$PEER`;
  - the HF cache path under `/home/<operator>`;
  - the WiFi SSID in `tools/soak.sh` and `setup/worker-wifi-keepalive.sh`, which should be removed before any public release.
- **Not yet exercised:** the edited launcher, abort and node-launch paths run only on the next start, stop or abort; the next boot validates them.

## 2026-10-26 — GitHub export preparation (autonome-research/dsv4-spark-1m)
- [measured] The user logged `gh` in as `<github-account>`, a member of `autonome-research`. Decisions by the user: repo `dsv4-spark-1m`, **private first**, the operator messages left out, licence Apache-2.0.
- Source changes, committed to the local repo:
  - `setup/worker-wifi-keepalive.sh` takes the NetworkManager connection name as its argument instead of hard-coding the SSID. The generated `/usr/local/sbin/wifi-keepalive` is equivalent to the installed one: diffed, only the quoting style of `CON=` differs. The installed timer on the worker is unchanged.
  - `tools/soak.sh` logs the pinned BSSID only when `SOAK_WIFI_CON` is set.
  - `setup/node-prep.sh` writes the sudoers rule for `$SUDO_USER` instead of a hard-coded account.
  - The handover note is no longer tracked in git (correspondence); its canonical copy stays in `~/reports/`. `commit-snapshot.sh` no longer copies it.
  - LICENSE (Apache-2.0) and NOTICE (vLLM/FlashInfer provenance of `patches/`) added.
- The GitHub copy is a fresh tree with no local history, because the local history contains the handover note. `tools/export-github.sh` builds it and redacts the WiFi SSID and BSSID/MAC addresses in `reports/*.md` of the export only. This log itself is not redacted.
- **Sensitive-information pass on the GitHub copy** (user request "make sure it does not contain any sensitive information"):
  - [measured] A scan of the first copy found: the operator's and the user's account names (about 60 mentions), the user's e-mail, the GitHub account name, router brands, an unrelated internal project, and the `/home/<operator>` cache path. It also found the export tools themselves, which contain those strings as patterns.
  - `tools/export-redact.py` now replaces them in the copy: people become roles ("the operator", "the user account"), plus `<author email>`, `<github-account>`, "an ISP gateway / a mesh router" and "another model". The benchmark word lists in `ctxbench.py`/`e12b_client.py` keep the ordinary word "the operator", so the exported prompts are byte-identical.
  - `.gitattributes` marks `tools/export-github.sh`, `tools/export-redact.py` and itself `export-ignore`, so they are not in the copy.
  - Export commits use the GitHub no-reply address of the pushing account, not a personal e-mail.
  - `node-launch.sh` takes the HF cache path from `DSV4_HF_CACHE`. The default is unchanged, so serving behaviour is unchanged; it takes effect at the next start.
  - [measured] The E16 bundle's commits are by `port <port@local>`. Strings in the FlashInfer `.so` show only container paths. Private CX7 and WiFi subnet addresses (192.168.200/201.x, 10.0.0.x, 192.168.86.x) are kept: they are not routable and are needed to follow INV-1.
  - [measured] Final scan of the copy for names, accounts, e-mail, SSID, BSSID/MAC, Tailscale addresses and brands: no matches.
