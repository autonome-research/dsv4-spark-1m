# DSV4 optimisation: final report

**Version 1.2, 2026-10-27.** It is compiled from `reports/DSV4 optimisation log.md`, entries E01–E16, F01–F02, I01–I03, R01.0–R01.7 and R02. It supersedes `DSV4 final report (draft).md` of 2026-10-24. v1.1 adds R02: the boot-compaction helper's first real intervention, and a further boot that widens the boot spread and the GEMV range. v1.2 corrects the boot counts (changes listed at the end).

**Scope:** serving DeepSeek-V4-Flash-0731 (NVFP4) at 1M context, one request at a time, on two DGX Spark GB10 nodes:
- spark-61dd: the leader, 192.168.200.10;
- gx10-6d0c: the worker, an ASUS Ascent GX10, 192.168.200.11.

vLLM runs TP=2 over a ConnectX-7 RoCE cable. The project ran from 2026-10-12 to 2026-10-26.

**Evidence tags** follow the log:
- **[measured]**: from a log, benchmark or counter;
- **[inferred]**: follows from measured facts;
- **[hypothesis]**: not tested.

A number with no tag is [measured] in the cited entry. Where the log corrects an earlier entry, this report uses the corrected version.

---

## 1. Summary

- **G1 (1M on 2×GB10, TP=2): met.** The first 1M request completed on 2026-10-12 (E06). 1M requests completed again in E07.6, E10, E12b, E15, FINAL and E16. None had an OOM, a wedge or a watchdog trip.
- **G2 (repeatable boot): met, with one open failure mode.** Boots reach serving in about 5 min. Some boots (4 of 10 before 2026-10-26; 2 of 3 rowshard boots) hit a KV-allocation compaction stall that the boot guard aborts (I02 class).
  - A boot-time compaction helper ("compact first, abort second") was added on 2026-10-26 (R01.4).
  - Its first two boots did not stall, so it has not yet been exercised (R01.5, R01.6).
- **G3 (decode): improved, target not met.**
  - 4K: 23.56 → 27.20 tok/s (E07.0 → FINAL, +15.4 %); from the first 1M-capable configuration with the same 256-token method, 22.79 → 27.20 (E06, +19 %, as in the README).
  - 1M: 21.08 → 25.12 tok/s (E06 → FINAL, +19.2 %).
  - Target 29–33 tok/s. The remaining large lever is speculative decoding (DSpark), which was not started.
- **G4 (prefill): improved.**
  - 1M TTFT, like for like: 1,113.3 → 866.0 s with ctxbench.py (E06 → FINAL, −22.2 %). With the deterministic e12b_client the current stack does 1M in 819–820 s (E15, E16); split on vs off on that client is −19.5 % (E12b).
  - 64K TTFT 46.1 s → 29.2 s (E07.6 → FINAL, −37 %).
- **G5 (latency after idle): met (E09).** After 30–60 min idle, the first request took 0.16–0.27 s.
- **Stability: met.**
  - F02 passed: one boot served for 35 h, the inference probe passed 66/66, and there was no serving fault.
  - Every failure found during the project has a fix or mitigation in place (section 6).

## 2. Goals

| # | Goal | Target | Start | Final | Status |
|---|---|---|---|---|---|
| G1 | One 1M request on 2×GB10, TP=2 | Completes with no OOM, wedge or crash | — | 1,036,267 tokens in 820.3 s (E16), correct retrieval. Minimum MemAvailable 16 GiB leader / 20 GiB worker | **Met** (E06, 2026-10-12) |
| G2 | Repeatable boot without manual recovery | Clean boot with guards on | Every boot aborted by the guard (E02, E05) | Serving in ~5 min. I02-class stall in ~30–40 % of boots; aborted cleanly, retry succeeds. Compaction helper added (R01.4), not yet exercised | **Met**, stall mitigation unproven |
| G3 | Decode | 29–33 tok/s (research-report estimate, DSpark off) | 4K 23.56 (E07.0); 1M 21.08 (E06) | 4K 27.20; 64K 26.61; 256K 26.15; 1M 25.12 (FINAL) | **Improved, not met** |
| G4 | Prefill | Improve 1M TTFT (18.5 min) | 1M 1,113.3 s (E06, ctxbench); 64K 46.1 s (E07.6) | 1M 866.0 s (FINAL, ctxbench, −22.2 %); 819.2 / 820.3 s with e12b_client (E15 / E16); 64K 29.2 s | **Improved** (−22 % at 1M, same client) |
| G5 | Latency after idle | ≈ warm latency | 3.6 s after 15 min; engine death after ~30 min (E08) | 0.16–0.27 s after 30–60 min (E09) | **Met** |

## 3. Performance

Each run was a single request, with DSpark off and thinking off. "Start" is configuration C1 (E06, 2026-10-12, `ctxbench.py`, 256 output tokens) unless the cell says otherwise. The source of each number is given in brackets.

### 3.1 Decode (tok/s, ctxbench.py)

| Context | Start | After async (E07.6) | Final benchmark (FINAL) |
|---|---|---|---|
| 4K | 22.79 [E06, 256 tok]; 23.56 [E07.0, 512 tok] | 26.64 | **27.20** (n=3, 27.12–27.32) |
| 32K | 22.25 [E06] | — | — |
| 64K | not measured | 26.29 | **26.61** (n=2) |
| 128K | 21.74 [E06] | — | — |
| 256K | not measured | — | **26.15** |
| 512K | 21.62 [E06] | — | — |
| 1M | 21.08 [E06] | 24.76 | **25.12** |

- **Boot-to-boot spread:** eight patch-on boots (E07.8, E10, F01, F02, R01.3, R01.5, R01.6, R02) gave 4K decode of 26.69–27.26 tok/s, a 2.1 % spread; R02 (26.69) widened it from 26.78–27.26 (1.8 %). Within one boot, three warm requests vary by 0.1–0.3 tok/s.
  - [inferred] Cross-boot differences below about 2 % cannot be attributed to a change.
  - Why some boots are faster is not known [hypothesis: thermal state or per-boot memory placement].
- The 4K step time fell from 42.4 ms (E07.0) to about 36.8 ms (FINAL).
- The decode rates from `tools/e12b_client.py` (about 60 output tokens) are not comparable to ctxbench and are not used (E12b baseline note).

### 3.2 Prefill (TTFT)

| Context | Start | Before split indexer | Final, e12b_client (E15 / E16 rowshard) | Final, ctxbench (FINAL) |
|---|---|---|---|---|
| 4K | 1.8 s [E06] | 1.7–1.8 s | — | 1.7 s (2,228 tok/s) |
| 64K | 46.1 s [E07.6, 4096 chunks] | 29.8 s [E10] | 29.9 s / 29.1 s | 29.2 s (2,251 tok/s) |
| 256K | 143.9 s [E13, split off] | 134.0 s [E12b, split] | 134.9 s / 134.4 s | 137.3 s (1,919 tok/s) |
| 1M | 1,113.3 s [E06] | 1,010.1 s [E10] | **819.2 s / 820.3 s** | 866.0 s (1,208 tok/s) |

- **Two clients, two 1M numbers.**
  - e12b_client.py (1,036,267 tokens, vault-code prompt) gives 813.7–820.3 s across four boots (E12b, E15, E16).
  - ctxbench.py (1,045,929 tokens, different prompt) gave 866.0 s.
  - [inferred] Today's E16 boot reproduced 820 s with e12b_client, so the current stack has not regressed. The 866 s figure reflects the client and prompt (and 0.9 % more tokens). The specific prompt-dependent cost has not been isolated.
- Like-for-like 1M improvements:
  - ctxbench: 1,113.3 → 866.0 s (−22.2 %);
  - e12b_client, split on vs off: 1,010.3 → 813.7 s (−19.5 %, E12b).

### 3.3 Where decode time goes (E07.5, E07.7, E07.8)

- The 4K step is about 36.8 ms. With a measured bandwidth ceiling of 257 GB/s, streaming about 5.9 GB of weights takes about 23 ms (≈ 43 tok/s roofline) [inferred].
- Weight-streaming kernels run at about 77 % of that ceiling. Non-weight work is about 8.4 ms per step. NCCL is 5–8 % of the step. Host scheduling gaps are ≈ 0 after async scheduling.
- [inferred] Without speculative decoding, the realistic remaining gains are small. Candidates: DeepGEMM M=1 FP8 efficiency (≤ 2–3 ms), fusing small kernels, NCCL tuning (≤ 1 ms). All of these are [hypothesis].

## 4. What changed (adopted)

| Change | Entry | Effect [measured] | Switch |
|---|---|---|---|
| KV cache pinned (`--kv-cache-memory-bytes 11274289152`, 1,094,398 tokens) | E04, E10 | Same KV size every boot; MemAvailable above the 6 GiB floor in every 1M run | `DSV4_KV_CACHE_BYTES` |
| Boot guards, pre-launch reclaim, MemFree ≥ 100 GiB gate, post-load reclaim | E02, E05 | Boots reach serving instead of aborting | — |
| Async scheduling | E07.6 | 4K decode +7.6 %; step-tail gap 0.8–1.6 ms → 0 | `DSV4_ASYNC` |
| GPU clocks uncapped | E07.4 | +5.1 % decode (not isolated from the HCA and profiler changes) | `setup/node-prep.sh` no longer caps clocks (R01.7) |
| Chunked compressor GEMV patch (`patches/dsv4-attention.py`) | E07.8, R01.1, R01.3 | **+1.2 % to +3.3 % decode** (patch-off boot 26.38 vs patch-on boots 26.69–27.26); task accuracy unchanged (41/52 on and off); 52/54 prompts token-identical, the 2 others at near-ties | `DSV4_PATCH` |
| 8192-token prefill chunks | E10 | 64K prefill +55 %; 1M +4.9 % | `DSV4_MAX_BATCH` |
| `VLLM_HOST_IP` pinned to CX7 | E09 | No vLLM control-plane socket on WiFi; idle-latency failure gone | — |
| Split Lightning Indexer (`patches/dsv4-sparse-attn-indexer.py`) | E11–E14 | 1M TTFT −19.5 % (same client, same prompt) | `DSV4_IDX_MODE=split` (default) |
| FlashInfer #4732 backport (`patches/dsv4-fi4732-sparse_mla_sm120.so`) | E15 | Removes a prefill-kernel hang (FlashInfer tests: prebuilt 1/3 hung, fixed 0/3); speed unchanged | `DSV4_FI4732` |
| Peerwatch (leader), leaderwatch (worker) | 2026-10-19, I01 | Lost peer → both sides stopped; worked as designed in I03 | `DSV4_PEERWATCH`, `DSV4_LEADERWATCH` |
| Stale-watcher retirement at start | 2026-10-24 | Removes the leaked-guard false alert that ended F02 soak part 1 | — |
| Abort: retry the worker `docker inspect` (3×) | R01.7 | Fixes the 2026-10-24 case where one timed-out inspect left the worker running; not yet exercised | — |
| Boot-compaction helper (`tools/boot-compact.sh`) | R01.4–R01.6, R02 | Runs `gb10-reclaim` when memory PSI > 20 for 6 s during boot, before the guard's 30 s rule. First real intervention in R02: PSI 41 with no free order ≥ 9 blocks, one reclaim restored 1,821 blocks, PSI below 20 within 8 s, boot served | `DSV4_BOOT_COMPACT` |
| Thermal logging (`tools/thermlog.sh`) | E13, R01.7 | Both nodes, after serving starts; restarted 2026-10-26 after a 4-day gap | — |

Smaller operational changes: launcher `down` cleanup, observer `nvidia-smi` tolerance, persisted TileLang/Triton caches, static CX7 IPv4 profiles, leader WiFi pinned to 5 GHz, worker WiFi keepalive timer, fstrim D-state exception in the guard.

## 5. Tried and not adopted

| Item | Result and reason |
|---|---|
| E16: upstream indexer row sharding (vLLM #54951 + #61647, backported) | Works at 1M with correct retrieval on 2×GB10. **No measurable gain** against our split: TTFT 29.1 / 134.4 / 820.3 s vs 29.9 / 134.9 / 819.2 s (64K / 256K / 1M), and 4K decode is the same. Split stays the default. The rowshard patches are kept in `patches/e16-upstream-rowshard/` |
| NCCL diagnostics off (`DSV4_DIAG=0`) | No measurable speed difference (26.86 vs 26.78–27.26 tok/s). Diagnostics stay on for debugging (R01.3) |
| E03: `vm.compaction_proactiveness=0` | No benefit; the kernel default of 20 is kept |
| E03: pausing Velvet's `nvtop` | The stall continued |
| E07.1 / E07.2: NCCL logging or communication as the decode cost | Refuted (E07.5): NCCL is 5–8 % of the step |
| Raising `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | Rejected (INV-1): it would only delay the WiFi failure |
| Warm-up prompts | Not needed; the persisted runtime cache already removes the JIT delay |
| 12288/16384-token chunks | Not measured; the launcher requires an explicit KV size for them |
| Fused top-k merge kernel | Not done; ≤ about 2 % of 1M TTFT [inferred, E14] |
| NCCL small-message tuning | Not tried; ceiling ≤ 1 ms per step [hypothesis] |
| `HandlePowerKey=ignore` in logind | Declined by the user (I03) |
| DSpark speculative decoding | Approved (D10), not started. Largest remaining decode lever, and the highest-risk item (TP=2 rank divergence, memory headroom) |

## 6. Stability and incidents

| Event | Cause | Fix or mitigation | Status |
|---|---|---|---|
| E08: WiFi control-plane stalls (2026-10-13) | vLLM's ZMQ control path used the WiFi default route; roams broke it | `VLLM_HOST_IP` pinned to CX7 (E09) | Fixed |
| INV-2: CX7 lost its IPv4 (2026-10-14/15) | Cables moved f1 → f0; static IPs existed only on f1 | Static NetworkManager profiles on f0 | Fixed |
| Worker reboot not detected (2026-10-16) | Nothing watched the worker while serving | Peerwatch | Fixed (worked in I03) |
| Relaunch failures (2026-10-15/18) | `down` left guards and observers behind; worker clock lag | Launcher hardening | Fixed |
| Boot aborts from observer `nvidia-smi` (2026-10-19) | A timeout killed the observer | Observer tolerates the timeout | Fixed |
| fstrim false alarms | Weekly fstrim sits in D-state for minutes | Guard exception for exactly `fstrim` | Fixed |
| I01: leader WiFi driver deadlock (2026-10-22) | mt7925 roam to 6 GHz deadlocked the wiphy mutex [hypothesis: driver bug] | Leader pinned to a 5 GHz BSSID; leaderwatch on the worker | Mitigated; a wired management network is still recommended |
| **I02 class: boot stall at KV allocation** (2026-10-23; again 2026-10-24, 2026-10-26) | High-order memory fragmentation forces direct compaction under the GPU driver lock [inferred]. PSI 70–95 %; in I02 it lasted 11 min until a manual `gb10-reclaim`; on 2026-10-26 two stalls cleared by themselves in ~67–100 s, after the guard had already aborted | Boot-compaction helper (R01.4) | **Mitigation in place, unproven**; root cause not proven |
| I03: worker power-off (2026-10-23) | Short power-button press; not a crash | Peerwatch and soak behaved as designed | Closed |
| Orphaned worker after a refused start (2026-10-24) | A timed-out remote `docker inspect` made the abort refuse to stop the worker | Retry in `dsv4-abort.sh` (R01.7) | Fixed, not yet exercised |
| Leaked guard false alert (F02 part 1, 2026-10-24) | A guard from the previous run outlived it | Stale-watcher retirement at start | Fixed |
| R01.1 start failure (2026-10-25 22:45) | A bounded remote call to the worker timed out 8 s after a worker WiFi roam [hypothesis]; the fail message was not captured | None specific; the retry succeeded | Open, not reproduced |

**Soak results.**
- **F01** (final settings): 8 h 08 min clean, ended by I03.
- **F02** (run `dsv4-t2-20261024T181019Z-188432`): **passed.**
  - Served for 35 h without interruption, and the inference probe passed 66/66 (median 26.02 tok/s end-to-end).
  - The monitor covered 23.5 h; it was ended twice by monitoring false positives, both since fixed.
  - Memory PSI stayed ≤ 0.26, there were no kernel errors, and greedy output was bit-stable within the boot over 26 h.

**Boot record, 2026-10-25/26 restart series (R01).**
- 7 starts: 4 reached serving (patch off, diag off, rowshard, defaults), 2 aborted on the boot stall (both rowshard, before the helper existed), and 1 failed at a remote call (R01.1 attempt 2). One further attempt was refused for an uncleared alert, an operator step since documented.
- Each abort left both nodes clean, with no orphan and no reboot needed.

## 7. Final configuration

The launcher is `/srv/dsv4/run/start-dsv4-default.sh`, a modified copy of the deployment by our collaborator Velvet in `/srv/dsv4/live/` (D1). Every item below is a launcher default.

| Item | Value |
|---|---|
| Image | `local/vllm-openai:46638857-fi0618` (vLLM 0.26.1rc1.dev1219+g46638857f, FlashInfer 0.6.18.dev20260819) |
| Model | deepseek-v4-flash-0731, NVFP4, 73.85 GiB per rank |
| Parallelism and kernels | TP=2 across both nodes. KV `fp8_ds_mla`, attention `FLASHINFER_MLA_SPARSE_DSV4`, MoE `deep_gemm`. DSpark off. max-num-seqs 1. CUDA graphs FULL_AND_PIECEWISE |
| Prefill chunk | `--max-num-batched-tokens 8192` |
| KV cache | `--kv-cache-memory-bytes 11274289152` (1,094,398 tokens, 1.04× of 1M) |
| Scheduling | `--async-scheduling` (DSpark off) |
| Other flags | `--enable-prefix-caching --enable-chunked-prefill --no-enable-flashinfer-autotune` |
| Patches (bind-mounted, read-only) | `dsv4-attention.py` (GEMV), `dsv4-sparse-attn-indexer.py` (split indexer), `dsv4-fi4732-sparse_mla_sm120.so` (FlashInfer #4732) |
| Control plane | `VLLM_HOST_IP` = CX7 address (192.168.201.10 / .11) |
| Interconnect | CX7 f0 ports, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`, `NCCL_SOCKET_IFNAME=enP2p1s0f0np0`, GID index 5 |
| Diagnostics | `DSV4_DIAG=1`: `NCCL_DEBUG=INFO`, `TORCH_NCCL_DESYNC_DEBUG=1`; no measurable cost (R01.3) |
| Host | `vm.compaction_proactiveness=20`; GPU clocks uncapped |
| Start sequence | Admission → guards and observers → blocker preflight → stale-watcher retirement → fabric check → `gb10-reclaim` on both nodes, MemFree ≥ 100 GiB → boot-compaction helper → coordinated launch → post-load reclaim → readiness → guards to serving rules → peerwatch and leaderwatch |
| Guards | `dsv4-t2-guard.py`: boot rules (PSI > 40 for 30 s, D-state > 60 s), serving rules (MemAvailable < 6 GiB, D-state > 10 s, residual growth with PSI). Alerts latch until a human clears them |
| Runtime cache | `/srv/dsv4/runtime-cache/46638857-fi0618-sm121` |

Code, documentation and history are in the git repository `/srv/dsv4/run` (README.md, `docs/operations.md`, `docs/patches.md`, `data/README.md` for the raw data index).

## 8. Open items

1. **Boot-compaction helper: one success, not yet proven.** It logs each boot to `runs/<id>/<role>/boot-compact.log`. Success means a boot whose PSI passes 20 for 6 s reaches serving without the guard firing. R02 met this once (PSI above 20 for 16 s, peak 41, one reclaim, served); whether that boot would have aborted without the helper cannot be known, so more boots are needed. If it does not help, the guard aborts as before; the next step would then be a pre-KV compaction, or a longer boot window with a hard D-state limit.
2. **Decode target (29–33 tok/s).** It needs DSpark speculative decoding, a separate phase. Plan (from the 2026-10-21 survey):
   - one change from today's config;
   - per-rank collective-count diffs after every request;
   - a ladder of context sizes, stopping at the first mismatch or hang.
3. **Leader cooling.** The leader runs about 8 °C hotter, with SW thermal slowdown in 9 % of busy samples (E13). Thermal logging resumed on 2026-10-26; the airflow needs a physical check.
4. **Wired management network.** Recommended for both nodes; it removes the WiFi dependence behind I01 and I03 and possibly the R01.1 failure.
5. **Upstream.**
   - The GEMV patch as a vLLM issue or PR. It needs an SM12x gate, and ideally a same-boot A/B, which the bind-mount design cannot do without a restart.
   - An E16 comment on vLLM #54951 / #61647: works on 2×GB10 at TP=2 and 1M, with no speed change and the threshold at 8192.
6. **Smaller items.**
   - Unexplained cross-boot variation (decode 2.1 % over eight boots; token-level nondeterminism, E12b).
   - The ctxbench/e12b_client 1M TTFT difference.
   - Leaderwatch and the abort retry have not run in a real failure.
   - mt7925 upstream status.

## 9. Appendix: data locations

| Path | Contents |
|---|---|
| `/srv/dsv4/run/data/E07/`–`E15/` | Per-experiment data (see the log entries) |
| `/srv/dsv4/run/data/E16/20261026T140659-rowshard/` | E16 serving A/B (rowshard) |
| `/srv/dsv4/run/data/FINAL/20261025T135749-f02-boot/` | Final benchmark, with config.txt (args, mounts, md5s) |
| `/srv/dsv4/run/data/F01/` | Soak and probe logs for F01 and F02 |
| `/srv/dsv4/run/data/GEMV/` | GEMV accuracy runs, patch on and off |
| `/srv/dsv4/run/data/RESTART/<ts>-<label>/` | Per-boot 4K checks for the R01 series |
| `/srv/dsv4/run/data/thermal/` (leader), `~/dsv4-thermal/` (worker) | Thermal logs |
| `/srv/dsv4/run/data/investigation/` | INV-1 network write-ups |
| `~/.cache/dsv4-t2/runs/<run-id>/` | Guard state, alerts, peerwatch, post-load reclaim and boot-compact logs |
| `/srv/dsv4/diagnostics/<host>/` | Profiler traces, NCCL logs |
| `reports/research_notes/` | vLLM upstream and DSpark surveys |

---

**Changes in v1.1 (2026-10-26, R02).** Added the boot-compaction helper's first real
intervention (sections 4 and 8). A sixth patch-on boot (26.69 tok/s) widened the
boot-to-boot spread from 1.8 % to 2.1 % and the GEMV gain range from +1.5–3.3 % to
+1.2–3.3 %.

**Changes in v1.2 (2026-10-27, results check before upstream posts).** The boot counts
were wrong; the ranges were right. There are eight patch-on boots (not six), seven of
them with 8192-token chunks (median 26.86 tok/s, so the GEMV gain is +1.8 % at the
median), and two patch-off boots (E07.6 with 4096-token chunks, R01.1 with 8192). The
F02 restart check and the final benchmark are the same boot. The within-boot spread is
0.1–0.3 tok/s, not ≤ 0.2 (E10: 0.34).
