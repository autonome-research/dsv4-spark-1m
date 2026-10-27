# Patches

All patches are **bind-mounted read-only** over files in the stock image
`local/vllm-openai:46638857-fi0618` (vLLM commit `46638857fdbb`) when the container
starts. The image itself is never modified. Each patch can be switched off per
start with an environment variable. `node-launch.sh` prints the md5 (8 hex digits)
of the mounted indexer files and `launch-dsv4.sh` refuses a start whose ranks differ;
`tools/final-bench.sh` records the full md5 of every mounted file in `config.txt`.

| Patch | File(s) in `patches/` | Mounted over | Switch | Default |
|---|---|---|---|---|
| [Chunked compressor GEMV](#chunked-compressor-gemv) | `dsv4-attention.py` | `vllm/models/deepseek_v4/attention.py` | `DSV4_PATCH` | on |
| [Split indexer](#split-indexer) | `dsv4-sparse-attn-indexer.py` | `vllm/model_executor/layers/sparse_attn_indexer.py` | `DSV4_IDX_MODE=split` | on |
| [Indexer row sharding (E16)](#indexer-row-sharding-e16) | `dsv4-rowshard-sparse-attn-indexer.py`, `dsv4-rowshard-mla-indexer.py`, `dsv4-rowshard-config-attention.py` | `.../sparse_attn_indexer.py`, `vllm/v1/attention/backends/mla/indexer.py`, `vllm/config/attention.py` | `DSV4_IDX_MODE=rowshard` | off (E16: no gain; kept for reference) |
| [FlashInfer #4732](#flashinfer-4732-backport) | `dsv4-fi4732-sparse_mla_sm120.so` | `flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so` | `DSV4_FI4732` | on |

Paths are relative to `/usr/local/lib/python3.12/dist-packages/` inside the
container. `*.orig.py` files are the unmodified image files, kept so every patch
can be reviewed with a plain `diff`:

```bash
diff -u patches/dsv4-attention.orig.py patches/dsv4-attention.py
diff -u patches/dsv4-sparse-attn-indexer.orig.py patches/dsv4-sparse-attn-indexer.py
```

> **Do not edit a patch file in place.** The optimisation log and every benchmark's
> `config.txt` identify the patches by md5. A change, even to a comment, makes
> earlier results unattributable. Add a new file and a new switch instead.

Reference md5s (first 10 hex digits):

| File | md5 |
|---|---|
| `dsv4-attention.py` | `cd6b032cee` |
| `dsv4-sparse-attn-indexer.py` | `ffcdc4a05c` |
| `dsv4-rowshard-sparse-attn-indexer.py` | `81456b6a7d` |
| `dsv4-rowshard-mla-indexer.py` | `98276bb8ef` |
| `dsv4-rowshard-config-attention.py` | `92d09a1145` |
| `dsv4-fi4732-sparse_mla_sm120.so` | `23acbe9f7a` |

---

## Chunked compressor GEMV

**Problem.** At decode (M ≤ 8 rows), the two BF16 `fused_wkv_wgate` projections in
the DeepSeek-V4 compressor run as cuBLAS GEMVs. On GB10, cuBLAS reaches about
160 GB/s for N = 1024/2048 but about 220 GB/s for N = 512.

**Change.** When M ≤ 8, the projection is issued in 512-row chunks. Prefill is
unchanged.

**Result.** 4K decode **+1.2 % to +3.3 %**: 26.38 tok/s with the patch off (R01.1)
against 26.69–27.26 tok/s on six patch-on boots (R01.3, R01.6, R02). The range reflects
the 2.1 % boot-to-boot spread, because a bind-mounted patch cannot be switched within
one boot. The first measurement (E07.8, 26.64 → 26.93, +1.1 %) was also across boots.

**Numerics.** In isolation, decode outputs differ only by fp32 summation order
(≤ 1.5e-7 relative); prefill is bit-exact. Served end to end (`tools/gemv-acc.py`,
52 fixed prompts, greedy):
- task accuracy 41/52 with the patch on (3 passes) and off (2 passes);
- 52 of 54 outputs token-identical, mean |Δlogprob| 0.0043; the two divergences are at
  near-tied tokens (p ≈ 0.5–0.6) and both versions are correct;
- each configuration is deterministic within its boot, and patch-on output was
  bit-stable over 26 h of uptime.

---

## Split indexer

**Problem (E11).** With TP = 2 the DeepSeek-V4 Lightning Indexer is replicated: both
ranks computed the full indexer logits and top-k for every prefill chunk. At 1M
context this was about 46 % of prefill time [inferred from the E11 profile].

**Change.** For a sub-chunk with at least 32,768 keys, each rank scores half of the
keys (`[0, s)` or `[s, n)`, with `s` a multiple of 64 because DeepGEMM rejects an
unaligned K-scale slice). Each rank takes its local top-512, the two candidate
lists are all-gathered over the TP group, and a final top-k merges them.

The merge is exact: a key in the global top-k is necessarily in the top-k of its
own half. Below 32,768 keys the gather costs more than it saves, so the stock path
is used. The path is also skipped for FP4 caches, DCP > 1, XPU, TP ≠ 2 and during
CUDA graph capture.

**Result.** 1M TTFT 1,010.3 s → 813.7 s (−19.5 %) on an identical prompt with the
same client (E12b). At ~230K context the step time fell 13 %: indexer logits −52 %,
top-k −70 % (E14).

**Validation.** Offline (E12a, E12b precheck): identical top-k rows apart from exact
score ties. Token-level comparison across boots is not usable as evidence because
greedy outputs already differ between boots with an unchanged configuration.

---

## Indexer row sharding (E16)

A port of upstream vLLM [#54951](https://github.com/vllm-project/vllm/pull/54951)
and [#61647](https://github.com/vllm-project/vllm/pull/61647) onto our vLLM commit.
Upstream shards the indexer's prefill *rows* (queries) across TP ranks instead of
splitting the *keys* as our split indexer does. If it matches the split indexer,
it replaces a local patch with an upstream design.

- Source, provenance checks and per-hunk notes:
  [`patches/e16-upstream-rowshard/PORT.md`](../patches/e16-upstream-rowshard/PORT.md).
- Combined diff: `port-combined.diff` (3 files, +185/−17). The working tree's git
  history is kept as `tree.bundle` (`git clone tree.bundle tree` restores it).
- Tests (CPU): `tests/sim_row_shard_cpu.py` (row bookkeeping, 400/400 cases at
  TP 2/3/4) and the upstream `test_indexer_tp_row_shard.py` (12/12).
- Upstream's default `indexer_tp_shard_min_rows` is 65,536, which never engages with
  our 8,192-token prefill chunks. The launcher passes
  `--attention-config '{"indexer_tp_shard_min_rows": 8192}'`
  (`DSV4_ROWSHARD_MIN_ROWS`).
- All three files must be mounted on **both** ranks, or the ranks disagree on the
  collective and NCCL hangs. `launch-dsv4.sh` aborts the start if the two ranks
  report different indexer modes or file md5s.

**Result (E16, 2026-10-26).** Served correctly at every size, with no measurable
speed difference from the split indexer (same client and prompts, `tools/e16-ab.sh`):

| Context | Row sharding | Split indexer (E15) |
|---|---|---|
| 64K | 29.1 s | 29.9 s |
| 256K | 134.4 s | 134.9 s |
| 1M | 820.3 s | 819.2 s |

4K decode was also unchanged (26.86 tok/s). The split indexer stays the default; the
port is kept as a reference and as GB10 TP = 2 data for the upstream threshold.

---

## FlashInfer #4732 backport

**Problem.** The image's FlashInfer (`0.6.18.dev20260819`) predates
[flashinfer#4732](https://github.com/flashinfer-ai/flashinfer/pull/4732), which fixes
a named-barrier hang in the SM120/SM121 sparse MLA prefill kernel
(`sparse_mla_prefill_mg_dual_kernel`). Consecutive double-buffered KV tiles shared
named barrier 1, so a delayed IO warp group could see two arrivals in one barrier
phase and hang with the GPU busy. The fix alternates the barrier ID by tile parity.

**Change.** The prebuilt module is loaded in preference to a JIT build, so patched
headers alone have no effect. The module was rebuilt offline from the patched
sources and the resulting `.so` is mounted over the prebuilt one. Sources, the
upstream diff and the analysis are in
[`patches/fi4732/README.md`](../patches/fi4732/README.md).

**Validation (E15).** FlashInfer's own sparse-MLA prefill tests (163 tests, 3 rounds
per module): the prebuilt module hung in 1/3 rounds, our unfixed build in 3/3 and
the fixed build in 0/3. Serving speed was unchanged (+0.7 % at each of 64K, 256K and
1M, within the boot-to-boot spread), and retrieval was correct at every size. Three rounds is a small sample; the
upstream analysis and merge remain the main evidence.
