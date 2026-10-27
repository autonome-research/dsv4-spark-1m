# Backport: vLLM #54951 + #61647 (indexer TP prefill row sharding) onto 46638857fdbb

Base = the image original (`local/vllm-openai:46638857-fi0618`, vllm at
`/usr/local/lib/python3.12/dist-packages/vllm`, `VLLM_BUILD_COMMIT=46638857fdbb30e0…`).
This is not our bind-mounted `dsv4-sparse-attn-indexer.py`.

## Provenance checks (step 1)
| file | image vs upstream raw @46638857 | other |
|---|---|---|
| vllm/model_executor/layers/sparse_attn_indexer.py | byte-identical | byte-identical to `/srv/dsv4/run/patches/dsv4-sparse-attn-indexer.orig.py` (sha256 fc112355…) |
| vllm/v1/attention/backends/mla/indexer.py | byte-identical | |
| vllm/config/attention.py | byte-identical | |

PRs: #54951 was merged as 3e8e1364ff41 (base 61f084b5). #61647 was merged as 24f632ca7e0d.
Upstream at 3e8e1364 is 2,429 commits ahead of 46638857, which explains the drift.

## Tree (`tree/`, git; not tracked, restore with `git clone tree.bundle tree`)
```
694f448 Port vllm#61647: engage indexer TP row sharding on total rows
23eeda9 Port vllm#54951: shard long-context indexer prefill rows across TP
a9298c1 base
```
Combined diff: `port-combined.diff`. Three files, +185/−17. #54951 is +182/−16 and #61647 is +8/−6.
The earlier port, whose files were lost, was recorded as +186/−17. The source of the one-line difference is not known. The likeliest cause is how the
`candidate_blocks` conditional was collapsed: here it is 2 lines, `# One row window per rank.` and a single-line assignment.

Source identity: an `ast` source-segment comparison against upstream @24f632ca matches every one of:
`PrefillRowShard`, `_reserve_row_shard_exchange_transient`,
`balanced_prefill_row_shard`, `tp_prefill_row_sharding_supported`,
`DeepseekV32IndexerPrefillMetadata.row_shard_sizes`, `AttentionConfig.indexer_tp_shard_min_rows`.
Only call sites differ.

## Per-hunk notes (applied with `patch -p1`, fuzz ≤ 2, then checked by hand)
### #54951 (16 hunks in scope)
sparse_attn_indexer.py
| # | result | note |
|---|---|---|
| H1 imports | clean (offset −1) | `get_current_vllm_config_or_none`, `get_tensor_model_parallel_rank`, `get_tp_group` are all exported at 46638857 (checked in `vllm/config/__init__.py` and `vllm/distributed/parallel_state.py`) |
| H2 import `tp_prefill_row_sharding_supported` | clean (offset) | |
| H3 `PrefillRowShard` + `_reserve_row_shard_exchange_transient` | clean (offset) | verbatim |
| H4 dummy-run reservation call | clean (offset) | |
| H5 `row_shard = …` | fuzz 2, then hand-adapted | Upstream reads `None if candidate_blocks is not None else …`. The DSV4.1 `candidate_blocks` path doesn't exist here, so it becomes `row_shard = PrefillRowShard.from_metadata(prefill_metadata, num_decode_tokens)`, and "DSV4.1 candidate rows stay replicated" is dropped from the comment |
| H6 narrowing block | rejected, hand-ported | Upstream anchors it before the "PCP + DCP KV all-gather" block, which doesn't exist here. Inserted verbatim right after `cp_gather_indexer_k_quant_cache`, before `q_slice`. This keeps the upstream invariant that the KV gather stays unconditional, because continuation sub-chunks reuse the workspace |
| H7 `q_slice`/`q_scale_slice`/`topk_indices` use `row_start:row_end` | fuzz 2 | Fuzz comes from the missing PCP+DCP context. It landed at the right spot and the content is verbatim |
| H8, H9 `weights[row_start:row_end]` (XPU + DeepGEMM) | clean (offset) | |
| H10 `row_starts=cu_seqlen_ks` + `exchange_topk` after loop | rejected, hand-ported | Upstream's `_merge_dcp_topk_global(...)` call sits one indent level deeper. The same change was made at our indentation |
| (base `if chunk.local_total_seq_lens == 0:`) | not changed | Upstream also checks `or q_slice.shape[0] == 0`, but that's a context line from another upstream commit. `narrow()` already `continue`s on empty windows |

indexer.py
| # | result | note |
|---|---|---|
| H1 `CUDAGraphMode` import | rejected, hand-ported | Different neighbouring imports here (no `get_current_vllm_config`, different jit_warmup import). Result: `from vllm.config import CUDAGraphMode, VllmConfig` |
| H2 `MIN_TP_SHARD_ROWS_PER_RANK`, `balanced_prefill_row_shard`, `tp_prefill_row_sharding_supported` | fuzz 1 (offset −55) | Placed before `class DeepseekV32IndexerBackend`, verbatim |
| H3 `row_shard_sizes` dataclass field | rejected, hand-ported | Base has no `max_prefill_seq_len` field. The new field goes right after `chunks` |
| H4 `self.tp_size = parallel_config.tensor_parallel_size` | clean (offset −408) | |
| H5 gate + `balanced_prefill_row_shard(seq_lens_cpu[num_decodes:], …)` | rejected, hand-ported | Verbatim block placed after the chunk loop. `seq_lens_cpu` here is `seq_lens_cpu_upper_bound`, which is uncompressed and exact for prefill rows, the same quantity upstream uses |
| H6 pass `row_shard_sizes=` | rejected, hand-ported | Base built `DeepseekV32IndexerPrefillMetadata(chunks)` with no `max_prefill_seq_len=`, so the call is now `(chunks, row_shard_sizes=row_shard_sizes)` |

Skipped in #54951:
- `vllm/models/glm5next/nvidia/sparse_indexer.py` (9 hunks). The file doesn't exist at 46638857 (HTTP 404). It is GLM's separate k-pool indexer and is not used by DeepSeek.
- The test file (1 hunk). Ported separately, see Tests.

### #61647 (3 hunks in scope)
| file / # | result | note |
|---|---|---|
| attention.py H1 `indexer_tp_shard_min_rows: int = Field(default=65536, ge=1)` | rejected, hand-ported | Upstream's anchors (the candidate-pool docstring and `hisparse_config`) don't exist here. Placed after `indexer_kv_dtype`. Also needs `from pydantic import Field, field_validator`, because base imported only `field_validator` |
| indexer.py H1 drop `MIN_TP_SHARD_ROWS_PER_RANK`, add `min_rows`, floor `max(min_rows, tp_size)` | fuzz 1 | verbatim |
| indexer.py H2 pass `attention_config.indexer_tp_shard_min_rows` | clean (offset −582) | |

Test file (3 hunks): the final upstream test @24f632ca was taken as a whole rather than replaying the hunks.

Totals over both PRs: 19 in-scope hunks. 8 applied clean (offset only), 4 with fuzz (one of them, H5, then hand-adapted), and 7 rejected and hand-ported.
The earlier, lost note said "5 of 18 failed", so its hunk accounting differed from this one.

No other file needs a change for the feature to work.
- `SparseAttnIndexer.forward_cuda` already calls `torch.ops.vllm.sparse_attn_indexer`, and DeepSeek-V4 uses that op (`models/deepseek_v4/attention.py`).
- The other `prefill.chunks` consumers (`flashmla_sparse.py`, `rocm_aiter_mla_sparse.py`) read MLA metadata, not indexer metadata.
- After the exchange every rank holds the full top-k buffer, so downstream code sees no difference.

## Tests
- `python3 -m py_compile` passes on all three files (and on the ported test).
- `tests/sim_row_shard_cpu.py`: plain Python, because the host has neither torch nor numpy.
  - It lifts the real `PrefillRowShard` class out of the ported file via `ast` and runs it per rank in threads, with a barrier-based `all_gatherv` and a list-backed int32 buffer.
  - It mirrors the ported chunk loop, with sub-chunks and with chunks skipped when they hold 0 KV.
  - Scores depend on the global row id, so a misaligned q/bounds window changes the answer.
  - Decode rows, padding rows and tail columns carry sentinels, so it also checks that the exchange doesn't touch them.
  - Row counts come from a transcription of `balanced_prefill_row_shard` (80%) or from random contiguous splits.

  `python3 -I tests/sim_row_shard_cpu.py --src ../dsv4-rowshard-sparse-attn-indexer.py --cases 400` (the mounted file is the ported `sparse_attn_indexer.py`):
  ```
  TP2: 400/400 match (sharded/engaged 382, replicated 18)
  TP3: 400/400 match (sharded/engaged 387, replicated 13)
  TP4: 400/400 match (sharded/engaged 380, replicated 20)
  RESULT: PASS
  ```
  Broken copies (`tests/make_broken.py` writes them to `tests/broken/`) all FAIL with exit 1:
  | mutation | TP2 / TP3 / TP4 matches out of 400 |
  |---|---|
  | `bounds_not_offset` (ks/ke not offset by window start) | 61 / 35 / 45 |
  | `no_decode_offset` (start forgets num_decode_tokens) | 118 / 167 / 153 |
  | `window_off_by_one` (stop − 1) | 26 / 17 / 32 |
  | `gather_misplaced` (exchange shifted by one row) | 18 / 13 / 20 (shape errors) |
  The cases that still match in the broken runs are the replicated or decode-free ones, where the mutation has no effect.
  - Caveat: the `balanced_prefill_row_shard` used in the simulation is a transcription, not the torch code itself, because torch is not installed on the host. The upstream test covers the real function.
- `tests/test_indexer_tp_row_shard.py`: the upstream test @24f632ca. The only change is dropping the `pcp_shard_decode_requests=False` kwarg, which `sparse_attn_indexer()` at 46638857 doesn't have.
  - It is CPU-tensor + monkeypatch based and needs torch and vllm importable. Run on 2026-10-24 in a throwaway container from the serving image (`--network none`, no GPU), installed as `tests/v1/attention/test_indexer_tp_row_shard.py`: **12 passed** in 14.8 s (optimisation log, E16).
- Helpers: `tests/_check_identical.py` and `tests/_cmp_defs.py` (identity checks), and `tests/make_broken.py`.

## How the feature engages (verified against the ported code)
Gate: `tp_prefill_row_sharding_supported()` in indexer.py. The builder evaluates it in every `build()` that has prefills, and it is also evaluated in the profiling run. All of these must hold:
1. `current_platform.is_cuda()`. This one was missing from the old notes.
2. `decode_context_parallel_size == 1` (DCP=1).
3. `not use_pcp`, i.e. `prefill_context_parallel_size == 1`.
4. `tensor_parallel_size > 1`.
5. `not VLLM_DISABLE_PYNCCL`. The exchange uses `get_tp_group().all_gatherv(sizes=…)`, and `CudaCommunicator.all_gatherv` asserts PyNCCL is enabled.
6. `not VLLM_USE_NCCL_SYMM_MEM`. With uniform sizes, `all_gatherv` would otherwise switch to the symmetric-memory path.
7. `not VLLM_BATCH_INVARIANT`.
8. `compilation_config.cudagraph_mode.mixed_mode() != FULL`. Only `FULL` is excluded; `NONE`, `PIECEWISE`, `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE` are allowed. The upstream test checks `FULL_AND_PIECEWISE`.

Then `balanced_prefill_row_shard()` returns sizes only if:

9. The total prefill query rows in the step are at least `max(attention_config.indexer_tp_shard_min_rows, tp_size)`.
   - The count is a sum over prefill requests only. Decode tokens, and short extends that the indexer classifies as decodes, don't count.
   - Since #61647 this is TOTAL rows, not per rank.
   - The default 65536 can never be reached with `max_num_batched_tokens=8192`. Set it with `--attention-config '{"indexer_tp_shard_min_rows": 8192}'`: `EngineArgs` builds `AttentionConfig(**dict)`. `ge=1` should be enforced by pydantic, the same `Field` pattern `CompilationConfig` uses; this was not tested.
   - With an 8192-token budget and min_rows 8192, sharding engages only on steps whose 8192 tokens are all prefill rows: any decode token in the step pushes prefill rows below 8192. A lower value such as 4096 would engage more often. Whether that is profitable is untested.
10. The total work estimate is > 0, i.e. Σ (first_key + row) // compress_ratio > 0.

In `sparse_attn_indexer()` it then runs only on the prefill path. It does not run when the op returns early for dense-MHA batches: non-FULL runtime mode, the main MLA's `prefill.use_dense_mha`, which is set when max prefill seq_len ≤ index_topk and `sparse_mla_force_mqa` is off, and `num_decode_tokens == 0`. Long-context prefills never take that early return.

Side effect: the profiling run reserves an extra `2 × rows × width × 4 B` transient whenever the gate (1–8) passes, even if condition 9 never triggers.

## Deployment and result
`DSV4_IDX_MODE=rowshard` (node-launch.sh) mounts the three ported files on both ranks in place of the split-indexer file, so the key split and the row sharding never run together, and passes `--attention-config '{"indexer_tp_shard_min_rows": 8192}'` (`DSV4_ROWSHARD_MIN_ROWS`). `launch-dsv4.sh` refuses a start whose ranks report different indexer files.

GPU serving A/B (optimisation log E16, 2026-10-26, `tools/e16-ab.sh`, same prompts as E15): TTFT 29.1 / 134.4 / 820.3 s at 64K / 256K / 1M against 29.9 / 134.9 / 819.2 s for the split indexer, with correct retrieval at every size and unchanged 4K decode. No measurable gain at TP = 2 on GB10, so the split indexer stays the default.
