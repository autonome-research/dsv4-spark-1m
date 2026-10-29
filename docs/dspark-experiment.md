# DSpark experiment

DSpark is the speculative decoder that ships inside the DeepSeek-V4-Flash-0731 checkpoint
(`mtp.0–2`, block of 5 draft tokens). This page is the procedure for testing it on this
cluster safely, and the record of what has been checked. Background, sources and the speed
estimate: `reports/research_notes/DSpark upstream survey/` (`notes.md`, `update-2026-10-28.md`).
Every result goes into `reports/DSV4 optimisation log.md`.

**Status (2026-10-28):** Stage 0 (offline checks) done. Stage 1 not started; it needs a
window agreed with Velvet. DSpark is **off** in the default start and stays off until
Stage 2.

## Contents

- [Why and what to expect](#why-and-what-to-expect)
- [Isolation model](#isolation-model)
- [Switches](#switches)
- [Stage 0: offline checks](#stage-0-offline-checks)
- [Stage 1: serving test](#stage-1-serving-test)
- [Stop rules](#stop-rules)
- [Rollback](#rollback)
- [Stage 2: adoption](#stage-2-adoption)

## Why and what to expect

Decode today is 26.7–27.3 tok/s at 4K and 25.1 at 1M, close to the memory-bandwidth
limit for one token per step. DSpark drafts 5 tokens and the target checks all of them in
one 6-token step, so one step can emit up to 6 tokens.

- **Estimate** [derived, research update §2]: about 35–60 tok/s at 4K and 26–60 tok/s at
  1M, depending on how many drafts are accepted.
- **Cost:** a 6-token verify step reads 3.3–5.7× more expert weights than a 1-token step,
  about 50–71 ms. A community measurement on 2× GB10 is 74 ms.
- **Memory:** the draft adds about 5 GiB per node. The estimate is about 16 GiB free while
  idle and about 11 GiB at the 1M peak, against the guard's 6 GiB floor.
- Our KV cache size is fixed (`kv_cache_memory_bytes`). vLLM then skips the
  `gpu_memory_utilization` check (`v1/worker/gpu_worker.py`), so the extra weights cannot
  make the start fail; they only reduce free memory.

## Isolation model

A separate copy of the stack is not possible: the model needs both nodes (about 74 GiB of
weights on each) and there is no spare GPU. Isolation comes from the design instead:

- **Per-start switches:** every change in this experiment is a switch. The default start
  is unchanged.
- **Read-only patches:** patches are mounted read-only over the stock image. The image is
  never modified. `/srv/dsv4/live` is never touched.
- **Per-run records:** each run has its own run ID, state directory and data directory.
- **Rank consistency:** the two ranks must agree on every setting that changes their
  collectives. `node-launch.sh` prints an `indexer mode=` line and a `decode spec= async=
  prefix_cache= gemv= dspark_loader=` line, and `launch-dsv4.sh` aborts the start if
  either line differs between the nodes.
- **Safety nets stay on:** the memory guard, peerwatch, leaderwatch and the boot
  helper.
- **Rollback** is the default start, about 6 minutes ([Rollback](#rollback)).

The one risk this cannot remove is a GPU left spinning at 96 % that only a reboot clears
(Velvet, 2026-08-21). We cannot reboot (no sudo) and do not press power buttons, so Velvet
must be reachable during every Stage 1 window.

## Switches

All are environment variables for `start-dsv4-default.sh`. Defaults are the adopted
configuration.

| Variable | Default | Experiment value | Effect |
|---|---|---|---|
| `DSV4_SPEC` | `off` | `on` | DSpark k=5, probabilistic draft sampling, adaptive verification off, CUDA-graph capture size 6. Requires `DSV4_DIAG=1` |
| `DSV4_PATCH` | `1` | `m1` | GEMV patch variant. `m1` chunks only at M = 1; `1` also chunks at M = 2–8, which is slower ([patches.md](patches.md#chunked-compressor-gemv)) |
| `DSV4_ASYNC` | `0` with DSpark | `0`, then `1` | `--async-scheduling`. Our image allows it with DSpark (`config/vllm.py`); untested here |
| `DSV4_PREFIX_CACHE` | `1` | `0`, then `1` | Prefix caching. DSpark plus prefix caching had draft-window bugs upstream, so test it last |
| `DSV4_DSPARK_LOADFIX` | `0` | `1` only if needed | Mounts `patches/dsv4-dspark.py`: the upstream #53984 guard for the #51916 `KeyError`, plus a warning that counts skipped draft expert params |

## Stage 0: offline checks

Done on 2026-10-28 without touching the serving model. Containers ran with no network and
no GPU.

| Check | Result | Evidence |
|---|---|---|
| vLLM's DSpark unit tests at our commit `46638857` (config, adaptive verification, top-k, synthetic rejection sampler, DSpark MLA model) | **42 passed**, 1 skipped (needs CUDA). The sparse-MLA kernel test was not run: GPU only | `data/DSPARK-S0/pytest-cpu-20261029T052547Z.log` |
| GEMV gate: `dsv4-attention-m1.py` chunks only at M = 1 | **PASS**. The test fails on the M ≤ 8 file, as required | `tools/test-gemv-gate.py` |
| Graph-mode corruption #62417 / #59986 applies to us? | **No.** MRV2 uses the weakest builder support, and ours is `UNIFORM_BATCH`, so prefill never enters FULL graphs | research update §1 |
| Async with DSpark allowed by the image? | **Yes** (`dspark` in the allowed list) | `config/vllm.py` in the image |
| Memory cap with the draft weights | Start cannot fail on it; free memory as estimated above | `v1/worker/gpu_worker.py` in the image |
| Draft loader `KeyError` (#51916) | **Cannot be settled offline.** The image has the unguarded line. It fails loudly at start if it happens; the fix is ready (`DSV4_DSPARK_LOADFIX=1`) | `patches/dsv4-dspark.py` |
| Ladder's rank check | Agrees on the serving run (5,771 eager collectives) and detects a difference between two runs | `data/DSPARK/20261028T222940-selftest/` |

Run the gate test again after any change to the patch files:

```bash
docker run --rm --network none -v /srv/dsv4/run:/r:ro --entrypoint python3 local/vllm-openai:46638857-fi0618 /r/tools/test-gemv-gate.py /r/patches/dsv4-attention-m1.py
```

## Stage 1: serving test

Each step is one change from the previous one. `tools/dspark-ladder.sh` runs the requests
and applies the [stop rules](#stop-rules) after each one. Its output goes to
`data/DSPARK/<timestamp>-<label>/`. The commands below are typed by the operator, one at a
time.

**Before the window:** agree the time with Velvet, and make sure Velvet can reboot a node
if needed (worker first, leader last).

**1. Reference with DSpark off**, on the default boot that is serving:

```bash
/srv/dsv4/run/tools/dspark-ladder.sh ref-off check tiny 4k acc 65536
```

**2. Stop the default run and clear the alerts on both nodes:**

```bash
/srv/dsv4/run/launch-dsv4.sh down
```

```bash
python3 /srv/dsv4/run/dsv4-t2-guard.py clear --confirm "CLEAR T2 ALERTS"
```

```bash
ssh gx10 'python3 ~/dsv4-t2-guard.py clear --confirm "CLEAR T2 ALERTS"'
```

**3. Arm A: start DSpark** with async off, prefix caching off and the M = 1 GEMV patch,
then type `START DSV4 T2`:

```bash
DSV4_SPEC=on DSV4_PATCH=m1 DSV4_ASYNC=0 DSV4_PREFIX_CACHE=0 /srv/dsv4/run/start-dsv4-default.sh
```

If the start fails with `KeyError` in `dspark.py` `load_weights` (#51916), clear the
alerts and start again with `DSV4_DSPARK_LOADFIX=1` added. Then look for the
`DSV4 patch: skipped N unregistered DSpark expert params` warning in the leader log, and
record N.

**4. Arm A, short context first:**

```bash
/srv/dsv4/run/tools/dspark-ladder.sh A-short check tiny 4k acc
```

Compare the 52 fixed prompts with the reference. The target model samples greedily, so
outputs should match except at near-tied tokens, as in the GEMV check:

```bash
/srv/dsv4/run/tools/gemv-acc.py compare data/GEMV/dspark-off-ref-off-<ts>.json data/GEMV/dspark-on-A-short-<ts>.json
```

**5. Arm A, long context**, one size per run so that a failure is unambiguous:

```bash
/srv/dsv4/run/tools/dspark-ladder.sh A-64k 65536
```

```bash
/srv/dsv4/run/tools/dspark-ladder.sh A-256k 262144
```

```bash
/srv/dsv4/run/tools/dspark-ladder.sh A-660k 660000
```

```bash
/srv/dsv4/run/tools/dspark-ladder.sh A-1m 1036288
```

**6. Arm B: async on.** Restart as in step 3, but with `DSV4_ASYNC=1`, then repeat steps
4 and 5 with labels `B-…`.

**7. Arm C: prefix caching on.** Restart with the better async setting and
`DSV4_PREFIX_CACHE=1`, then repeat steps 4 and 5. Also send the same 64K prompt twice
(`dspark-ladder.sh C-64k 65536 65536`): the second request is a prefix-cache hit, which is
the case that broke DSpark upstream.

**8. Return to the default configuration** ([Rollback](#rollback)), unless Stage 2 starts.

## Stop rules

`dspark-ladder.sh` stops at the first broken rule (exit code 4, reason in `ladder.log`)
and leaves the server running for diagnosis.

1. **The ranks ran different eager NCCL collectives.** This is the signature of Velvet's
   2026-08 hangs: rank 0 did an extra 6-row logits gather. Collectives inside CUDA graphs
   are not logged (E07.1), so this rule covers eager collectives only.
2. **Wrong answer or failed request:** a hidden code is not found, or the client fails.
3. **MemAvailable below 10 GiB** on either node. The request is aborted below 6 GiB.
4. **Draft acceptance below 40 %**, which points to a draft-loader defect.

Also stop, by hand, if a GPU sits at about 96 % while no request runs, or a request makes
no progress for 5 minutes. Before rolling back, take the evidence:

```bash
/srv/dsv4/run/launch-dsv4.sh status
```

```bash
nvidia-smi
```

Keep the run directory (`~/.cache/dsv4-t2/runs/<run>/`), the ladder directory and
`/srv/dsv4/diagnostics/<host>/` on both nodes. Record everything in the log.

## Rollback

1. Stop: `/srv/dsv4/run/launch-dsv4.sh down`.
2. Clear the alerts on both nodes (the commands in Stage 1 step 2).
3. Start the default configuration, with no `DSV4_*` variables set, and type
   `START DSV4 T2`:

   ```bash
   /srv/dsv4/run/start-dsv4-default.sh
   ```

4. Run the 4K check (`tools/boot-4k.sh rollback`). Expect 26.7–27.3 tok/s.

If a GPU still shows about 96 % after the stop, with no process using it: run
`sudo -n /usr/local/sbin/gb10-reclaim` on that node, wait a minute, and check again. If
it persists, message Velvet for a reboot (worker first, leader last). Do not press power
buttons.

## Stage 2: adoption

DSpark becomes the default only if one arm meets all of the following:

- it passes every rung up to 1M with no stop;
- 4K decode is at least 1.2× the reference (`4k` rung);
- the 52-prompt accuracy is no worse than DSpark off;
- it then passes a 24 h soak with the inference probe (`tools/soak.sh`,
  `tools/infer-probe.py`).

Changing the default is Velvet's decision. It is recorded as a decision in the log, and
`start-dsv4-default.sh`, the README and `docs/operations.md` are updated together.
