#!/usr/bin/env python3
"""CPU simulation of the indexer TP prefill row-sharding bookkeeping
(vllm#54951 + #61647 as ported onto 46638857fdbb). Plain Python only.

What it exercises:
  * The REAL ``PrefillRowShard`` class, lifted by ``ast`` out of the ported
    sparse_attn_indexer.py (``--src``), run against a list-backed 2-D int32
    buffer and a threaded ``all_gatherv`` (one thread per TP rank, barrier).
  * A mirror of the ported prefill chunk loop: per (sub-)chunk each rank
    narrows its query rows and causal bounds (cu_seqlen_ks/ke), "scores" only
    those rows, writes top-k into its buffer, then exchanges int32 top-k
    rows. Rows' scores depend on the GLOBAL row id, so a misaligned q/bounds
    window changes the answer.
  * Row counts come from a pure-Python transcription of
    ``balanced_prefill_row_shard`` (incl. the #61647 total-row floor
    ``max(min_rows, tp_size)``), plus some arbitrary splits.

Every rank's final buffer must equal the unsharded (replicated) result
exactly, including decode rows, CUDA-graph padding rows and tail columns
beyond the top-k width, which the exchange must not touch.

Usage: python3 -I sim_row_shard_cpu.py --src PATH [--cases N] [--seed S]
Exit status 0 = all cases match; 1 = any mismatch/exception.
"""

import argparse
import ast
import random
import sys
import threading
from types import SimpleNamespace

DECODE_SENTINEL = -5
PAD_SENTINEL = -7
TAIL_SENTINEL = -9


# ---------------------------------------------------------------- fake tensor
class Buf:
    """Minimal 2-D int buffer: [rows, cols] slicing returns copies, slice
    assignment writes in place with strict shape checks."""

    def __init__(self, rows):
        self.r = [list(x) for x in rows]

    @property
    def shape(self):
        return (len(self.r), len(self.r[0]) if self.r else 0)

    def __getitem__(self, key):
        rs, cs = key
        return Buf([row[cs] for row in self.r[rs]])

    def contiguous(self):
        return self

    def __setitem__(self, key, val):
        rs, cs = key
        idx = range(*rs.indices(len(self.r)))
        if len(idx) != len(val.r):
            raise ValueError(f"row mismatch {len(idx)} != {len(val.r)}")
        for i, row in zip(idx, val.r):
            a, b, _ = cs.indices(len(self.r[i]))
            if b - a != len(row):
                raise ValueError(f"col mismatch {b - a} != {len(row)}")
            self.r[i][a:b] = row


# ------------------------------------------------------- fake TP collectives
class FakeTP:
    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=10)
        self.slots = [None] * world
        self.tls = threading.local()

    def rank(self):
        return self.tls.rank

    def group(self):
        tp = self

        def all_gatherv(local, dim=0, sizes=None):
            assert dim == 0
            r = tp.rank()
            if sizes is not None and len(local.r) != sizes[r]:
                raise ValueError(f"rank {r}: {len(local.r)} != sizes {sizes[r]}")
            tp.slots[r] = [list(x) for x in local.r]
            tp.barrier.wait()
            out = [row for s in tp.slots for row in s]
            tp.barrier.wait()
            return Buf(out)

        return SimpleNamespace(all_gatherv=all_gatherv, world_size=tp.world)


def load_prefill_row_shard(src_path, tp):
    """Exec only the PrefillRowShard class source from the ported file."""
    src = open(src_path).read()
    tree = ast.parse(src)
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrefillRowShard"
    )
    seg = ast.get_source_segment(src, node)
    ns = {
        "torch": SimpleNamespace(Tensor=object),
        "get_tensor_model_parallel_rank": tp.rank,
        "get_tp_group": tp.group,
    }
    exec(compile(seg, src_path + ":PrefillRowShard", "exec"), ns)
    return ns["PrefillRowShard"]


# ------------------------------------- transcription of balanced_prefill_row_shard
def balanced_prefill_row_shard(seq_lens, query_lens, compress_ratio, tp_size, min_rows):
    num_rows = sum(query_lens)
    if tp_size < 2 or num_rows < max(min_rows, tp_size):
        return None
    cost, acc = [], 0
    for s, q in zip(seq_lens, query_lens):
        first_key = s - q + 1
        for j in range(q):
            acc += (first_key + j) // compress_ratio
            cost.append(acc)
    total = cost[-1] if cost else 0
    if total <= 0:
        return None

    def searchsorted_left(a, v):
        lo, hi = 0, len(a)
        while lo < hi:
            m = (lo + hi) // 2
            if a[m] < v:
                lo = m + 1
            else:
                hi = m
        return lo

    targets = [i * total // tp_size for i in range(1, tp_size)]
    bounds = [0, *[searchsorted_left(cost, t) for t in targets], num_rows]
    for i in range(1, tp_size):
        bounds[i] = max(bounds[i], bounds[i - 1] + 1)
    for i in range(tp_size - 1, 0, -1):
        bounds[i] = min(bounds[i], bounds[i + 1] - 1)
    return [bounds[i + 1] - bounds[i] for i in range(tp_size)]


# ----------------------------------------------------------- batch generation
def make_case(rng, tp):
    num_decode_tokens = rng.choice([0, 0, 1, 3, 5, 8])
    nreq = rng.randint(1, 5)
    q_lens = [rng.randint(1, 160) for _ in range(nreq)]
    seq_lens = [q + rng.choice([0, 0, rng.randint(0, 300)]) for q in q_lens]
    ratio = rng.choice([1, 1, 4, 128])
    pad = rng.choice([0, 0, 2, 7])
    topk = rng.choice([4, 8, 16])
    width = topk + rng.choice([0, 0, 3])  # buffer wider than top-k (tail cols)

    # Chunks: greedy request groups, then optional query sub-chunks (as in
    # split_indexer_prefill_chunks); groups with 0 compressed KV are skipped
    # (build_prefill_chunk_metadata returns None).
    qsl = [num_decode_tokens]
    for q in q_lens:
        qsl.append(qsl[-1] + q)
    chunks, i = [], 0
    while i < nreq:
        j = min(nreq, i + rng.randint(1, 3))
        kv_lens = [s // ratio for s in seq_lens[i:j]]
        if sum(kv_lens) > 0:
            kv_off = [0]
            for kl in kv_lens:
                kv_off.append(kv_off[-1] + kl)
            ks_all, ke_all = [], []
            for r in range(i, j):
                base = kv_off[r - i]
                for p in range(q_lens[r]):
                    ks_all.append(base)
                    ke_all.append(base + (seq_lens[r] - q_lens[r] + p + 1) // ratio)
            m = len(ks_all)
            max_q = rng.choice([m, m, max(1, m // 3), rng.randint(1, m)])
            for off in range(0, m, max_q):
                sub = min(max_q, m - off)
                chunks.append(
                    SimpleNamespace(
                        token_start=qsl[i] + off,
                        token_end=qsl[i] + off + sub,
                        cu_seqlen_ks=ks_all[off : off + sub],
                        cu_seqlen_ke=ke_all[off : off + sub],
                        local_total_seq_lens=sum(kv_lens),
                    )
                )
        i = j

    if rng.random() < 0.8:
        min_rows = rng.randint(1, max(1, sum(q_lens)))
        sizes = balanced_prefill_row_shard(seq_lens, q_lens, ratio, tp, min_rows)
    else:  # arbitrary contiguous split, >= 1 row per rank
        n = sum(q_lens)
        if n >= tp:
            cuts = sorted(rng.sample(range(1, n), tp - 1))
            b = [0, *cuts, n]
            sizes = [b[k + 1] - b[k] for k in range(tp)]
        else:
            sizes = None
    return SimpleNamespace(
        num_decode_tokens=num_decode_tokens,
        num_tokens=qsl[-1] + pad,
        prefill_end=qsl[-1],
        chunks=chunks,
        sizes=sizes,
        topk=topk,
        width=width,
        salt=rng.getrandbits(32),
    )


def score(salt, global_row, rel_key):
    h = (global_row * 1000003) ^ (rel_key * 7919) ^ salt
    h = (h * 2654435761) & 0xFFFFFFFF
    return h ^ (h >> 13)


def row_topk(salt, global_row, ks, ke, topk):
    keys = sorted(range(ks, ke), key=lambda k: (-score(salt, global_row, k - ks), k))
    out = [k - ks for k in keys[:topk]]
    return out + [-1] * (topk - len(out))


def fresh_buffer(case):
    rows = []
    for t in range(case.num_tokens):
        if t < case.num_decode_tokens:
            v = DECODE_SENTINEL
        elif t >= case.prefill_end:
            v = PAD_SENTINEL
        else:
            v = -1  # cleared prefill rows ("no token")
        rows.append([v] * case.topk + [TAIL_SENTINEL] * (case.width - case.topk))
    return Buf(rows)


def prefill_pass(case, buf, row_shard):
    """Mirror of the ported prefill chunk loop in sparse_attn_indexer()."""
    for chunk in case.chunks:
        cu_seqlen_ks = chunk.cu_seqlen_ks
        cu_seqlen_ke = chunk.cu_seqlen_ke
        row_start, row_end = chunk.token_start, chunk.token_end
        if row_shard is not None:
            narrowed = row_shard.narrow(chunk)
            if narrowed is None:
                continue
            row_start, row_end, cu_seqlen_ks, cu_seqlen_ke = narrowed
        # q_slice = q_quant[row_start:row_end]: score rows by GLOBAL row id.
        n = row_end - row_start
        if len(cu_seqlen_ks) != n or len(cu_seqlen_ke) != n:
            raise ValueError("bounds/rows length mismatch")
        if chunk.local_total_seq_lens == 0:
            out = [[-1] * case.topk for _ in range(n)]
        else:
            out = [
                row_topk(case.salt, row_start + i, cu_seqlen_ks[i], cu_seqlen_ke[i], case.topk)
                for i in range(n)
            ]
        buf[row_start:row_end, : case.topk] = Buf(out)
    if row_shard is not None:
        row_shard.exchange_topk(buf, case.topk)


def run_case(cls, tp, case):
    ref = fresh_buffer(case)
    prefill_pass(case, ref, None)
    results, errors = [None] * tp.world, []
    meta = SimpleNamespace(row_shard_sizes=case.sizes)

    def worker(r):
        tp.tls.rank = r
        try:
            buf = fresh_buffer(case)
            shard = cls.from_metadata(meta, case.num_decode_tokens)
            prefill_pass(case, buf, shard)
            results[r] = buf.r
        except Exception as e:  # noqa: BLE001
            errors.append(f"rank {r}: {type(e).__name__}: {e}")
            tp.barrier.abort()

    ths = [threading.Thread(target=worker, args=(r,)) for r in range(tp.world)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errors:
        tp.barrier.reset()
        return False, errors[0]
    for r in range(tp.world):
        if results[r] != ref.r:
            bad = sum(a != b for a, b in zip(results[r], ref.r))
            return False, f"rank {r}: {bad} rows differ"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--tp", default="2,3,4")
    a = ap.parse_args()
    ok_all = True
    for world in map(int, a.tp.split(",")):
        rng = random.Random(a.seed * 31 + world)
        tp = FakeTP(world)
        cls = load_prefill_row_shard(a.src, tp)
        n_ok = n_engaged = 0
        first_fail = None
        for c in range(a.cases):
            case = make_case(rng, world)
            n_engaged += case.sizes is not None
            ok, msg = run_case(cls, tp, case)
            n_ok += ok
            if not ok and first_fail is None:
                first_fail = f"case {c}: {msg}"
        ok_all &= n_ok == a.cases
        print(
            f"TP{world}: {n_ok}/{a.cases} match (sharded/engaged {n_engaged}, "
            f"replicated {a.cases - n_engaged})"
            + (f"  FIRST FAIL {first_fail}" if first_fail else "")
        )
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
