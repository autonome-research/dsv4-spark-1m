# FlashInfer #4732 backport (sparse MLA SM120/SM121 prefill hang)

Upstream: https://github.com/flashinfer-ai/flashinfer/pull/4732 (merged 2026-09-01, merge d2090811), fixing issue #3700.
Our image's FlashInfer 0.6.18.dev20260819 predates it.

- `barrier.orig.cuh` / `prefill_kernel.orig.cuh`: the image's files (git blobs a06c86d / 1d2b344 = the PR's base).
- `barrier.cuh` / `prefill_kernel.cuh`: after applying `flashinfer-pr4732.diff` (blobs 7aaad22 / 9b0e8ac = the PR's result).
- `test_sparse_mla_sm120.f1e2a5c.py`: FlashInfer's test at the PR's parent commit f1e2a5c.

What it fixes (read from `prefill_kernel.orig.cuh` lines 146-158 and 520-525): KV tiles are double-buffered.
Per tile ti, the IO warps issue the load of tile ti+1 and then block on named barrier 1 (bar_sync, line 157)
until the math warps have finished tile ti. The math warps signal that with a non-blocking arrive on barrier 1
(bar_arrive, line 522), then wait only on the mbarrier for tile ti+1's data, which the IO warps issued *before*
their sync. So if the IO warps are delayed, the math warps can finish tile ti+1 and arrive on barrier 1 again
before the IO warps have completed the sync for tile ti: two arrivals in one barrier phase, which
desynchronises the barrier and can hang the kernel with the GPU busy. The fix alternates the barrier ID by tile
parity (ID 1 for even tiles, ID 5 for odd), so consecutive tiles never share a barrier. The same pattern in
`prefill_mg_impl` (lines 785, 826 sync; 1517 arrive) gets the same change. Other named barriers in this file: 2, 3;
ID 5 is free.

The prebuilt module (`flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so`) is loaded in
preference to a JIT build, so header changes alone do nothing: the fixed module is built offline and its .so
is bind-mounted over the prebuilt one (`../dsv4-fi4732-sparse_mla_sm120.so`, md5 `23acbe9f7a…`;
`DSV4_FI4732=1`, the default). The build tree (`build/`) is not tracked.

Validation (optimisation log E15): FlashInfer's sparse-MLA prefill tests, 163 tests × 3 rounds per module.
The prebuilt module hung in 1/3 rounds, an unfixed build of the same sources in 3/3, and the fixed build in
0/3 (all 163 passed each round). Serving speed was unchanged and retrieval was correct at 64K, 256K and 1M.
