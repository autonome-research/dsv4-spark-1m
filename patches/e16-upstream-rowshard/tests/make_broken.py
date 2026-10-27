"""Write deliberately broken copies of the ported sparse_attn_indexer.py."""
import sys, os
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
muts = {
    "bounds_not_offset": ("chunk.cu_seqlen_ks[lo:hi], chunk.cu_seqlen_ke[lo:hi]",
                          "chunk.cu_seqlen_ks[: hi - lo], chunk.cu_seqlen_ke[: hi - lo]"),
    "no_decode_offset": ("self.start = num_decode_tokens + sum(sizes[:rank])",
                         "self.start = sum(sizes[:rank])"),
    "window_off_by_one": ("end = min(chunk.token_end, self.stop)",
                          "end = min(chunk.token_end, self.stop - 1)"),
    "gather_misplaced": ("topk_buffer[self.num_decode_tokens : end, :width] =",
                         "topk_buffer[self.num_decode_tokens + 1 : end + 1, :width] ="),
}
for name, (a, b) in muts.items():
    assert s.count(a) == 1, name
    open(os.path.join(out, name + ".py"), "w").write(s.replace(a, b))
    print("wrote", name)
