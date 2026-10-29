#!/usr/bin/env bash
# Start one DSV4 node. Runs LOCALLY on whichever box it is invoked on.
#   ./node-launch.sh 0     leader  (serves the API)
#   ./node-launch.sh 1     worker  (headless)
# Optional arguments: [breakable_cudagraph_value] [max_num_batched_tokens]
#   [max_num_seqs] [spec_mode=on|off] [diagnostics=0|1] [max_model_len].
#
# Called by launch-dsv4.sh up (directly on the leader, over ssh on the worker); it
# refuses to run without the attended run descriptor. Settings come from DSV4_*
# environment variables (README.md, "Configuration"); the patch files are expected
# next to this script (leader: /srv/dsv4/run/patches/, worker: ~/).
#
# This exists so the docker command is never re-parsed by a second shell.
# Passing it through `ssh "docker run ... $ARGS"` strips the quotes out of
# --speculative-config's JSON and vLLM rejects it:
#   argument --speculative-config: Value method:dspark cannot be converted
set -uo pipefail

RANK="${1:-}"
# Optional 2nd arg: VLLM_USE_BREAKABLE_CUDAGRAPH value.
#   (unset) -> let vLLM auto-enable it, which DISABLES torch.compile
#   0       -> keep torch.compile on
# The boot log auto-enables it and then says "disabling vLLM's torch.compile
# pipeline. Equivalent to -cc.mode=none". Measured prefill was ~1300-1550 tok/s
# against the runbook's ~2600, and the runbook credits a working torch.compile
# path as one of two reasons its Lane B build is faster. So this is worth one
# controlled experiment.
# MEASURED 2026-08-13: setting this to 0 is a REGRESSION. torch.compile turns on
# but the model does not support it ("does not support it. Please open an issue"),
# so it is a no-op for decode (median delta -0.5%, inside a +-17% noise floor)
# while the side effects are real: custom_ops 'all'->'none' and rms_norm
# ['vllm_c','native']->['native'] cost 31-37% of PREFILL throughput at depth
# (8.7K: 1388->963 tok/s, 34.7K: 1543->967). Leave it unset.
BREAKABLE_CG="${2:-}"

# Favor the model's full native context over request concurrency. Keep prefill
# chunks small; override arguments 3 and 4 only for controlled experiments.
MAXBATCH="${3:-4096}"
MAXSEQS="${4:-1}"
SPEC_MODE="${5:-on}"
DIAGNOSTICS="${6:-0}"
# DeepSeek V4's native context is 1,048,576. A prior 660,556-token request
# exposed an unresolved GPU-side sample_tokens hang, so full-context operation
# stays diagnostic and intentionally uses max_num_seqs=1 by default.
MAX_MODEL_LEN="${7:-1048576}"
GPU_MEMORY_UTILIZATION="${DSV4_GPU_MEMORY_UTILIZATION:-0.85}"
# Pin the KV pool instead of letting the profiler size it from free memory.
# Prior boots got 17-20 GiB = ~3.2M tokens per rank; 9 GiB covers ~1.45M,
# enough for one 1M request, and leaves the rest as unified-memory headroom.
KV_CACHE_BYTES="${DSV4_KV_CACHE_BYTES:-9663676416}"
PREFIX_CACHE="${DSV4_PREFIX_CACHE:-1}"
if [[ "$SPEC_MODE" != on && "$SPEC_MODE" != off ]]; then
  echo "spec_mode must be on or off (got: $SPEC_MODE)" >&2
  exit 2
fi
# dspark uses k=5 draft tokens, so captured size must be max-seqs * (k+1).
# Without speculative decoding, the ordinary capture size is max-num-seqs.
if [[ "$SPEC_MODE" == on ]]; then MAXCG=$((MAXSEQS * 6)); else MAXCG="$MAXSEQS"; fi

case "$RANK" in
  0) NAME=dsv4-leader; ROLE=leader; EXTRA=() ;;
  1) NAME=dsv4-worker; ROLE=worker; EXTRA=(--headless) ;;
  *) echo "usage: $0 <0|1> [breakable_cudagraph_value] [max_num_batched_tokens] [max_num_seqs] [spec_mode] [diagnostics] [max_model_len]" >&2; exit 2 ;;
esac

T2_GUARD="${DSV4_T2_GUARD:-$HOME/.cache/dsv4-t2/guard.py}"
T2_STATE="${DSV4_T2_STATE_DIR:-$HOME/.cache/dsv4-t2}"
RUN_ID="${DSV4_T2_RUN_ID:-}"
[[ -r "$T2_GUARD" && "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
  echo "refusing direct DSV4 launch: attended T2 run is missing" >&2; exit 1;
}
python3 -B "$T2_GUARD" admission --run-id "$RUN_ID" --role "$ROLE" --state-dir "$T2_STATE" --max-age 6 || {
  echo "refusing DSV4 launch: attended admission/guard is not fresh" >&2; exit 1;
}

STACK="${DSV4_STACK:-anemll}"
case "$STACK" in
  anemll) DEFAULT_IMG=ghcr.io/anemll/dspark-vllm-gx10:0.1.1 ;;
  upstream) DEFAULT_IMG=local/vllm-openai:46638857 ;;
  *) echo "DSV4_STACK must be anemll or upstream (got: $STACK)" >&2; exit 2 ;;
esac
IMG="${DSV4_IMAGE:-$DEFAULT_IMG}"
# Persist runtime-generated kernels across coordinated container replacement.
# The cache namespace is pinned to this exact vLLM/FlashInfer/SM121 stack so a
# future image cannot accidentally consume incompatible binaries.
RUNTIME_CACHE_ROOT="${DSV4_RUNTIME_CACHE_ROOT:-/srv/dsv4/runtime-cache/46638857-fi0618-sm121}"
mkdir -p "$RUNTIME_CACHE_ROOT"/{torchinductor,vllm,tilelang,triton,flashinfer}
SNAP=7872f01b1d1fe23eabc4c98b48bffcef5a386062
MODEL="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/$SNAP"
PORT=8888

# cx7-nccl-env.sh lives at a different path on each node
for p in /srv/dsv4/run/cx7-nccl-env.sh "$HOME/cx7-nccl-env.sh"; do
  [[ -r "$p" ]] && { source "$p"; break; }
done
[[ -n ${NCCL_SOCKET_IFNAME:-} ]] || { echo "node-launch: cx7-nccl-env.sh not found or CX7 not detected" >&2; exit 1; }

RDMA=()
for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do RDMA+=(--device="$d"); done

ENVS=()
# Keep private Anemll knobs out of official vLLM. Current upstream implements
# DSV4/DSpark natively and warns or ignores several of these legacy variables.
if [[ "$STACK" == anemll ]]; then
  for kv in \
    VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_WO_PROJECTION=1 VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1 VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
    VLLM_DSPARK_CONFIDENCE_SCHEDULER=off VLLM_DSPARK_LOCAL_ARGMAX=1 \
    VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0 VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0 \
    VLLM_DSV4_B12X_COMPRESSED_MLA=0 VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0 \
    B12X_W4A16_TC_DECODE=0
  do ENVS+=(-e "$kv"); done
fi
# CUDA-graph memory estimation is left at vLLM's default (enabled): the pinned
# commit includes PR #53682's throwaway graph-pool fix, so it is safe and keeps the
# KV allocation from taking memory later needed by runtime JIT/autotuning.
ENVS+=(-e HF_HUB_OFFLINE=1)
# Step-1 hardening (2026-10-11). UCX hooks leaked registered memory on GB10;
# expandable segments fragment unified memory under long-context KV growth.
ENVS+=(-e UCX_MEM_MMAP_HOOK_MODE=none -e UCX_RCACHE_MAX_UNRELEASED=1024)
ENVS+=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False)

# Diagnostic mode (DSV4_DIAG, default 1 in start-dsv4-default.sh) keeps scheduler
# dumps and NCCL traces across container shutdown, in $DIAG_DIR. Its cost is not
# measurable at 4K (optimisation log R01.3).
DIAG_DIR="/srv/dsv4/diagnostics/$(hostname)"
mkdir -p "$DIAG_DIR"
if [[ "$DIAGNOSTICS" == 1 ]]; then
  ENVS+=(--env VLLM_DEBUG_DUMP_PATH=/tmp/vllm-debug)
  ENVS+=(--env VLLM_RINGBUFFER_WARNING_INTERVAL=10)
  ENVS+=(--env NCCL_DEBUG_SUBSYS=INIT,NET,COLL)
  ENVS+=(--env NCCL_DEBUG_FILE=/tmp/vllm-debug/nccl.%h.%p.log)
  ENVS+=(--env TORCH_NCCL_DUMP_ON_TIMEOUT=1)
  ENVS+=(--env TORCH_NCCL_DESYNC_DEBUG=1)
  ENVS+=(--env TORCH_NCCL_TRACE_BUFFER_SIZE=2000)
  ENVS+=(--env TORCH_NCCL_ENABLE_TIMING=1)
fi
[[ -n "$BREAKABLE_CG" ]] && ENVS+=(-e "VLLM_USE_BREAKABLE_CUDAGRAPH=$BREAKABLE_CG")
for v in NCCL_IB_HCA NCCL_NET NCCL_IB_DISABLE NCCL_IB_MERGE_NICS NCCL_IB_GID_INDEX \
         NCCL_IB_ADDR_FAMILY NCCL_IB_ROCE_VERSION_NUM NCCL_CROSS_NIC \
         NCCL_IGNORE_CPU_AFFINITY NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE NCCL_DEBUG \
         NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME; do
  ENVS+=(-e "$v=${!v}")
done
# Pin vLLM's control plane (engine->worker ZMQ broadcast, get_ip()) to the CX7
# link. Unset, get_ip() follows the default route (WiFi), which roams between
# networks and stalled/killed the engine on 2026-10-12 (optimisation log E08).
VLLM_HOST_IP=$(ip -4 -o addr show dev "$NCCL_SOCKET_IFNAME" | awk '{split($4,a,"/"); print a[1]; exit}')
[[ -n "$VLLM_HOST_IP" ]] || { echo "node-launch: no IPv4 on $NCCL_SOCKET_IFNAME for VLLM_HOST_IP" >&2; exit 1; }
ENVS+=(-e "VLLM_HOST_IP=$VLLM_HOST_IP")
# The inherited NCCL_DEBUG value is appended above; diagnostic mode must be
# appended last so INFO is not overwritten by cx7-nccl-env.sh's default WARN.
if [[ "$DIAGNOSTICS" == 1 ]]; then
  ENVS+=(--env NCCL_DEBUG=INFO)
fi

if timeout --kill-after=2 10 docker inspect --type container "$NAME" >/dev/null 2>&1; then
  echo "refusing launch: container name $NAME already exists" >&2
  exit 1
else
  collision_rc=$?
  (( collision_rc == 1 )) || { echo "could not safely inspect container name $NAME" >&2; exit 1; }
fi
CID_DIR="$T2_STATE/runs/$RUN_ID/cids"
mkdir -p "$CID_DIR" && chmod 700 "$T2_STATE" "$T2_STATE/runs" "$T2_STATE/runs/$RUN_ID" "$CID_DIR"
CIDFILE="$CID_DIR/$ROLE.cid"
[[ ! -e "$CIDFILE" && ! -L "$CIDFILE" ]] || { echo "protected cidfile already exists: $CIDFILE" >&2; exit 1; }
OWNED_FILE="$T2_STATE/owned-process.json"
[[ ! -e "$OWNED_FILE" && ! -L "$OWNED_FILE" ]] || { echo "owned launcher descriptor already exists" >&2; exit 1; }
write_owned() {
  python3 -B - "$OWNED_FILE" "$RUN_ID" "$ROLE" "$BASHPID" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
run_id, role, pid = sys.argv[2], sys.argv[3], int(sys.argv[4])
raw = Path('/proc') / str(pid) / 'stat'
text = raw.read_text()
starttime = text[text.rfind(')') + 2:].split()[19]
tmp = path.with_name(f'.{path.name}.{pid}.tmp')
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps({'pid': pid, 'starttime': starttime, 'run_id': str(run_id), 'role': str(role)}, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    # Exclusive publish: os.link fails if a descriptor already exists, so the
    # anchor an abort cancels through can never be silently replaced by a
    # second launcher for the same run and role.
    os.link(tmp, path)
finally:
    os.unlink(tmp)
PY
}
clear_owned() { rm -f -- "$OWNED_FILE"; }
on_signal() { clear_owned; exit 143; }
trap clear_owned EXIT
trap on_signal INT TERM HUP

CACHE_ARGS=(--enable-prefix-caching)
[[ "$PREFIX_CACHE" == 0 ]] && CACHE_ARGS=(--no-enable-prefix-caching)
# DSV4_PROFILE=1 configures vLLM's torch profiler (inactive until POST /start_profile).
# Traces land in $DIAG_DIR/prof on each node. Optimisation log E07.
PROFILE_ARGS=()
[[ "${DSV4_PROFILE:-0}" == 1 ]] && PROFILE_ARGS=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm-debug/prof","torch_profiler_with_stack":false,"ignore_frontend":true,"delay_iterations":10,"max_iterations":20}')
# Async scheduling (optimisation log E07.6, D7): default ON when DSpark is off
# (tested: 4K/64K/1M + idle probes), default OFF when DSpark is on (untested with
# DSpark; it was disabled while debugging the DSpark rank-divergence deadlock).
# DSV4_ASYNC=1 or DSV4_ASYNC=0 overrides either default.
ASYNC="${DSV4_ASYNC:-}"
[[ -z "$ASYNC" ]] && { [[ "$SPEC_MODE" == on ]] && ASYNC=0 || ASYNC=1; }
[[ "$ASYNC" == 0 || "$ASYNC" == 1 ]] || { echo "DSV4_ASYNC must be 0 or 1 (got: $ASYNC)" >&2; exit 1; }
ASYNC_FLAG=--no-async-scheduling
[[ "$ASYNC" == 1 ]] && ASYNC_FLAG=--async-scheduling

# Local vLLM patch (optimisation log E07.8): compressor GEMV issued in 512-row
# chunks at decode. Bind-mounted over the image's file. DSV4_PATCH:
#   1  (default) dsv4-attention.py, chunks for M <= 8
#   m1 dsv4-attention-m1.py, chunks for M = 1 only (2026-10-27 benchmark: chunking is
#      slower at M = 2-8, which DSpark's verify and draft steps use)
#   0  the image's own file
PATCH_ARGS=()
case "${DSV4_PATCH:-1}" in
  0) PATCH_FILE= ;;
  1) PATCH_FILE=dsv4-attention.py ;;
  m1) PATCH_FILE=dsv4-attention-m1.py ;;
  *) echo "DSV4_PATCH must be 0, 1 or m1 (got: $DSV4_PATCH)" >&2; exit 1 ;;
esac
if [[ -n $PATCH_FILE ]]; then
  for p in "/srv/dsv4/run/patches/$PATCH_FILE" "$HOME/$PATCH_FILE"; do
    [[ -f "$p" ]] && { PATCH_ARGS=(-v "$p:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/attention.py:ro"); break; }
  done
  [[ ${#PATCH_ARGS[@]} -gt 0 ]] || { echo "DSV4_PATCH=${DSV4_PATCH:-1} but $PATCH_FILE was not found" >&2; exit 1; }
fi

# Indexer prefill mode. DSV4_IDX_MODE:
#   split    (default) our key split, optimisation log E12: each TP rank scores half of the
#            indexer keys for long prefixes and the two top-K lists are merged.
#   rowshard upstream vLLM #54951 + #61647 backport (E16): each rank scores its share of query
#            rows and the top-K indices are all-gathered. Mounts three files and passes
#            indexer_tp_shard_min_rows (DSV4_ROWSHARD_MIN_ROWS, default 8192; upstream's 65,536
#            never engages with 8192-token batches).
#   off      the image's own indexer.
# The two modes replace the same file and must match on both ranks, or the collectives differ
# and NCCL hangs; launch-dsv4.sh passes the same values to both nodes.
# Legacy: DSV4_IDX_SPLIT=0 with no DSV4_IDX_MODE means off.
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
IDX_MODE="${DSV4_IDX_MODE:-}"
if [[ -z "$IDX_MODE" ]]; then
  case "${DSV4_IDX_SPLIT:-1}" in 1) IDX_MODE=split ;; 0) IDX_MODE=off ;;
    *) echo "DSV4_IDX_SPLIT must be 0 or 1 (got: $DSV4_IDX_SPLIT)" >&2; exit 1 ;; esac
fi
find_patch() {  # find_patch <file name>: the leader's patches dir, else the worker's $HOME copy
  local p
  for p in "/srv/dsv4/run/patches/$1" "$HOME/$1"; do [[ -f "$p" ]] && { echo "$p"; return 0; }; done
  echo "DSV4_IDX_MODE=$IDX_MODE but $1 was not found" >&2; return 1
}
IDX_ARGS=() ATTN_CFG_ARGS=()
case "$IDX_MODE" in
  split)
    p=$(find_patch dsv4-sparse-attn-indexer.py) || exit 1
    IDX_ARGS=(-v "$p:$VLLM_PKG/model_executor/layers/sparse_attn_indexer.py:ro") ;;
  rowshard)
    MIN_ROWS="${DSV4_ROWSHARD_MIN_ROWS:-8192}"
    [[ "$MIN_ROWS" =~ ^[1-9][0-9]*$ ]] || { echo "DSV4_ROWSHARD_MIN_ROWS must be a positive integer (got: $MIN_ROWS)" >&2; exit 1; }
    p1=$(find_patch dsv4-rowshard-sparse-attn-indexer.py) || exit 1
    p2=$(find_patch dsv4-rowshard-mla-indexer.py) || exit 1
    p3=$(find_patch dsv4-rowshard-config-attention.py) || exit 1
    IDX_ARGS=(-v "$p1:$VLLM_PKG/model_executor/layers/sparse_attn_indexer.py:ro"
              -v "$p2:$VLLM_PKG/v1/attention/backends/mla/indexer.py:ro"
              -v "$p3:$VLLM_PKG/config/attention.py:ro")
    ATTN_CFG_ARGS=(--attention-config "{\"indexer_tp_shard_min_rows\": $MIN_ROWS}") ;;
  off) ;;
  *) echo "DSV4_IDX_MODE must be split, rowshard or off (got: $IDX_MODE)" >&2; exit 1 ;;
esac
# Record what this rank mounts, so a leader/worker mismatch is visible in the launch output.
echo "node-launch: indexer mode=$IDX_MODE${MIN_ROWS:+ min_rows=$MIN_ROWS} files: $(for ((i = 1; i < ${#IDX_ARGS[@]}; i += 2)); do md5sum "${IDX_ARGS[i]%%:*}" | cut -c1-8; done | tr '\n' ' ')" >&2

# FlashInfer #4732 backport (optimisation log E15): sparse MLA SM120/121 prefill kernel built with the
# upstream named-barrier fix, mounted over the prebuilt module. On by default since 2026-10-23; DSV4_FI4732=0 disables.
FI_ARGS=()
if [[ "${DSV4_FI4732:-1}" == 1 ]]; then
  for p in /srv/dsv4/run/patches/dsv4-fi4732-sparse_mla_sm120.so "$HOME/dsv4-fi4732-sparse_mla_sm120.so"; do
    [[ -f "$p" ]] && { FI_ARGS=(-v "$p:/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so:ro"); break; }
  done
  [[ ${#FI_ARGS[@]} -gt 0 ]] || { echo "DSV4_FI4732=1 but dsv4-fi4732-sparse_mla_sm120.so was not found" >&2; exit 1; }
elif [[ "${DSV4_FI4732:-1}" != 0 ]]; then
  echo "DSV4_FI4732 must be 0 or 1 (got: $DSV4_FI4732)" >&2; exit 1
fi

SPEC_ARGS=()
if [[ "$SPEC_MODE" == on ]]; then
  SPEC_ARGS=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}')
fi

# DSpark draft-loader patch (docs/dspark-experiment.md, Stage 0): upstream vLLM #53984's
# guard for draft expert params the draft does not register (#51916 KeyError), plus a
# warning that counts them. Opt-in, and only meaningful with spec_mode=on.
DSPARK_ARGS=()
case "${DSV4_DSPARK_LOADFIX:-0}" in
  0) ;;
  1)
    [[ "$SPEC_MODE" == on ]] || { echo "DSV4_DSPARK_LOADFIX=1 needs spec_mode=on" >&2; exit 1; }
    for p in /srv/dsv4/run/patches/dsv4-dspark.py "$HOME/dsv4-dspark.py"; do
      [[ -f "$p" ]] && { DSPARK_ARGS=(-v "$p:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py:ro"); break; }
    done
    [[ ${#DSPARK_ARGS[@]} -gt 0 ]] || { echo "DSV4_DSPARK_LOADFIX=1 but dsv4-dspark.py was not found" >&2; exit 1; }
    ;;
  *) echo "DSV4_DSPARK_LOADFIX must be 0 or 1 (got: $DSV4_DSPARK_LOADFIX)" >&2; exit 1 ;;
esac

# Settings that must be identical on both ranks, or they run different collectives and
# NCCL hangs. launch-dsv4.sh compares this line between the nodes and aborts on a mismatch.
md5_of() { if [[ -n $1 ]]; then md5sum "${1%%:*}" | cut -c1-8; else echo none; fi; }
echo "node-launch: decode spec=$SPEC_MODE async=$ASYNC prefix_cache=$PREFIX_CACHE" \
  "gemv=$(md5_of "${PATCH_ARGS[1]:-}") dspark_loader=$(md5_of "${DSPARK_ARGS[1]:-}")" >&2

if [[ "$STACK" == upstream ]]; then
  BACKEND_ARGS=(
    --attention-backend FLASHINFER_MLA_SPARSE_DSV4
    --kv-cache-dtype fp8_ds_mla
    --moe-backend deep_gemm
  )
else
  BACKEND_ARGS=(
    --kv-cache-dtype nvfp4_ds_mla
    --moe-backend flashinfer_b12x
  )
fi

# Publish the exact owned PID/starttime FIRST, then recheck admission. The order
# is the fix for the cancellation race: dsv4-abort.sh latches CANCELLED before it
# signals, so a cancellation landing before the descriptor exists is caught by the
# recheck below, and one landing after it is caught by `cancel-owned` finding the
# descriptor. Rechecking first left a window in which cancel-owned saw no
# descriptor, reported "nothing owned", and this launcher still reached docker run.
write_owned || { echo "could not record owned launcher identity" >&2; exit 1; }
python3 -B "$T2_GUARD" admission --run-id "$RUN_ID" --role "$ROLE" --state-dir "$T2_STATE" --max-age 6 || {
  echo "refusing DSV4 launch: admission was cancelled before container creation" >&2; exit 1;
}
if ! timeout --kill-after=2 180 docker run -d --name "$NAME" --restart=no \
  --label "dsv4.t2.run=$RUN_ID" --label "dsv4.t2.role=$ROLE" --label dsv4.t2.owner=attended \
  --cidfile "$CIDFILE" \
  --network host --ipc host --gpus all \
  --ulimit memlock=-1 --shm-size 64gb \
  --cap-add=IPC_LOCK --cap-add=SYS_PTRACE "${RDMA[@]}" \
  -v "${DSV4_HF_CACHE:-/home/<collaborator>/.cache/huggingface}:/root/.cache/huggingface" \
  -v "$RUNTIME_CACHE_ROOT/torchinductor:/tmp/torchinductor_root" \
  -v "$RUNTIME_CACHE_ROOT/vllm:/root/.cache/vllm" \
  -v "$RUNTIME_CACHE_ROOT/tilelang:/root/.tilelang" \
  -v "$RUNTIME_CACHE_ROOT/triton:/root/.triton" \
  -v "$RUNTIME_CACHE_ROOT/flashinfer:/root/.cache/flashinfer" \
  -v "$DIAG_DIR:/tmp/vllm-debug" \
  "${PATCH_ARGS[@]}" \
  "${DSPARK_ARGS[@]}" \
  "${IDX_ARGS[@]}" \
  "${FI_ARGS[@]}" \
  "${ENVS[@]}" \
  --entrypoint vllm "$IMG" serve "$MODEL" \
    --served-model-name deepseek-v4-flash-0731 \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$RANK" \
    --master-addr "$CX7_MASTER_ADDR" --master-port "$CX7_MASTER_PORT" \
    "${BACKEND_ARGS[@]}" --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "$MAXBATCH" \
    --max-cudagraph-capture-size "$MAXCG" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
    --tokenizer-mode deepseek_v4 \
    "${SPEC_ARGS[@]}" \
    "${ATTN_CFG_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${CACHE_ARGS[@]}" "$ASYNC_FLAG" --enable-chunked-prefill --no-enable-flashinfer-autotune \
    --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --generation-config vllm \
    "${EXTRA[@]}" >/dev/null; then
  echo "docker run failed; no container ID was accepted" >&2
  exit 1
fi

# A cancellation delivered while docker run was in flight cannot interrupt it:
# bash defers the trap until the foreground command returns, so a container may
# now exist for a run that was already cancelled or alerted. Report that instead
# of a healthy start. Stopping it belongs to the exact-ID abort path; this
# launcher never stops or removes containers.
if ! python3 -B "$T2_GUARD" admission --run-id "$RUN_ID" --role "$ROLE" --state-dir "$T2_STATE" --max-age 6 >/dev/null 2>&1; then
  echo "container $NAME was created for run $RUN_ID but admission is no longer active" >&2
  echo "reconcile the exact container through: dsv4-abort.sh --run-id $RUN_ID" >&2
  exit 1
fi

timeout --kill-after=2 6 sleep 4
timeout --kill-after=2 10 docker inspect --type container "$NAME" --format '{{.Id}} {{.Name}} {{.State.Status}}' >/dev/null || exit 1
chmod 600 "$CIDFILE" || exit 1
cid=$(<"$CIDFILE")
[[ "$cid" =~ ^[0-9a-f]{64}$ ]] || { echo "docker wrote an invalid protected cidfile" >&2; exit 1; }
[[ "$(timeout --kill-after=2 10 docker inspect --type container "$cid" --format '{{.Id}} {{.Name}} {{index .Config.Labels "dsv4.t2.role"}} {{index .Config.Labels "dsv4.t2.run"}}')" == "$cid /$NAME $ROLE $RUN_ID" ]] || {
  echo "container identity/labels failed validation" >&2; exit 1;
}
printf '%s\n' "$cid"
