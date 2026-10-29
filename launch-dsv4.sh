#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-0731 across both Sparks, TP=2.
#   ./launch-dsv4.sh up [cudagraph] [max_batch] [max_seqs] [spec] [diag] [max_len]
#                            start worker then leader (arguments as passed by
#                            start-dsv4-default.sh; see node-launch.sh)
#   ./launch-dsv4.sh down    stop both
#   ./launch-dsv4.sh logs    follow leader logs
#   ./launch-dsv4.sh status  container + endpoint health
#   ./launch-dsv4.sh wlogs   follow worker logs
#
# `up` is normally called by start-dsv4-default.sh, which provides the attended run
# descriptor, guards and settings; on its own it refuses to start. `down` and
# `status` can be run directly. The DSV4_STACK default (anemll) is the original
# DSpark image; start-dsv4-default.sh sets DSV4_STACK=upstream and DSV4_IMAGE.
# The docker command itself is built by node-launch.sh on each node; the DSV4_*
# settings it reads are listed in README.md, "Configuration".
set -uo pipefail

STACK="${DSV4_STACK:-anemll}"
case "$STACK" in
  anemll) DEFAULT_IMG=ghcr.io/anemll/dspark-vllm-gx10:0.1.1 ;;
  upstream) DEFAULT_IMG=local/vllm-openai:46638857 ;;
  *) echo "DSV4_STACK must be anemll or upstream (got: $STACK)" >&2; exit 2 ;;
esac
IMG="${DSV4_IMAGE:-$DEFAULT_IMG}"
GPU_MEMORY_UTILIZATION="${DSV4_GPU_MEMORY_UTILIZATION:-0.85}"
PEER=192.168.200.11
HERE=/srv/dsv4/run
PORT=8888

T2_GUARD="${DSV4_T2_GUARD:-$HERE/dsv4-t2-guard.py}"
T2_STATE="${DSV4_T2_STATE_DIR:-$HOME/.cache/dsv4-t2}"
RUN_ID="${DSV4_T2_RUN_ID:-}"
SSH_TIMEOUT=${DSV4_T2_SSH_TIMEOUT:-30}
LAUNCH_TIMEOUT=${DSV4_T2_LAUNCH_TIMEOUT:-180}
# Teardown must not depend on a currently usable fabric/NCCL environment.
if [[ ${1:-} == down ]]; then
  DOWN_RUN="${RUN_ID:-$(head -n 1 "$T2_STATE/active-run" 2>/dev/null || true)}"
  # Stop the worker watch first: a deliberate down must not look like a lost worker.
  [[ "$DOWN_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] && pkill -TERM -f "dsv4-peerwatch[.]sh --run-id $DOWN_RUN " 2>/dev/null
  "$HERE/dsv4-abort.sh" --run-id "$DOWN_RUN" --reason 'launch-dsv4 down'
  rc=$?
  # down also ends this run's guards and observers (optimisation log 2026-10-18):
  # a serving-mode guard left running alerts on the next boot's weight load and
  # aborts it. Exact run ID or verified pid file only; the [.] keeps the pattern
  # from matching the remote shell that carries it.
  if [[ "$DOWN_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    pkill -TERM -f "guard[.]py run --run-id $DOWN_RUN " 2>/dev/null
    pkill -TERM -f "allocator-observer[.]sh --data $T2_STATE/runs/$DOWN_RUN/observer/" 2>/dev/null
    timeout --kill-after=2 "${DSV4_T2_SSH_TIMEOUT:-30}" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
      "pkill -TERM -f 'guard[.]py run --run-id $DOWN_RUN '; p=\$(cat ~/.cache/dsv4-t2/observer.pid 2>/dev/null) && grep -qa allocator-observer /proc/\$p/cmdline 2>/dev/null && kill -TERM \$p; true" >/dev/null 2>&1
    sleep 2
    left_l=$(pgrep -fc "guard[.]py run --run-id $DOWN_RUN |allocator-observer[.]sh --data $T2_STATE/runs/$DOWN_RUN/" || true)
    left_w=$(timeout --kill-after=2 "${DSV4_T2_SSH_TIMEOUT:-30}" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
      "pgrep -fc 'guard[.]py run --run-id $DOWN_RUN |allocator-observer[.]sh' || true" 2>/dev/null || echo unknown)
    echo "dsv4-down: guards/observers left for $DOWN_RUN: leader=${left_l:-0} worker=${left_w:-unknown}"
  fi
  exit $rc
fi

source "$HERE/cx7-nccl-env.sh"
timed() { timeout --kill-after=2 "$1" "${@:2}"; }
require_attended_admission() {
  [[ -r "$T2_GUARD" && "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || { echo 'launch requires an active attended T2 run descriptor' >&2; return 1; }
  python3 -B "$T2_GUARD" admission --run-id "$RUN_ID" --role leader --state-dir "$T2_STATE" --max-age 6 || return 1
  timed "$SSH_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" \
    "python3 -B '$HOME/.cache/dsv4-t2/guard.py' admission --run-id '$RUN_ID' --role worker --state-dir '$HOME/.cache/dsv4-t2' --max-age 6"
}

case "${1:-}" in
up)
  require_attended_admission || { echo 'refusing DSV4 up: both fresh guards and attended admission are required' >&2; exit 1; }
  # Both ends go through node-launch.sh, which builds the docker command with
  # bash arrays and runs it locally. Never interpolate the args into an ssh
  # string -- the remote shell strips the quotes out of --speculative-config.
  timed "$SSH_TIMEOUT" scp -q "$HERE/node-launch.sh" "$HERE/cx7-nccl-env.sh" "$HERE/dsv4-t2-guard.py" "$HERE/patches/dsv4-attention.py" "$HERE/patches/dsv4-attention-m1.py" "$HERE/patches/dsv4-dspark.py" "$HERE/patches/dsv4-sparse-attn-indexer.py" "$HERE/patches/dsv4-rowshard-sparse-attn-indexer.py" "$HERE/patches/dsv4-rowshard-mla-indexer.py" "$HERE/patches/dsv4-rowshard-config-attention.py" "$HERE/patches/dsv4-fi4732-sparse_mla_sm120.so" "$HERE/dsv4-leaderwatch.sh" "$PEER:" || {
    echo 'bounded worker preparation failed' >&2; exit 1;
  }
  timed "$SSH_TIMEOUT" ssh -o BatchMode=yes "$PEER" 'mkdir -p ~/.cache/dsv4-t2 && chmod 700 ~/.cache/dsv4-t2 && cp -p ~/dsv4-t2-guard.py ~/.cache/dsv4-t2/guard.py && chmod 700 ~/.cache/dsv4-t2/guard.py && chmod +x ~/node-launch.sh ~/cx7-nccl-env.sh ~/dsv4-leaderwatch.sh' || {
    echo 'bounded worker admission preparation failed' >&2; exit 1;
  }

  # up [breakable_cudagraph] [max_num_batched_tokens] [max_num_seqs]
  #    [spec_mode=on|off] [diagnostics=0|1] [max_model_len]
  # Prefer the model's native 1M context over concurrency. A 660K request
  # previously exposed an unresolved sample_tokens GPU hang, so diagnostics
  # remain enabled by the managed launcher while this mode is observed.
  CG="${2:-}"
  MB="${3:-4096}"
  MS="${4:-1}"
  SPEC="${5:-on}"
  DIAG="${6:-0}"
  MAXLEN="${7:-1048576}"

  # Worker first -- the leader hosts the rendezvous and a leader-first start races.
  echo "=== starting worker (rank 1) on $PEER ==="
  printf -v REMOTE_IMG '%q' "$IMG"
  printf -v REMOTE_STACK '%q' "$STACK"
  printf -v REMOTE_GPU_UTIL '%q' "$GPU_MEMORY_UTILIZATION"
  printf -v REMOTE_IDX_MODE '%q' "${DSV4_IDX_MODE:-}"
  printf -v REMOTE_MIN_ROWS '%q' "${DSV4_ROWSHARD_MIN_ROWS:-8192}"
  worker_output=$(timed "$LAUNCH_TIMEOUT" ssh -o BatchMode=yes "$PEER" "DSV4_PROFILE=${DSV4_PROFILE:-0} DSV4_ASYNC=${DSV4_ASYNC:-} DSV4_PATCH=${DSV4_PATCH:-1} DSV4_DSPARK_LOADFIX=${DSV4_DSPARK_LOADFIX:-0} DSV4_IDX_SPLIT=${DSV4_IDX_SPLIT:-1} DSV4_IDX_MODE=$REMOTE_IDX_MODE DSV4_ROWSHARD_MIN_ROWS=$REMOTE_MIN_ROWS DSV4_FI4732=${DSV4_FI4732:-1} DSV4_KV_CACHE_BYTES=${DSV4_KV_CACHE_BYTES:-9663676416} DSV4_PREFIX_CACHE=${DSV4_PREFIX_CACHE:-1} DSV4_STACK=$REMOTE_STACK DSV4_IMAGE=$REMOTE_IMG DSV4_GPU_MEMORY_UTILIZATION=$REMOTE_GPU_UTIL DSV4_T2_RUN_ID='$RUN_ID' ~/node-launch.sh 1 '$CG' '$MB' '$MS' '$SPEC' '$DIAG' '$MAXLEN'" 2>&1)
  worker_rc=$?
  printf '%s\n' "$worker_output"
  if (( worker_rc != 0 )); then
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'worker launch failed' || true
    exit 1
  fi
  worker_cid=$(tail -n 1 <<<"$worker_output")
  [[ $worker_cid =~ ^[0-9a-f]{64}$ ]] || {
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'worker identity missing' || true; exit 1;
  }
  mkdir -p "$T2_STATE/runs/$RUN_ID/cids" && chmod 700 "$T2_STATE/runs" "$T2_STATE/runs/$RUN_ID" "$T2_STATE/runs/$RUN_ID/cids" || {
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'worker cid transfer failed' || true; exit 1;
  }
  printf '%s\n' "$worker_cid" >"$T2_STATE/runs/$RUN_ID/cids/worker.cid"
  chmod 600 "$T2_STATE/runs/$RUN_ID/cids/worker.cid"

  echo "=== starting leader (rank 0) here ==="
  leader_output=$(DSV4_STACK="$STACK" DSV4_IMAGE="$IMG" DSV4_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" DSV4_T2_RUN_ID="$RUN_ID" DSV4_T2_STATE_DIR="$T2_STATE" DSV4_T2_GUARD="$T2_GUARD" timeout --kill-after=2 "$LAUNCH_TIMEOUT" "$HERE/node-launch.sh" 0 "$CG" "$MB" "$MS" "$SPEC" "$DIAG" "$MAXLEN" 2>&1)
  leader_rc=$?
  printf '%s\n' "$leader_output"
  if (( leader_rc != 0 )); then
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'leader launch failed' || true
    exit 1
  fi
  [[ $(tail -n 1 <<<"$leader_output") =~ ^[0-9a-f]{64}$ ]] || {
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'leader identity missing' || true; exit 1;
  }
  # Both ranks must mount the same indexer files (optimisation log E16): a mismatch makes the
  # ranks run different collectives and NCCL hangs.
  w_idx=$(grep -m1 '^node-launch: indexer mode=' <<<"$worker_output"); l_idx=$(grep -m1 '^node-launch: indexer mode=' <<<"$leader_output")
  [[ -n $w_idx && $w_idx == "$l_idx" ]] || {
    echo "indexer mismatch: worker [$w_idx] leader [$l_idx]" >&2
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'indexer mode mismatch between ranks' || true; exit 1;
  }
  # The same for the decode path (docs/dspark-experiment.md): speculative decoding, async
  # scheduling, prefix caching and the GEMV / DSpark-loader patch files.
  w_dec=$(grep -m1 '^node-launch: decode ' <<<"$worker_output"); l_dec=$(grep -m1 '^node-launch: decode ' <<<"$leader_output")
  [[ -n $w_dec && $w_dec == "$l_dec" ]] || {
    echo "decode settings mismatch: worker [$w_dec] leader [$l_dec]" >&2
    "$HERE/dsv4-abort.sh" --run-id "$RUN_ID" --reason 'decode settings mismatch between ranks' || true; exit 1;
  }
  echo
  echo "stack=$STACK image=$IMG"
  echo "mode: spec=$SPEC diagnostics=$DIAG max_batch=$MB max_seqs=$MS max_model_len=$MAXLEN gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "cold start is ~5 min (weights + cudagraph capture)."
  echo "  follow:  $0 logs"
  echo "  check:   $0 status"
  ;;
logs)   docker logs -f dsv4-leader 2>&1 ;;
wlogs)  ssh -o BatchMode=yes "$PEER" 'docker logs -f dsv4-worker' 2>&1 ;;
status)
  echo "=== containers ==="
  timed "$SSH_TIMEOUT" docker ps -a --filter name=dsv4-leader --format "  leader: {{.Status}}"
  timed "$SSH_TIMEOUT" ssh -o BatchMode=yes "$PEER" 'docker ps -a --filter name=dsv4-worker --format "  worker: {{.Status}}"' 2>&1
  echo "=== endpoint ==="
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300 || echo "  not serving yet"
  echo
  echo "=== GPU state ==="
  printf '  leader: '
  timeout --kill-after=2 10 nvidia-smi --query-gpu=utilization.gpu,pstate,power.draw --format=csv,noheader 2>/dev/null || echo "unavailable"
  printf '  worker: '
  timeout --kill-after=2 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$PEER" 'nvidia-smi --query-gpu=utilization.gpu,pstate,power.draw --format=csv,noheader 2>/dev/null' 2>/dev/null || echo "unavailable"
  echo "=== key boot lines ==="
  if docker inspect dsv4-leader >/dev/null 2>&1; then
    docker logs dsv4-leader 2>&1 | grep -iE "MoE backend|KV cache size|Maximum concurrency|startup complete|error|Traceback" | tail -12
  else
    echo "  no leader container"
  fi
  ;;
*) echo "usage: $0 <up|down|logs|wlogs|status>" >&2; exit 2 ;;
esac
