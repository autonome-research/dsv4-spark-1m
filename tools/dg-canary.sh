#!/usr/bin/env bash
# Start, check or stop the kernel-load canary (tools/dg-load-canary.py; I04, optimisation
# log 2026-10-29). Leader only. It runs in the serving image with no network, a 4 GiB
# memory cap and the DeepGEMM JIT cache mounted read-only; it uses well under 1 GiB.
#
# Usage: dg-canary.sh start <label> | status | stop
#   start   one container, dsv4-dg-canary; output data/CANARY/<timestamp>-<label>/canary.jsonl
#   status  container state, last load rounds, failure count
#   stop    docker stop (the container is kept for inspection, as with the server)
#
# The canary is a GPU process, so the start preflight lists it as a blocker. Stop it
# before a restart (docs/operations.md); a new canary per boot also restarts its clock.
set -uo pipefail
NAME=dsv4-dg-canary
IMG=local/vllm-openai:46638857-fi0618
CACHE=/srv/dsv4/runtime-cache/46638857-fi0618-sm121/vllm/deep_gemm/cache
# The I04 kernel (einsum, 17 tokens), a linear-layer GEMM and an hc_prenorm GEMM.
KERNELS=(kernel.sm120_fp8_fp4_bmm.fd4d2cfacb0749f433c43400155ed3ac
         kernel.sm120_fp8_fp4_gemm_1d1d.01f6ce270eff2ffa81edcf78a3b6c2c5
         kernel.sm120_tf32_hc_prenorm_gemm.0b69afeb8890e0d43bc7ea8e6545e023)
INTERVAL=${CANARY_INTERVAL:-600}

case "${1:-}" in
  start)
    LABEL=${2:?usage: dg-canary.sh start <label>}
    docker inspect "$NAME" >/dev/null 2>&1 && { echo "$NAME exists; stop it and rename or remove it first" >&2; exit 1; }
    OUT=/srv/dsv4/run/data/CANARY/$(date +%Y%m%dT%H%M%S)-$LABEL
    mkdir -p "$OUT"
    cubins=$(printf '/dgcache/%s/kernel.cubin:' "${KERNELS[@]}")
    docker run -d --name "$NAME" --label dsv4.canary=I04 --gpus all --network none \
      --memory 4g --restart no \
      -v "$CACHE:/dgcache:ro" -v /srv/dsv4/run/tools/dg-load-canary.py:/canary.py:ro -v "$OUT:/out" \
      -e CANARY_OUT=/out/canary.jsonl -e CANARY_INTERVAL="$INTERVAL" -e CANARY_CUBINS="${cubins%:}" \
      --entrypoint python3 "$IMG" -I /canary.py >"$OUT/container.id" || exit 1
    { echo "label: $LABEL"; echo "started: $(date -Is)"; echo "image: $IMG"; echo "interval_s: $INTERVAL"
      echo "kernels: ${KERNELS[*]}"
      echo "server_run_id: $(docker inspect dsv4-leader --format '{{index .Config.Labels "dsv4.t2.run"}}' 2>/dev/null)"
      echo "server_started: $(docker inspect dsv4-leader --format '{{.State.StartedAt}}' 2>/dev/null)"
      echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; } >"$OUT/config.txt"
    echo "started $NAME; output $OUT" ;;
  status)
    docker inspect "$NAME" --format 'state={{.State.Status}} started={{.State.StartedAt}} exit={{.State.ExitCode}}' || exit 1
    OUT=$(docker inspect "$NAME" --format '{{range .Mounts}}{{if eq .Destination "/out"}}{{.Source}}{{end}}{{end}}')
    J=$OUT/canary.jsonl
    echo "log: $J"
    echo "load rounds: $(grep -c '"event": "load"' "$J"), failed: $(grep -c '"ok": false' "$J")"
    tail -n 2 "$J" | cut -c1-400 ;;
  stop)
    docker stop -t 30 "$NAME" ;;
  *) echo "usage: dg-canary.sh start <label> | status | stop" >&2; exit 2 ;;
esac
