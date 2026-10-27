#!/usr/bin/env bash
# Emit the NCCL environment for the CX-7 fabric. Source it, or run it to print.
#   source /srv/dsv4/run/cx7-nccl-env.sh      (leader; the worker copy is ~/cx7-nccl-env.sh)
# When sourced it returns 1 if no CX-7 link is Up; node-launch.sh then refuses to start.
#
# HCA names are DISCOVERED, never hardcoded. The runbook hardcodes the "f1
# twins"; an earlier revision of this file hardcoded f0 because that was where
# the cable sat at the time. The cable has moved between f0 and f1 more than once
# (optimisation log INV-2; on f0 since 2026-10-15), and each hardcoding was correct
# exactly once, so derive it instead.

# --- data plane: every CX-7 device whose link is actually Up ------------------
_UP_NETDEV=$(ibdev2netdev | awk '/\(Up\)/{print $5}')
_UP_HCA=$(ibdev2netdev | awk '/\(Up\)/{print $1}' | paste -sd,)

if [[ -z "$_UP_HCA" ]]; then
  echo "cx7-nccl-env: no CX-7 device is Up -- check the QSFP cable" >&2
  return 1 2>/dev/null || exit 1
fi

export NCCL_IB_HCA="$_UP_HCA"
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_IB_MERGE_NICS=1
# Discover the local IPv4 RoCE-v2 GID instead of hardcoding its slot. GID
# indices can change after a link cycle or address recreation (for example,
# rank 0 may use index 3 while rank 1 uses index 4). Each NCCL rank selects its
# own local GID and exchanges the resulting address with its peer.
_FIRST_HCA="${_UP_HCA%%,*}"
NCCL_IB_GID_INDEX=""
for _type_file in /sys/class/infiniband/"$_FIRST_HCA"/ports/1/gid_attrs/types/*; do
  [[ "$(cat "$_type_file" 2>/dev/null)" == "RoCE v2" ]] || continue
  _idx="${_type_file##*/}"
  _gid="$(cat "/sys/class/infiniband/$_FIRST_HCA/ports/1/gids/$_idx" 2>/dev/null)"
  [[ "$_gid" == *:ffff:* && "$_gid" != "0000:0000:0000:0000:0000:0000:0000:0000" ]] || continue
  NCCL_IB_GID_INDEX="$_idx"
  break
done
if [[ -z "$NCCL_IB_GID_INDEX" ]]; then
  echo "cx7-nccl-env: no IPv4 RoCE-v2 GID found on $_FIRST_HCA" >&2
  return 1 2>/dev/null || exit 1
fi
export NCCL_IB_GID_INDEX
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_CROSS_NIC=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# --- control plane -----------------------------------------------------------
# The runbook says "pin to the management NIC". Do NOT do that here: management
# is Wi-Fi on both boxes, it roams, it renegotiates rates, and its IPv6 is
# broken. Worse, tailscale0 is up on spark-61dd -- unpinned, NCCL will happily
# route tensor-parallel traffic over the overlay and everything will "work" at
# terrible speed. Pin all three selectors to a fabric interface.
#
# Use the SECOND plane for control so rendezvous chatter and the Gloo CPU group
# do not share a NIC with the plane NCCL picks first for bulk transfer. Traffic
# is kilobytes either way; this is tidiness, not bandwidth.
_CTRL=$(echo "$_UP_NETDEV" | grep '^enP2p' | head -1)
[[ -z "$_CTRL" ]] && _CTRL=$(echo "$_UP_NETDEV" | head -1)

export NCCL_SOCKET_IFNAME="$_CTRL"
export GLOO_SOCKET_IFNAME="$_CTRL"    # torch.distributed's CPU group is Gloo and
export TP_SOCKET_IFNAME="$_CTRL"      # needs its own pin -- the classic omission

# --- rendezvous --------------------------------------------------------------
# Leader is spark-61dd on the .200 plane. Fixed by hostname so both nodes agree.
# CX7_NODE_RANK is informational (printed below); node-launch.sh takes the rank as
# its first argument.
export CX7_MASTER_ADDR=192.168.200.10
export CX7_MASTER_PORT=25000
case "$(hostname)" in
  spark-61dd) export CX7_NODE_RANK=0 ;;
  gx10-6d0c)  export CX7_NODE_RANK=1 ;;
esac

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "# discovered on $(hostname)"
  echo "#   up netdevs: $(echo $_UP_NETDEV | tr '\n' ' ')"
  echo "#   up HCAs:    $_UP_HCA"
  echo "#   control:    $_CTRL"
  for v in NCCL_IB_HCA NCCL_NET NCCL_IB_DISABLE NCCL_IB_MERGE_NICS NCCL_IB_GID_INDEX \
           NCCL_IB_ADDR_FAMILY NCCL_IB_ROCE_VERSION_NUM NCCL_CROSS_NIC \
           NCCL_IGNORE_CPU_AFFINITY NCCL_CUMEM_ENABLE NCCL_NVLS_ENABLE NCCL_DEBUG \
           NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME \
           CX7_MASTER_ADDR CX7_MASTER_PORT CX7_NODE_RANK; do
    printf '%s=%s\n' "$v" "${!v}"
  done
fi
