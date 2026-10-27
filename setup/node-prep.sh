#!/usr/bin/env bash
# One-time root setup for a DSV4 node. Run on each node: sudo bash setup/node-prep.sh
#   - installs the sudoers rule that lets the service account run
#     /usr/local/sbin/gb10-reclaim without a password (used before and after launch)
#   - sets vm.compaction_proactiveness=20 (E03, D5). This is the kernel default, so it
#     is not persisted; the line only undoes a non-default value until the next boot
#
# GPU clocks are left UNCAPPED (D8, E07.4: +5.1 % decode once the cap was cleared).
# The former `nvidia-smi -lgc 0,2100` line was removed on 2026-10-26 (optimisation
# log R01.7) so that re-running this script cannot silently re-apply the cap.
set -euo pipefail
[[ $EUID == 0 ]] || { echo "run with sudo" >&2; exit 1; }
u=${SUDO_USER:?run with sudo from the serving account}
f=/etc/sudoers.d/gb10-reclaim-$u
echo "$u ALL=(root) NOPASSWD: /usr/local/sbin/gb10-reclaim" > "$f.tmp"
chmod 440 "$f.tmp"
visudo -cf "$f.tmp" && mv "$f.tmp" "$f"
sysctl vm.compaction_proactiveness=20
echo "prep done on $(hostname)"
