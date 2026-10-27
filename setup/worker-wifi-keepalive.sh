#!/usr/bin/env bash
# Worker (gx10) WiFi keepalive — optimisation log I03, 2026-10-23. Run as root ON THE WORKER.
# Why: at 10:24:13 the 4-way handshake timed out (wpa_supplicant reason=15) on 6 GHz.
# NetworkManager read that as a wrong key, asked for a new one, found no secret agent
# (`no-secrets`) and stopped autoconnecting for good. The worker was off WiFi/Tailscale for
# about 9 h until a reboot. The stored PSK (psk-flags=0) was fine.
# This installs a timer that brings the WiFi profile back up when wlP9s9 is not connected.
# Usage: sudo setup/worker-wifi-keepalive.sh "<NetworkManager connection name>"
# It touches only the WiFi device, never the CX7 links that serving uses.
# Undo: systemctl disable --now wifi-keepalive.timer; rm /etc/systemd/system/wifi-keepalive.* /usr/local/sbin/wifi-keepalive
set -euo pipefail
CON=${1:?usage: worker-wifi-keepalive.sh "<NetworkManager connection name>"}
[[ $EUID == 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ $(hostname) == gx10-6d0c ]] || { echo "this is for the worker gx10-6d0c only" >&2; exit 1; }

{ printf '#!/usr/bin/env bash\n# Bring the WiFi profile back when NetworkManager has given up on it (see I03).\nCON=%q DEV=wlP9s9\n' "$CON"
  cat <<'EOS'
state=$(nmcli -t -f DEVICE,STATE dev | awk -F: -v d=$DEV '$1==d{print $2}')
[[ $state == connected ]] && exit 0
logger -t wifi-keepalive "$DEV state=${state:-absent}; reactivating '$CON'"
timeout 90 nmcli --wait 60 connection up "$CON" ifname $DEV 2>&1 | logger -t wifi-keepalive
EOS
} >/usr/local/sbin/wifi-keepalive
chmod 755 /usr/local/sbin/wifi-keepalive

cat >/etc/systemd/system/wifi-keepalive.service <<'EOS'
[Unit]
Description=Reactivate the worker WiFi profile if NetworkManager gave up (DSV4 I03)
After=NetworkManager.service
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/wifi-keepalive
EOS

cat >/etc/systemd/system/wifi-keepalive.timer <<'EOS'
[Unit]
Description=Check worker WiFi every 5 minutes (DSV4 I03)
[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
EOS

# Keep retrying autoconnect without a limit (default gives up after 4 tries).
nmcli connection modify "$CON" connection.autoconnect-retries 0

systemctl daemon-reload
systemctl enable --now wifi-keepalive.timer
systemctl start wifi-keepalive.service
echo "--- installed"
systemctl list-timers wifi-keepalive.timer --no-pager | head -3
nmcli -g connection.autoconnect-retries connection show "$CON"
journalctl -t wifi-keepalive -n 3 --no-pager || true
nmcli -t -f DEVICE,STATE dev | grep '^wlP9s9:'
