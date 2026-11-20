#!/usr/bin/env bash
# Connect the Pi's Mosquitto to the cloud broker (the laptop's docker-compose stack).
#   bash pi/install_bridge.sh 10.93.8.158      # laptop IP (check with: ip -4 addr on the laptop)
#   bash pi/install_bridge.sh --remove
set -euo pipefail
cd "$(dirname "$0")"
DEST=/etc/mosquitto/conf.d/bridge.conf
if [[ "${1:-}" == "--remove" ]]; then
  sudo rm -f "$DEST" && sudo systemctl restart mosquitto && echo "bridge removed"; exit 0
fi
HOST="${1:?usage: install_bridge.sh <cloud-ip>}"
sed "s/__CLOUD_HOST__/$HOST/" mosquitto-bridge.conf.template | sudo tee "$DEST" >/dev/null
sudo systemctl restart mosquitto
sleep 3
systemctl is-active mosquitto
mosquitto_sub -h localhost -t plant/bridge/pi-node-01 -C 1 -W 10 -v || echo "no bridge notification yet (cloud reachable?)"
