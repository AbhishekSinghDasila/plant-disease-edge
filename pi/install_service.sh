#!/usr/bin/env bash
# Start the plant app automatically on boot (fills in this user and folder).
#   bash pi/install_service.sh          logs: journalctl -u plant-app -f
#   sudo systemctl disable --now plant-app   to turn it off
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
sed -e "s|^User=.*|User=$USER|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=$DIR|" \
    -e "s|^ExecStart=.*|ExecStart=$DIR/.venv/bin/python app/server.py|" \
    "$DIR/pi/plant-app.service" | sudo tee /etc/systemd/system/plant-app.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now plant-app
sleep 3
systemctl --no-pager status plant-app | head -5
echo "Open http://$(hostname -I | awk '{print $1}'):8000"
