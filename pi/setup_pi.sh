#!/usr/bin/env bash
# One-time setup on the Raspberry Pi (64-bit Raspberry Pi OS Bookworm).
# Run from the project folder:
#   bash pi/setup_pi.sh            # phone hotspot / normal internet
#   bash pi/setup_pi.sh --proxy    # college Wi-Fi (proxy 172.31.2.3:8080, see Pi_gateway_guide Step 5)
set -e
cd "$(dirname "$0")/.."

PROXY_URL="http://172.31.2.3:8080"
if [ "$1" = "--proxy" ]; then
  echo ">> Using college proxy $PROXY_URL for apt and pip"
  if [ ! -f /etc/apt/apt.conf.d/95proxy ]; then
    printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$PROXY_URL" "$PROXY_URL" \
      | sudo tee /etc/apt/apt.conf.d/95proxy
  fi
  export http_proxy=$PROXY_URL https_proxy=$PROXY_URL
fi

echo ">> apt packages"
sudo apt update
sudo apt install -y python3-venv python3-pip mosquitto mosquitto-clients
sudo apt install -y python3-picamera2 || echo "(picamera2 not installed - fine without a Pi camera)"

echo ">> Mosquitto"
# Accept connections from the NodeMCU and laptop (same as the Pi-gateway lab).
# Only create the file if no config already opens port 1883 (two listeners stop Mosquitto).
if ! grep -rqs "listener 1883" /etc/mosquitto/conf.d/; then
  printf "listener 1883\nallow_anonymous true\n" | sudo tee /etc/mosquitto/conf.d/local.conf
fi
sudo systemctl enable mosquitto
sudo systemctl restart mosquitto
systemctl is-active mosquitto

echo ">> Python venv (can also see the apt-installed picamera2)"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
# TFLite runtime: ai-edge-litert (64-bit OS, newer Python), else tflite-runtime (32-bit OS / Bullseye)
ARCH=$(dpkg --print-architecture)
echo "OS architecture: $ARCH, $(python3 --version)"
.venv/bin/pip install "numpy<2" pillow pyyaml flask "paho-mqtt>=2.0"
CORAL_WHL="https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl"
if [ "$ARCH" = "armhf" ]; then
  # 32-bit Raspberry Pi OS. tflite-runtime >= 2.13 (if already installed for this user in
  # ~/.local, visible through --system-site-packages) runs FP32 and INT8 correctly.
  # Fallback: Google Coral's 2.5 wheel - runs FP32 correctly but gives WRONG INT8 results.
  if python3 -c "import tflite_runtime as t, sys; v=tuple(map(int, t.__version__.split('.')[:2])); sys.exit(0 if v >= (2, 13) else 1)" 2>/dev/null; then
    echo "Using the existing tflite-runtime $(python3 -c 'import tflite_runtime as t; print(t.__version__)') (FP32 + INT8 OK)"
    .venv/bin/pip uninstall -y tflite-runtime 2>/dev/null || true
  else
    .venv/bin/pip install "$CORAL_WHL"
    echo "WARNING: tflite-runtime 2.5 - use the *_fp32.tflite models only (INT8 output is wrong)."
  fi
else
  .venv/bin/pip install ai-edge-litert || .venv/bin/pip install tflite-runtime
fi

echo ">> Check"
.venv/bin/python -c "import flask, paho.mqtt, PIL, yaml; print('python packages OK')"
.venv/bin/python -c "
try:
    from ai_edge_litert.interpreter import Interpreter; print('runtime: ai-edge-litert OK')
except ImportError:
    from tflite_runtime.interpreter import Interpreter; print('runtime: tflite-runtime OK')"
.venv/bin/python -c "import picamera2; print('picamera2 OK')" || echo "picamera2 not importable (USB webcam needs: sudo apt install python3-opencv)"
mkdir -p models
echo
echo "Done. Next: copy the trained .tflite files into $(pwd)/models/"
