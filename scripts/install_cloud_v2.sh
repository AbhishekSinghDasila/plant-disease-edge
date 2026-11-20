#!/usr/bin/env bash
# Phase 5 v2: install the Colab-trained cloud model, compare it with v1, and switch the cloud service to it.
#
#   bash scripts/install_cloud_v2.sh ~/Downloads            # folder with cloud_v2.tflite + cloud_v2.labels.json
#   bash scripts/install_cloud_v2.sh ~/Downloads --force    # switch even if v2 is not better
#   bash scripts/install_cloud_v2.sh --rollback             # back to v1 (pv38_fp32)
#   CPUS=0-1 bash scripts/install_cloud_v2.sh ~/Downloads   # gentler on battery (2 cores)
#
# It switches only if v2 makes fewer confident-wrong answers on the field test photos than v1, is at least as
# accurate on them, and still gets >= 95 % of lab photos right; otherwise it keeps v1 and prints the numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/cloud/docker-compose.yml"
PY="$ROOT/.venv/bin/python"

switch() {   # $1 = model file name in models/, $2 = labels file name
  sed -i -E "s#(MODEL_PATH: )/models/[^ ]+#\1/models/$1#; s#(LABELS_PATH: )/models/[^ ]+#\1/models/$2#" "$COMPOSE"
  (cd "$ROOT/cloud" && docker-compose rm -fs cloud-inference >/dev/null 2>&1; docker-compose up -d cloud-inference)
  sleep 6
  docker logs --tail 3 cloud_cloud-inference_1 2>&1 | grep -i "model" || true
}

if [[ "${1:-}" == "--rollback" ]]; then
  switch pv38_fp32.tflite pv38.labels.json
  echo "Rolled back to v1 (pv38_fp32)."
  exit 0
fi

SRC="${1:?give the folder that holds cloud_v2.tflite and cloud_v2.labels.json}"
FORCE="${2:-}"
for f in cloud_v2.tflite cloud_v2.labels.json; do
  [[ -f "$SRC/$f" ]] || { echo "missing $SRC/$f"; exit 1; }
done
cp "$SRC/cloud_v2.tflite" "$SRC/cloud_v2.labels.json" "$ROOT/models/"
echo "Copied to models/. Training metrics from Colab:"
"$PY" -c "import json; d=json.load(open('$ROOT/models/cloud_v2.labels.json')); print(' ', d.get('model'), d.get('trained'), len(d['ids']), 'classes'); [print('  ', k, v) for k, v in (d.get('test_metrics') or {}).items()]"

echo "Evaluating v1 and v2 on the laptop's held-out lab + field test photos (~5 min, charger in!) ..."
cd "$ROOT"
taskset -c "${CPUS:-0-3}" nice -n 10 "$PY" scripts/eval_cloud_tta.py --out results/cloud_tta.csv 2>/dev/null | grep -E "^(set|lab|field) "
taskset -c "${CPUS:-0-3}" nice -n 10 "$PY" scripts/eval_cloud_tta.py --cloud-model models/cloud_v2.tflite \
  --out results/cloud_v2_eval.csv 2>/dev/null | grep -E "^(set|lab|field) "

VERDICT=$("$PY" - <<'EOF'
import csv
def get(path):
    return {(r["set"], r["setup"]): r for r in csv.DictReader(open(path))}
v1, v2 = get("results/cloud_tta.csv"), get("results/cloud_v2_eval.csv")
k = ("field", "edge+cloud_agree")
better = float(v2[k]["confident_wrong"]) < float(v1[k]["confident_wrong"])
lab_ok = float(v2[("lab", "cloud_fp32_tta6")]["accuracy"]) >= 0.95
# a model that refuses everything has 0 % confident-wrong, so it must also be at least as accurate on field photos
lab_ok = lab_ok and float(v2[("field", "cloud_fp32_tta6")]["accuracy"]) >= float(v1[("field", "cloud_fp32_tta6")]["accuracy"])
print(f"field confident-wrong (edge+cloud): v1 {float(v1[k]['confident_wrong']):.1%} -> v2 {float(v2[k]['confident_wrong']):.1%}; "
      f"v2 lab accuracy {float(v2[('lab', 'cloud_fp32_tta6')]['accuracy']):.1%}")
print("SWITCH" if better and lab_ok else "KEEP")
EOF
)
echo "$VERDICT" | head -1
if [[ "$(echo "$VERDICT" | tail -1)" == "SWITCH" || "$FORCE" == "--force" ]]; then
  switch cloud_v2.tflite cloud_v2.labels.json
  echo "Cloud service now uses cloud_v2. Undo: bash scripts/install_cloud_v2.sh --rollback"
else
  echo "v2 is not better on these tests: kept v1. Use --force to switch anyway."
fi
