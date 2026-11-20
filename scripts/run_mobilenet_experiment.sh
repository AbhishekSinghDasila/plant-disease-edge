#!/usr/bin/env bash
# Full MobileNetV3-Small experiment from the proposal, on a laptop CPU (~1 h). Log: results/train_log.txt
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p results
step() { echo; echo "===== $(date +%H:%M) $*"; }
step "1/6 baseline (standard augmentation)"; $PY scripts/train.py --model mobilenetv3small --augment standard
step "2/6 strong augmentation";              $PY scripts/train.py --model mobilenetv3small --augment strong
step "3/6 fine-tune on field images";        $PY scripts/finetune_field.py --base models/mobilenetv3small_strong
step "4/6 quantize";                         for m in standard field; do $PY scripts/quantize.py models/mobilenetv3small_$m; done
step "5/6 evaluate";                         $PY scripts/evaluate.py models/mobilenetv3small_standard models/mobilenetv3small_strong models/mobilenetv3small_field models/mobilenetv3small_*.tflite
step "6/6 done"
