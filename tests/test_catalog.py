"""38-class catalog and multi-crop predictor."""
import json
from pathlib import Path

import pytest

from pde import catalog

ROOT = Path(__file__).resolve().parent.parent


def test_catalog_complete():
    assert len(catalog.PV38) == 38 and len(set(catalog.PV38_IDS)) == 38
    assert catalog.PV38 == sorted(catalog.PV38)          # byte-sorted = model output order
    assert len(catalog.CROPS) == 14
    for i in catalog.BY_ID.values():
        assert i["advice"] and i["crop_name"] and i["disease_name"]


def test_labels_file_matches_catalog():
    f = ROOT / "models" / "pv38.labels.json"
    if not f.exists():
        pytest.skip("run scripts/import_pretrained.py --all-crops first")
    assert json.loads(f.read_text())["ids"] == catalog.PV38_IDS


@pytest.mark.parametrize("folder", ["Apple___Black_rot", "Tomato___healthy", "Potato___Late_blight"])
def test_pv38_predicts_crop(folder):
    model = ROOT / "models" / "pv38_int8.tflite"
    d = ROOT / "data" / "pv_allcrops" / folder
    if not d.exists():
        d = ROOT / "data" / "lab" / "test" / folder
    if not model.exists() or not d.exists():
        pytest.skip("model or images missing")
    from pde.predictor import Predictor
    p = Predictor(model).predict(sorted(d.glob("*"))[0])
    want = catalog.BY_ID[catalog.class_id(folder)]
    assert p.crop == want["crop"]
    assert p.display == want["display"] and p.class_id == want["id"]
