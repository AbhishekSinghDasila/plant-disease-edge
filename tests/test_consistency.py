"""The class ids must be the same everywhere: cloud KB, irrigation bands, Colab notebook class map.
(Catalog vs edge model labels is checked in test_catalog.py.)"""
import json
from pathlib import Path

import yaml

from pde import catalog

ROOT = Path(__file__).resolve().parent.parent


def test_cloud_kb_covers_catalog():
    kb = yaml.safe_load((ROOT / "cloud/services/advisor/kb.yaml").read_text())
    for cid in catalog.BY_ID:
        assert cid in kb or cid.endswith(".healthy"), cid


def test_colab_class_map_matches():
    c = json.loads((ROOT / "notebooks/cloud_v2/classes.json").read_text())
    edge = json.loads((ROOT / "models/pv38.labels.json").read_text())["ids"]
    assert c["ids"] == edge                                  # same order: v2 ids = these 38 + 2 "other"
    assert sorted(c["pv_folder_to_id"].values()) == sorted(edge)


def test_every_crop_has_a_water_band():
    bands = yaml.safe_load((ROOT / "cloud/services/irrigation/crops.yaml").read_text())
    for crop in catalog.CROPS:
        assert crop in bands, crop
