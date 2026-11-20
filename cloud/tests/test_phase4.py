"""Phase 4 tests: disease advisor + notifier rules (run: cd cloud && PYTHONPATH=. python -m pytest -q tests)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


adv = load("advisor_logic", "services/advisor/logic.py")
rules_mod = load("notifier_rules", "services/notifier/rules.py")
IDS = json.loads((ROOT.parent / "models/pv38.labels.json").read_text())["ids"]
EDGE = {"image_id": "x1", "device": "pi-node-01", "class_id": "tomato.late_blight", "crop": "tomato",
        "status": "disease", "confidence": 0.93}


def test_kb_covers_all_38_classes():
    assert len(IDS) == 38
    for cid in IDS:
        e = adv.entry(cid)
        assert e is not None, cid
        assert e["type"] in ("none", "fungal", "bacterial", "viral", "pest"), cid
        assert e["severity"] in adv.SEVERITY, cid
        if e["type"] != "none":
            assert e["immediate"] and e["prevention"] and e["symptoms"], cid
    extra = set(adv.KB) - set(IDS) - {"_healthy"}
    assert not extra, f"KB ids not in the model: {extra}"


def test_disease_verdict_and_humid_bump():
    a = adv.advise(EDGE, None, {})
    assert a["verdict"] == "disease" and a["severity"] == "critical" and a["chemical"]
    b = adv.advise({**EDGE, "class_id": "tomato.early_blight"}, None, {"temp_c": 24, "humidity": 92})
    assert b["severity"] == "high" and "Humid" in b["climate_note"]       # medium -> high


def test_not_sure_gives_no_measures():
    a = adv.advise({**EDGE, "status": "not_sure", "confidence": 0.7})
    assert a["verdict"] == "uncertain" and not a["chemical"] and "70 %" in a["summary"]
    q = adv.advise({**EDGE, "status": "not_sure", "quality": {"ok": False, "message": "Photo is too blurry."}})
    assert "blurry" in q["summary"]


def test_cloud_agree_and_disagree():
    assert adv.advise(EDGE, {"class_id": "tomato.late_blight", "confidence": 0.9})["agreement"] == "agree"
    d = adv.advise(EDGE, {"class_id": "tomato.early_blight", "confidence": 0.8})
    assert d["verdict"] == "uncertain" and d["agreement"] == "disagree"


def test_healthy():
    a = adv.advise({**EDGE, "class_id": "apple.healthy", "status": "healthy"})
    assert a["verdict"] == "healthy" and a["title"].startswith("Apple")


def test_notifier_disease_cooldown():
    r = rules_mod.Rules(disease_cooldown_s=600)
    a = adv.advise(EDGE)
    assert r.on_advice(a, 0)[0]["level"] == "critical"
    assert r.on_advice(a, 100) == []                      # same disease again soon: no spam
    assert len(r.on_advice(a, 700)) == 1
    assert r.on_advice(adv.advise({**EDGE, "class_id": "corn.common_rust"}), 0) == []   # low severity: no alert


def test_notifier_water_status_watchdog():
    r = rules_mod.Rules(water_cooldown_s=1800, silent_after_s=300)
    hi = {"need_water": True, "urgency": "high", "reason": "soil 18 %"}
    assert len(r.on_irrigation(hi, 0)) == 1 and r.on_irrigation(hi, 60) == []
    r.on_irrigation({"need_water": False}, 120)
    assert len(r.on_irrigation(hi, 180)) == 1             # new dry spell alerts again
    assert r.on_status({"device": "pi", "state": "online"}, 0) == []
    assert r.on_status({"device": "pi", "state": "offline"}, 1)[0]["level"] == "critical"
    assert r.on_status({"device": "pi", "state": "online"}, 2)[0]["level"] == "info"
    r.seen("soil", 0)
    assert r.watchdog(100) == []
    assert len(r.watchdog(400)) == 1 and r.watchdog(500) == []
    assert r.seen("soil", 600)[0]["level"] == "info"


# ---- Phase 5: cloud second opinion
def test_cloud_unsure_blocks_diagnosis():
    a = adv.advise(EDGE, {"class_id": "tomato.late_blight", "status": "not_sure", "confidence": 0.6,
                          "not_sure_reason": "low confidence"})
    assert a["verdict"] == "uncertain" and a["agreement"] == "cloud_unsure" and not a["chemical"]
    ok = adv.advise(EDGE, {"class_id": "tomato.late_blight", "status": "disease", "confidence": 0.95})
    assert ok["verdict"] == "disease" and ok["agreement"] == "agree"


def test_notifier_waits_for_cloud():
    r = rules_mod.Rules()
    a = adv.advise(EDGE)
    assert r.on_advice({**a, "pending_cloud": True}, 0) == []
    assert len(r.on_advice({**a, "pending_cloud": False}, 1)) == 1


def test_tta_model_on_sample():
    import pytest
    model = ROOT.parent / "models/pv38_fp32.tflite"
    photo = ROOT.parent / "samples/Tomato___Late_blight.jpg"
    if not model.exists() or not photo.exists():
        pytest.skip("model or sample photo missing")
    tta = load("tta", "services/cloud_inference/tta.py")
    m = tta.CloudModel(model, ROOT.parent / "models/pv38.labels.json")
    r = m.predict(photo)
    assert r["class_id"] == "tomato.late_blight" and r["status"] == "disease"
    assert len(r["views"]) == 6 and r["agreement"] == 1.0


def test_cloud_v2_rejects_non_leaf():
    a = adv.advise(EDGE, {"class_id": "other.not_a_leaf", "status": "not_sure", "confidence": 0.97,
                          "not_sure_reason": "no plant leaf in the photo"})
    assert a["verdict"] == "uncertain" and a["agreement"] == "cloud_rejects" and a["title"].startswith("No leaf")
    b = adv.advise(EDGE, {"class_id": "other.unknown_plant", "status": "not_sure", "confidence": 0.9,
                          "not_sure_reason": "a plant the model does not know (not one of the 14 crops)"})
    assert b["title"].startswith("Unknown plant")


def test_tta_loads_v2_style_model():
    """The smoke-test export of the Colab notebook (40 classes, 64 px) loads and answers in the right format."""
    import os, pytest
    smoke = os.environ.get("V2_SMOKE_DIR")
    if not smoke:
        pytest.skip("set V2_SMOKE_DIR to the smoke export folder")
    tta = load("tta", "services/cloud_inference/tta.py")
    m = tta.CloudModel(f"{smoke}/cloud_v2.tflite", f"{smoke}/cloud_v2.labels.json")
    r = m.predict(ROOT.parent / "samples/Tomato___Late_blight.jpg")
    assert len(m.ids) == 40 and m.size == 64 and r["status"] in ("disease", "healthy", "not_sure")
