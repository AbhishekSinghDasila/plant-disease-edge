"""Unit tests for cloud logic (run: cd cloud && PYTHONPATH=. python -m pytest -q tests)."""
import importlib.util
from pathlib import Path

from common.agro import vpd_kpa

spec = importlib.util.spec_from_file_location("logic", Path(__file__).parent.parent / "services/irrigation/logic.py")
logic = importlib.util.module_from_spec(spec); spec.loader.exec_module(logic)
BAND = {"dry": 40, "target": 65, "wilting": 25}


def test_vpd():
    assert abs(vpd_kpa(25, 50) - 1.584) < 0.01
    assert vpd_kpa(None, 50) is None


def test_decide_bands_and_hysteresis():
    assert logic.decide(70, BAND)["need_water"] is False
    r = logic.decide(35, BAND)
    assert r["need_water"] and r["urgency"] == "medium"
    assert logic.decide(35, BAND, temp_c=35, rh=30)["urgency"] == "high"
    assert logic.decide(20, BAND)["urgency"] == "high"
    assert logic.decide(50, BAND, was_needed=True)["need_water"] is True    # stays on until target
    assert logic.decide(50, BAND, was_needed=False)["need_water"] is False


def test_trend_and_hours_to_dry():
    pts = [(i * 600, 60 - i * 0.5) for i in range(7)]       # -3 %/h
    s = logic.trend_pct_per_hour(pts)
    assert abs(s + 3) < 1e-6
    assert logic.decide(55, BAND, slope=s)["hours_to_dry"] == 5.0


def test_detect_watering():
    assert logic.detect_watering(None, 50) is None
    assert logic.detect_watering(40, 39.2) is None          # normal drying
    assert logic.detect_watering(40, 44) is None            # small rise (noise / light drizzle)
    w = logic.detect_watering(30, 65)
    assert w == {"rise": 35.0, "from_pct": 30.0, "to_pct": 65.0}
