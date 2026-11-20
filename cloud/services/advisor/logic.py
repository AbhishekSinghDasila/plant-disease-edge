"""Disease advisor: diagnosis (+ optional cloud second opinion + climate) -> verdict and measures.

Pure functions, no MQTT/DB (unit-tested in cloud/tests/test_advisor.py).

Verdicts:
  healthy    edge (and cloud, if any) say healthy
  disease    edge is sure of a disease, and the cloud model agrees and is sure (or is unreachable)
  uncertain  edge not sure (low confidence / bad photo), cloud not sure, or edge and cloud disagree
             -> no measures, retake tips
"""
from pathlib import Path

import yaml

KB = yaml.safe_load((Path(__file__).parent / "kb.yaml").read_text())
SEVERITY = ["none", "low", "medium", "high", "critical"]
OTHER_TITLES = {"no plant leaf in the photo": "No leaf found: retake the photo",
                "a plant the model does not know (not one of the 14 crops)": "Unknown plant: not one of the 14 crops"}
RETAKE = [
    "Take a new photo: one leaf, 40-60 cm from the camera, plain background, good light, held still.",
    "Photograph the side of the leaf with the clearest spots.",
    "The model only knows 14 crops (apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato, "
    "raspberry, soybean, squash, strawberry, tomato). Other plants get a wrong label.",
]


def entry(class_id):
    """class id 'tomato.late_blight' -> KB entry (healthy classes share one entry). None if unknown."""
    if not class_id:
        return None
    if class_id.endswith(".healthy"):
        return KB["_healthy"]
    return KB.get(class_id)


def crop_name(class_id):
    return (class_id or "?").split(".")[0].capitalize()


def label(class_id, e):
    """'Tomato: Late blight' style title, without repeating the crop ('Tomato late blight' stays as is)."""
    crop = crop_name(class_id)
    return e["name"] if e["name"].lower().startswith(crop.lower()) else f"{crop}: {e['name']}"


def nice(class_id):
    e = entry(class_id)
    if e is None:
        return (class_id or "?").replace(".", ": ").replace("_", " ")
    return f"{crop_name(class_id)}: {e['name']}" if e["name"] == "Healthy" else e["name"]


def bump(sev, n=1):
    i = SEVERITY.index(sev) if sev in SEVERITY else 0
    return SEVERITY[min(len(SEVERITY) - 1, i + n)] if i > 0 else sev


def climate_note(e, climate):
    """-> (note, raise_severity). climate = {temp_c, humidity, fungal_risk} or {}."""
    rh, temp = (climate or {}).get("humidity"), (climate or {}).get("temp_c")
    if rh is None:
        return "No climate reading (DHT11 not wired), so the weather risk is not included.", False
    if e["humid"] and rh >= 85:
        return f"Humid air ({rh:.0f} % RH) helps this disease spread: act sooner than usual.", True
    if e["type"] == "pest" and temp is not None and temp >= 30 and rh < 50:
        return f"Hot, dry air ({temp:.0f} °C, {rh:.0f} % RH) makes spider mites multiply fast.", True
    if e["humid"]:
        return f"Air humidity {rh:.0f} % RH: keep leaves dry to slow the spread.", False
    return f"Climate now: {temp:.0f} °C, {rh:.0f} % RH.", False


def advise(edge, cloud=None, climate=None):
    """edge/cloud = diagnosis dicts (class_id or top_label, status, confidence, quality...). -> advice dict."""
    cid = edge.get("class_id") or edge.get("top_label")
    base = {"image_id": edge.get("image_id"), "device": edge.get("device"), "class_id": cid,
            "crop": edge.get("crop"), "edge_confidence": edge.get("confidence"),
            "cloud_class_id": (cloud or {}).get("class_id"), "cloud_confidence": (cloud or {}).get("confidence"),
            "immediate": [], "cultural": [], "chemical": [], "prevention": []}
    e = entry(cid)

    # 1. edge not sure -> no measures
    if edge.get("status") == "not_sure" or e is None:
        q = edge.get("quality") or {}
        msg = (q.get("message") or "the photo check failed").split(". ")[0].rstrip(".")
        why = msg[:1].lower() + msg[1:] if q.get("ok") is False else \
            f"the model is only {100 * (edge.get('confidence') or 0):.0f} % sure"
        guess = nice(cid)
        look = f" If it really is {guess}, look for: {e['symptoms']}" if e and e["type"] != "none" else ""
        return {**base, "verdict": "uncertain", "status": "not_sure", "severity": "none",
                "title": "Not sure: retake the photo",
                "summary": f"Best guess {guess}, but {why}. No treatment advice for an uncertain result.{look}",
                "immediate": RETAKE, "agreement": "edge_unsure"}

    # 2. cloud second opinion disagrees -> uncertain, show both
    agreement = "edge_only"
    if cloud and cloud.get("class_id"):
        if cloud["class_id"].startswith("other.") and cloud.get("not_sure_reason") in OTHER_TITLES:
            # v2 cloud model is sure the photo is not a known crop leaf: the edge answer cannot be right
            reason = cloud["not_sure_reason"]
            return {**base, "verdict": "uncertain", "status": "not_sure", "severity": "none",
                    "title": OTHER_TITLES[reason],
                    "summary": f"The cloud model sees {reason} ({100 * (cloud.get('confidence') or 0):.0f} %), so the edge "
                               f"answer ({nice(cid)}, {100 * edge['confidence']:.0f} %) is ignored.",
                    "immediate": RETAKE, "agreement": "cloud_rejects"}
        if cloud["class_id"] == cid and cloud.get("status") == "not_sure":
            # measured on field photos (results/cloud_tta.csv): requiring a sure cloud cuts confident-wrong answers 56 % -> 30 %
            reason = cloud.get("not_sure_reason") or "low confidence"
            return {**base, "verdict": "uncertain", "status": "not_sure", "severity": "none",
                    "title": "Not sure: the cloud check failed",
                    "summary": f"The edge model says {nice(cid)} ({100 * edge['confidence']:.0f} %), but the bigger cloud model "
                               f"is not sure ({100 * (cloud.get('confidence') or 0):.0f} %, {reason}). "
                               f"If it really is {nice(cid)}, look for: {e['symptoms']}",
                    "immediate": RETAKE, "agreement": "cloud_unsure"}
        if cloud["class_id"] == cid:
            agreement = "agree"
        else:
            other = entry(cloud["class_id"])
            return {**base, "verdict": "uncertain", "status": "not_sure", "severity": "none",
                    "title": "Models disagree: check the leaf by eye",
                    "summary": f"Edge model: {nice(cid)} ({100 * edge['confidence']:.0f} %). "
                               f"Cloud model: {nice(cloud['class_id'])} ({100 * (cloud.get('confidence') or 0):.0f} %).",
                    "immediate": [f"{nice(cid)} looks like: {e['symptoms']}"] +
                                 ([f"{nice(cloud['class_id'])} looks like: {other['symptoms']}"] if other else []) + RETAKE[:1],
                    "agreement": "disagree"}

    # 3. healthy
    if e["type"] == "none":
        return {**base, "verdict": "healthy", "status": "healthy", "severity": "none",
                "title": f"{crop_name(cid)}: healthy" + (" (both models agree)" if agreement == "agree" else ""),
                "summary": e["symptoms"], "immediate": e["immediate"], "cultural": e["cultural"],
                "prevention": e["prevention"], "agreement": agreement}

    # 4. disease
    note, worse = climate_note(e, climate)
    sev = bump(e["severity"]) if worse else e["severity"]
    return {**base, "verdict": "disease", "status": "disease", "severity": sev, "type": e["type"],
            "title": label(cid, e) + (" (both models agree)" if agreement == "agree" else ""),
            "summary": f"{e['symptoms']} Cause: {e['cause']}. Spread: {e['spread']}.",
            "climate_note": note, "immediate": e["immediate"], "cultural": e["cultural"],
            "chemical": e["chemical"] + (["Follow the product label and local extension advice."] if e["chemical"] else []),
            "prevention": e["prevention"], "agreement": agreement}
