"""Irrigation decision: pure functions, no MQTT/DB (unit-tested in cloud/tests)."""
from common.agro import vpd_kpa


def trend_pct_per_hour(points):
    """points: [(unix_seconds, moisture_pct)] -> least-squares slope in %/h (None if < 3 points or < 5 min)."""
    if len(points) < 3 or points[-1][0] - points[0][0] < 300:
        return None
    n = len(points)
    mx = sum(t for t, _ in points) / n
    my = sum(m for _, m in points) / n
    sxx = sum((t - mx) ** 2 for t, _ in points)
    if sxx == 0:
        return None
    sxy = sum((t - mx) * (m - my) for t, m in points)
    return sxy / sxx * 3600


def decide(moisture, band, temp_c=None, rh=None, slope=None, was_needed=False):
    """-> dict(need_water, urgency none|low|medium|high, reason, hours_to_dry, vpd_kpa).

    need_water turns on below band['dry'] and stays on until band['target'] (hysteresis), so it does not
    flap around the threshold. Heat or dry air (high VPD) raise the urgency because the soil will dry faster.
    """
    vpd = vpd_kpa(temp_c, rh)
    hot = (temp_c is not None and temp_c >= 32) or (vpd is not None and vpd >= 1.6)
    hours = None
    if slope is not None and slope < -0.01 and moisture > band["dry"]:
        hours = round((moisture - band["dry"]) / -slope, 1)

    need = moisture < band["dry"] or (was_needed and moisture < band["target"])
    if moisture < band["wilting"]:
        urgency, reason = "high", f"soil {moisture:.0f} % is below wilting level {band['wilting']} %"
    elif moisture < band["dry"]:
        urgency = "high" if hot else "medium"
        reason = f"soil {moisture:.0f} % is below the dry level {band['dry']} %"
        if hot:
            reason += " and it is hot/dry air" + (f" (VPD {vpd:.1f} kPa)" if vpd is not None else "")
    elif need:
        urgency, reason = "low", f"watering in progress: keep going until {band['target']} % (now {moisture:.0f} %)"
    elif hours is not None and hours < 6:
        urgency, reason = "low", f"soil {moisture:.0f} % is OK but drying: dry in ~{hours} h"
    else:
        urgency, reason = "none", f"soil {moisture:.0f} % is fine (dry below {band['dry']} %)"
    return {"need_water": need, "urgency": urgency, "reason": reason, "hours_to_dry": hours,
            "vpd_kpa": vpd, "moisture_pct": round(moisture, 1), "band": band}


def detect_watering(prev, now, min_rise=8.0):
    """Watering = soil moisture jumps up by >= min_rise points between two readings.

    Drying is slow (about -1 point per reading) and sensor noise is small (+-0.5), so a sudden rise can only
    come from water being added (by hand, rain or a pump). Works the same for the simulator and a real probe.
    -> dict(rise, from_pct, to_pct) or None.
    """
    if prev is None or now is None:
        return None
    rise = now - prev
    if rise >= min_rise:
        return {"rise": round(rise, 1), "from_pct": round(prev, 1), "to_pct": round(now, 1)}
    return None
