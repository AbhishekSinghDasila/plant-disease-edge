"""Small agronomy helpers shared by services (pure functions, unit-tested)."""
import math


def svp_kpa(temp_c):
    """Saturation vapour pressure (Tetens), kPa."""
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def vpd_kpa(temp_c, rh):
    """Vapour pressure deficit, kPa. ~0.4-1.2 comfortable; >1.5 plants lose water fast; <0.4 very humid."""
    if temp_c is None or rh is None:
        return None
    return round(svp_kpa(temp_c) * (1 - max(0.0, min(100.0, rh)) / 100), 3)
