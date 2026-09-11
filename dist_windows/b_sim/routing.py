"""Observation-only route-order heuristics for omnidirectional sources.

The sample centroid below is an ordering proxy, not a localization estimate
or an emptiness/clearance certificate.  In particular, a centroid can fall
inside a removed silence disk when the remaining sample set is nonconvex.
"""

from __future__ import annotations

import math
from typing import Sequence

from .arena import ARENA_RADIUS, ERROR_ABS_MAX_DEG, RECEPTION_MAX_M, RECEPTION_MIN_M
from .geometry import Bearing

Point = tuple[float, float]


def single_bearing_proxy(bearing: Bearing, silent_points: Sequence[Point]) -> Point | None:
    """Return a Q3-only visit-order proxy using one bearing and its silences.

    Five rays and 60 radial samples give at most 300 candidate positions.
    Radius weights approximate area in polar coordinates.  Public source
    and reception bounds remove implausible samples, as do this channel's
    actual no-signal observations within the guaranteed 1000 m range.

    The +/-1.005 degree fan includes the same serialization guard as the
    homing region.  Sampling remains a heuristic: missing/invalid samples
    return None so the caller can use its ordinary nominal-range proxy.
    Callers must not apply silence-disk exclusion to Q4 directional sources.
    This function neither mutates observations nor classifies a channel.
    """
    if not all(math.isfinite(v) for v in (bearing.x, bearing.y, bearing.svd_deg)):
        return None
    if any(not all(math.isfinite(v) for v in point) for point in silent_points):
        return None
    error = ERROR_ABS_MAX_DEG + 0.005
    source_radius2 = ARENA_RADIUS ** 2
    guaranteed_range2 = RECEPTION_MIN_M ** 2
    xsum = ysum = total = 0.0
    for factor in (-1.0, -0.5, 0.0, 0.5, 1.0):
        angle = math.radians(bearing.svd_deg + factor * error)
        ux, uy = math.cos(angle), math.sin(angle)
        for radius in range(12, int(RECEPTION_MAX_M), 25):
            x = bearing.x + radius * ux
            y = bearing.y + radius * uy
            if x*x + y*y > source_radius2:
                continue
            if any((x-px)**2 + (y-py)**2 < guaranteed_range2
                   for px, py in silent_points):
                continue
            xsum += radius * x
            ysum += radius * y
            total += radius
    if not total:
        return None
    return xsum / total, ysum / total
