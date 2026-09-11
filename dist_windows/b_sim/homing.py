"""Bounded homing regions derived only from public bounds and bearings.

The source disk is enclosed by a circumscribed polygon.  Its intersection
with every observed bearing wedge remains an outer bound, suitable for the
existing strategy's cover-radius calculation.  This module does not plan
routes or choose clearing actions.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

from .arena import ARENA_RADIUS, ERROR_ABS_MAX_DEG
from .geometry import Bearing, centroid, clip_polygon, polygon_diameter, wedge_half_planes

Point = tuple[float, float]


@lru_cache(maxsize=8)
def _outer_disk(radius: float, sides: int) -> tuple[Point, ...]:
    """Vertices of a regular polygon with sides tangent to the source disk."""
    circumradius = radius / math.cos(math.pi / sides)
    return tuple((circumradius * math.cos(2 * math.pi * k / sides),
                  circumradius * math.sin(2 * math.pi * k / sides))
                 for k in range(sides))


def source_region(
    bearings: Sequence[Bearing], *, source_radius_m: float = ARENA_RADIUS,
    error_deg: float = ERROR_ABS_MAX_DEG + 0.005, sides: int = 128,
) -> list[Point]:
    """Enclose all source positions consistent with all observed bearings.

    The 0.005 degree guard allows for bearing serialization and roundoff.
    Robot observation coordinates may be outside the source disk.  Missing
    or inconsistent observations yield no region and never certify a clear.
    """
    if not bearings:
        return []
    if not math.isfinite(source_radius_m) or source_radius_m <= 0 or sides < 8 or not 0 < error_deg < 90:
        raise ValueError("invalid source-region bounds")
    polygon = list(_outer_disk(source_radius_m, sides))
    for bearing in bearings:
        if not all(math.isfinite(v) for v in (bearing.x, bearing.y, bearing.svd_deg)):
            return []
        for plane in wedge_half_planes(bearing, error_deg):
            polygon = clip_polygon(polygon, plane)
            if not polygon:
                return []
    return polygon


def bounded_estimate(bearings: Sequence[Bearing]) -> tuple[Point, float, float] | None:
    """Return centroid, diameter, and centroid cover radius for one region.

    A single bearing remains unlocalized so existing second-observation
    acquisition is preserved.  Inconsistent observations do not fall back
    to least squares.  The cover radius measures the farthest region vertex
    from this same centroid, preserving ChannelState.cover_radius semantics.
    """
    if len(bearings) < 2:
        return None
    polygon = source_region(bearings)
    if not polygon:
        return None
    point = centroid(polygon)
    return point, polygon_diameter(polygon)[0], max(math.dist(point, v) for v in polygon)
