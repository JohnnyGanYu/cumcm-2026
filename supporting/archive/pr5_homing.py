"""Conservative source regions for homing, using only public problem bounds.

The polygon encloses the 1800 m source disk rather than clipping to an
inscribed approximation.  A clear certificate therefore covers every source
consistent with all bearing observations; it never uses simulator truth.
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
    """Vertices of a regular polygon whose sides are tangent to the disk."""
    circumradius = radius / math.cos(math.pi / sides)
    return tuple((circumradius * math.cos(2 * math.pi * k / sides),
                  circumradius * math.sin(2 * math.pi * k / sides))
                 for k in range(sides))


def source_region(
    bearings: Sequence[Bearing], *, source_radius_m: float = ARENA_RADIUS,
    error_deg: float = ERROR_ABS_MAX_DEG + 0.005, sides: int = 128,
) -> list[Point]:
    """Outer approximation of the source disk intersected with every wedge.

    The 0.005 degree guard covers bearing serialization and roundoff.  Empty
    or inconsistent observations return an empty region, never a certificate.
    Robot observation coordinates are deliberately not limited to this disk.
    """
    if not bearings:
        return []
    if source_radius_m <= 0 or sides < 8 or not 0 < error_deg < 90:
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


def bounded_estimate(bearings: Sequence[Bearing]) -> tuple[Point, float] | None:
    """Return region centroid and diameter after two or more observations.

    Keeping a single bearing unlocalized preserves the strategy's cheap
    co-located second-baseline acquisition.  No least-squares extrapolation
    outside the source prior is returned if the observations are inconsistent.
    """
    if len(bearings) < 2:
        return None
    polygon = source_region(bearings)
    if not polygon:
        return None
    point = centroid(polygon)
    return point, polygon_diameter(polygon)[0]


def certified_clear_point(bearings: Sequence[Bearing], *, radius_m: float = 18.0) -> Point | None:
    """Return a point whose radius-m disk covers the full feasible polygon.

    A disk is convex, so checking all polygon vertices also checks every
    possible source within the polygon.  The default leaves 2 m clearance
    below the official 20 m success radius.  Failure means 'not certified'.
    """
    if len(bearings) < 2 or not 0 < radius_m <= 20.0:
        return None
    polygon = source_region(bearings)
    if not polygon:
        return None
    diameter, a, b = polygon_diameter(polygon)
    if diameter > 2 * radius_m:
        return None
    # The diameter midpoint can beat the centroid for skinny wedges; both
    # are merely candidates and receive the same conservative coverage test.
    candidates = [centroid(polygon), ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)]
    for point in candidates:
        if max(math.dist(point, vertex) for vertex in polygon) <= radius_m:
            return point
    return None
