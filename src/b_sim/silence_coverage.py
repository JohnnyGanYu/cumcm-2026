"""Conservative continuous certificates for a channel's silent observations.

The source lies in the radius-1800 disk; a receiver may stand outside it.
Only points at which THIS channel was actually measured silent belong here.
The sampled gain is a routing heuristic and is never an empty-channel proof.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
import math
from typing import Iterable

from .arena import ARENA_RADIUS, RECEPTION_MIN_M

Point = tuple[float, float]
_SAFE_RANGE = RECEPTION_MIN_M - 1e-6


def certified_stations(kind: str = "rings25") -> list[Point]:
    """Return stations covering every source position and 180-degree heading.

    Each template triangulates its convex hull using edges shorter than 1000 m.
    The hull contains the entire source disk.  A source in a triangle is within
    1000 m of every vertex, and at least one vertex lies in each closed emitting
    half-plane through it (a convex combination cannot have all negative dots).
    """
    if kind == "rings25":
        return [(0.0, 0.0)] + [
            (r * math.cos(math.radians(30 * k + phase)),
             r * math.sin(math.radians(30 * k + phase)))
            for r, phase in ((970.0, 15.0), (1870.0, 0.0)) for k in range(12)
        ]
    if kind == "mesh25":
        a = 980.0
        points = [(a * (q + r / 2), a * math.sqrt(3) * r / 2)
                  for q in range(-2, 3) for r in range(-2, 3)
                  if max(abs(q), abs(r), abs(q + r)) <= 2]
        return points + [
            (1850 * math.cos(math.radians(30 + 60 * k)),
             1850 * math.sin(math.radians(30 + 60 * k))) for k in range(6)
        ]
    raise ValueError(f"unknown station template: {kind}")


def _cross(a: Point, b: Point, c: Point) -> float:
    left = (b[0] - a[0]) * (c[1] - a[1])
    right = (b[1] - a[1]) * (c[0] - a[0])
    determinant = left - right
    if abs(determinant) > 1e-14 * (abs(left) + abs(right)):
        return determinant
    # Near collinearity, roundoff must not put a point on the wrong side of a
    # certificate edge.  Exact arithmetic is limited to this uncommon case.
    ax, ay, bx, by, cx, cy = (Fraction(v) for v in (*a, *b, *c))
    return float((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


def _hull(points: tuple[Point, ...]) -> list[Point]:
    lower: list[Point] = []
    upper: list[Point] = []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _triangulate(points: tuple[Point, ...]) -> list[tuple[Point, Point, Point]]:
    """Bowyer-Watson triangulation; no SciPy or platform DLL dependency."""
    scale = max(1.0, *(abs(v) for p in points for v in p)) * 32
    vertices = list(points) + [(-scale, -scale), (scale, -scale), (0.0, scale)]
    n = len(points)
    triangles = [(n, n + 1, n + 2)]
    for i, p in enumerate(points):
        bad = []
        for tri in triangles:
            a, b, c = (vertices[j] for j in tri)
            ax, ay = a[0] - p[0], a[1] - p[1]
            bx, by = b[0] - p[0], b[1] - p[1]
            cx, cy = c[0] - p[0], c[1] - p[1]
            determinant = ((ax * ax + ay * ay) * (bx * cy - by * cx)
                           - (bx * bx + by * by) * (ax * cy - ay * cx)
                           + (cx * cx + cy * cy) * (ax * by - ay * bx))
            if determinant > 0:
                bad.append(tri)
        boundary: dict[tuple[int, int], int] = {}
        for a, b, c in bad:
            for edge in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(edge))
                boundary[key] = boundary.get(key, 0) + 1
        bad_set = set(bad)
        triangles = [tri for tri in triangles if tri not in bad_set]
        for (a, b), count in boundary.items():
            if count != 1:
                continue
            orient = _cross(vertices[a], vertices[b], p)
            if orient > 0:
                triangles.append((a, b, i))
            elif orient < 0:
                triangles.append((b, a, i))
    return [tuple(vertices[j] for j in tri) for tri in triangles if max(tri) < n]


def _segment_origin_distance(a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    t = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / denom)) if denom else 0.0
    return math.hypot(a[0] + t * dx, a[1] + t * dy)


def _triangle_touches_disk(tri: tuple[Point, Point, Point]) -> bool:
    a, b, c = tri
    if all(_cross(x, y, (0.0, 0.0)) >= 0 for x, y in ((a, b), (b, c), (c, a))):
        return True
    return min(_segment_origin_distance(x, y) for x, y in ((a, b), (b, c), (c, a))) <= ARENA_RADIUS + 1e-6


def _verified_covering_mesh(triangles: list[tuple[Point, Point, Point]], hull: list[Point]) -> bool:
    """Fail closed on a missing, reversed, overlapping or degenerate triangle.

    Interior directed edges must cancel exactly.  Every uncancelled directed
    edge must have the complete source disk on its left.  Thus no mesh hole or
    cut intersecting the source disk can pass, even if its area is tiny.  The
    area check is an additional audit, not the reason a gap is considered safe.
    """
    edges: dict[tuple[Point, Point], list[tuple[Point, Point]]] = {}
    areas = []
    for a, b, c in triangles:
        area2 = _cross(a, b, c)
        if area2 <= 1e-8:
            return False
        areas.append(area2 / 2)
        for edge in ((a, b), (b, c), (c, a)):
            edges.setdefault(tuple(sorted(edge)), []).append(edge)
    boundary = []
    for occurrences in edges.values():
        if len(occurrences) == 1:
            boundary.append(occurrences[0])
        elif len(occurrences) != 2 or occurrences[0] != tuple(reversed(occurrences[1])):
            return False
    if not boundary or any(_cross(a, b, (0.0, 0.0)) <= (ARENA_RADIUS + 1e-6) * math.dist(a, b)
                           for a, b in boundary):
        return False
    hull_area = math.fsum(a[0] * b[1] - a[1] * b[0]
                          for a, b in zip(hull, hull[1:] + hull[:1])) / 2
    return math.isclose(math.fsum(areas), hull_area, rel_tol=1e-10, abs_tol=1e-4)


def _in_hull(point: Point, hull: list[Point]) -> bool:
    if len(hull) == 1:
        return point == hull[0]
    if len(hull) == 2:
        a, b = hull
        return (_cross(a, b, point) == 0
                and min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))
    return len(hull) >= 3 and all(_cross(a, b, point) >= 0
                                for a, b in zip(hull, hull[1:] + hull[:1]))


def _cover_triangle(tri: tuple[Point, Point, Point], points: tuple[Point, ...], depth: int = 0) -> bool:
    """Certify a triangle using nearby receivers, subdividing only when needed.

    Every retained receiver is within 1000 m of all triangle vertices and hence
    the complete triangle.  If their convex hull contains the triangle, every
    emitting half-plane through each possible source contains such a receiver.
    This also uses measurements that are not vertices of the triangulation.
    """
    if not _triangle_touches_disk(tri):
        return True
    near_all = tuple(p for p in points if all(math.dist(p, v) <= _SAFE_RANGE for v in tri))
    hull = _hull(near_all) if len(near_all) > 1 else list(near_all)
    if all(_in_hull(v, hull) for v in tri):
        return True
    center = (sum(p[0] for p in tri) / 3, sum(p[1] for p in tri) / 3)
    if math.hypot(*center) <= ARENA_RADIUS:
        nearby = tuple(p for p in points if math.dist(p, center) <= _SAFE_RANGE)
        center_hull = _hull(nearby) if len(nearby) > 1 else list(nearby)
        if not _in_hull(center, center_hull):
            return False
    if depth >= 9:
        return False
    a, b, c = tri
    ab = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    bc = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
    ca = ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2)
    return all(_cover_triangle(child, points, depth + 1)
               for child in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))


def _directional_certificate(points: tuple[Point, ...]) -> bool:
    if len(points) < 3:
        return False
    hull = _hull(points)
    if len(hull) < 3:
        return False
    # A strict geometric margin avoids interpreting a numerical boundary error
    # as coverage of the closed source disk.
    if any(_cross(a, b, (0.0, 0.0)) <= (ARENA_RADIUS + 1e-6) * math.dist(a, b)
           for a, b in zip(hull, hull[1:] + hull[:1])):
        return False
    triangles = _triangulate(points)
    if not triangles or not _verified_covering_mesh(triangles, hull):
        return False
    for a, b, c in triangles:
        if _triangle_touches_disk((a, b, c)) and max(math.dist(a, b), math.dist(b, c), math.dist(c, a)) > _SAFE_RANGE:
            if not _cover_triangle((a, b, c), points):
                return False
    return True


def _omni_certificate(points: tuple[Point, ...]) -> bool:
    if not points:
        return False
    # Every accepted square lies in one guaranteed reception disk.  Squares
    # touching the source boundary are retained until certified, never dropped
    # because a sampled source grid happens to miss them.
    todo = [(-ARENA_RADIUS, -ARENA_RADIUS, ARENA_RADIUS, ARENA_RADIUS, 0)]
    range2 = _SAFE_RANGE * _SAFE_RANGE
    while todo:
        x0, y0, x1, y1, depth = todo.pop()
        nx = max(x0, min(0.0, x1))
        ny = max(y0, min(0.0, y1))
        if nx * nx + ny * ny > ARENA_RADIUS * ARENA_RADIUS:
            continue
        if any(max((x0 - px) ** 2, (x1 - px) ** 2)
               + max((y0 - py) ** 2, (y1 - py) ** 2) <= range2 for px, py in points):
            continue
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if xm * xm + ym * ym <= ARENA_RADIUS * ARENA_RADIUS:
            if all((xm - px) ** 2 + (ym - py) ** 2 > range2 for px, py in points):
                return False
        if depth >= 10:
            return False
        todo.extend(((x0, y0, xm, ym, depth + 1), (xm, y0, x1, ym, depth + 1),
                     (x0, ym, xm, y1, depth + 1), (xm, ym, x1, y1, depth + 1)))
    return True


@lru_cache(maxsize=2048)
def _certificate_cached(points: tuple[Point, ...], directional: bool) -> bool:
    if not directional:
        return _omni_certificate(points)
    if _directional_certificate(points):
        return True
    # Adding receivers can expose a nearly flat outer Delaunay triangle that
    # fails the conservative mesh audit.  It cannot invalidate a proof using
    # an already observed subset.  Recheck known witnesses only when EVERY
    # station was actually silent on this channel; never fill in planned stops.
    observed = set(points)
    for kind in ("rings25", "mesh25"):
        witness = tuple(sorted(certified_stations(kind)))
        if len(witness) < len(points) and all(point in observed for point in witness):
            if _certificate_cached(witness, True):
                return True
    return False


def silence_certificate(points: Iterable[Point], directional: bool = False) -> bool:
    """Sufficient proof of absence, using actual silent points with no rounding.

    False means unknown, not that a source necessarily exists.  The cache is
    shared by channels with identical observation histories.
    """
    unique = tuple(sorted(set((float(x), float(y)) for x, y in points)))
    if any(not math.isfinite(v) for p in unique for v in p):
        return False
    return _certificate_cached(unique, directional)


_SAMPLES = tuple((float(x), float(y)) for x in range(-1800, 1801, 300)
                 for y in range(-1800, 1801, 300) if x * x + y * y <= ARENA_RADIUS ** 2)


def _heading_fraction(source: Point, points: Iterable[Point]) -> float:
    angles = sorted(math.atan2(p[1] - source[1], p[0] - source[0])
                    for p in points if math.dist(p, source) <= RECEPTION_MIN_M)
    if not angles:
        return 0.0
    gaps = [b - a for a, b in zip(angles, angles[1:])] + [angles[0] + 2 * math.pi - angles[-1]]
    return 1.0 - sum(max(0.0, gap - math.pi) for gap in gaps) / (2 * math.pi)


class SilenceCoverage:
    """Incremental observation container; sampled gain is only a heuristic."""

    def __init__(self, directional: bool = False):
        self.directional = directional
        self.points: list[Point] = []
        self._fractions: list[float] | None = None
        self._proved = False

    def add(self, point: Point) -> None:
        p = (float(point[0]), float(point[1]))
        if p not in self.points:
            self.points.append(p)
            self._fractions = None

    @property
    def complete(self) -> bool:
        # A later silent observation cannot invalidate an existing certificate.
        # Keeping the proof also avoids triangulating an ever-growing history.
        if not self._proved:
            self._proved = silence_certificate(self.points, self.directional)
        return self._proved

    def _scores(self) -> list[float]:
        if self._fractions is None:
            if self.directional:
                self._fractions = [_heading_fraction(s, self.points) for s in _SAMPLES]
            else:
                self._fractions = [float(any(math.dist(s, p) <= RECEPTION_MIN_M for p in self.points)) for s in _SAMPLES]
        return self._fractions

    @property
    def uncovered_cells(self) -> list[tuple[float, float, float, float]]:
        """Heuristic cells centered on uncovered samples; NOT a proof mesh."""
        return [(x - 150, y - 150, x + 150, y + 150)
                for (x, y), fraction in zip(_SAMPLES, self._scores()) if fraction < 1 - 1e-12]

    def gain(self, point: Point) -> float:
        """Estimated newly excluded source/heading mass, independent of route."""
        scores = self._scores()
        if self.directional:
            augmented = self.points + [point]
            return sum(_heading_fraction(s, augmented) - old for s, old in zip(_SAMPLES, scores)
                       if old < 1 - 1e-12 and math.dist(s, point) <= RECEPTION_MIN_M)
        return sum(1.0 - old for s, old in zip(_SAMPLES, scores)
                   if old < 1 and math.dist(s, point) <= RECEPTION_MIN_M)
