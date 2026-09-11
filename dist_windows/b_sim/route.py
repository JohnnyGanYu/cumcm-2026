"""开放路径巡回排序：最近邻构造 + 2-opt / Or-opt 局部改进（纯标准库）。

为什么需要它：评分指标 T̄ = 定位清除总时间 / 已清除个数，而总时间的约 80% 是移动
耗时（`timeline.py` 实测）。机器狗最终必须走进每个干扰源的 20 m 内，因此"访问一批
已定位目标"本质是一个**起点固定、不要求回到起点的旅行商问题（开放 TSP / 哈密顿路径）**。

原实现按"当前位置最近者优先"逐个归航，等价于纯最近邻启发式。最近邻的期望长度约为
最优解的 1.25 倍，且会留下典型的"最后一个远点"长尾。本模块在最近邻之上加两种标准
局部搜索：

* **2-opt**：反转一段子路径，消除自交叉；
* **Or-opt**：把长度 1-3 的连续段移动到别处（含反向插入），修正"顺手漏掉一个点、
  最后再折返"的模式。

两者都只做**改进步**（严格下降），因此必然终止且不会比最近邻更差。规模 n ≤ 16，
一次完整改进在毫秒级，对 20 分钟现实时间预算而言可以在每次重规划时无脑调用。

注意：目标位置是**估计值**且会随逼近过程刷新，所以排序必须能廉价地反复重算，而不是
一次性求解后照本宣科。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(start: Point, order: Sequence[Point]) -> float:
    """开放路径长度：start -> order[0] -> ... -> order[-1]（不回到 start）。"""
    total = 0.0
    cur = start
    for point in order:
        total += distance(cur, point)
        cur = point
    return total


def nearest_neighbour(start: Point, points: Sequence[Point]) -> list[Point]:
    """最近邻构造。并列时按 (x, y) 升序取首个，保证确定性。"""
    remaining = list(points)
    order: list[Point] = []
    cur = start
    while remaining:
        best = min(range(len(remaining)),
                   key=lambda i: (distance(cur, remaining[i]), remaining[i]))
        cur = remaining.pop(best)
        order.append(cur)
    return order


def two_opt(start: Point, order: Sequence[Point], *, max_passes: int = 40) -> list[Point]:
    """开放路径 2-opt：反转 order[i:j+1]，只接受严格改进。

    开放路径没有"回到起点"的边，因此末端子路径反转也可能改进（这与闭合 TSP 不同），
    循环上界取到 n-1。
    """
    best = list(order)
    n = len(best)
    if n < 3:
        return best
    nodes = [start] + best
    for _ in range(max_passes):
        improved = False
        for i in range(1, n):            # nodes[i] 起（含），i>=1 保证 start 不动
            for j in range(i + 1, n + 1):
                # 反转 nodes[i..j]：受影响的边是 (i-1,i) 与 (j,j+1)
                before = distance(nodes[i - 1], nodes[i])
                after = distance(nodes[i - 1], nodes[j])
                if j + 1 <= n:
                    before += distance(nodes[j], nodes[j + 1])
                    after += distance(nodes[i], nodes[j + 1])
                if after < before - 1e-9:
                    nodes[i:j + 1] = reversed(nodes[i:j + 1])
                    improved = True
        if not improved:
            break
    return nodes[1:]


def or_opt(start: Point, order: Sequence[Point], *, max_segment: int = 3,
           max_passes: int = 40) -> list[Point]:
    """Or-opt：把长度 1..max_segment 的连续段搬到别处（含反向），只接受严格改进。"""
    best = list(order)
    n = len(best)
    if n < 3:
        return best
    for _ in range(max_passes):
        base = path_length(start, best)
        improved = False
        for length in range(1, min(max_segment, n - 1) + 1):
            for i in range(0, n - length + 1):
                segment = best[i:i + length]
                rest = best[:i] + best[i + length:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    for piece in (segment, segment[::-1]):
                        candidate = rest[:j] + list(piece) + rest[j:]
                        if path_length(start, candidate) < base - 1e-9:
                            best = candidate
                            base = path_length(start, best)
                            improved = True
            if improved:
                break
        if not improved:
            break
    return best


def order_route(start: Point, points: Sequence[Point], *,
                improve: bool = True, max_points: int = 24) -> list[Point]:
    """从 start 出发访问 points 的开放路径排序：最近邻 + 2-opt + Or-opt。

    `max_points` 是局部搜索的规模闸门：本题目标数 ≤ 16（干扰源上限），补测站位也在
    十余个量级，超过阈值时只做最近邻，避免 O(n³) 改进拖慢现实时间。
    """
    unique: list[Point] = []
    seen: set[Point] = set()
    for point in points:
        key = (round(point[0], 6), round(point[1], 6))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    if len(unique) <= 1:
        return unique
    order = nearest_neighbour(start, unique)
    if not improve or len(unique) > max_points:
        return order
    order = two_opt(start, order)
    order = or_opt(start, order)
    # 兜底：局部搜索只接受改进步，结果不应比最近邻差；万一浮点抖动导致变差就回退。
    fallback = nearest_neighbour(start, unique)
    if path_length(start, order) > path_length(start, fallback) + 1e-6:
        return fallback
    return order


def insertion_cost(start: Point, order: Sequence[Point], point: Point) -> tuple[float, int]:
    """把 point 插入已排定路径的最小增量代价及插入位置（廉价插入启发式）。

    用于把"补测站位"顺路塞进清除巡回，而不是打断巡回专门跑一趟。
    """
    nodes = [start] + list(order)
    best_cost = float("inf")
    best_index = len(order)
    for k in range(1, len(nodes) + 1):
        prev = nodes[k - 1]
        if k < len(nodes):
            nxt = nodes[k]
            cost = distance(prev, point) + distance(point, nxt) - distance(prev, nxt)
        else:
            cost = distance(prev, point)      # 追加到末尾（开放路径无回程边）
        if cost < best_cost - 1e-9:
            best_cost, best_index = cost, k - 1
    return best_cost, best_index


__all__ = ["Point", "distance", "insertion_cost", "nearest_neighbour", "or_opt",
           "order_route", "path_length", "two_opt"]
