"""B 题目标区域仿真核心：案例生成 + 物理规则。

依据：problem/B题/B题.pdf 附录1-2、附件1《模拟器使用说明》、附件2《模拟器通信接口说明》。
这是按文档复刻的本地仿真器，用于离线无限次调优策略；官方模拟器为最终判据，
任何本文件与官方行为不一致处，以官方模拟器实测为准。

文档未明确、由本实现设定的假设（写入论文时须声明）：
1. 示向度误差在 [-1, 1] 度内，且"同一地点的电磁环境干扰固定"。本实现将误差实现为
   仅由检测坐标决定的确定性伪随机场（对坐标取 0.01 m 分辨率哈希），因此同点重复
   检测返回同一误差，不同点误差近似独立均匀。误差不依赖频道。
2. 干扰源位置在半径 1800 m 圆域内均匀采样；有效接收半径在 [1000, 1500] m 均匀采样；
   频道从 1..20 无重复采样；定向方向在 [0, 360) 均匀采样。
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

ARENA_RADIUS = 1800.0
SPEED_MPS = 5.0
MEASURE_SECONDS = 5.0
CHANNEL_SWITCH_SECONDS = 1.0
CLEAR_FAIL_SECONDS = 3.0
CLEAR_SUCCESS_SECONDS = 5.0
NEAR_THRESHOLD_M = 5.0
CLEAR_RADIUS_M = 20.0
MAX_VIRTUAL_SECONDS = 360000.0
MAX_REAL_SECONDS = 1200
CHANNEL_MIN = 1
CHANNEL_MAX = 20
RECEPTION_MIN_M = 1000.0
RECEPTION_MAX_M = 1500.0
SOURCE_COUNT_MIN = 10
SOURCE_COUNT_MAX = 16
DIRECTIONAL_HALF_ANGLE_DEG = 90.0
ERROR_ABS_MAX_DEG = 1.0
ERROR_GRID_M = 0.01


def norm_deg(angle: float) -> float:
    """归一化到 [0, 360)。"""
    value = math.fmod(angle, 360.0)
    if value < 0.0:
        value += 360.0
    return value


def angular_gap_deg(a: float, b: float) -> float:
    """两方位角之间的最小夹角，范围 [0, 180]。"""
    diff = abs(norm_deg(a) - norm_deg(b))
    return diff if diff <= 180.0 else 360.0 - diff


@dataclass
class Jammer:
    """一个干扰源。directional_deg 为 None 表示全向。"""

    channel: int
    x: float
    y: float
    reception_radius_m: float
    directional_deg: float | None = None
    cleared: bool = False

    @property
    def is_directional(self) -> bool:
        return self.directional_deg is not None

    def covers(self, px: float, py: float) -> bool:
        """检测点是否落在有效覆盖角度范围内（定向为定向方向两侧各 90 度，含边界）。"""
        if self.directional_deg is None:
            return True
        if px == self.x and py == self.y:
            return True
        bearing_source_to_point = norm_deg(math.degrees(math.atan2(py - self.y, px - self.x)))
        return angular_gap_deg(bearing_source_to_point, self.directional_deg) <= DIRECTIONAL_HALF_ANGLE_DEG

    def distance_to(self, px: float, py: float) -> float:
        return math.hypot(px - self.x, py - self.y)

    def true_bearing_from(self, px: float, py: float) -> float:
        """检测点指向干扰源的真实方位角（x 轴正向为 0，逆时针为正）。"""
        return norm_deg(math.degrees(math.atan2(self.y - py, self.x - px)))


def bearing_error_deg(px: float, py: float, salt: str) -> float:
    """由检测坐标确定的固定误差，落在 [-1, 1] 度。同一地点重复检测误差不变。"""
    key = f"{salt}|{round(px / ERROR_GRID_M)}|{round(py / ERROR_GRID_M)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)  # [0, 1)
    return (unit * 2.0 - 1.0) * ERROR_ABS_MAX_DEG


@dataclass
class Case:
    """一局案例。"""

    jammers: list[Jammer]
    error_salt: str
    seed: int
    directional_count: int = 0

    @property
    def total(self) -> int:
        return len(self.jammers)

    @property
    def cleared_count(self) -> int:
        return sum(1 for j in self.jammers if j.cleared)

    def by_channel(self, channel: int) -> Jammer | None:
        for jammer in self.jammers:
            if jammer.channel == channel:
                return jammer
        return None

    def truth_table(self) -> list[dict]:
        return [
            {
                "channel": j.channel,
                "x": round(j.x, 3),
                "y": round(j.y, 3),
                "reception_radius_m": round(j.reception_radius_m, 3),
                "directional_deg": None if j.directional_deg is None else round(j.directional_deg, 3),
                "cleared": j.cleared,
            }
            for j in sorted(self.jammers, key=lambda j: j.channel)
        ]


def generate_case(
    seed: int | None = None,
    *,
    count: int | None = None,
    directional_count: int = 0,
    min_pair_distance_m: float = 0.0,
) -> Case:
    """生成一局案例。directional_count=0 对应问题 3，>0 对应问题 4。"""
    rng = random.Random(seed)
    if count is None:
        count = rng.randint(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX)
    if not SOURCE_COUNT_MIN <= count <= SOURCE_COUNT_MAX:
        raise ValueError(f"干扰源总数必须在 {SOURCE_COUNT_MIN}-{SOURCE_COUNT_MAX} 之间，收到 {count}")
    if not 0 <= directional_count <= count:
        raise ValueError("定向干扰源个数必须在 0..count 之间")

    channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), count)
    positions: list[tuple[float, float]] = []
    for _ in range(count):
        for _attempt in range(10000):
            radius = ARENA_RADIUS * math.sqrt(rng.random())
            theta = rng.uniform(0.0, 2.0 * math.pi)
            px, py = radius * math.cos(theta), radius * math.sin(theta)
            if min_pair_distance_m <= 0.0 or all(
                math.hypot(px - qx, py - qy) >= min_pair_distance_m for qx, qy in positions
            ):
                positions.append((px, py))
                break
        else:
            raise RuntimeError("无法在约束下放置干扰源，请降低 min_pair_distance_m")

    directional_flags = [True] * directional_count + [False] * (count - directional_count)
    rng.shuffle(directional_flags)

    jammers = [
        Jammer(
            channel=channel,
            x=px,
            y=py,
            reception_radius_m=rng.uniform(RECEPTION_MIN_M, RECEPTION_MAX_M),
            directional_deg=rng.uniform(0.0, 360.0) if is_dir else None,
        )
        for channel, (px, py), is_dir in zip(channels, positions, directional_flags)
    ]
    return Case(
        jammers=jammers,
        error_salt=f"salt-{seed}",
        seed=-1 if seed is None else seed,
        directional_count=directional_count,
    )


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    channel: int = 1
    virtual_time_us: int = 0
    entered: bool = False
    finished: bool = False
    exit_reason: str | None = None
    measure_calls: int = 0
    clear_calls: int = 0
    clear_success: int = 0
    move_distance_m: float = 0.0
    trace: list[dict] = field(default_factory=list)

    @property
    def virtual_time_s(self) -> float:
        return self.virtual_time_us / 1_000_000.0


class Arena:
    """虚拟目标区域：推进虚拟时钟、判定检测与清除结果。协议/HTTP 层不在此处。"""

    def __init__(self, case: Case, *, max_virtual_seconds: float = MAX_VIRTUAL_SECONDS) -> None:
        self.case = case
        self.robot = RobotState()
        self.max_virtual_us = int(round(max_virtual_seconds * 1_000_000))

    # ---- 动作 ----

    def enter(self) -> dict:
        if self.robot.entered:
            raise ArenaRejected("重复调用 /enter")
        self.robot.entered = True
        return {
            "max_virtual_duration_s": self.max_virtual_us / 1_000_000.0,
            "max_real_duration_s": MAX_REAL_SECONDS,
        }

    def measure(self, x: float, y: float, channel: int) -> dict:
        self._require_active()
        seconds = self._move_seconds(x, y) + MEASURE_SECONDS
        if channel != self.robot.channel:
            seconds += CHANNEL_SWITCH_SECONDS
        self._commit_move(x, y, seconds)
        self.robot.channel = channel
        self.robot.measure_calls += 1

        jammer = self.case.by_channel(channel)
        result: dict
        if jammer is None or jammer.cleared:
            result = {"measure_result": "no_signal"}
        else:
            distance = jammer.distance_to(x, y)
            covered = jammer.covers(x, y)
            if not covered or distance > jammer.reception_radius_m:
                result = {"measure_result": "no_signal"}
            elif distance <= NEAR_THRESHOLD_M:
                result = {"measure_result": "near"}
            else:
                svd = jammer.true_bearing_from(x, y) + bearing_error_deg(x, y, self.case.error_salt)
                result = {"measure_result": "direction", "svd_deg": round(norm_deg(svd), 2)}
        self.robot.trace.append(
            {"action": "measure", "x": x, "y": y, "channel": channel,
             "virtual_time_s": self.robot.virtual_time_s, **result}
        )
        return result

    def clear(self, x: float, y: float, channel: int) -> dict:
        self._require_active()
        jammer = self.case.by_channel(channel)
        hit = jammer is not None and not jammer.cleared and jammer.distance_to(x, y) <= CLEAR_RADIUS_M
        action_seconds = CLEAR_SUCCESS_SECONDS if hit else CLEAR_FAIL_SECONDS
        seconds = self._move_seconds(x, y) + action_seconds
        self._commit_move(x, y, seconds)
        self.robot.clear_calls += 1
        if hit:
            assert jammer is not None
            jammer.cleared = True
            self.robot.clear_success += 1
            result = {"clear_result": "success"}
        else:
            result = {"clear_result": "no_target_in_range"}
        self.robot.trace.append(
            {"action": "clear", "x": x, "y": y, "channel": channel,
             "virtual_time_s": self.robot.virtual_time_s, **result}
        )
        return result

    def exit(self, reason: str = "user_exit") -> dict:
        self._require_active()
        self.robot.finished = True
        self.robot.exit_reason = reason
        return {"exit_reason": reason}

    # ---- 统计 ----

    def stats(self) -> dict:
        cleared = self.case.cleared_count
        total_time = self.robot.virtual_time_s
        return {
            "total_jammers": self.case.total,
            "directional_jammers": self.case.directional_count,
            "cleared": cleared,
            "cleared_ratio": cleared / self.case.total if self.case.total else 0.0,
            "total_virtual_time_s": total_time,
            "mean_clear_time_s": total_time / cleared if cleared else None,
            "measure_calls": self.robot.measure_calls,
            "clear_calls": self.robot.clear_calls,
            "move_distance_m": self.robot.move_distance_m,
            "exit_reason": self.robot.exit_reason,
        }

    # ---- 内部 ----

    def _require_active(self) -> None:
        if not self.robot.entered:
            raise ArenaRejected("尚未成功调用 /enter")
        if self.robot.finished:
            raise ArenaRejected("测试已经结束")

    def _move_seconds(self, x: float, y: float) -> float:
        return math.hypot(x - self.robot.x, y - self.robot.y) / SPEED_MPS

    def _commit_move(self, x: float, y: float, seconds: float) -> None:
        delta_us = int(round(seconds * 1_000_000))
        if self.robot.virtual_time_us + delta_us > self.max_virtual_us:
            self.robot.finished = True
            self.robot.exit_reason = "virtual_time_exhausted"
            raise ArenaRejected("虚拟时间超限")
        self.robot.move_distance_m += math.hypot(x - self.robot.x, y - self.robot.y)
        self.robot.x, self.robot.y = x, y
        self.robot.virtual_time_us += delta_us


class ArenaRejected(Exception):
    """业务上被拒绝的动作（对应 accepted=false）。"""
