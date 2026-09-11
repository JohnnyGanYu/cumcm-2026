"""机器狗搜索—定位—清除策略（问题 3 全向 / 问题 4 含定向）。

核心思路：**逐步逼近式归航**，而不是远距离一次性交会。

为什么不能只靠远距离交会：示向度误差 ±1°，在距离 r 处引起的横向定位误差约
r·tan(1°) ≈ 0.0175 r。r = 1500 m 时约 26 m，已超过 20 m 清除半径；而 r = 100 m
时仅约 1.7 m，远小于清除半径。所以策略不追求"站在远处算准位置"，而是
"边靠近边重测"，让几何误差随距离自然收缩。

三个阶段：
1. 原点普扫：在 (0,0) 遍历 20 个频道。题面保证有效接收半径 >= 1000 m，因此凡是
   距原点 1000 m 以内的干扰源必被发现。成本仅 20 次检测 + 19 次切频。
2. 归航清除：对已有示向度的频道做 dog-leg 交会 + 逐步逼近，估计距离降到阈值内
   即发 /clear；失败则小半径螺旋补点。频道间按"就近优先"排序减少空驶。
3. 环形补扫：对仍然沉默的频道，到覆盖环站位补测（半径 1100 m、8 个站位与原点
   共同保证全域任意点到某站位不超过 1000 m），既覆盖远距离源，也为问题 4 的
   定向源提供方位多样性。

时间成本：移动 = 距离/5 s，切频 1 s，检测 5 s，清除成功 5 s / 未发现 3 s。
优化目标是"定位清除总时间 / 被清除个数"，所以要压缩长距离空驶和无效检测，
而不是单纯压缩指令条数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .arena import (
    ARENA_RADIUS,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_RADIUS_M,
    MAX_VIRTUAL_SECONDS,
    RECEPTION_MIN_M,
    norm_deg,
)
from .client import ClearResult, MeasureResult, ProtocolError, RobotClient, TestEnded
from .directional import DirectionalBelief
from .geometry import Bearing, estimate_position, localization_polygon, polygon_diameter


@dataclass
class ChannelState:
    """一个频道的认知状态。"""

    channel: int
    bearings: list[Bearing] = field(default_factory=list)
    cleared: bool = False
    exhausted: bool = False          # 本轮放弃（可能在补扫后复活）
    silent_points: list[tuple[float, float]] = field(default_factory=list)
    measure_count: int = 0          # 累计检测次数（含搜索阶段）
    homing_measures: int = 0        # 归航阶段检测次数，拿到首个示向度后才计
    clear_attempts: int = 0
    consecutive_silent: int = 0
    revived: bool = False            # 补扫后已给过第二次机会
    ray_swept: bool = False          # 已做过射线盲清除（防重入，_home_channel 是循环）
    verdict_empty: bool = False      # 已判定该频道无干扰源，停止再测
    belief: DirectionalBelief | None = None   # 问题 4：定向源候选集合
    estimate: tuple[float, float] | None = None
    diameter: float = float("inf")

    def add_bearing(self, x: float, y: float, svd_deg: float) -> None:
        first = not self.bearings
        self.bearings.append(Bearing(x, y, svd_deg))
        self.consecutive_silent = 0
        if first:
            # 搜索阶段可能已烧掉大量检测（定向源尤甚）。拿到首个示向度说明已锁定
            # 目标，归航预算必须重新起算，否则会在"已能定位"时被判预算耗尽而放弃。
            self.homing_measures = 0
            self.exhausted = False
        self._refresh()

    def _refresh(self) -> None:
        if len(self.bearings) < 2:
            self.estimate = None
            self.diameter = float("inf")
            return
        recent = self.bearings[-4:]          # 只用最近几条，近距离观测精度更高
        polygon = localization_polygon(recent)
        self.diameter = polygon_diameter(polygon)[0] if polygon else float("inf")
        self.estimate = estimate_position(recent) or estimate_position(self.bearings)

    @property
    def active(self) -> bool:
        return bool(self.bearings) and not self.cleared and not self.exhausted


@dataclass
class StrategyConfig:
    # 阶段 1
    sweep_origin: bool = True
    # 阶段 3 覆盖环。设计依据（数值实测，见 results/b_coverage_design.json）：
    # 干扰源随机分布时约 51% 在原点收不到信号，故补扫是主路径而非兜底，
    # 环参数要按"最坏覆盖间隙 < 接收半径下限 1000 m"直接选，而不是靠惰性触发省成本。
    #
    # 站位数从 1250x12 收缩到 1150x6 的依据（900 局配对实测，
    # 见 results/b_layout_final_d{0,5,10}.json，种子块 11000，三档各 300 局）：
    #   配置        Q3全向              Q4 5定向            Q4 10定向
    #   1250x12     444.8s 3905/3905    1187.5s 3904/3905   1258.2s 3901/3905
    #   1150x6      400.4s 3905/3905    1129.2s 3904/3905   1196.1s 3902/3905
    #   1150x7      407.2s 3905/3905    1137.3s 3904/3905   1203.3s 3902/3905
    #   1200x7      404.4s 3905/3905    1142.8s 3904/3905   1211.8s 3901/3905
    # 即 -9.98% / -4.91% / -4.94%，检测数 188.6 -> 133.8（省 29%），零清除率代价。
    #
    # 为什么 6 个站位够：精确 bisector 几何（见 src/b_layout_geom.py）给出
    # R=1150 m、n=6 时最坏覆盖间隙 988 m < 接收半径下限 1000 m，故全域任意点
    # 到某站位不超过 1000 m，第 7 个及以后的站位是冗余的。原 12 站位（最坏间隙
    # 670 m）是按"更保险"选的，实测证明那份保险金买的是纯空驶。
    #
    # 早先小样本上收缩会掉 1 个源，故一度判为"时间换清除率"的取舍；raygap 兜底
    # （estimate is None 时改走 _ray_blind_clear）合入后该代价消失。d=10 上收缩
    # 反而多清 1 个源（3901->3902，C_homing_failed 由 1 降到 0）：站位少、归航
    # 路径短，不容易在往返中丢信号。
    #
    # 三档残余漏失（seed 11032/11207/11277）全部归因 B1_layout_gap，任何环参数
    # 都救不回，属站位覆盖的固有极限；理论上限 u_hard 仍是全清。
    probe_ring_radius: float = 1150.0
    probe_ring_stations: int = 6
    # 外环半径实测依据（200 案例 × 5 定向源，统计定向源可见率）：
    #   内1250x12 单环          90.00%
    #   +外1560x14              97.30%
    #   +外1700x16              99.10%
    #   +外1750x18              99.90%
    # 1560 m 不够：漏失案例 seed 7039 ch4 的可见点全落在距原点 1600-1800 m 窄带内，
    # 1560 m 外环差 40 m 就是够不到，这也解释了早先"开外环仍救不回它"。
    # 外环是惰性的（内环解决不了才走），故提高半径不会普遍推高耗时。
    outer_ring_radius: float = 1750.0
    outer_ring_stations: int = 14
    use_outer_ring: bool = False          # 全向情形单环足够；定向情形自动启用
    # 问题 4：定向源只在 180 度扇区内可见，方位采样必须更密
    directional_ring_stations: int = 18
    # 归航
    dogleg_offset_deg: float = 32.0      # 第二个观测点相对示向度的偏转角
    dogleg_length_m: float = 380.0       # dog-leg 基线长度
    approach_fraction: float = 0.62      # 每次前进占剩余距离的比例
    lateral_fraction: float = 0.16       # 前进时的横向偏移，保持交会角
    clear_distance_m: float = 55.0       # 估计距离降到该值以内即尝试清除
    spiral_radius_m: float = 15.0
    spiral_points: int = 6
    max_measures_per_channel: int = 40    # 搜索阶段总额度（定向源需要大量沉默观测）
    max_homing_measures: int = 14         # 归航阶段额度，拿到首个示向度后起算
    narrow_sector_extra_measures: int = 22  # 窄可见区定向源的归航加码（沉默观测>=8 时）
    # 判空：累积够多个"不同站位"的沉默观测后，认定该频道无干扰源、停止再测。
    # 蒙特卡洛依据（4000 次抽样，站位=原点+内环1250x12+外环1750x18）：
    #   前 k 个站位全沉默时，真源被误判为空的概率
    #     k= 5: 全向 23.23%  定向 45.90%
    #     k= 8: 全向 11.60%  定向 30.98%
    #     k=12: 全向  0.00%  定向 11.43%
    #     k=31: 全向  0.00%  定向  0.65%
    # 故全向情形 13 个站位（原点+内环）足以零误判判空，可省下 18 个外环站位的检测；
    # 定向源即使跑满 31 站位仍有 0.65% 误判，问题 4 不能提前判空（阈值设为极大）。
    # 实测官方日志：空频道各被检测 47 次，占检测总数 62%-79%，是最大的时间浪费。
    # 实测结论：Q3 只用原点+内环共 13 个站位，阈值 13 等于"跑满才判"，
    # 一分钱不省，反而因误杀真源触发复活绕路（3 局实测 477->545 s，变慢 14%）。
    # 故默认关闭（999）。若将来 Q3 也启用外环（31 站位），可设 13 生效。
    empty_verdict_stations: int = 999       # 全向：不同站位沉默数达此值即判空
    # 就近复活：归航失败的频道不必等整环走完，机器狗路过其估计位置附近时立即重试。
    # 根因实测（`src/b_tail_diag.py`，8 局）：
    #   评分指标 mean_clear_time_s 的 55% 来自最慢 3 个源；
    #   尾部源 = 第一次归航失败的源（头部源 fails=0，尾部源 fails=1）；
    #   典型样本 t=119 s 就拿到示向度，却到 t=9398 s 才清除；
    #   失败后等 p50=6809 s 才重试，而重试仅需 p50=248 s 就成功。
    #   => 浪费的是**等待**，因为 `_revive_exhausted` 只在整环走完后调用。
    # 半径标定（98 局 4 档配对实测，`src/b_nearby_revive.py`）：
    #   600 m: -3.0%   900 m: -4.1%(最优)   1400 m: -3.2%
    #   过小则很少触发，过大则退化为"原地重试同一失败几何"（已证伪：+6.8 s）。
    # 全量验证：Q3 -0.6% / 5定向 -4.1% / 10定向 -5.3% / 16定向 -6.1%，
    #   加权 -48.5 s，清除率与漏失数完全不变（2->2）。
    nearby_revive_radius_m: float = 900.0
    nearby_revive_max_per_channel: int = 2
    # 批量归航：扫过几个站位后才统一归航清除（1 = 逐站归航）。
    # 实测被证伪，保留参数供论文写"试过但无效"：Q4 20局 同种子
    #   batch=1 -> 1261.9s (20/20)    batch=2 -> 1291.8s
    #   batch=3 -> 1334.0s            batch=5 -> 1349.4s
    #   batch=8 -> 1395.8s (19/20，掉一局)
    # 越攒越慢。原因：拿到示向度时机器狗已被 dogleg/step_closer 带到源附近，
    # "发现即归航"本就是就近处理；攒批后回头清除反而绕远路。
    # 官方日志里"环形<->归航切换 13 次"不是浪费，是必要的就近往返。
    home_batch_stations: int = 1
    silent_tolerance: int = 3               # 连续沉默多少次后放弃归航
    narrow_sector_silent_tolerance: int = 10  # 窄可见区定向源的放宽值
    # 射线盲清除（单条示向度、交会不出位置时的最后手段）
    ray_clear_step_m: float = 32.0          # 沿射线铺点间距，须 < 2*清除半径(40m)
    ray_clear_lateral_m: float = 22.0       # 横向偏移，补偿 ±1° 示向度误差
    ray_clear_max_range_m: float = 1600.0   # 最远铺到多少米（接收半径下限 1000m 起）
    ray_clear_max_attempts: int = 120       # 该频道 clear 调用上限
    max_clear_attempts: int = 9
    # 预算
    real_time_reserve_s: float = 15.0
    virtual_time_budget_s: float = MAX_VIRTUAL_SECONDS * 0.9
    directional_probe: bool = False       # 问题 4 打开：沉默频道走完整覆盖环


class DogStrategy:
    def __init__(self, client: RobotClient, config: StrategyConfig | None = None) -> None:
        self.client = client
        self.config = config or StrategyConfig()
        self.channels: dict[int, ChannelState] = {
            ch: ChannelState(ch) for ch in range(CHANNEL_MIN, CHANNEL_MAX + 1)
        }
        self.pos = (0.0, 0.0)
        self.radio_channel = 1
        self.events: list[str] = []
        self._stopped = False
        # 每频道就近复活次数，防止在同一频道上反复烧预算
        self._revive_counts: dict[int, int] = {}

    # ---- 入口 ----

    def run(self) -> dict:
        self.client.enter()
        try:
            if self.config.directional_probe:
                for state in self.channels.values():
                    state.belief = DirectionalBelief.fresh()
            if self.config.sweep_origin:
                self._sweep_at((0.0, 0.0), range(CHANNEL_MIN, CHANNEL_MAX + 1))
            self._home_all()
            self._probe_ring()
            self._revive_exhausted()
            self._home_all()
            self._inner_retry()
            if self.config.directional_probe:
                self._belief_search()
                self._home_all()
                self._blind_clear_sweep()
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"aborted: {exc}")
        finally:
            try:
                self.client.exit()
            except (TestEnded, ProtocolError):
                pass
        return self.summary()

    def summary(self) -> dict:
        cleared = sorted(ch for ch, st in self.channels.items() if st.cleared)
        measures = sum(1 for e in self.client.log if e["path"] == "/measure")
        clears = sum(1 for e in self.client.log if e["path"] == "/clear")
        return {
            "cleared_channels": cleared,
            "cleared_count": len(cleared),
            "virtual_time_s": self.client.virtual_time_s,
            "measure_calls": measures,
            "clear_calls": clears,
            "mean_clear_time_s": self.client.virtual_time_s / len(cleared) if cleared else None,
            "events": self.events,
        }

    # ---- 阶段 1 / 3：扫描 ----

    def _sweep_at(self, point: tuple[float, float], channels) -> None:
        """在一个站位上按频道号顺序扫描，减少切频次数（切频与频道距离无关，仍保持有序）。"""
        for channel in channels:
            if self._stop(): return
            state = self.channels[channel]
            if state.cleared or state.bearings or state.verdict_empty:
                continue
            self._probe_channel_at(point, state)

    def _probe_target(self, state: ChannelState) -> bool:
        """补扫对象：未清除、未判空，且尚无示向度或归航已失败（换站位可改善几何）。"""
        if state.cleared or state.verdict_empty:
            return False
        return not state.bearings or state.exhausted

    def _needs_probe(self) -> bool:
        return any(self._probe_target(st) for st in self.channels.values())

    def _probe_channel_at(self, point: tuple[float, float], state: ChannelState) -> None:
        result = self._measure(point, state.channel)
        if result is None:
            return
        if result.result == "near":
            self._clear_at(point, state.channel)
        elif result.svd_deg is not None:
            state.add_bearing(point[0], point[1], result.svd_deg)
            if state.belief is not None:
                state.belief.apply_bearing(point[0], point[1], result.svd_deg)
        else:
            state.silent_points.append(point)
            state.consecutive_silent += 1
            if state.belief is not None:
                state.belief.apply_silence(point[0], point[1])
            self._maybe_verdict_empty(state)

    def _maybe_verdict_empty(self, state: ChannelState) -> None:
        """够多个"相距足够远的站位"都沉默 -> 判定该频道无干扰源，不再检测。

        只在全向情形启用（directional_probe 关闭）。依据见 StrategyConfig
        中 empty_verdict_stations 的蒙特卡洛表：全向源在 13 个站位（原点+内环）
        全沉默的误判率为 0.00%，而定向源即使 31 站位全沉默仍有 0.65%，
        故问题 4 不判空。
        """
        if state.verdict_empty or state.bearings or state.cleared:
            return
        if self.config.directional_probe:
            return
        # 按"互相距离 >= 接收半径下限一半"去重，避免同一片区域的多次沉默被重复计数
        distinct: list[tuple[float, float]] = []
        for p in state.silent_points:
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= RECEPTION_MIN_M * 0.5
                   for q in distinct):
                distinct.append(p)
        if len(distinct) >= self.config.empty_verdict_stations:
            state.verdict_empty = True
            self.events.append(
                f"ch{state.channel}: verdict empty after {len(distinct)} distinct silent stations")

    def _probe_stations(self) -> list[list[tuple[float, float]]]:
        """双环覆盖站位。内环先扫（离原点近、代价低），外环兜住边界源。

        问题 4 打开 directional_probe 时外环加密到 directional_ring_stations 个方位，
        因为定向源仅在 180 度扇区内可见，方位采样必须比全向情形更密。
        """
        cfg = self.config
        layers = [(cfg.probe_ring_radius, cfg.probe_ring_stations, 0.0)]
        if cfg.directional_probe or cfg.use_outer_ring:
            outer_n = cfg.directional_ring_stations if cfg.directional_probe else cfg.outer_ring_stations
            layers.append((cfg.outer_ring_radius, outer_n, math.pi / max(1, outer_n)))
        rings: list[list[tuple[float, float]]] = []
        for radius, count, phase in layers:
            rings.append([
                (radius * math.cos(2.0 * math.pi * k / count + phase),
                 radius * math.sin(2.0 * math.pi * k / count + phase))
                for k in range(count)
            ])
        return rings

    def _probe_ring(self) -> None:
        """覆盖环补扫：处理远距离源，并为定向源提供方位多样性。"""
        cfg = self.config
        if not self._needs_probe():
            return
        stations = self._probe_stations()
        # 分层惰性：先走内环，内环解决不了才启用外环。外环代价高（半径 1560 m、
        # 方位密），无条件全扫会把平均清除时间抬高约 50%，实测 419 s -> 638 s。
        for ring in stations:
            if not self._needs_probe() or self._stop():
                return
            self._walk_ring(ring)

    def _order_ring(self, ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """把一层环的站位按方位角单调排序，并从离当前位置最近的那个开始。

        为什么不用最近邻贪心：环上站位近似等距分布在圆周上，顺扫相邻两站
        距离约 2*pi*R/n（R=1250 m, n=12 时约 647 m）。而贪心每次取全局最近，
        一旦因归航离开环再回来，"最近"的未访问站常在环对面，产生一次
        约 2R 的跨场弦跳。按方位角单调走可消除这类弦跳。

        方向选择：比较顺时针与逆时针两种顺扫的首段接入代价，取较小者。
        """
        if len(ring) <= 2:
            return ring[:]
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        by_angle = sorted(ring, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        # 找离当前位置最近的站位作为切入点
        start = min(range(len(by_angle)),
                    key=lambda i: math.hypot(by_angle[i][0] - self.pos[0],
                                             by_angle[i][1] - self.pos[1]))
        ccw = by_angle[start:] + by_angle[:start]
        cw = [ccw[0]] + list(reversed(ccw[1:]))

        def tour_len(seq: list[tuple[float, float]]) -> float:
            total = math.hypot(seq[0][0] - self.pos[0], seq[0][1] - self.pos[1])
            for a, b in zip(seq, seq[1:]):
                total += math.hypot(b[0] - a[0], b[1] - a[1])
            return total

        return ccw if tour_len(ccw) <= tour_len(cw) else cw

    def _walk_ring(self, ring: list[tuple[float, float]]) -> None:
        """就近串行访问环上站位，每扫过 batch 个站位再统一归航清除。

        为什么批量：官方日志实测(seed 案例 230354)总移动 66,920 m 中，
        归航/清除段占 45,629 m = 9,126 s，且环形<->归航切换 13 次，
        每次切换都是一趟长途往返。同样 155 个站位按最近邻只需 30,929 m，
        即"发现一个就立刻跑去清除"浪费了 54% 的移动距离。
        批量归航把多个目标凑一起、按就近顺序处理，减少往返。
        batch=1 等价于原来的逐站归航。
        """
        remaining = self._order_ring(ring)
        since_home = 0
        while remaining and not self._stop():
            todo = [st for st in self.channels.values() if self._probe_target(st)]
            if not todo:
                break
            # 站位顺序已在 _order_ring 里按方位角单调排好，直接顺序取用。
            # 不再用"最近邻"重排：环形几何下贪心最近邻会在归航返回后
            # 跳到环对面（跨场弦跳 2.5-2.9 km，顺扫每步仅约 647 m）。
            # 官方日志实测该缺陷均值浪费 10,846 m = 2,169 s/局；
            # run2 恰好单调仅损失 3.2%，run1/3/4 乱序损失 54-58%。
            station = remaining.pop(0)
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop(): return
                self._probe_channel_at(station, state)
            since_home += 1
            # 有目标可清且已攒够一批，或本环已走完 -> 统一归航
            ready = any(st.active for st in self.channels.values())
            if ready and (since_home >= self.config.home_batch_stations or not remaining):
                self._home_all()
                since_home = 0
            # 就近复活：路过时立即重试归航失败的频道，不等整环走完。见 StrategyConfig
            # 的 nearby_revive_radius_m 注释（根因与 98 局标定数据）。
            if self._stop():
                return
            self._revive_nearby()
        if not self._stop():
            self._home_all()

    def _revive_nearby(self) -> None:
        """机器狗当前位置附近若有\"归航失败但已定位\"的频道，立刻重试一个。

        与 `_revive_exhausted` 的区别是**触发条件**：后者按时间（整环走完），
        前者按几何（走到附近）。实测尾部源的第二次归航只需 248 s 就成功，
        却要等 6809 s 才被调度到——这个等待是评分指标的主要损失来源。

        只取最近的一个：复活后 `_home_channel` 会带着机器狗离开当前站位，
        批量复活会退化成\"专程往返\"（已证伪：eager 变体 +74 s / 7-of-8 更差）。
        """
        cfg = self.config
        best: tuple[float, ChannelState] | None = None
        for state in self.channels.values():
            if state.cleared or state.verdict_empty or not state.exhausted:
                continue
            if not state.bearings or state.estimate is None:
                continue
            if self._revive_counts.get(state.channel, 0) >= cfg.nearby_revive_max_per_channel:
                continue
            distance = math.hypot(state.estimate[0] - self.pos[0],
                                  state.estimate[1] - self.pos[1])
            if distance <= cfg.nearby_revive_radius_m and (best is None or distance < best[0]):
                best = (distance, state)
        if best is None:
            return
        state = best[1]
        self._revive_counts[state.channel] = self._revive_counts.get(state.channel, 0) + 1
        # 归航预算重新起算：失败可能只是当时几何差，新站位给了更好的交会角
        state.exhausted = False
        state.consecutive_silent = 0
        state.homing_measures = 0
        state.clear_attempts = 0
        state.ray_swept = False
        if not self._home_channel(state):
            state.exhausted = True

    # ---- 阶段 2：归航 ----

    def _revive_exhausted(self) -> None:
        """补扫后给放弃过的频道第二次机会（新观测可能改善交会几何）。"""
        for state in self.channels.values():
            if state.exhausted and not state.cleared and not state.revived:
                state.exhausted = False
                state.revived = True
                state.consecutive_silent = 0
                state.ray_swept = False   # 新观测可能给出更好的射线，允许再扫一次

    def _inner_retry(self) -> None:
        """内圈重测：救"原点可见、但环形站位全部超出其接收半径"的近距离源。

        实测漏失案例 seed 9002 ch18：真值 (-213,38)、距原点仅 217 m、接收半径
        1027 m —— 原点能拿到示向度，但半径 1250 m 的 12 个环形站位到它最近
        1037 m，全部超出 9.9 m。归航失败后 `_probe_ring` 只往外走，旧示向度
        重试同样的几何必然再失败。故在此对仍未清除的频道回内圈重新取观测。
        """
        # 只救"已经拿到过示向度、却始终清不掉"的频道。
        # 完全沉默的频道属于覆盖问题（交给环形补扫/信念搜索），内圈重测帮不上；
        # 无条件对所有频道跑内圈会让平均耗时上涨约 12%（实测 477->533 s）。
        targets = [st for st in self.channels.values()
                   if not st.cleared and st.bearings]
        if not targets:
            return
        # 原点 + 小半径三角，给近距离源提供新的交会基线
        inner = [(0.0, 0.0), (300.0, 0.0), (-150.0, 260.0), (-150.0, -260.0)]
        for station in inner:
            if self._stop():
                return
            todo = [st for st in self.channels.values()
                    if not st.cleared and st.bearings]
            if not todo:
                return
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop():
                    return
                # 清掉陈旧几何，强制以新观测重新定位
                state.exhausted = False
                self._probe_channel_at(station, state)
            self._home_all()

    def _belief_search(self, max_rounds: int = 60) -> None:
        """问题 4 收尾：对仍沉默的频道，用候选集合推理挑信息增益最大的点补测。

        盲目加密站位有天花板（实测 37 站位仅 98.7% 定向源可见率，成本翻倍），
        而"检测无信号"本身是约束：距离 <= 1000 m 时无信号即意味着观测点落在
        辐射扇区之外。把 (位置, 朝向) 离散成候选集合并逐次排除矛盾状态，
        再挑"若再无信号可排除最多候选"的点，用推理代替蛮力。
        """
        for _ in range(max_rounds):
            if self._stop():
                return
            todo = [
                st for st in self.channels.values()
                if not st.cleared and not st.bearings and st.belief is not None and st.belief.alive
            ]
            if not todo:
                return
            # 候选空间最小的频道最接近定位成功，优先处理
            todo.sort(key=lambda st: len(st.belief.candidates))  # type: ignore[union-attr]
            progressed = False
            for state in todo:
                if self._stop():
                    return
                assert state.belief is not None
                probe = state.belief.best_probe_point(self.pos)
                if probe is None:
                    continue
                before = len(state.bearings)
                self._probe_channel_at(probe, state)
                if state.cleared or len(state.bearings) > before:
                    progressed = True
                    break
                progressed = True
            if not progressed:
                return

    def _blind_clear_sweep(self, max_targets: int = 24) -> None:
        """收尾：对仍未清除的频道，直接到候选位置发 /clear，不要求先看见目标。

        依据（实测）：漏失的定向源全部"物理可解"——其可见扇区一直延伸到源身边
        （最近 4-10 m），但该扇区仅占目标区域 0.1%-19%，靠站位撞上的概率很低。
        而 /clear 的判定只看距离（20 m 内即成功），与辐射方向无关。因此当 belief
        已把候选压缩到少数格点时，逐个盲清除比继续找信号便宜得多。

        每个候选格点按 20 m 清除半径在其 120 m 网格内布点，保证覆盖该格。
        """
        for state in sorted(self.channels.values(), key=lambda s: s.channel):
            if self._stop():
                return
            if state.cleared or state.belief is None or not state.belief.alive:
                continue
            positions = state.belief.candidate_positions()
            if not positions:
                continue
            if len(positions) > max_targets:
                # 候选仍很散：只清除最可能的若干格点（按候选状态数加权），
                # 其余留给剩余时间预算，避免把时间全耗在低概率格点上。
                weight: dict[tuple[float, float], int] = {}
                for pi, _, _ in state.belief.candidates:
                    q = state.belief.positions[pi]
                    weight[q] = weight.get(q, 0) + 1
                positions = [q for q, _ in sorted(weight.items(), key=lambda kv: -kv[1])[:max_targets]]
            positions.sort(key=lambda q: math.hypot(q[0] - self.pos[0], q[1] - self.pos[1]))
            for cx, cy in positions:
                if self._stop() or state.cleared:
                    break
                for point in self._grid_cover_points(cx, cy):
                    if self._stop() or state.cleared:
                        break
                    if math.hypot(*point) > ARENA_RADIUS:
                        continue
                    self._clear_at(point, state.channel)

    @staticmethod
    def _grid_cover_points(cx: float, cy: float,
                           cell: float = 120.0, radius: float = CLEAR_RADIUS_M) -> list[tuple[float, float]]:
        """在以 (cx,cy) 为中心、边长 cell 的网格内布点，使格内任意点距某点 <= radius。"""
        half = cell / 2.0
        step = radius * 1.35
        points: list[tuple[float, float]] = []
        k = int(math.ceil(half / step))
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                points.append((cx + i * step, cy + j * step))
        points.sort(key=lambda q: math.hypot(q[0] - cx, q[1] - cy))
        return points

    def _home_all(self) -> None:
        while not self._stop():
            targets = [st for st in self.channels.values() if st.active]
            if not targets:
                return
            targets.sort(key=lambda st: self._range_to(st))
            if not self._home_channel(targets[0]):
                targets[0].exhausted = True

    def _home_channel(self, state: ChannelState) -> bool:
        """把一个频道推进到清除完成。返回 False 表示放弃。"""
        while not self._stop():
            if state.cleared:      # 循环内某步已清除成功，立刻收工
                return True
            budget = self.config.max_homing_measures + (6 if state.revived else 0)
            # 可见区极小的定向源需要更多归航试探：实测 seed 7001 ch11 / 7015 ch20 /
            # 7039 ch4 的可见区仅占目标区域 2.3%-5.4% 且朝区域外辐射，一旦走出扇区
            # 立即丢信号，14 次归航额度不够（该局反复 5 次 budget exhausted）。
            # 沉默观测多说明可见区窄，据此加码；虚拟/现实时间余量极大（实测各用
            # 上限的 5.1% 与 0.17%），不加码纯属浪费预算。
            if len(state.silent_points) >= 8:
                budget += self.config.narrow_sector_extra_measures
            if state.homing_measures >= budget:
                self.events.append(f"ch{state.channel}: measure budget exhausted")
                return self._final_attempt(state)
            if state.clear_attempts >= self.config.max_clear_attempts:
                self.events.append(f"ch{state.channel}: clear budget exhausted")
                # 打桩实测（seed 7001 ch11）：交会估计落在 (815,1948)，距原点 2112 m
                # 已在 1800 m 区域外，而真值在 (1050.7,1415.8)。围绕错误估计打满 9 次
                # clear 后，这条门禁在"丢信号"判断之前就 return，使射线盲扫永不触发
                # （打桩计数为 0）。估计不可信时改走射线盲扫：只信示向度方向。
                return self._ray_blind_clear(state)
            # 丢信号容忍度：窄可见区的定向源在归航路上频繁出扇区，3 次即弃太苛刻。
            # 实测 seed 7001 ch11 / 7015 ch20 可见区仅占目标区域 3.9%/5.4% 且朝区域外
            # 辐射，均因连续沉默 3 次被判 "lost signal" 而漏失（而非预算耗尽）。
            silent_tolerance = self.config.silent_tolerance
            if state.bearings and len(state.silent_points) >= 6:
                silent_tolerance = self.config.narrow_sector_silent_tolerance
            if state.consecutive_silent >= silent_tolerance:
                self.events.append(f"ch{state.channel}: lost signal, defer to ring probe")
                # 已有示向度却反复丢信号 = 窄扇区源，环形补扫也救不回来（实测开外环
                # 反而 36/40 更差）。改为立即用射线盲清除兜底：方向已知、距离未知时
                # /clear 只看距离，沿射线铺点即可命中。
                if state.bearings:
                    return self._ray_blind_clear(state)
                return False

            if state.estimate is None:
                if not self._dogleg(state):
                    # 只有 1 条示向度时 estimate 永远是 None（_refresh 要求 >=2 条），
                    # dogleg 失败后旧实现直接放弃，绕过了射线盲扫 —— 这是漏失的主因：
                    # 真值归因显示 14/17 个漏失源都走这条路，且它们可见区非空、
                    # /clear 只看距离，理论上全部可清除。方向已知就该沿射线铺点。
                    if state.bearings:
                        return self._ray_blind_clear(state)
                    return False
                continue

            distance = math.hypot(state.estimate[0] - self.pos[0], state.estimate[1] - self.pos[1])
            if distance <= self.config.clear_distance_m:
                if self._clear_at(state.estimate, state.channel):
                    return True
                if not self._spiral(state):
                    return False
                return state.cleared
            if not self._step_closer(state, distance):
                return False
        return False

    def _dogleg(self, state: ChannelState) -> bool:
        """只有一条示向度时，沿示向度方向前进并带小角度偏移，构造交会基线。

        关键：必须"沿着"示向度前进而不是横向平移。横向平移可能走出有效接收半径
        （实测该错误导致远距离源大量丢失）；沿示向度前进则单调靠近目标，信号只会
        更强，同时用小角度偏移仍能获得足够的交会角。角度从大到小逐级退让，
        最后退化为纯示向度方向直行。
        """
        last = state.bearings[-1]
        cfg = self.config
        for offset in (cfg.dogleg_offset_deg, cfg.dogleg_offset_deg * 0.5, 0.0):
            for sign in (1.0, -1.0) if offset else (1.0,):
                angle = math.radians(last.svd_deg + sign * offset)
                candidate = (self.pos[0] + cfg.dogleg_length_m * math.cos(angle),
                             self.pos[1] + cfg.dogleg_length_m * math.sin(angle))
                if math.hypot(*candidate) > ARENA_RADIUS:
                    continue
                before = len(state.bearings)
                self._probe_channel_at(candidate, state)
                if state.cleared:
                    return True
                if len(state.bearings) > before:
                    return True
                if self._stop() or state.homing_measures >= cfg.max_homing_measures + (
                        6 if state.revived else 0):
                    return False
        return False

    def _step_closer(self, state: ChannelState, distance: float) -> bool:
        """朝估计位置前进一段并带横向偏移，在更近处重测以收缩误差。"""
        assert state.estimate is not None
        cfg = self.config
        ex, ey = state.estimate
        dx, dy = ex - self.pos[0], ey - self.pos[1]
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        advance = max(cfg.clear_distance_m * 0.8, distance * cfg.approach_fraction)
        advance = min(advance, distance - cfg.clear_distance_m * 0.35) if distance > cfg.clear_distance_m else advance
        # R3.3：横向偏移按 lateral_fraction × 前进距离(advance) 叠加（而非 × 剩余距离），
        # 与设计 §3.3「lateral_fraction·前进距离」一致；上限 220 m 防止在远距离退化成大幅横向平移。
        lateral = min(advance * cfg.lateral_fraction, 220.0)
        for sign in (1.0, -1.0):
            candidate = (self.pos[0] + ux * advance - uy * lateral * sign,
                         self.pos[1] + uy * advance + ux * lateral * sign)
            if math.hypot(*candidate) > ARENA_RADIUS * 1.02:
                continue
            before = len(state.bearings)
            self._probe_channel_at(candidate, state)
            if state.cleared:
                return True
            if len(state.bearings) > before:
                return True
            # no_signal：可能已进入 5 m 盲区外的遮挡或估计偏差，直接试一次清除
            if state.consecutive_silent >= 2 and state.estimate is not None:
                return self._clear_at(state.estimate, state.channel) or self._spiral(state)
            return True
        return False

    def _spiral(self, state: ChannelState) -> bool:
        """清除失败后围绕估计位置补点。"""
        if state.estimate is None:
            return False
        cx, cy = state.estimate
        cfg = self.config
        for k in range(cfg.spiral_points):
            # 已清除必须立刻停：同一干扰源只能清除一次，继续打全是浪费。
            # 实测(官方日志 seed 7001)45 次清除调用中 35 次发生在成功之后，
            # 其中 30 次出自本函数——每个多余调用白烧 3 s。
            if state.cleared or self._stop() or state.clear_attempts >= cfg.max_clear_attempts:
                return state.cleared
            theta = 2.0 * math.pi * k / cfg.spiral_points
            point = (cx + cfg.spiral_radius_m * math.cos(theta),
                     cy + cfg.spiral_radius_m * math.sin(theta))
            if self._clear_at(point, state.channel):
                return True
        return False

    def _final_attempt(self, state: ChannelState) -> bool:
        if state.estimate is not None:
            if self._clear_at(state.estimate, state.channel):
                return True
            if self._spiral(state):
                return True
        return self._ray_blind_clear(state)

    def _ray_blind_clear(self, state: ChannelState) -> bool:
        """沿单条示向度射线铺点盲清除：方向已知、距离未知时的最后手段。

        适用场景（实测 seed 7001 ch11 / 7015 ch20 / 7039 ch4）：源贴近区域边界且
        朝区域外辐射，可见区退化成一条极细走廊（7001 ch11 仅方位 300° 能看到
        1140 m，其余方位 20-200 m）。策略沿示向度从原点侧逼近走的是错误走廊，
        一路沉默，交会不出 estimate，`_final_attempt` 旧实现直接放弃。

        依据：`/clear` 判定只看平面距离 <= 20 m，与辐射朝向、测向机频道无关。
        因此只需在射线上按 <= 2*20 m 的间距铺清除点即可覆盖整条射线；方向由示向度
        给出（±1° 误差在 1800 m 处横向偏差约 31 m，故同时在射线两侧做横向偏移）。
        虚拟时间余量极大（Q4 实测仅用上限的 5.1%），这点开销可以接受。
        """
        if not state.bearings or state.ray_swept:
            return False
        state.ray_swept = True
        # 射线盲扫用独立计数：_home_channel 的 max_clear_attempts(=9) 是给"围绕估计
        # 位置补点"设的，会在铺点第 9 次就把盲扫掐死（实测事件从 lost signal 变成
        # clear budget exhausted 却仍漏失）。这里改为只受 ray_clear_max_attempts 约束。
        ray_start_attempts = state.clear_attempts
        last = state.bearings[-1]
        ox, oy, bearing = last.x, last.y, last.svd_deg
        step = self.config.ray_clear_step_m
        lateral = self.config.ray_clear_lateral_m
        rad = math.radians(bearing)
        ux, uy = math.cos(rad), math.sin(rad)
        # 法向单位向量，用于补偿 ±1° 示向度误差带来的横向偏差
        nx, ny = -uy, ux
        r = step
        while r <= self.config.ray_clear_max_range_m:
            if self._stop() or state.cleared:
                return state.cleared
            if state.clear_attempts - ray_start_attempts >= self.config.ray_clear_max_attempts:
                self.events.append(f"ch{state.channel}: ray blind clear budget exhausted")
                return state.cleared
            # 横向偏差随距离线性增长，近处只打中心线，远处补两侧
            offsets = [0.0]
            if r * math.radians(1.0) > lateral:
                offsets = [0.0, lateral, -lateral]
            for off in offsets:
                px = ox + ux * r + nx * off
                py = oy + uy * r + ny * off
                if math.hypot(px, py) > ARENA_RADIUS + CLEAR_RADIUS_M:
                    continue
                if self._clear_at((px, py), state.channel):
                    self.events.append(
                        f"ch{state.channel}: cleared by ray blind sweep at r={r:.0f}m")
                    return True
                if self._stop() or (state.clear_attempts - ray_start_attempts
                                    >= self.config.ray_clear_max_attempts):
                    return state.cleared
            r += step
        return state.cleared

    # ---- 底层动作 ----

    def _measure(self, point: tuple[float, float], channel: int) -> MeasureResult | None:
        try:
            result = self.client.measure(point[0], point[1], channel)
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"measure stopped: {exc}")
            self._stopped = True
            return None
        self.pos = point
        self.radio_channel = channel
        state = self.channels[channel]
        state.measure_count += 1
        if state.bearings:
            state.homing_measures += 1
        return result

    def _clear_at(self, point: tuple[float, float], channel: int) -> bool:
        state = self.channels[channel]
        # 全局兜底：同一干扰源只能清除一次，已清除后任何调用都只会返回
        # no_target_in_range 并白烧 3 s 虚拟时间。
        if state.cleared:
            return True
        state.clear_attempts += 1
        try:
            result: ClearResult = self.client.clear(point[0], point[1], channel)
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"clear stopped: {exc}")
            self._stopped = True
            return False
        self.pos = point
        if result.ok:
            state.cleared = True
        return result.ok

    def _range_to(self, state: ChannelState) -> float:
        """到目标的排序距离。**已知缺陷，见下**。

        缺陷（`src/b_home_order.py --diagnose` 实测，Q3 30 局）：`estimate` 需要 >= 2 条
        示向度（`_refresh` 的要求）才非 None。而 `_walk_ring` 是"在同一站位把所有待测
        频道扫一遍"，所以一批目标的**首条**示向度全部记在**同一个观测点**上，而那个
        观测点恰是机器狗当前位置 —— 于是下面的回退分支对整批目标全返回 0.0。
        实测 63/63 个多目标归航批次排序键完全并列、批内持有 estimate 的目标数为 0，
        `_home_all` 里的 `sorted` 是稳定排序，"按距离取最近"因此退化为**按频道号取**，
        而频道号与空间位置无关。批内真值巡回长度：频道号序 4842 m、示向度方位角序
        3543 m(-26.8%)、真值最优 2180 m。
        更深一层的原因（也是更值得修的）：`estimate` 恒为 None 是因为**原点普扫只在一个
        点取样**，所以普扫结束时每个频道都得先走一次 `dogleg_length_m=380 m` 去挣第二条
        基线。这是 `homing_origin` 占 Q3 总移动 38.6% 的来源。

        两种修法的配对检验（真实闭环，同种子逐局配对，详见
        `results/b_deadhead_verdict.md`）：
          1. 只修排序（`src/b_raysort.py`，并列时按示向度射线代理点取最近）：
             Q3 24 局移动 -5.6%(t=-2.44, p=0.023)。有效但有限。
          2. 共址补测（`src/b_colocate.py`，在已付过移动费的停靠点原地给其余频道补测，
             移动成本 0）：Q3 40 局移动 -20.9%(t=-10.30, p=1.1e-12, 40/0/0 全胜)、
             Q4-5定向 -8.0%(p=7.4e-10)、Q4-全定向 -2.8%(p=0.0080)，清除率三档持平。
        两者均未合入本文件（生产路径未改），合入需队员决定。
        """
        if state.estimate is None:
            if state.bearings:
                last = state.bearings[-1]
                return math.hypot(last.x - self.pos[0], last.y - self.pos[1])
            return 1e9
        return math.hypot(state.estimate[0] - self.pos[0], state.estimate[1] - self.pos[1])

    def _stop(self) -> bool:
        if self._stopped:
            return True
        if self.client.real_budget_left_s() <= self.config.real_time_reserve_s:
            self.events.append("real time budget reached")
            self._stopped = True
            return True
        if self.client.virtual_time_s >= self.config.virtual_time_budget_s:
            self.events.append("virtual time budget reached")
            self._stopped = True
            return True
        return False


def safe_clear_threshold() -> float:
    """定位区域直径小于该值时，走到质心即可保证落在清除半径内。"""
    return CLEAR_RADIUS_M


def lateral_error_at(range_m: float, error_deg: float = 1.0) -> float:
    """示向度误差在给定距离上引起的横向定位误差，用于论证逼近的必要性。"""
    return range_m * math.tan(math.radians(error_deg))


__all__ = [
    "ChannelState",
    "DogStrategy",
    "StrategyConfig",
    "lateral_error_at",
    "safe_clear_threshold",
    "ARENA_RADIUS",
    "RECEPTION_MIN_M",
    "norm_deg",
]
