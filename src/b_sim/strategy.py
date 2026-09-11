"""机器狗搜索—定位—清除策略（问题 3 全向 / 问题 4 含定向）。

## 优化目标决定了架构

评分指标是 T̄ = 定位清除总时间 / 被清除个数，而总时间由四类成本相加（附件 1 表 1）：

    移动 = 距离 / 5 m·s⁻¹      检测 = 5 s/次      切频 = 1 s/次      清除 = 成功 5 s / 失败 3 s

实测分解（`timeline.py`）显示**移动占 ~80%**，因此策略的主要战场是路径规划，而不是
压缩指令条数。两条由此推出的量纲关系贯穿全文：

* **1 次检测 ≈ 30 m 位移**（6 s ÷ 5 m·s⁻¹）。检测极其便宜，"顺手多测一个频道"几乎
  免费，而"专门跑一趟去测"很贵。
* 机器狗最终必须走进每个源的 20 m 内，所以"访问已定位目标"是一个**起点固定、无需
  回程的开放 TSP**，其最近邻巡回长度是移动距离的天然基准（`timeline.detour_ratio`）。

## 四条机制

1. **逐步逼近式归航**（原有）。示向度误差 ±1° 在距离 r 处引起横向定位误差
   r·tan(1°) ≈ 0.0175 r：r = 1500 m 时约 26 m，已超过 20 m 清除半径；r = 100 m 时仅
   1.7 m。所以不追求"站在远处算准位置"，而是边靠近边重测，让几何误差随距离自然收缩。

2. **机会式搭载检测**（`_piggyback`）。每次停车时顺手补测其它尚未定位的频道。边际
   成本仅 5-6 s ≡ 25-30 m 位移，却带来两样东西：未发现频道的**覆盖排除**（每次
   no_signal 挖掉一个半径 ≥1000 m 的圆盘，见 `feasible.py`），以及已发现频道的**长基线
   第二条示向度**（交会角天然优良）。实测：清除巡回自身的停车点已足以覆盖整个目标
   区域，于是**固定 12 站位环扫整体变成可选兜底**。

3. **可行域驱动的自适应站位**（`feasible.FeasibleSet` + `cover_route`）。把"沉默即
   超距"当集合约束序贯收缩每个频道的可行域，据此
   (a) 判定"该频道确实没有未清除的源"从而**主动停止搜索**（原实现只能靠预算耗尽被动
   放弃，为不存在的源白跑站位）；
   (b) 用贪心最大覆盖选出**有覆盖保证的最小站位集**，替代与案例无关的固定环。

4. **全局巡回排序**（`route.order_route`）。已定位目标按最近邻 + 2-opt + Or-opt 排序，
   替代"每次取最近者"的纯贪心；并且**不再在环扫与归航之间来回乒乓**（原实现每访问一个
   环站位就立刻归航一次，一进一出的空驶是绕路的主要来源）。

## 情节顺序

    原点普扫（20 次检测、0 位移）
      → 主巡回：已定位目标（最短开放巡回）→ 单示向度目标（dog-leg 交会）
                → 可行域补测站位（自适应），每次停车都做搭载检测
      → 兜底：内圈重测、射线盲清除、（问题 4）候选推理搜索 + 盲清除
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
    SOURCE_COUNT_MAX,
    SOURCE_COUNT_MIN,
    norm_deg,
)
from .client import ClearResult, MeasureResult, ProtocolError, RobotClient, TestEnded
from .directional import DirectionalBelief
from .feasible import FeasibleSet, cover_route
from .geometry import Bearing, estimate_position, localization_polygon, polygon_diameter
from .homing import bounded_estimate
from .route import order_route


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
    probe_rounds: int = 0            # 已在覆盖站位上重搜过几轮（有界，防止无进展死循环）
    abandoned: bool = False          # 全部兜底手段用尽，永久放弃（不再消耗任何预算）
    certified_empty: bool = False    # 仅实际沉默观测的连续覆盖证明；不等同于 abandoned
    belief: DirectionalBelief | None = None   # 问题 4：定向源候选集合 (x, y, φ)
    feasible: FeasibleSet | None = None       # 可行域（位置边缘），驱动"是否还要找"与"站哪"
    estimate: tuple[float, float] | None = None
    diameter: float = float("inf")
    # 估计位置到定位区域最远顶点的距离。≤ 清除半径 ⟹ 可证明"直接走过去清除必成功"。
    cover_radius: float = float("inf")
    bounded_homing: bool = False

    def add_bearing(self, x: float, y: float, svd_deg: float) -> None:
        first = not self.bearings
        self.bearings.append(Bearing(x, y, svd_deg))
        self.consecutive_silent = 0
        if first or self.exhausted:
            # 搜索阶段可能已烧掉大量检测（定向源尤甚）。拿到首个示向度说明已锁定目标，
            # 归航预算必须重新起算，否则会在"已能定位"时被判预算耗尽而放弃。
            # 同理，归航失败后在新站位重新拿到示向度 = 换了一套交会几何，值得重新起算；
            # 重搜轮数由 `probe_rounds` 上界兜住，不会因此无限重试。
            self.homing_measures = 0
            self.exhausted = False
        self._refresh()

    def _refresh(self) -> None:
        if self.bounded_homing:
            result = bounded_estimate(self.bearings)
            self.estimate, self.diameter, self.cover_radius = (
                result if result is not None else (None, float("inf"), float("inf"))
            )
            return
        if len(self.bearings) < 2:
            self.estimate = None
            self.diameter = float("inf")
            self.cover_radius = float("inf")
            return
        recent = self.bearings[-4:]          # 只用最近几条，近距离观测精度更高
        polygon = localization_polygon(recent)
        self.diameter = polygon_diameter(polygon)[0] if polygon else float("inf")
        self.estimate = estimate_position(recent) or estimate_position(self.bearings)
        if polygon and self.estimate is not None:
            # 覆盖半径 = 估计位置到定位区域最远顶点的距离。定位区域是凸多边形且必含真值
            # （每条示向度的 ±1° 楔形都含真值），故区域内任意点到估计位置的距离都不超过
            # 到某个顶点的距离 —— 覆盖半径 ≤ 20 m 时清除**必然**成功。见问题 1。
            self.cover_radius = max(math.dist(self.estimate, v) for v in polygon)
        else:
            self.cover_radius = float("inf")

    @property
    def verdict_empty(self) -> bool:
        """兼容别名：TSP 重构（23ca2ac）把「该频道已放弃」统一表达为 abandoned。

        11 个离线分析脚本仍按旧名读取：b_miss_rescue / b_miss_attrib / b_colocate /
        b_colocate_guard / b_combo / b_nearby_revive / b_survey_first / b_tail_fix /
        b_attrib_diag 等。缺这个属性时它们只在「恰好触发判空」的种子上抛
        AttributeError，那一局被 harness 当成 error 静默丢弃 —— 实测 seed 7052 因此
        使两臂变成 11 vs 12 局、142 vs 157 源，一度被误判为「变体影响了案例生成」。

        主策略路径不读这个字段，所以 215 项测试全绿也发现不了。只读别名，零行为改动。
        """
        return self.abandoned

    @property
    def active(self) -> bool:
        return bool(self.bearings) and not self.cleared and not self.exhausted

    @property
    def possible(self) -> bool:
        """该频道是否仍可能存在未清除的源。

        判据优先级：已清除 → 否；有示向度 → 是（看见过就一定存在）；否则看可行域/候选集
        是否已被观测约束排空。可行域为空是**主动停止搜索**的依据——原实现只能靠"检测预算
        耗尽"被动放弃，会为根本不存在的源反复跑站位。
        """
        if self.cleared or self.abandoned or self.certified_empty:
            return False
        if self.bearings:
            return True
        if self.belief is not None and not self.belief.alive:
            return False
        if self.feasible is not None and not self.feasible.possible:
            return False
        return True


@dataclass
class StrategyConfig:
    # None 按题型选择默认值：Q3 开启，Q4 保留 main；显式布尔值用于独立消融。
    # Q4 大样本出现新增漏清，未通过清除保持验收，不能仅因均值下降默认开启。
    certified_silence: bool | None = None
    bounded_homing: bool | None = None
    # 阶段 1
    sweep_origin: bool = True
    # 阶段 3 覆盖环。设计依据（数值实测，见 results/b_coverage_design.json）：
    # 干扰源随机分布时约 51% 在原点收不到信号，故补扫是主路径而非兜底，
    # 环参数要按"最坏覆盖间隙 < 接收半径下限 1000 m"直接选，而不是靠惰性触发省成本。
    # 半径 1250 m / 12 站位 + 原点：最坏间隙 670 m，13 个站位即可全域覆盖；
    # 双环方案（950x8 + 1560x14）需 23 站位，最坏间隙 539 m，覆盖更好但成本高 77%。
    # 实测标定（300 局/档 × 3 档, seed 17000, 见 results/b_margin_d*.json）：
    #   1150x6  间隙 987.0 m (余量 13 m)  Q3 329.1s(漏1) Q4d5 1084.4s Q4d10 1149.4s
    #   1200x7  间隙 885.9 m (余量 114 m) Q3 333.5s(全清) Q4d5 1094.6s Q4d10 1169.4s
    #   1100x8  间隙 887.3 m  Q3 340.8s(全清) Q4d5 1021.9s(-5.8%) Q4d10 1097.5s(-4.5%)
    # 选 1100x8：Q4 两档最快(-4.5%~-5.8%)，余量 113 m 与 1200x7 相当。
    # 弃 1150x6 的理由是余量仅 13 m：arena 判定为 distance > radius 才沉默、
    # radius~U(1000,1500) 下界可达，13 m 余量不足以吸收半径取值贴近下界的情形，
    # 这是 Q3 那 1 例漏失的直接原因（拉开余量后 1200x7 与 1100x8 均全清）。
    probe_ring_radius: float = 1100.0
    probe_ring_stations: int = 8
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
    dogleg_offset_deg: float = 18.0      # 第二个观测点相对示向度的偏转角
    dogleg_length_m: float = 380.0       # dog-leg 基线长度（固定回退值）
    # dog-leg 基线按"到可行域质心距离"的比例取（0 = 用固定值）。交会精度取决于相对基线。
    dogleg_range_fraction: float = 0.65
    dogleg_min_length_m: float = 180.0
    dogleg_max_length_m: float = 900.0
    approach_fraction: float = 0.62      # 每次前进占剩余距离的比例
    lateral_fraction: float = 0.16       # 前进时的横向偏移，保持交会角
    clear_distance_m: float = 55.0       # 估计距离降到该值以内即尝试清除
    spiral_radius_m: float = 15.0
    spiral_points: int = 6
    max_measures_per_channel: int = 40    # 搜索阶段总额度（定向源需要大量沉默观测）
    max_homing_measures: int = 14         # 归航阶段额度，拿到首个示向度后起算
    # 窄可见区定向源的归航加码（沉默观测>=8 时）。消融实测（60 局 d10, 旧 1150x6 环、
    # 无 TSP 路径规划）显示关闭它对清除率与 T̄ 均无差别（T̄ 甚至微降 0.5 s），疑为
    # 上游机制（射线盲扫）已覆盖其目标场景。但该消融未在本分支的 TSP 路径下重测，
    # 且它是"放宽预算"而非收紧，删除会使归航更早放弃、有清除率风险，故保留默认值。
    narrow_sector_extra_measures: int = 22
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

    # ---- 路径规划优化（本轮新增，见模块 docstring 的"四条机制"）----
    # 机会式搭载检测总开关。边际成本 5-6 s ≡ 25-30 m 位移。
    piggyback: bool = True
    # 分支 A：**免费的第二条示向度**。只有一条示向度的频道，交会需要第二条；顺手在
    # 一个远离首个观测点的位置补测，既不多走一步，交会角还天然优良。
    # 实测这是搭载检测的全部价值所在：关掉后 T̄ 从 267 s 涨到 349 s（+30%）。
    piggyback_bearing: bool = True
    piggyback_min_baseline_m: float = 250.0  # 免费第二条示向度所需的最小交会基线
    # 分支 B：**免费的覆盖排除**（顺手测未发现的频道，挖掉一个 ≥1000 m 圆盘）。
    # 实测**不划算**，默认关闭：覆盖排除必须按频道逐个付 6 s（每个频道都要独立认证
    # 自己的可行域，信息不能共享），而归航途中的停车点彼此挨得近、圆盘高度重叠；
    # 专门规划的补测站位每次检测能排掉近满额的 271 格，效率高得多。保留为消融开关。
    piggyback_coverage: bool = False
    # 覆盖排除分支的增益门槛（格点数，1 格 = 100×100 m；满额约 271 格）。
    piggyback_min_gain_cells: int = 80
    piggyback_max_per_stop: int = 6       # 单次停车的搭载检测上限，防止在一点上磨时间
    # 可行域的沉默约束（no_signal ⟹ 距离 > 1000 m 或落在扇区外）。留作消融开关。
    use_silence_constraint: bool = True
    # 问题 4 的定向方向离散档数。12 档 = 30°/档，保守裕度 Δ/2 = 15°。
    heading_count: int = 12
    # 盲清除（`/clear` 只看距离、与辐射方向无关）：处理"可行域非空但一直收不到信号"的
    # 窄扇区定向源。按候选格点由可能性高到低铺点，受站位数与清除次数双重上界约束。
    blind_clear_max_cells: int = 26
    blind_clear_max_calls: int = 320
    # 自适应覆盖站位（贪心最大覆盖 + 冗余消除 + 最短巡回）替代固定 12 站位环扫。
    adaptive_cover: bool = True
    cover_max_stations: int = 12
    # 站位增益闸门（格点数）：低于此值不值得为它专门跑一趟。收尾阶段会放开到 1，
    # 以满足"确保所有干扰源被清除"的硬前提。
    cover_min_gain_cells: int = 4
    # 单局补测站位总预算。问题 3 一般只用 8-12 站就把可行域排空，该上界不起作用；问题 4
    # 则**必须**有这个闸门：定向源的完全认证需要对每个格点从 3 个以上方向各看一眼，实测
    # 完全排空要 47 站、21.5 km 空驶且仍剩 120 个格点（贴边格点需要从区域外回看）。
    # 因此问题 4 采用"有界搜索 + 盲清除收尾"，而不是追求可证明的完全认证。
    cover_station_budget: int = 30
    # 站位打分里"1 个覆盖格点折算多少米位移"（净收益 = 覆盖格点数 − 距离/该值）。
    # 实测该项宜取大（弱折价）：折价过强会让贪心选近处的低增益站位，站位数变多、总路程
    # 反而更长（0.5 m/格 → T̄ 365 s；20 m/格 → 272 s）。站位次序另由 2-opt 巡回负责，
    # 因此"选站位"环节应以覆盖效率为主、移动为辅。
    cover_travel_weight_m: float = 20.0
    # 站位选择的"并列最优"容差：只在增益 ≥ 该比例 × 最大增益的候选里比移动距离。
    # 覆盖某块残余可行域的站位位置自由度很大，让出一点覆盖效率往往能换来大段省路。
    cover_gain_tolerance: float = 0.0
    # 每一轮都重算补测站位计划（消融/对照用）。默认按"待搜索频道集合变化"触发重算。
    cover_replan_always: bool = False
    # 让"解析式正 n 边形覆盖布局"与贪心解同台竞争（按巡回长度择优）。见 feasible._ring_layouts。
    cover_ring_layouts: bool = True
    # 推迟补测站位：先把有线索的目标清完，认证性检测尽量靠搭载检测顺手完成。
    cover_defer: bool = False
    # 把 dog-leg 偏转方向与逼近横向偏移方向朝"巡回里下一个航点"那一侧偏（零成本自由度）。
    steer_toward_next: bool = False
    # 可行域格点边长。它同时决定**保守裕度** h = (√2/2)·step，而"保证覆盖半径"= 1000 − h：
    # 格子越细，裕度越小、保证覆盖半径越大，覆盖整个区域所需的站位环半径就越小、巡回越短。
    # 代价是约束更新与覆盖表预计算变慢（O(格点数)，随 step⁻² 增长）。
    feasible_cell_step_m: float = 75.0
    # 巡回局部改进（2-opt + Or-opt）。关掉即退化为原来的"每次取最近者"纯贪心。
    route_improve: bool = True
    # 有界重搜：归航失败的频道最多再在覆盖站位上重搜几轮，之后永久放弃。
    # 没有这个上界时会出现"站位给出新示向度 → 归航再失败 → 再回站位"的活锁。
    max_probe_rounds: int = 3
    # 跳过"确定无收益"的检测：候选位置全在接收半径上界 1500 m 之外时，既不可能发现目标，
    # 沉默也排除不了任何候选。留作消融开关。
    skip_hopeless_probes: bool = True
    # 更激进的门禁：站位对该频道的"保证覆盖"增益低于此值就跳过这次检测。0 = 关闭。
    # 与上一条的区别：这条会放弃 929-1500 m 之间"可能撞上"的机会以换取时间。
    station_min_gain_cells: int = 0
    # 安全阀：单局检测总次数上限。触发后只收尾已定位目标，不再开新的搜索。
    # 依据附件 1 §4.6"正式测试加密日志需 < 2 MB，机器狗程序不得有异常高频循环"。
    max_total_measures: int = 900
    # 定位区域直径小于该值时，直接走到估计位置发 /clear，不再在半路多测一轮。
    # 取 150 m 的依据：真值必在定位区域内，故走到估计位置后距真值 ≤ 150 m；此时再测一次
    # 的横向误差仅 150·tan(1°) ≈ 2.6 m ≪ 20 m 清除半径，最多一轮就能收敛。
    # 直径 ≤ 20√3 ≈ 34.6 m 时甚至无需重测（Jung 平面上界，见问题 1）。
    direct_clear_diameter_m: float = 150.0
    legacy_ring_fallback: bool = True    # 可行域路线走完仍有漏时，回落到固定环扫

    def __post_init__(self) -> None:
        if self.certified_silence is None:
            self.certified_silence = not self.directional_probe
        if self.bounded_homing is None:
            self.bounded_homing = not self.directional_probe


class DogStrategy:
    # 类级默认值：这些是纯计数/缓存字段，给出类级默认后，即使实例被绕过 `__init__` 构造
    # （测试里用 `DogStrategy.__new__` + 手工注入替身 client 的做法），底层动作也不会因为
    # 缺属性而抛 AttributeError。`__init__` 仍会覆盖成实例属性。
    measure_calls: int = 0
    cover_visits: int = 0
    _next_hint: tuple[float, float] | None = None
    _cover_cache: tuple[tuple[int, ...], list[tuple[float, float]]] | None = None

    def __init__(self, client: RobotClient, config: StrategyConfig | None = None) -> None:
        self.client = client
        self.config = config or StrategyConfig()
        self.channels: dict[int, ChannelState] = {
            ch: ChannelState(ch, bounded_homing=self.config.bounded_homing)
            for ch in range(CHANNEL_MIN, CHANNEL_MAX + 1)
        }
        self.pos = (0.0, 0.0)
        self.radio_channel = 1
        self.events: list[str] = []
        self.measure_calls = 0
        self.cover_visits = 0
        self._next_hint: tuple[float, float] | None = None
        self._stopped = False
        # 覆盖站位计划缓存：(待搜索频道元组, 剩余站位列表)
        self._cover_cache: tuple[tuple[int, ...], list[tuple[float, float]]] | None = None

    # ---- 入口 ----

    def run(self) -> dict:
        cfg = self.config
        # 问题 3 与问题 4 的唯一差别就是可行域的模式集合：前者只含"全向"，后者再加上
        # `heading_count` 档定向方向。沉默约束在两种情形下都成立（定向源的沉默只是能
        # 排除的模式更少），因此不再需要两套策略。
        headings = cfg.heading_count if cfg.directional_probe else 0
        for state in self.channels.values():
            state.feasible = FeasibleSet(step=cfg.feasible_cell_step_m, headings=headings)
        self.client.enter()
        try:
            if cfg.sweep_origin:
                self._sweep_at((0.0, 0.0), range(CHANNEL_MIN, CHANNEL_MAX + 1))
            self._tour()
            # ---- 兜底 ----
            # 每一级兜底都必须有触发条件。`_unfinished()`（"还有频道可能有源"）在问题 4 里
            # 几乎恒为真——空频道永远无法被证否——用它当闸门会让昂贵的兜底每局都跑：实测
            # 固定环扫兜底吃掉了问题 4 的 46.7% 移动（18.6 km / 3727 s）却几乎没有产出。
            self._revive_exhausted()
            self._tour()
            if self._stuck_with_bearings():
                # 只有"见过信号却清不掉"的频道才值得回内圈换交会几何
                self._inner_retry()
                self._tour()
            if cfg.legacy_ring_fallback and self._must_find_more():
                # 固定环扫只在"可断言还有源没找到"时启用；自适应站位规划本就是它的超集，
                # 这一级纯粹是对可行域推理出意外时的保险。
                self._probe_ring()
                self._revive_exhausted()
                self._tour()
            if self._unfinished():
                # 收尾：仍有频道可行域非空但一直找不到信号 —— 典型是朝区域外辐射的窄扇区
                # 定向源，其可见区仅占目标区域百分之几。`/clear` 只看距离、与辐射方向无关
                # （附录 2(8)），所以直接对候选格点盲清除比继续找信号便宜。
                self._blind_clear_feasible()
        except (TestEnded, ProtocolError) as exc:
            self.events.append(f"aborted: {exc}")
        finally:
            try:
                self.client.exit()
            except (TestEnded, ProtocolError):
                pass
        return self.summary()

    def _unfinished(self) -> bool:
        """是否还有"可能存在且未清除"的频道。可行域已排空的频道不算未完成。"""
        return any(st.possible for st in self.channels.values())

    def _cleared_count(self) -> int:
        return sum(1 for st in self.channels.values() if st.cleared)

    def _stuck_with_bearings(self) -> bool:
        """是否存在"见过信号却还没清掉"的频道 —— 这种频道一定有源，值得花兜底预算。"""
        return any(st.bearings and not st.cleared and not st.abandoned
                   for st in self.channels.values())

    def _must_find_more(self) -> bool:
        """是否**可以断言**还有未清除的干扰源。

        题面给出干扰源总数在 10-16 之间（具体个数未知）。于是已清除个数 < 10 时一定还有源
        没清掉，此时继续搜索、乃至启用昂贵的盲清除都是**有据可依**的；一旦清除数达到 10，
        剩下那些"从未收到过信号"的频道就有可能全都是空频道，再为它们盲清除是纯亏。

        这条判据修掉了一个致命的成本失控：问题 4 里通常有 7-10 个空频道，若无差别地对它们
        的候选格点盲清除，实测单局会多打 3000 多次 /clear、多走 35 万米（T̄ 从数百秒炸到
        8000 s 以上）。而空频道恰恰无法与"藏起来的定向源"区分——两者都是"没见过信号"，
        可行域完全相同——所以只能靠总数下界这个外部信息来止损。
        """
        return self._cleared_count() < SOURCE_COUNT_MIN

    # ---- 主巡回：统一的路线规划 ----

    def _tour(self, max_rounds: int = 400) -> None:
        """统一路线规划主循环。

        每轮按优先级挑一件事做，做完立刻重规划（估计位置会随逼近刷新，站位计划会随
        可行域收缩改变，所以是滚动式重规划而非一次排定）：

        1. **已定位目标** —— 全部估计位置交给 `route.order_route` 排成最短开放巡回，
           取巡回第一个目标归航清除。这一步替代了原来的"每次取当前最近者"：纯最近邻
           期望长度约为最优的 1.25 倍，且会留下"最后一个远点"长尾。
        2. **单示向度目标** —— 还交会不出位置，就近做 dog-leg 构造基线。
        3. **覆盖补测** —— 没有任何线索时，按剩余可行域规划有覆盖保证的站位巡回，
           走一站就重规划（一旦拿到示向度立即转去清除，不把整条环走完）。

        原实现的顺序是"环扫每到一站就归航一次"，一进一出的空驶是绕路倍数 2.58x 的
        主要来源；这里把两类动作放进同一个代价空间统一排序，天然消除乒乓。
        """
        for _ in range(max_rounds):
            if self._stop():
                return
            if not self._next_waypoint():
                return

    def _next_waypoint(self) -> bool:
        """把三类待办**放进同一个开放 TSP 里排序**，处理排在最前的那一个。

        三类待办都可以归约成"必须去的一个坐标"，因此可以放进同一个代价空间：

        | 待办 | 代表坐标 | 到达后做什么 |
        |---|---|---|
        | 已定位目标 | 交会估计位置 | 逐步逼近并清除 |
        | 单示向度目标 | 沿示向度前进 `dogleg_length_m` 的点 | dog-leg 构造交会基线 |
        | 覆盖补测站位 | 贪心最大覆盖选出的格心 | 扫所有待搜索频道 |

        这一步是把"清除巡回"与"搜索补测"**合并规划**的关键。分开规划时，机器狗先把
        能定位的目标全清掉、再折回去跑补测站位，两段路径各自最优但合起来绕路；合并后
        补测站位会被顺路吃掉（一个站位只要落在剩余可行域里就有效，位置自由度很大）。

        处理完一个航点就重新规划：估计位置会随逼近刷新、可行域会随观测收缩，滚动重规划
        比一次排定更贴合真实信息。为避免每轮重算站位集合造成抖动，站位计划做了缓存
        （`_cover_plan`），只在"待搜索频道集合变化"或"计划走空"时重算。
        """
        waypoints: list[tuple[float, float]] = []
        kinds: list[tuple[str, ChannelState | None]] = []

        for state in self.channels.values():
            if not state.active:
                continue
            if state.estimate is not None:
                waypoints.append(state.estimate)
                kinds.append(("target", state))
            else:
                point = self._seed_waypoint(state)
                if point is not None:
                    waypoints.append(point)
                    kinds.append(("seed", state))
        # `cover_defer`：只有在没有任何待清除线索时才把补测站位放进巡回。
        # 用意是把"认证性检测"尽可能安排在**已经要去的地方**（源位置、逼近途中的停车点）顺手
        # 完成（配合 `piggyback_coverage`），从而缩短专门的补测环。
        if not (self.config.cover_defer and waypoints):
            for station in self._cover_plan():
                waypoints.append(station)
                kinds.append(("station", None))

        if not waypoints:
            return False
        order = order_route(self.pos, waypoints, improve=self.config.route_improve)
        head = order[0] if order else waypoints[0]
        pick = min(range(len(waypoints)), key=lambda i: math.dist(waypoints[i], head))
        kind, state = kinds[pick]
        # 巡回里的**下一个**航点。dog-leg 的偏转方向与逼近的横向偏移方向在几何上都是左右对称、
        # 代价相同的自由度，把它们朝下一个航点那一侧偏，就等于顺手把去下一个目标的路走了一段。
        self._next_hint = order[1] if len(order) > 1 else None

        if kind == "station":
            return self._visit_cover_station(waypoints[pick])
        assert state is not None
        if not self._home_channel(state):
            state.exhausted = True
            self._invalidate_cover_plan()
        elif kind == "target":
            # 刚清除完，机器狗正站在一个源的位置上——目标区域深处的优质观测点，
            # 顺手把其它未定位频道扫一遍（每次仅 5-6 s，且零位移）。
            self._piggyback(exclude=state.channel)
            self._invalidate_cover_plan()
        return True

    def _seed_waypoint(self, state: ChannelState) -> tuple[float, float] | None:
        """"只有一条示向度"的目标在巡回里的代表坐标 —— 取**可行域质心**。

        为什么这个代表点很要紧：巡回排序只认坐标。早先版本用"沿示向度前进一个 dog-leg 基线
        长度（380 m）"当代表点，结果所有还没交会出位置的目标都挤在原点附近的一小圈里，
        开放 TSP 根本分不出它们的方位，于是先随便挑一个走过去、再横穿区域去下一个，路径反复
        交叉。

        可行域质心才是"这个源大概在哪"的正确估计：一条示向度把候选压成一个楔形，再叠上
        "距观测点 ≤ 1500 m（接收半径上界）"与"在半径 1800 m 区域内"，质心自然落在射线上
        合理的距离处（实测约 700-900 m，而不是 380 m）。这样单示向度目标也能按方位正确参与
        全局排序，与已定位目标、补测站位混排。
        """
        if not state.bearings:
            return None
        if state.feasible is not None:
            centre = state.feasible.centroid()
            if centre is not None:
                return centre
        last = state.bearings[-1]
        angle = math.radians(last.svd_deg)
        length = self.config.dogleg_length_m
        point = (last.x + length * math.cos(angle), last.y + length * math.sin(angle))
        if math.hypot(*point) > ARENA_RADIUS:
            norm = math.hypot(*point) or 1.0
            point = (point[0] / norm * ARENA_RADIUS, point[1] / norm * ARENA_RADIUS)
        return point

    # ---- 覆盖站位计划（缓存 + 失效）----

    def _invalidate_cover_plan(self) -> None:
        self._cover_cache = None

    def _cover_plan(self) -> list[tuple[float, float]]:
        """当前的覆盖补测站位计划（缓存）。待搜索频道集合变化或计划走空时重算。"""
        cfg = self.config
        if not cfg.adaptive_cover:
            return []
        pending = tuple(sorted(st.channel for st in self.channels.values()
                               if self._search_target(st)))
        if not pending:
            self._cover_cache = None
            return []
        cache = self._cover_cache
        if cache is not None and cache[0] == pending and cache[1] and not cfg.cover_replan_always:
            # 待搜索频道集合不变时沿用原计划，避免同一批站位被反复重挑重排。
            # 但集合一变就必须重算：某频道被发现而退出待搜索集合后，**并集覆盖要求变小**，
            # 常常能少跑一整个站位。实测"发现后不重算"会让 T̄ 从 254 s 退到 261 s，
            # 说明这里重算带来的覆盖收益大于路线抖动的代价。
            return list(cache[1])
        sets = [self.channels[ch].feasible for ch in pending
                if self.channels[ch].feasible is not None]
        # 锚点 = 机器狗接下来反正要去的地方（当前位置 + 待清除目标估计位置）。
        # 补测站位落在锚点附近时几乎不产生额外空驶，是"顺路吃掉搜索补测"的关键。
        anchors = [st.estimate for st in self.channels.values()
                   if st.active and st.estimate is not None]
        stations = cover_route(sets, self.pos, cell_step=cfg.feasible_cell_step_m,
                              max_stations=cfg.cover_max_stations,
                              min_gain=cfg.cover_min_gain_cells,
                              travel_weight_m=cfg.cover_travel_weight_m,
                              gain_tolerance=cfg.cover_gain_tolerance,
                              use_ring_layouts=cfg.cover_ring_layouts,
                              anchors=anchors)
        if not stations and cfg.cover_min_gain_cells > 1:
            # 主相位的增益闸门挡住了"只剩零星碎格"的收尾；为满足"确保所有干扰源被清除"
            # 的硬前提，收尾时放开闸门做一次精确覆盖。
            stations = cover_route(sets, self.pos, cell_step=cfg.feasible_cell_step_m,
                                   max_stations=cfg.cover_max_stations,
                                   min_gain=1,
                                   travel_weight_m=cfg.cover_travel_weight_m,
                                   gain_tolerance=cfg.cover_gain_tolerance,
                                   use_ring_layouts=cfg.cover_ring_layouts,
                                   anchors=anchors)
        if not stations:
            for ch in pending:
                state = self.channels[ch]
                if state.feasible is not None and not state.feasible.possible:
                    self.events.append(f"ch{ch}: 可行域已空，判定该频道无源")
        self._cover_cache = (pending, list(stations))
        return list(stations)

    def _visit_cover_station(self, station: tuple[float, float]) -> bool:
        """走到一个覆盖站位，扫完值得扫的待搜索频道，并把该站位从计划中划掉。

        两个省时细节：

        * **无效检测门禁**：一个站位是为"待搜索频道的并集"选出来的，其中往往有几个频道的
          剩余可行域离它很远。若该频道所有候选位置都在接收半径上界 1500 m 之外，则此处检测
          既不可能发现目标、沉默也排除不了任何候选，属于确定无收益的 5-6 s，直接跳过
          （`FeasibleSet.detection_possible`）。
        * **切频顺序**：检测顺序对结果无影响，但相邻两次检测同频道可省 1 s 切频。因此从
          测向机**当前频道**开始按环形顺序扫，而不是无条件从小到大——每个站位省 1 s。
        """
        todo = [st for st in self.channels.values() if self._search_target(st)]
        if not todo:
            self._invalidate_cover_plan()
            return True
        self.cover_visits += 1
        for state in todo:
            if state.exhausted:
                state.probe_rounds += 1
        found = False
        for state in self._scan_order(todo):
            if self._stop():
                return True
            if not state.possible:
                continue
            if state.feasible is not None and self._skip_probe(state, station):
                continue
            before = len(state.bearings)
            self._probe_channel_at(station, state)
            if state.cleared or len(state.bearings) > before:
                found = True
        self._piggyback()
        cache = self._cover_cache
        if cache is not None:
            self._cover_cache = (cache[0], [p for p in cache[1] if p != station])
        if found:
            self._invalidate_cover_plan()
        return True

    def _next_localized(self) -> bool:
        """处理一个已交会出位置的目标；无此类目标返回 False。"""
        targets = [st for st in self.channels.values()
                   if st.active and st.estimate is not None]
        if not targets:
            return False
        order = order_route(self.pos, [st.estimate for st in targets],  # type: ignore[misc]
                            improve=self.config.route_improve)
        head = order[0] if order else targets[0].estimate
        state = min(targets, key=lambda st: math.dist(st.estimate, head))  # type: ignore[arg-type]
        if not self._home_channel(state):
            state.exhausted = True
        else:
            # 刚清除完，机器狗正站在一个源的位置上——目标区域深处的优质观测点，
            # 顺手把其它未定位频道扫一遍（每次仅 5-6 s，且零位移）。
            self._piggyback(exclude=state.channel)
            self._invalidate_cover_plan()
        return True

    def _next_seed(self) -> bool:
        """处理一个"只有示向度、还交会不出位置"的目标；无此类目标返回 False。"""
        seeds = [st for st in self.channels.values()
                 if st.active and st.estimate is None]
        if not seeds:
            return False
        seeds.sort(key=lambda st: (self._range_to(st), st.channel))
        state = seeds[0]
        if not self._home_channel(state):
            state.exhausted = True
        return True

    def _next_cover_station(self) -> bool:
        """按剩余可行域规划一条有覆盖保证的补测巡回并**整条走完**；无事可做返回 False。

        为什么整条走完而不是"走一站就重规划"：重规划会让机器狗每次都奔向**最近**的站位，
        而最近不等于增益高，于是在几个低增益站位之间来回蹭（实测退化成上百轮、单频道
        近 400 次检测）。整条走完保持路线连贯，只在**真的发现了新目标**时才中断——此时
        中断是有价值的，因为新目标通常就在附近。
        """
        cfg = self.config
        if not cfg.adaptive_cover:
            return False
        pending = [st for st in self.channels.values() if self._search_target(st)]
        if not pending:
            return False
        sets = [st.feasible for st in pending if st.feasible is not None]
        if not sets:
            return False
        stations = cover_route(sets, self.pos, max_stations=cfg.cover_max_stations,
                              min_gain=cfg.cover_min_gain_cells)
        if not stations and cfg.cover_min_gain_cells > 1:
            # 主相位的增益闸门挡住了"只剩零星碎格"的收尾；为满足"确保所有干扰源被清除"
            # 的硬前提，收尾时放开闸门做一次精确覆盖。
            stations = cover_route(sets, self.pos, max_stations=cfg.cover_max_stations,
                                   min_gain=1)
        if not stations:
            for state in pending:
                if state.feasible is not None and not state.feasible.possible:
                    self.events.append(f"ch{state.channel}: 可行域已空，判定该频道无源")
            return False
        for state in pending:
            if state.exhausted:
                state.probe_rounds += 1
        for station in stations:
            if self._stop():
                return True
            todo = [st for st in pending if self._search_target(st)]
            if not todo:
                return True
            found = False
            for state in sorted(todo, key=lambda st: st.channel):
                if self._stop():
                    return True
                if not state.possible:
                    continue
                before = len(state.bearings)
                self._probe_channel_at(station, state)
                if state.cleared or len(state.bearings) > before:
                    found = True
            # 顺手把"虽然已有示向度、但还缺第二条基线"的频道也测一遍（零位移）
            self._piggyback()
            if found:
                return True          # 有新线索 —— 就近清除优先于继续扫站位
        return True

    def _skip_probe(self, state: ChannelState, station: tuple[float, float]) -> bool:
        """该站位上对该频道的这次检测是否确定无收益（可跳过，省 5-6 s）。"""
        cfg = self.config
        feasible = state.feasible
        assert feasible is not None
        if cfg.skip_hopeless_probes and not feasible.detection_possible(*station):
            return True          # 候选全在接收半径上界之外：不可能发现、也排除不了任何候选
        if cfg.station_min_gain_cells > 0:
            return feasible.coverage_gain(*station) < cfg.station_min_gain_cells
        return False

    def _scan_order(self, states: list[ChannelState]) -> list[ChannelState]:
        """一站多频道的扫描顺序：从测向机当前频道起按频道号环形排列。

        任意两频道间切换都是 1 s（附录 2(4)），与频道号距离无关，所以顺序不影响切频**总数**
        ——除了一点：若首个被扫的频道恰是测向机当前频道，这一次切频就省掉了。环形顺序同时
        保持了确定性（便于复盘与回归测试）。
        """
        ordered = sorted(states, key=lambda st: st.channel)
        pivot = next((i for i, st in enumerate(ordered)
                      if st.channel >= self.radio_channel), 0)
        return ordered[pivot:] + ordered[:pivot]

    def _search_target(self, state: ChannelState) -> bool:
        """需要靠站位搜索的频道：可能存在、且当前没有可用线索（没测到过或归航已失败）。"""
        if not state.possible or state.feasible is None:
            return False
        if self._cleared_count() >= SOURCE_COUNT_MAX:
            # 题面给出干扰源总数上界 16。清除数达到 16 ⟹ **可以断言**再没有未清除的源，
            # 剩余站位一次都不用跑。这是一条免费且严格的提前收工判据（约 1/7 的案例命中）。
            return False
        if self._measure_calls() >= self.config.max_total_measures:
            return False          # 安全阀：只收尾已定位目标，不再开新搜索
        if (self.cover_visits >= self.config.cover_station_budget
                and not self._must_find_more()):
            # 站位预算用尽且已清除个数达到总数下界 10 —— 剩下的频道可能全是空频道，收手。
            # 但若清除数仍 < 10，则一定还有源没找到，继续加站位是有据可依的。
            return False
        if not state.bearings:
            return True
        # 有示向度却归航失败：换站位重取几何值得再试，但必须有界，否则
        # "站位给新示向度 → 归航再失败 → 再回站位"会活锁。
        return state.exhausted and state.probe_rounds < self.config.max_probe_rounds

    def _measure_calls(self) -> int:
        return self.measure_calls

    # ---- 机会式搭载检测 ----

    def _piggyback(self, exclude: int | None = None) -> None:
        """在**当前位置**顺手补测其它未定位频道（零位移，边际成本仅 5-6 s）。

        两个分支，取舍是实测出来的（1 次检测 ≈ 30 m 位移）：

        * **分支 A（默认开）免费的第二条示向度**：只有一条示向度的频道，交会定位缺的就是
          第二条。顺手在一个远离首个观测点（基线 ≥ `piggyback_min_baseline_m`）的位置补测，
          既不多走一步，交会角还天然优良——**这是搭载检测的全部价值**，关掉后 T̄ +30%。
        * **分支 B（默认关）免费的覆盖排除**：顺手测未发现的频道以挖掉一个 ≥1000 m 圆盘。
          实测不划算：覆盖认证**不能跨频道共享**（每个频道都得独立排空自己的可行域，
          所以要按频道逐个付 6 s），而归航途中的停车点彼此很近、圆盘高度重叠；专门规划
          的补测站位每次检测能排掉近满额的 271 格，效率高得多。
        * 已能交会定位的频道一律跳过：更多示向度不改变"要走过去清除"这件事，逼近/直清
          过程本身会在更近处重测（那时精度更高）。
        """
        cfg = self.config
        if not cfg.piggyback or self._stop():
            return
        picks: list[tuple[int, int, ChannelState]] = []
        for state in self.channels.values():
            if state.channel == exclude or state.cleared or state.estimate is not None:
                continue
            if not state.possible or state.feasible is None:
                continue
            gain = state.feasible.coverage_gain(*self.pos)
            if gain <= 0:
                continue          # 当前位置不可能收到该频道的信号，测了必沉默且无信息
            if state.bearings:
                if not cfg.piggyback_bearing:
                    continue
                baseline = min(math.dist(self.pos, (b.x, b.y)) for b in state.bearings)
                if baseline < cfg.piggyback_min_baseline_m:
                    continue
                priority = 1 << 30          # 免费的第二条示向度优先于纯覆盖排除
            elif not cfg.piggyback_coverage or gain < cfg.piggyback_min_gain_cells:
                continue
            else:
                priority = gain
            picks.append((-priority, state.channel, state))
        picks.sort(key=lambda item: (item[0], item[1]))
        for _, _, state in picks[:cfg.piggyback_max_per_stop]:
            if self._stop():
                return
            self._probe_channel_at(self.pos, state)

    def _coverage_gain_at(self, point: tuple[float, float]) -> int:
        """一个候选停车点对所有未定位频道的合计覆盖增益（用于在等价方案间择优）。"""
        total = 0
        for state in self.channels.values():
            if state.cleared or state.estimate is not None or not state.possible:
                continue
            if state.feasible is None:
                continue
            total += state.feasible.coverage_gain(*point)
        return total

    def summary(self) -> dict:
        cleared = sorted(ch for ch, st in self.channels.items() if st.cleared)
        certified = sorted(ch for ch, st in self.channels.items() if st.certified_empty and not st.cleared)
        unresolved = sorted(set(self.channels) - set(cleared) - set(certified))
        basis = ("source_count_upper_bound" if len(cleared) >= SOURCE_COUNT_MAX else
                 "continuous_coverage" if not unresolved else "unresolved")
        measures = sum(1 for e in self.client.log if e["path"] == "/measure")
        clears = sum(1 for e in self.client.log if e["path"] == "/clear")
        return {
            "cleared_channels": cleared,
            "cleared_count": len(cleared),
            "virtual_time_s": self.client.virtual_time_s,
            "measure_calls": measures,
            "clear_calls": clears,
            "mean_clear_time_s": self.client.virtual_time_s / len(cleared) if cleared else None,
            "certified_empty_channels": certified,
            "unresolved_channels": [] if basis == "source_count_upper_bound" else unresolved,
            "completion_basis": basis,
            "abandoned_channels": sorted(ch for ch, st in self.channels.items() if st.abandoned),
            "events": self.events,
        }

    # ---- 阶段 1 / 3：扫描 ----

    def _sweep_at(self, point: tuple[float, float], channels) -> None:
        """在一个站位上按频道号顺序扫描，减少切频次数（切频与频道距离无关，仍保持有序）。"""
        for channel in channels:
            if self._stop(): return
            state = self.channels[channel]
            if state.cleared or state.bearings or not state.possible:
                continue
            self._probe_channel_at(point, state)

    def _probe_target(self, state: ChannelState) -> bool:
        """补扫对象：仍可能存在，且尚无示向度或归航已失败（换站位可改善几何）。"""
        return state.possible and (not state.bearings or state.exhausted)

    def _needs_probe(self) -> bool:
        return any(self._probe_target(st) for st in self.channels.values())

    def _probe_channel_at(self, point: tuple[float, float], state: ChannelState) -> None:
        """在 point 对 state.channel 检测一次，并把结果并入该频道的全部信念。

        三类结果都是信息（这是本轮优化的关键前提）：
        * near      —— 源距该点 ≤ 5 m，直接清除；
        * direction —— 楔形约束（源在 ±1° 楔形内、且距离 ≤ 1500 m 接收半径上限）；
        * no_signal —— 全向情形下 ⟹ 距离 > 1000 m，挖掉一个圆盘（`use_silence_constraint`）。
        """
        result = self._measure(point, state.channel)
        if result is None:
            return
        if result.result == "near":
            if state.feasible is not None:
                state.feasible.apply_near(point[0], point[1])
            self._clear_at(point, state.channel)
        elif result.result == "direction":
            if result.svd_deg is None:
                raise ProtocolError("direction response has no bearing")
            state.add_bearing(point[0], point[1], result.svd_deg)
            if state.feasible is not None:
                state.feasible.apply_bearing(point[0], point[1], result.svd_deg)
            if state.belief is not None:
                state.belief.apply_bearing(point[0], point[1], result.svd_deg)
        elif result.result == "no_signal":
            state.silent_points.append(point)
            state.consecutive_silent += 1
            if self._certify_silence(state):
                return
            if state.feasible is not None and self.config.use_silence_constraint:
                state.feasible.apply_silence(point[0], point[1])
            if state.belief is not None:
                state.belief.apply_silence(point[0], point[1])
        else:
            raise ProtocolError(f"unknown measurement result: {result.result!r}")

    def _certify_silence(self, state: ChannelState) -> bool:
        """Only received no-signal points may justify this distinct verdict."""
        if not self.config.certified_silence or state.bearings or state.cleared:
            return False
        if state.certified_empty:
            return True
        from .silence_coverage import silence_certificate
        if not silence_certificate(state.silent_points, self.config.directional_probe):
            return False
        state.certified_empty = True
        self._invalidate_cover_plan()
        self.events.append(f"ch{state.channel}: empty by continuous silence certificate")
        return True

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

    def _walk_ring(self, ring: list[tuple[float, float]]) -> None:
        """就近串行访问一个环上的站位，每站扫完立刻归航清除，避免回头空驶。"""
        remaining = ring[:]
        while remaining and not self._stop():
            todo = [st for st in self.channels.values() if self._probe_target(st)]
            if not todo:
                return
            remaining.sort(key=lambda p: math.hypot(p[0] - self.pos[0], p[1] - self.pos[1]))
            station = remaining.pop(0)
            for state in sorted(todo, key=lambda s: s.channel):
                if self._stop(): return
                self._probe_channel_at(station, state)
            self._home_all()

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

    def _blind_clear_feasible(self) -> None:
        """收尾兜底：对"可行域非空但一直收不到信号"的频道，直接到候选格点发 /clear。

        依据（实测）：漏失的定向源全部"物理可解"——其可见扇区一直延伸到源身边（最近 4-10 m），
        但该扇区仅占目标区域 0.1%-19%，靠站位撞上的概率很低。而 `/clear` 的判定只看平面距离
        （20 m 内即成功），**与辐射方向、测向机频道无关**（附录 2(8)）。因此当可行域已压缩到
        少数格点时，逐格盲清除比继续找信号便宜得多。

        铺点顺序：候选格点按"剩余模式数"（该位置还有多少种朝向可能）降序，即按可能性排序；
        格内按 20 m 清除半径铺点保证覆盖整格。移动顺序交给最短巡回，并受
        `blind_clear_max_cells` / `blind_clear_max_calls` 双重上界约束。
        """
        cfg = self.config
        for state in sorted(self.channels.values(), key=lambda s: s.channel):
            if self._stop():
                return
            if state.cleared or state.certified_empty or state.feasible is None or not state.feasible.possible:
                continue
            # 收到过示向度 ⟹ 这个频道**确实**有一个未清除的源，盲清除无论多贵都值得。
            # 没收到过 ⟹ 与空频道不可区分，只有在"已清除个数 < 总数下界 10"时才有据可依。
            proven = bool(state.bearings)
            if not proven and not self._must_find_more():
                # 盲清除的代价（每格约 12 次 /clear + 格内空驶）远超收益，且问题 4 通常有
                # 7-10 个空频道。见 `_must_find_more`。
                self.events.append(
                    f"ch{state.channel}: 从未收到信号且已清除 {self._cleared_count()} 个 "
                    f"≥ 总数下界 {SOURCE_COUNT_MIN}，不为其做盲清除")
                state.abandoned = True
                continue
            ranked = state.feasible.weighted_points()[:cfg.blind_clear_max_cells]
            if not ranked:
                continue
            self.events.append(
                f"ch{state.channel}: 盲清除 {len(ranked)} 个候选格点"
                f"（剩余候选位置 {len(state.feasible)}）")
            calls = 0
            for centre in order_route(self.pos, [p for p, _ in ranked]):
                if self._stop() or state.cleared or calls >= cfg.blind_clear_max_calls:
                    break
                for point in self._grid_cover_points(centre[0], centre[1],
                                                     cell=state.feasible.step):
                    if self._stop() or state.cleared or calls >= cfg.blind_clear_max_calls:
                        break
                    if math.hypot(*point) > ARENA_RADIUS + CLEAR_RADIUS_M:
                        continue
                    calls += 1
                    self._clear_at(point, state.channel)
            if state.cleared:
                self.events.append(f"ch{state.channel}: 盲清除命中")
            else:
                state.abandoned = True

    @staticmethod
    def _grid_cover_points(cx: float, cy: float,
                           cell: float = 120.0, radius: float = CLEAR_RADIUS_M) -> list[tuple[float, float]]:
        """在以 (cx,cy) 为中心、边长 cell 的方格内布点，使格内任意点距某点 ≤ radius。

        取每轴 n = ⌈cell / (radius·√2)⌉ 个点、等距居中排布（点在各子格中心，而不是从中心
        往外按固定步长铺）。这样最坏距离恰为子格半对角 = (cell/n)·√2/2 ≤ radius，点数
        n² 达到方格覆盖的下界量级：边长 100 m、半径 20 m 时 n = 4，16 个点即可保证覆盖；
        原先"从中心按 1.35·radius 步长外扩"要 25 个点且铺到格外，多打 9 次 /clear（27 s）。
        """
        span = radius * math.sqrt(2.0)
        n = max(1, int(math.ceil(cell / span)))
        sub = cell / n
        offsets = [(-cell / 2.0) + sub * (k + 0.5) for k in range(n)]
        points = [(cx + dx, cy + dy) for dx in offsets for dy in offsets]
        points.sort(key=lambda q: math.hypot(q[0] - cx, q[1] - cy))
        return points

    def _home_all(self) -> None:
        """把当前所有有线索的频道推进到清除完成。

        已定位的目标按开放路径最短巡回（最近邻 + 2-opt + Or-opt）挑下一个，而不是纯
        "当前最近者优先"；尚未交会出位置的目标没有可排序的坐标，仍按最近线索处理。
        """
        while not self._stop():
            if self._next_localized() or self._next_seed():
                continue
            return

    def _home_channel(self, state: ChannelState) -> bool:
        """把一个频道推进到清除完成。返回 False 表示放弃。"""
        while not self._stop():
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
                    # 单条示向度时 estimate 恒为 None（_refresh 要求 >=2 条）。
                    # 此处直接 return False 会绕过射线盲扫：真值归因显示 14/17 个
                    # 漏失源走这条路，且可见区非空、理论可清除。消融实测（60 局 d10）
                    # 关闭射线盲扫使漏失 0 -> 9（配对 8 局变差 / 0 局变好, p=0.0056）。
                    if state.bearings:
                        return self._ray_blind_clear(state)
                    return False
                self._piggyback(exclude=state.channel)
                continue

            distance = math.hypot(state.estimate[0] - self.pos[0], state.estimate[1] - self.pos[1])
            if distance <= self.config.clear_distance_m or self._can_clear_directly(state):
                if self._clear_at(state.estimate, state.channel):
                    return True
                # 清除失败 ⟹ 估计偏差 > 20 m。机器狗此刻正**站在估计点上**，原地重测
                # 一次只花 5-6 s、零位移，而近距离处示向度误差引起的横向偏差仅
                # r·tan(1°)（r 已很小），几何会迅速收缩 —— 比立刻围着错误估计螺旋盲撒
                # 点划算得多（螺旋每点 3 s 且可能全落空）。
                before = len(state.bearings)
                self._probe_channel_at(self.pos, state)
                if state.cleared:
                    return True
                if len(state.bearings) > before and state.estimate is not None:
                    continue
                if not self._spiral(state):
                    return False
                return state.cleared
            if not self._step_closer(state, distance):
                return False
            # 逼近途中每停一次车都是一个免费的观测点：顺手补测其它未定位频道。
            self._piggyback(exclude=state.channel)
        return False

    def _can_clear_directly(self, state: ChannelState) -> bool:
        """能否跳过中途重测、直接走到估计位置发 /clear。

        两条判据，前者可证、后者是经验加速：

        * **可证明分支**：定位区域是凸多边形且必含真值（每条示向度的 ±1° 楔形都含
          真值，交会区域是它们的交）。若估计位置到**所有顶点**的距离 ≤ 20 m，则由凸性
          区域内任意点（含真值）到估计位置的距离 ≤ 20 m，`/clear` 必然成功。这正是问题 1
          "定位区域直径 / 覆盖判定"算法在策略里的直接用途。
        * **启发分支**：直径 ≤ `direct_clear_diameter_m` 时也直接去。此时不保证成功，
          但失败只多花 3 s，且随后原地重测（零位移）会迅速收敛，实测净收益为正。

        每命中一次就省掉一轮"中途停车重测"：6 s 检测 + 一段横向绕路。
        """
        if state.estimate is None:
            return False
        if state.cover_radius <= CLEAR_RADIUS_M:
            return True
        return state.diameter <= self.config.direct_clear_diameter_m

    def _dogleg(self, state: ChannelState) -> bool:
        """只有一条示向度时，沿示向度方向前进并带小角度偏移，构造交会基线。

        关键：必须"沿着"示向度前进而不是横向平移。横向平移可能走出有效接收半径
        （实测该错误导致远距离源大量丢失）；沿示向度前进则单调靠近目标，信号只会
        更强，同时用小角度偏移仍能获得足够的交会角。角度从大到小逐级退让，
        最后退化为纯示向度方向直行。
        """
        last = state.bearings[-1]
        cfg = self.config
        length = self._dogleg_length(state)
        for offset in (cfg.dogleg_offset_deg, cfg.dogleg_offset_deg * 0.5, 0.0):
            for sign in self._offset_signs(last.svd_deg, offset, length):
                angle = math.radians(last.svd_deg + sign * offset)
                candidate = (self.pos[0] + length * math.cos(angle),
                             self.pos[1] + length * math.sin(angle))
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

    def _offset_signs(self, bearing_deg: float, offset_deg: float,
                      length: float) -> tuple[float, ...]:
        """dog-leg 偏转方向的尝试次序：优先偏向巡回里下一个航点的那一侧。

        ±offset 两侧在交会几何上完全等价（交会角、前进距离、耗时都一样），是一个**零成本的
        自由度**。dog-leg 基线约 0.65×距离（常有 600-900 m），18° 偏转带来的横向位移约
        0.31×基线 ≈ 200-280 m，选错一侧等于白走两倍这个量。按"哪一侧离下一个航点更近"排序，
        相当于顺手把去下一个目标的路走了一段。
        """
        if offset_deg == 0.0:
            return (1.0,)
        signs = (1.0, -1.0)
        hint = self._next_hint
        if hint is None or not self.config.steer_toward_next:
            return signs

        def candidate(sign: float) -> tuple[float, float]:
            angle = math.radians(bearing_deg + sign * offset_deg)
            return (self.pos[0] + length * math.cos(angle),
                    self.pos[1] + length * math.sin(angle))

        return tuple(sorted(signs, key=lambda s: math.dist(candidate(s), hint)))

    def _dogleg_length(self, state: ChannelState) -> float:
        """dog-leg 基线长度。按"到可行域质心的距离"的一个固定比例取，而不是固定米数。

        交会精度由**相对**基线决定，不是绝对基线：交会角 ≈ (基线/距离)·sin(偏转角)，定位横向
        误差 ≈ 距离·tan(1°)/sin(交会角)。所以固定 380 m 对 1400 m 外的源基线太短（交会角
        ~3.6°，误差 ~390 m，交会不出可直接清除的精度），对 500 m 内的源又白走。改成
        `dogleg_range_fraction × 估计距离` 后，无论远近都能一次交会到"可以直接走过去清除"的
        精度（配合 `direct_clear_diameter_m`），省掉中途的重测停车。

        `dogleg_range_fraction = 0` 时退回固定长度（消融用）。
        """
        cfg = self.config
        if cfg.dogleg_range_fraction <= 0.0 or state.feasible is None:
            return cfg.dogleg_length_m
        centre = state.feasible.centroid()
        if centre is None:
            return cfg.dogleg_length_m
        expected = math.dist(self.pos, centre)
        return min(max(cfg.dogleg_range_fraction * expected, cfg.dogleg_min_length_m),
                   cfg.dogleg_max_length_m)

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
        # 左右两侧的横向偏移在几何上等价（交会角相同、前进距离相同、耗时相同），
        # 所以侧别是一个**零成本的自由度**：选"顺带能排除更多其它频道可行域"的一侧，
        # 让逼近路径同时充当覆盖搜索。这是搭载检测的免费增强，不引入任何额外位移。
        def side_point(sign: float) -> tuple[float, float]:
            return (self.pos[0] + ux * advance - uy * lateral * sign,
                    self.pos[1] + uy * advance + ux * lateral * sign)

        signs = (1.0, -1.0)
        if cfg.steer_toward_next and self._next_hint is not None:
            # 优先偏向巡回里下一个航点的那一侧（零成本自由度，顺手走一段去下一个目标的路）
            signs = tuple(sorted(signs, key=lambda s: math.dist(side_point(s), self._next_hint)))
        elif cfg.piggyback_coverage:
            # 没有下一航点提示时，退而选"顺带能排除更多其它频道可行域"的一侧
            signs = tuple(sorted(signs, key=lambda s: -self._coverage_gain_at(side_point(s))))
        for sign in signs:
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
            if self._stop() or state.clear_attempts >= cfg.max_clear_attempts:
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
        self.measure_calls += 1
        state = self.channels[channel]
        state.measure_count += 1
        if state.bearings:
            state.homing_measures += 1
        return result

    def _clear_at(self, point: tuple[float, float], channel: int) -> bool:
        state = self.channels[channel]
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
