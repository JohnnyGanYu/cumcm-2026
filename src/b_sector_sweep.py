"""环内站位「角度单调顺扇」优化：相位重对齐 + 前瞻延后归航。

## 现状缺陷（baseline，`strategy._walk_ring`）

```python
remaining = self._order_ring(ring)   # 进入本环时排一次序，此后不再重排
while remaining:
    station = remaining.pop(0)
    ...                              # 在该站位扫所有待测频道
    self._home_all()                 # 把机器狗拖到某个源附近（可能离环 1-2 km）
```

两个独立的浪费源，都出在「环上顺扫」与「归航外出」的交界处：

1. **相位失配**：`_order_ring` 的切入点按「进入环那一刻的 self.pos」选。一旦
   `_home_all()` 把机器狗带走，`remaining.pop(0)` 仍按当初排好的顺序取下一站，
   该站可能在环的另一侧 —— 一次约 2R 的跨场弦跳。
2. **贪心外出**：`_home_all()` 只按「当前距离」排序就立刻出发，不看后面的顺扫
   路线是否会自然经过该源附近。对一个角度上还在前方 120° 的源，现在专程往返
   要走 2·d，而顺扫到它那一段时只需绕一小段。

评分指标 `mean_clear_time_s = 总虚拟时间 / 已清除个数`（见 `arena.stats`），
而虚拟时间里行进占 85%~90%，所以上述两项都是直接的分数损失。

## 本模块的两个候选算子

`phase`（角度单调相位重对齐）
    每次归航返回后，保持进入本环时选定的单调转向（顺/逆时针），把顺扫**相位**
    重新对齐到「机器狗当前方位角在该转向上的下一个未访问站位」。
    与「最近邻贪心」的区别：贪心会反向折回并在环上来回横跳（已被证伪，见
    `strategy._order_ring` 注释）；这里方向始终单调，只是不再从旧相位续接。

`defer`（前瞻延后归航）
    归航前先算一次前瞻：`d_now = dist(机器狗, 源估计位置)`，
    `d_future = min over 本环剩余未访问站位 dist(站位, 源估计位置)`。
    若 `d_now > d_future + slack`，说明顺扫本来就会更近地经过它，
    此刻专程往返纯亏 —— 延后。本环走完（remaining 空）时 `d_future = inf`，
    门自动全开，因此**不会漏源**。

`bsector`（信念搜索的角度单调顺扇 —— 真正有效的那一个）
    见 `SectorSweepStrategy._belief_search` 的 docstring。

## 为什么先做的两个算子都无效：分账定位（`src/b_sector_diag.py`，8 局）

|阶段|Q3 时间占比|Q4 时间占比|Q4 移动|
|---|---|---|---|
|原点普扫|2.2%|0.8%|0 m|
|`_walk_ring` 环形补扫|37.7%|35.1%|17,647 m|
|`_home_all` 归航|60.2%|24.2%|16,324 m|
|`_belief_search` 信念搜索|0%|**38.9%**|**25,510 m**|

环形补扫本身已接近最优：Q4 每局 17,647 m，而内环(1250×12)+外环(1750×18)
周长合计 18,850 m —— 顺扫基本只绕了一圈，**没有可回收的跨场弦跳**。
这解释了 `phase`/`defer` 为何都轻微变差（实测 6 局：+1.8% / +0.3%）：
它们在优化一个已经没有冗余的阶段，只是打乱了原有的「发现即就近归航」节奏。

真正的浪费在 `_belief_search`：逐步长统计（6 局 704 步）显示
**84% 的移动来自 60 次超过 1000 m 的跨场跳**（均 2.1 km），
而 p50 步长是 0 m。根因见 `_belief_search` 的 docstring。

用法：

    python -m src.b_sector_sweep --episodes 40 --directional 5 \
        --variants base bsector --label q4

输出 `results/b_sector_sweep_<label>.json`，含逐局配对数据、配对 t 检验与
Wilcoxon 符号秩检验（scipy）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import threading

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import ChannelState, DogStrategy, StrategyConfig

ROBOT_ID = "dog-sector"


class SectorSweepStrategy(DogStrategy):
    """在 `DogStrategy` 上叠加「角度单调相位重对齐」与「前瞻延后归航」。

    只改站位取用顺序与归航时机，不动任何检测/清除/预算/判空逻辑，
    因此清除率的变化只能来自路径调度本身。
    """

    def __init__(self, client, config: StrategyConfig | None = None, *,
                 phase_realign: bool = False,
                 defer_lookahead: bool = False,
                 belief_sector: bool = False,
                 belief_timecost: bool = False,
                 serve_all: bool = False,
                 fan_order: bool = False,
                 defer_slack_m: float = 0.0,
                 gain_weight_s: float = 1.0,
                 fan_k: int = 8) -> None:
        super().__init__(client, config)
        self.phase_realign = phase_realign
        self.defer_lookahead = defer_lookahead
        self.belief_sector = belief_sector
        self.belief_timecost = belief_timecost
        self.serve_all = serve_all
        self.fan_order = fan_order
        self.defer_slack_m = defer_slack_m
        # 「排除一个候选状态」折算多少秒，用于把信息增益与行进时间放到同一量纲
        self.gain_weight_s = gain_weight_s
        # 每轮扇面取多少个高增益探测点。太小退化为原贪心（弦跳依旧），
        # 太大则扇面里塞进低增益点、绕路成本超过顺扫收益。默认 8 经扫描标定。
        self.fan_k = fan_k
        # 诊断计数
        self.phase_shifts = 0
        self.phase_saved_m = 0.0
        self.defer_events = 0
        self.defer_saved_m = 0.0
        self.fan_points = 0
        self.fan_measures = 0
        # 前瞻门禁的作用域：本环剩余未访问站位；None 表示门全开
        self._remaining_stations: list[tuple[float, float]] | None = None

    # ---- 目标位置代理 ----

    def _target_point(self, state: ChannelState) -> tuple[float, float] | None:
        if state.estimate is not None:
            return state.estimate
        if state.bearings:
            last = state.bearings[-1]
            return (last.x, last.y)
        return None

    # ---- 算子 2：前瞻延后归航 ----

    def _defer(self, state: ChannelState) -> bool:
        """True 表示「顺扫后面会更近地经过它」，此刻不该专程往返。"""
        if not self.defer_lookahead:
            return False
        stations = self._remaining_stations
        if not stations:
            return False
        point = self._target_point(state)
        if point is None:
            return False
        d_now = math.hypot(point[0] - self.pos[0], point[1] - self.pos[1])
        d_future = min(math.hypot(point[0] - s[0], point[1] - s[1]) for s in stations)
        if d_now > d_future + self.defer_slack_m:
            self.defer_events += 1
            self.defer_saved_m += d_now - d_future
            return True
        return False

    def _home_all(self) -> None:
        """按前瞻门禁过滤后的就近归航。门全开时与父类行为一致。"""
        while not self._stop():
            targets = [stt for stt in self.channels.values() if stt.active]
            if self.defer_lookahead:
                targets = [stt for stt in targets if not self._defer(stt)]
            if not targets:
                return
            targets.sort(key=lambda stt: self._range_to(stt))
            if not self._home_channel(targets[0]):
                targets[0].exhausted = True

    # ---- 算子 1：角度单调相位重对齐 ----

    @staticmethod
    def _ring_center(ring: list[tuple[float, float]]) -> tuple[float, float]:
        return (sum(p[0] for p in ring) / len(ring),
                sum(p[1] for p in ring) / len(ring))

    def _realign(self, remaining: list[tuple[float, float]],
                 center: tuple[float, float], ccw: bool) -> list[tuple[float, float]]:
        """保持单调转向，把相位对齐到机器狗当前方位角之后的第一个未访问站位。

        `remaining` 已是该转向下的单调序列，只做循环左移（不改相对顺序、不反向）。
        """
        if len(remaining) <= 1:
            return remaining
        cx, cy = center
        theta_dog = math.atan2(self.pos[1] - cy, self.pos[0] - cx)

        def ahead(p: tuple[float, float]) -> float:
            delta = math.atan2(p[1] - cy, p[0] - cx) - theta_dog
            if not ccw:
                delta = -delta
            return delta % (2.0 * math.pi)   # [0, 2π)：沿转向方向的前进量

        idx = min(range(len(remaining)), key=lambda i: ahead(remaining[i]))
        if idx == 0:
            return remaining
        d_old = math.hypot(remaining[0][0] - self.pos[0], remaining[0][1] - self.pos[1])
        d_new = math.hypot(remaining[idx][0] - self.pos[0], remaining[idx][1] - self.pos[1])
        self.phase_shifts += 1
        self.phase_saved_m += max(0.0, d_old - d_new)
        return remaining[idx:] + remaining[:idx]

    @staticmethod
    def _is_ccw(ordered: list[tuple[float, float]],
                center: tuple[float, float]) -> bool:
        """判定 `_order_ring` 给出的序列是顺时针还是逆时针。"""
        if len(ordered) < 2:
            return True
        cx, cy = center
        a = math.atan2(ordered[0][1] - cy, ordered[0][0] - cx)
        b = math.atan2(ordered[1][1] - cy, ordered[1][0] - cx)
        return ((b - a) % (2.0 * math.pi)) <= math.pi

    def _walk_ring(self, ring: list[tuple[float, float]]) -> None:
        remaining = self._order_ring(ring)
        center = self._ring_center(ring)
        ccw = self._is_ccw(remaining, center)
        since_home = 0
        try:
            while remaining and not self._stop():
                todo = [stt for stt in self.channels.values() if self._probe_target(stt)]
                if not todo:
                    break
                if self.phase_realign:
                    remaining = self._realign(remaining, center, ccw)
                station = remaining.pop(0)
                # 前瞻门禁看的是「本环剩余未访问站位」，须在 pop 之后设置
                self._remaining_stations = remaining
                for state in sorted(todo, key=lambda s: s.channel):
                    if self._stop():
                        return
                    self._probe_channel_at(station, state)
                since_home += 1
                ready = any(stt.active for stt in self.channels.values())
                if ready and (since_home >= self.config.home_batch_stations or not remaining):
                    self._home_all()
                    since_home = 0
                if self._stop():
                    return
                self._revive_nearby()
        finally:
            # 本环结束：门全开，保证延后的目标一定被处理，不漏源
            self._remaining_stations = None
        if not self._stop():
            self._home_all()

    # ---- 算子 3：信念搜索的角度单调顺扇 ----

    def _belief_search(self, max_rounds: int = 60) -> None:
        """把「按候选数排序、每轮换一个频道」改成「按方位角单调顺扇、就近服务」。

        ## 基线缺陷（`strategy._belief_search`）

        ```python
        todo.sort(key=lambda st: len(st.belief.candidates))  # 候选最少的频道优先
        for state in todo:
            probe = state.belief.best_probe_point(self.pos)
            self._probe_channel_at(probe, state)
            if 有进展: break                                 # 换回外层，重新排序
        ```

        排序键是**候选集合大小**，与机器狗当前位置无关。多个沉默频道的候选区往往
        分布在区域相对的两侧，于是每轮 `break` 回去重排后，下一个被选中的频道
        经常在场地对面 —— 产生一次跨场长跳。

        实测（`src/b_sector_diag.py` + 逐步长统计，6 局 704 步）：
        信念搜索占 Q4 总时的 **38.9%**、移动 25,510 m/局；
        其中 **84% 的移动来自 60 次 >1000 m 的跨场跳**（均 2.1 km），
        而步长中位数是 0 m（同一探测点连续换频道检测，不移动）。
        探测点平均半径 1473 m —— 它们本质上就是贴边界的环上站位，
        所以「环内站位角度单调顺扇」这条思路用在这里才对症。

        ## 关键观测：所有沉默频道共享同一个 belief

        打桩（seed 6000，`_belief_search` 入口）实测：9 个待办频道的候选集合
        **完全相同**（各 141 个候选，distinct belief states = 1），
        `best_probe_point` 对 9 个频道返回**同一个点**。

        原因：候选集合只由「在哪些点检测过、结果是否沉默」决定
        （见 `DirectionalBelief.apply_silence`），而普扫/环形补扫对每个沉默频道
        走的是同一组站位，因此沉默历史逐字相同 ⇒ 约束相同 ⇒ 候选相同。

        所以跨场跳**不是频道间乒乓**，而是「信息贪心点序列」本身：
        `best_probe_point` 的评分是 `gain - d/500*0.05 = gain - d*1e-4`，
        一次 2000 m 的跨场跳只被罚 0.2，而 gain 上界是 1.0 ——
        距离项被严重低估（2000 m 行进 = 400 s ≡ 80 次检测）。

        ## 本实现：K 点扇面 + 角度单调顺扇 + 顺手全测

        每轮：
        1. **取扇面**：从共享 belief 的探测点池（`_probe_candidates`）里按
           「若无信号可排除的候选数」取增益最高的 `K` 个点。
           这保留了信息贪心的选点质量，但一次取 K 个而不是 1 个。
        2. **角度单调排序**：把这 K 个点按方位角 `atan2(y,x)` 排成单调序列，
           从离机器狗当前方位最近的点切入，比较顺/逆时针两个转向的巡回长度取短者，
           此后方向不反转 —— 消除「打一个点就横穿场地去下一个」的弦跳。
        3. **顺手全测**：每到一个点，把所有仍沉默且该点对其有正信息增益的频道
           一并检测（增益判据与 `apply_silence` 完全一致，见 `_informative`）。
           移动成本已付，一次检测仅 5 s ≡ 25 m 行进。
        4. 拿到示向度即跳出重建扇面（候选结构已变，且该让归航接手）。

        信息利用与基线同源（同一探测点池、同一约束更新），只改**访问顺序**与
        **每轮取点数**，所以清除率不受影响。
        """
        if self.belief_timecost:
            return self._belief_search_timecost(max_rounds)
        if not self.belief_sector:
            return super()._belief_search(max_rounds)
        for _ in range(max_rounds):
            if self._stop():
                return
            todo = [
                stt for stt in self.channels.values()
                if not stt.cleared and not stt.bearings
                and stt.belief is not None and stt.belief.alive
            ]
            if not todo:
                return

            # 1) 取扇面：以候选集合最小的频道为代表 belief（它们通常完全相同）
            lead = min(todo, key=lambda stt: len(stt.belief.candidates))  # type: ignore[union-attr]
            fan = self._top_gain_points(lead, self.fan_k)
            if not fan:
                return

            # 2) 角度单调排序 + 选较短转向
            fan = self._angle_monotone(fan)

            # 3) 沿扇面顺序访问。`progressed` 的语义必须与基线一致 ——
            # 「本轮是否做了任何有效检测」，而不是「是否拿到示向度」。
            # 一旦收紧成后者，扇面走完一圈没拿到示向度就会 return，
            # 使 belief 候选集合远未收敛就交给 `_blind_clear_sweep` 蛮力清除
            # （实测该 bug 让 T̄ 从 1256 s 爆到 19903 s，+1485%）。
            progressed = False
            got_bearing = False
            for probe in fan:
                if self._stop():
                    return
                served = 0
                for state in sorted(self.channels.values(), key=lambda s: s.channel):
                    if self._stop():
                        return
                    if (state.cleared or state.bearings or state.belief is None
                            or not state.belief.alive):
                        continue
                    if not self._informative(state, probe):
                        continue
                    before = len(state.bearings)
                    self._probe_channel_at(probe, state)
                    served += 1
                    progressed = True
                    self.fan_measures += 1
                    if state.cleared or len(state.bearings) > before:
                        got_bearing = True
                if served:
                    self.fan_points += 1
                if got_bearing:
                    break
            if not progressed:
                return

    def _top_gain_points(self, state, k: int,
                         pool_limit: int = 260) -> list[tuple[float, float]]:
        """从探测点池里取「若无信号可排除候选数」最高的 k 个点。

        与 `DirectionalBelief.best_probe_point` 用同一个池与同一个增益定义，
        差别只是取 top-k 而非 argmax，且**不做距离折价** ——
        距离由后续的角度单调排序统一处理，避免贪心把距离项算错。
        """
        from .b_sim.directional import RECEPTION_MIN_M, in_beam

        belief = state.belief
        if belief is None:
            return []
        probes = belief._probe_candidates(pool_limit)
        if not probes:
            return []
        scored: list[tuple[int, tuple[float, float]]] = []
        cand = list(belief.candidates)
        for px, py in probes:
            gain = 0
            for pi, hi, _ in cand:
                sx, sy = belief.positions[pi]
                if math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
                        sx, sy, belief.headings[hi], px, py):
                    gain += 1
            if gain:
                scored.append((gain, (px, py)))
        if not scored:
            return []
        scored.sort(key=lambda item: -item[0])
        return [p for _, p in scored[:max(1, k)]]

    def _angle_monotone(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """按方位角排成单调序列，从最近点切入，取较短转向。"""
        if len(pts) <= 2:
            return pts
        by_angle = sorted(pts, key=lambda p: math.atan2(p[1], p[0]))
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

    def _informative(self, state, probe: tuple[float, float]) -> bool:
        """该探测点若返回无信号，能否排除 `state` 的至少一个候选。

        判据与 `DirectionalBelief.apply_silence` 完全一致（距离 <= 接收半径下限
        且探测点落在候选扇区内），所以「顺手扫」扫到的都是真有信息量的检测，
        不会退化成盲目多测。为控算力只抽样部分候选。
        """
        from .b_sim.directional import RECEPTION_MIN_M, in_beam

        belief = state.belief
        if belief is None:
            return False
        px, py = probe
        for k, (pi, hi, _) in enumerate(belief.candidates):
            if k >= 400:
                break
            sx, sy = belief.positions[pi]
            if math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
                    sx, sy, belief.headings[hi], px, py):
                return True
        return False


    # ---- 算子 4：探测点选择的距离折价改按真实时间成本定标 ----

    def _probe_point_timecost(self, state) -> tuple[float, float] | None:
        """替代 `best_probe_point`：把「信息增益」与「行进时间」折算到同一量纲。

        基线评分（`DirectionalBelief.best_probe_point`）：

            score = gain - d/500*0.05 = gain - d*1e-4        # gain ∈ [0,1]

        量纲问题：一次 2000 m 的跨场跳只被罚 0.2，而 gain 上界是 1.0，
        于是「候选排除比例高一点」轻易压过「多走 2 km」。但按题面成本表，
        2000 m 行进 = 400 s，而一次检测只要 5 s —— 相当于 80 次检测的代价。

        本实现改为直接比较**秒**：

            score_s = gain * candidates * seconds_per_candidate - d / SPEED

        其中 `seconds_per_candidate` 由 `--gain-weight` 给出，
        含义是「排除一个候选值多少秒」。基线等价于该值约
        `1e-4 * SPEED / |candidates| ≈ 极大`，即几乎不看距离。
        """
        from .b_sim.arena import SPEED_MPS
        from .b_sim.directional import RECEPTION_MIN_M, in_beam

        belief = state.belief
        if belief is None or not belief.candidates:
            return None
        probes = belief._probe_candidates(260)
        if not probes:
            return None
        cand = list(belief.candidates)
        best_score, best_point = -1e18, None
        for px, py in probes:
            gain = 0
            for pi, hi, _ in cand:
                sx, sy = belief.positions[pi]
                if math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
                        sx, sy, belief.headings[hi], px, py):
                    gain += 1
            if not gain:
                continue
            d = math.hypot(px - self.pos[0], py - self.pos[1])
            score = gain * self.gain_weight_s - d / SPEED_MPS
            if score > best_score:
                best_score, best_point = score, (px, py)
        return best_point

    def _topk_timecost_points(self, state, k: int) -> list[tuple[float, float]]:
        """按时间定标评分取前 k 个探测点（同 `_probe_point_timecost` 的评分）。"""
        from .b_sim.arena import SPEED_MPS
        from .b_sim.directional import RECEPTION_MIN_M, in_beam

        belief = state.belief
        if belief is None or not belief.candidates:
            return []
        probes = belief._probe_candidates(260)
        if not probes:
            return []
        cand = list(belief.candidates)
        scored: list[tuple[float, tuple[float, float]]] = []
        for px, py in probes:
            gain = 0
            for pi, hi, _ in cand:
                sx, sy = belief.positions[pi]
                if math.hypot(px - sx, py - sy) <= RECEPTION_MIN_M and in_beam(
                        sx, sy, belief.headings[hi], px, py):
                    gain += 1
            if not gain:
                continue
            d = math.hypot(px - self.pos[0], py - self.pos[1])
            scored.append((gain * self.gain_weight_s - d / SPEED_MPS, (px, py)))
        if not scored:
            return []
        scored.sort(key=lambda item: -item[0])
        return [p for _, p in scored[:max(1, k)]]

    def _belief_search_fan(self, max_rounds: int = 60) -> None:
        """时间定标选点 + 角度单调顺扇（本任务的「环内站位角度单调顺扇」落点）。

        每轮取时间定标评分最高的 K 个探测点，按方位角排成单调序列、选较短转向后
        依次访问；每到一点把所有该点有信息增益的沉默频道一并检测。
        目的是消化 `bcost` 之后仍剩的跨场跳（实测 36 次 / 48% 的信念阶段移动）。
        """
        for _ in range(max_rounds):
            if self._stop():
                return
            todo = [
                stt for stt in self.channels.values()
                if not stt.cleared and not stt.bearings
                and stt.belief is not None and stt.belief.alive
            ]
            if not todo:
                return
            lead = min(todo, key=lambda stt: len(stt.belief.candidates))  # type: ignore[union-attr]
            fan = self._angle_monotone(self._topk_timecost_points(lead, self.fan_k))
            if not fan:
                return
            progressed = False
            got = False
            for probe in fan:
                if self._stop():
                    return
                for state in sorted(self.channels.values(), key=lambda s: s.channel):
                    if self._stop():
                        return
                    if (state.cleared or state.bearings or state.belief is None
                            or not state.belief.alive):
                        continue
                    if not self._informative(state, probe):
                        continue
                    before = len(state.bearings)
                    self._probe_channel_at(probe, state)
                    progressed = True
                    self.fan_measures += 1
                    if state.cleared or len(state.bearings) > before:
                        got = True
                self.fan_points += 1
                if got:
                    break
            if not progressed:
                return

    def _belief_search_timecost(self, max_rounds: int = 60) -> None:
        """与基线 `_belief_search` 结构完全一致，只把选点换成时间定标版。"""
        if self.fan_order:
            return self._belief_search_fan(max_rounds)
        for _ in range(max_rounds):
            if self._stop():
                return
            todo = [
                stt for stt in self.channels.values()
                if not stt.cleared and not stt.bearings
                and stt.belief is not None and stt.belief.alive
            ]
            if not todo:
                return
            todo.sort(key=lambda stt: len(stt.belief.candidates))  # type: ignore[union-attr]
            progressed = False
            for state in todo:
                if self._stop():
                    return
                probe = self._probe_point_timecost(state)
                if probe is None:
                    continue
                before = len(state.bearings)
                self._probe_channel_at(probe, state)
                self.fan_measures += 1
                self.fan_points += 1
                got = state.cleared or len(state.bearings) > before
                progressed = True
                if self.serve_all and not self._stop():
                    # 顺手全测：移动成本已付（该点可能离上一位置 1-2 km），
                    # 再测一个频道只要 5 s ≡ 25 m 行进。只测该点确有信息增益的频道
                    # （判据同 apply_silence），因此不是盲目多测。
                    for other in sorted(self.channels.values(), key=lambda s: s.channel):
                        if self._stop():
                            break
                        if (other is state or other.cleared or other.bearings
                                or other.belief is None or not other.belief.alive):
                            continue
                        if not self._informative(other, probe):
                            continue
                        b2 = len(other.bearings)
                        self._probe_channel_at(probe, other)
                        self.fan_measures += 1
                        if other.cleared or len(other.bearings) > b2:
                            got = True
                if got:
                    break
            if not progressed:
                return


VARIANTS = {
    "base":    dict(phase_realign=False, defer_lookahead=False, belief_sector=False),
    "phase":   dict(phase_realign=True,  defer_lookahead=False, belief_sector=False),
    "defer":   dict(phase_realign=False, defer_lookahead=True,  belief_sector=False),
    "sector":  dict(phase_realign=True,  defer_lookahead=True,  belief_sector=False),
    "bsector": dict(phase_realign=False, defer_lookahead=False, belief_sector=True),
    "bcost":   dict(phase_realign=False, defer_lookahead=False, belief_sector=False,
                    belief_timecost=True),
    # 最终方案：时间定标选点 + 顺手全测（同点把所有有增益的沉默频道一并检测）
    "bcost2":  dict(phase_realign=False, defer_lookahead=False, belief_sector=False,
                    belief_timecost=True, serve_all=True),
    # 最终方案：时间定标选点 + 角度单调顺扇（在 bcost 之上再消除残余跨场跳）
    "final":   dict(phase_realign=False, defer_lookahead=False, belief_sector=False,
                    belief_timecost=True, serve_all=True, fan_order=True),
    "all":     dict(phase_realign=True,  defer_lookahead=True,  belief_sector=True),
}


def run_one(seed: int, *, directional: int, variant: str,
            count: int | None = None, defer_slack_m: float = 0.0,
            fan_k: int = 8, gain_weight_s: float = 1.0,
            real_limit_s: float = 1200.0) -> dict:
    case = generate_case(seed, count=count, directional_count=directional)
    arena = Arena(case)
    session = Session(ROBOT_ID, arena, real_limit_s=real_limit_s)
    httpd = make_server(session, port=0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional > 0)
    extra: dict = {}
    try:
        client = RobotClient(ROBOT_ID, base_url=f"http://127.0.0.1:{port}")
        kw = VARIANTS[variant]
        strat = SectorSweepStrategy(client, cfg, defer_slack_m=defer_slack_m,
                                    fan_k=fan_k, gain_weight_s=gain_weight_s, **kw)
        strat.run()
        extra = {"phase_shifts": strat.phase_shifts,
                 "phase_saved_m": round(strat.phase_saved_m),
                 "defer_events": strat.defer_events,
                 "defer_saved_m": round(strat.defer_saved_m),
                 "fan_points": strat.fan_points,
                 "fan_measures": strat.fan_measures}
    finally:
        httpd.shutdown()
        httpd.server_close()
    s = arena.stats()
    return {"seed": seed, "variant": variant, "total": s["total_jammers"],
            "cleared": s["cleared"], "ratio": s["cleared_ratio"],
            "virtual_s": s["total_virtual_time_s"],
            "mean_clear_time_s": s["mean_clear_time_s"],
            "measures": s["measure_calls"], "clears": s["clear_calls"],
            "move_m": s["move_distance_m"],
            "missed": [r["channel"] for r in case.truth_table() if not r["cleared"]],
            **extra}


def aggregate(rows: list[dict], variant: str) -> dict:
    rs = [r for r in rows if r["variant"] == variant]
    mt = [r["mean_clear_time_s"] for r in rs if r["mean_clear_time_s"]]
    return {
        "variant": variant,
        "episodes": len(rs),
        "full_clear_rate": round(sum(1 for r in rs if r["ratio"] >= 1 - 1e-9) / len(rs), 4) if rs else 0.0,
        "clear_rate": round(sum(r["cleared"] for r in rs) / sum(r["total"] for r in rs), 4) if rs else 0.0,
        "total_missed": sum(len(r["missed"]) for r in rs),
        "mean_clear_time_s": round(st.fmean(mt), 1) if mt else None,
        "move_m_mean": round(st.fmean(r["move_m"] for r in rs)) if rs else None,
        "measures_mean": round(st.fmean(r["measures"] for r in rs), 1) if rs else None,
        "virtual_s_mean": round(st.fmean(r["virtual_s"] for r in rs), 1) if rs else None,
    }


def paired_test(rows: list[dict], base: str, cand: str) -> dict:
    """对 `mean_clear_time_s` 做配对 t 检验 + Wilcoxon 符号秩检验。"""
    from scipy import stats as sps

    by_seed: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["variant"]] = r
    pairs = [(p[base], p[cand]) for p in by_seed.values()
             if base in p and cand in p
             and p[base]["mean_clear_time_s"] and p[cand]["mean_clear_time_s"]]
    if len(pairs) < 2:
        return {"comparison": f"{cand} vs {base}", "pairs": len(pairs),
                "error": "配对样本不足"}
    b = [x["mean_clear_time_s"] for x, _ in pairs]
    c = [y["mean_clear_time_s"] for _, y in pairs]
    d = [ci - bi for bi, ci in zip(b, c)]
    n = len(d)
    mean_d = st.fmean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n) if sd else 0.0
    tt = sps.ttest_rel(c, b)
    try:
        wx = sps.wilcoxon(c, b)
        w_p = float(wx.pvalue)
        w_stat = float(wx.statistic)
    except ValueError as exc:          # 全为零差时 wilcoxon 会拒绝
        w_p, w_stat = None, None
        _ = exc
    # 相对降幅（以配对 baseline 均值为分母，与题目「降低 15%」口径一致）
    base_mean = st.fmean(b)
    rel = mean_d / base_mean if base_mean else None
    # 相对降幅的 95% CI（对 d 的 CI 除以 baseline 均值）
    ci_lo, ci_hi = (mean_d - 1.96 * se, mean_d + 1.96 * se) if se else (mean_d, mean_d)
    # 清除率配对差
    d_cleared = sum(y["cleared"] - x["cleared"] for x, y in pairs)
    return {
        "comparison": f"{cand} vs {base}",
        "pairs": n,
        "baseline_mean_clear_time_s": round(base_mean, 1),
        "candidate_mean_clear_time_s": round(st.fmean(c), 1),
        "d_mean_s": round(mean_d, 1),
        "d_median_s": round(st.median(d), 1),
        "d_sd_s": round(sd, 1),
        "rel_change": round(rel, 4) if rel is not None else None,
        "rel_change_ci95": [round(ci_lo / base_mean, 4), round(ci_hi / base_mean, 4)] if base_mean else None,
        "d_ci95_s": [round(ci_lo, 1), round(ci_hi, 1)],
        "t_stat": round(float(tt.statistic), 3),
        "t_pvalue": float(tt.pvalue),
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_p,
        "better": sum(1 for x in d if x < -1),
        "worse": sum(1 for x in d if x > 1),
        "tie": sum(1 for x in d if abs(x) <= 1),
        "net_extra_cleared": d_cleared,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="环内站位角度单调顺扇 A/B 配对实验")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--directional", type=int, default=5)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--variants", nargs="+", default=["base", "sector"],
                    choices=sorted(VARIANTS))
    ap.add_argument("--defer-slack", type=float, default=0.0,
                    help="前瞻延后的松弛量（米），越大越倾向立刻归航")
    ap.add_argument("--fan-k", type=int, default=8,
                    help="信念搜索每轮扇面取的高增益探测点个数")
    ap.add_argument("--gain-weight", type=float, default=1.0,
                    help="bcost 变体：排除一个候选状态折算多少秒")
    ap.add_argument("--label", default="q4")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    if args.count is not None and args.directional > args.count:
        ap.error("--directional 不能超过 --count")

    rows: list[dict] = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        line = [f"seed={seed}"]
        for variant in args.variants:
            r = run_one(seed, directional=args.directional, variant=variant,
                        count=args.count, defer_slack_m=args.defer_slack,
                        fan_k=args.fan_k, gain_weight_s=args.gain_weight)
            rows.append(r)
            mt = r["mean_clear_time_s"]
            line.append(f"{variant}: {r['cleared']}/{r['total']} "
                        f"T={'-' if mt is None else round(mt)} "
                        f"move={r['move_m']:.0f}")
        print(" | ".join(line), flush=True)

    summary = {v: aggregate(rows, v) for v in args.variants}
    tests = []
    base = args.variants[0]
    for v in args.variants[1:]:
        tests.append(paired_test(rows, base, v))
    report = {
        "config": {"episodes": args.episodes, "seed_start": args.seed_start,
                   "directional": args.directional, "count": args.count,
                   "variants": args.variants, "defer_slack_m": args.defer_slack,
                   "fan_k": args.fan_k, "gain_weight_s": args.gain_weight},
        "summary": summary,
        "paired_tests": tests,
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    out = f"{args.out_dir}/b_sector_sweep_{args.label}.json"
    with open(out, "w", encoding="utf-8") as h:
        json.dump({"report": report, "rows": rows}, h, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
