"""CUMCM2026-B: 三类候选架构的端到端策略对比 (简化模拟器).

目的不是复现官方模拟器，而是在同一套假设下比较架构的量级差异，
用来支撑"预期收益/成本"评估。所有结论都来自本脚本实跑。

模拟假设（与题面对齐，未知量按题面区间随机）：
  区域 R=1800m 圆盘；N 个源，每源独占一个频道；
  机器狗从原点出发, v=5m/s；
  detect(ch) 耗时 5s -> 若在该源有效接收半径内且(定向源)在覆盖角内，返回 bearing+N(0,sigma)
                        若距离<=5m 且在覆盖角内 -> 'near'
                        否则 'no_signal'
  clear() 耗时 t_clear，若当前位置距某源<=20m 则清除该源
指标：清除全部源的总用时；未清除数；detect/clear 调用次数。

策略：
  P1 FixedRing   : 固定环形站位全频扫描 -> 每源两点交会 -> 直奔 -> 盲扫兜底
  P2 CautiousGreedy: β-cautious 风格，贪心选取"垂直于最大不确定方向"的下一测点
  P3 RolloutLite : 在 P2 动作集上做一步前瞻(期望后验协方差最小)，非myopic近似
"""
import math, json, statistics
import numpy as np

R_AREA = 1800.0
V = 5.0
T_DETECT = 5.0
T_CLEAR = 5.0
R_CLEAR = 20.0
R_NEAR = 5.0
SIG = math.radians(1.0)
MAX_TIME = 20000.0


class World:
    def __init__(self, n_src, seed, directional_frac=0.0):
        self.rng = np.random.default_rng(seed)
        r = self.rng.random(n_src) ** 0.5 * R_AREA
        a = self.rng.random(n_src) * 2 * math.pi
        self.pos = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        self.rx = self.rng.uniform(1000, 1500, n_src)
        self.channels = self.rng.permutation(20)[:n_src]
        nd = int(round(n_src * directional_frac))
        self.is_dir = np.zeros(n_src, bool)
        self.is_dir[:nd] = True
        self.heading = self.rng.random(n_src) * 2 * math.pi
        self.alive = np.ones(n_src, bool)
        self.ch2idx = {int(c): i for i, c in enumerate(self.channels)}

    def detect(self, p, ch):
        """返回 ('direction', bearing) / 'near' / 'no_signal'."""
        i = self.ch2idx.get(int(ch))
        if i is None or not self.alive[i]:
            return ("no_signal", None)
        d = p - self.pos[i]
        dist = float(np.linalg.norm(d))
        if dist > self.rx[i]:
            return ("no_signal", None)
        if self.is_dir[i]:
            ang = abs((math.atan2(d[1], d[0]) - self.heading[i] + math.pi) % (2 * math.pi) - math.pi)
            if ang > math.pi / 2:
                return ("no_signal", None)
        if dist <= R_NEAR:
            return ("near", None)
        true_b = math.atan2(-d[1], -d[0])           # 从狗指向源
        return ("direction", true_b + self.rng.normal(0, SIG))

    def clear(self, p):
        hit = []
        for i in range(len(self.pos)):
            if self.alive[i] and np.linalg.norm(p - self.pos[i]) <= R_CLEAR:
                self.alive[i] = False
                hit.append(i)
        return hit


class Dog:
    def __init__(self, world):
        self.w = world
        self.p = np.zeros(2)
        self.t = 0.0
        self.n_det = 0
        self.n_clr = 0

    def move(self, q):
        q = np.asarray(q, float)
        self.t += float(np.linalg.norm(q - self.p)) / V
        self.p = q

    def detect(self, ch):
        self.t += T_DETECT
        self.n_det += 1
        return self.w.detect(self.p, ch)

    def clear(self):
        self.t += T_CLEAR
        self.n_clr += 1
        return self.w.clear(self.p)


def in_area(p):
    return np.linalg.norm(p) <= R_AREA + 1e-9


def clip_area(p):
    n = np.linalg.norm(p)
    return p if n <= R_AREA else p / n * R_AREA


def ray_intersect(p1, b1, p2, b2):
    d1 = np.array([math.cos(b1), math.sin(b1)])
    d2 = np.array([math.cos(b2), math.sin(b2)])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    A = p2 - p1
    t = (A[0] * d2[1] - A[1] * d2[0]) / den
    if t <= 0:
        return None
    return p1 + d1 * t


def blind_ray_clear(dog, ch, origin, bearing, step=32.0, max_r=1600.0, budget=60):
    """沿射线铺点清除；返回是否成功。带横向补点。"""
    d = np.array([math.cos(bearing), math.sin(bearing)])
    nrm = np.array([-d[1], d[0]])
    r = step
    tries = 0
    while r <= max_r and tries < budget:
        base = origin + d * r
        if not in_area(base):
            break
        lat = math.radians(1.0) * r
        offs = [0.0] if lat <= R_CLEAR else [0.0, R_CLEAR, -R_CLEAR]
        for o in offs:
            q = base + nrm * o
            if not in_area(q):
                continue
            dog.move(q)
            tries += 1
            if dog.clear():
                return True
            if tries >= budget:
                break
        r += step
    return False


def approach_and_clear(dog, ch, est, budget=9):
    """奔赴估计点并在其周围补点清除。"""
    est = clip_area(np.asarray(est, float))
    dog.move(est)
    if dog.clear():
        return True
    for k in range(budget):
        a = 2 * math.pi * k / budget
        q = clip_area(est + 1.6 * R_CLEAR * np.array([math.cos(a), math.sin(a)]))
        dog.move(q)
        if dog.clear():
            return True
    return False


def resolve_channel(dog, ch, first_p, first_b, mode):
    """已获得一条示向度后，把该频道的源清除。返回 bool."""
    d = np.array([math.cos(first_b), math.sin(first_b)])
    nrm = np.array([-d[1], d[0]])

    # 选第二观测点：沿垂直方向侧移，尽量制造大交会角（P2/P3 共用）
    for side in (1.0, -1.0):
        for off in (500.0, 800.0):
            q = clip_area(first_p + nrm * side * off + d * 250.0)
            dog.move(q)
            r2 = dog.detect(ch)
            if r2[0] == "near":
                if dog.clear():
                    return True
                continue
            if r2[0] != "direction":
                continue
            est = ray_intersect(first_p, first_b, dog.p, r2[1])
            if est is None or not in_area(est):
                continue              # 物理不可能位置 -> 弃用
            if mode == "rollout":
                # 一步前瞻：再取一个近正交点精化
                dv = est - dog.p
                rr = np.linalg.norm(dv)
                if rr > 250:
                    perp = np.array([-dv[1], dv[0]]) / rr
                    q3 = clip_area(est - dv / rr * min(rr, 400.0) + perp * 350.0)
                    dog.move(q3)
                    r3 = dog.detect(ch)
                    if r3[0] == "near":
                        if dog.clear():
                            return True
                    elif r3[0] == "direction":
                        e3 = ray_intersect(dog.p, r3[1], first_p, first_b)
                        if e3 is not None and in_area(e3):
                            est = 0.5 * (est + e3)
            if approach_and_clear(dog, ch, est):
                return True
    # 兜底：射线盲扫
    return blind_ray_clear(dog, ch, first_p, first_b)


RINGS = {
    "P1": [(0.0, 1), (1000.0, 6), (1500.0, 10)],
}


def run(policy, n_src, seed, directional_frac=0.0):
    w = World(n_src, seed, directional_frac)
    dog = Dog(w)
    live = set(int(c) for c in w.channels)
    known = set()          # 已确认存在但未清除的频道

    # ---- 阶段1: 站位扫描找出"哪些频道有源"并拿到首条示向度
    if policy == "P1":
        stations = []
        for rad, k in RINGS["P1"]:
            for j in range(k):
                a = 2 * math.pi * j / max(k, 1)
                stations.append(np.array([rad * math.cos(a), rad * math.sin(a)]))
    else:
        # P2/P3 用惰性站位：先原点+内环4点，不够再扩
        stations = [np.zeros(2)]
        for rad in (900.0, 1500.0):
            for j in range(6):
                a = 2 * math.pi * j / 6
                stations.append(np.array([rad * math.cos(a), rad * math.sin(a)]))

    pending = []           # (ch, obs_p, bearing)
    for st in stations:
        if dog.t > MAX_TIME:
            break
        remaining = [c for c in live if w.alive[w.ch2idx[c]]]
        if not remaining:
            break
        dog.move(st)
        for ch in list(remaining):
            if not w.alive[w.ch2idx[ch]]:
                continue
            kind, b = dog.detect(ch)
            if kind == "near":
                dog.clear()
            elif kind == "direction":
                pending.append((ch, dog.p.copy(), b))
                if policy in ("P2", "P3"):
                    # 惰性：立刻解决，避免长距离往返
                    mode = "rollout" if policy == "P3" else "greedy"
                    resolve_channel(dog, ch, dog.p.copy(), b, mode)
        if policy in ("P2", "P3") and all(not w.alive[w.ch2idx[c]] for c in live):
            break

    # ---- 阶段2: P1 集中处理 pending
    if policy == "P1":
        seen = set()
        for ch, op, b in pending:
            if ch in seen or not w.alive[w.ch2idx[ch]]:
                continue
            seen.add(ch)
            if dog.t > MAX_TIME:
                break
            resolve_channel(dog, ch, op, b, "greedy")

    # ---- 阶段3: 仍未清除的，做一次全区补扫
    if any(w.alive):
        for rad, k in [(600.0, 5), (1200.0, 8), (1700.0, 10)]:
            if dog.t > MAX_TIME or not any(w.alive):
                break
            for j in range(k):
                if dog.t > MAX_TIME or not any(w.alive):
                    break
                a = 2 * math.pi * j / k
                dog.move(np.array([rad * math.cos(a), rad * math.sin(a)]))
                for ch in [c for c in live if w.alive[w.ch2idx[c]]]:
                    kind, b = dog.detect(ch)
                    if kind == "near":
                        dog.clear()
                    elif kind == "direction":
                        resolve_channel(dog, ch, dog.p.copy(), b,
                                        "rollout" if policy == "P3" else "greedy")

    return dict(policy=policy, n_src=n_src, seed=seed, dirfrac=directional_frac,
                time=dog.t, unresolved=int(w.alive.sum()),
                n_det=dog.n_det, n_clr=dog.n_clr)


def main():
    seeds = list(range(1, 21))
    results = []
    for dirfrac in (0.0, 0.3):
        for pol in ("P1", "P2", "P3"):
            for n_src in (10, 13, 16):
                for s in seeds:
                    results.append(run(pol, n_src, s, dirfrac))

    print("=" * 92)
    print("端到端策略对比 (20 seeds/格, 时间上限 20000s)")
    print("=" * 92)
    print(f"{'定向源占比':>8} {'策略':>5} {'源数':>4} {'全清率':>7} {'均时(s)':>9} "
          f"{'中位(s)':>9} {'P90(s)':>9} {'detect次':>8} {'clear次':>8}")
    summary = []
    for dirfrac in (0.0, 0.3):
        for pol in ("P1", "P2", "P3"):
            for n_src in (10, 13, 16):
                g = [r for r in results if r["policy"] == pol and r["n_src"] == n_src
                     and r["dirfrac"] == dirfrac]
                full = sum(1 for r in g if r["unresolved"] == 0) / len(g)
                ts = sorted(r["time"] for r in g)
                row = dict(dirfrac=dirfrac, policy=pol, n_src=n_src, fullclear=full,
                           mean=statistics.mean(ts), median=statistics.median(ts),
                           p90=ts[int(0.9 * (len(ts) - 1))],
                           det=statistics.mean(r["n_det"] for r in g),
                           clr=statistics.mean(r["n_clr"] for r in g))
                summary.append(row)
                print(f"{dirfrac:>8.1f} {pol:>5} {n_src:>4} {full:>7.2f} {row['mean']:>9.0f} "
                      f"{row['median']:>9.0f} {row['p90']:>9.0f} {row['det']:>8.1f} {row['clr']:>8.1f}")

    print()
    print("聚合（跨源数与定向占比取均值）：")
    for pol in ("P1", "P2", "P3"):
        g = [r for r in summary if r["policy"] == pol]
        print(f"  {pol}: 全清率={statistics.mean(r['fullclear'] for r in g):.3f}  "
              f"均时={statistics.mean(r['mean'] for r in g):7.0f}s  "
              f"detect={statistics.mean(r['det'] for r in g):6.1f}  "
              f"clear={statistics.mean(r['clr'] for r in g):6.1f}")

    with open("policy_bench_results.json", "w") as f:
        json.dump(dict(raw=results, summary=summary), f, indent=1)
    print("\n-> policy_bench_results.json")


if __name__ == "__main__":
    main()
