"""既然行进占 85~90% 总时间，检验"站位配置"与"解算顺序"这两个真正的杠杆。

变量:
  A. 扫描站位配置 (环半径/站数) —— 惰性触发
  B. 解算顺序: 就近立即解决(lazy) vs 收集后按最近邻/TSP 顺序解决(batch)
"""
import math, statistics, json
import numpy as np
import policy_bench as pb

CONFIGS = {
    "orig+900x6+1500x6": [(0.0, 1), (900.0, 6), (1500.0, 6)],
    "orig+1100x5":       [(0.0, 1), (1100.0, 5)],
    "orig+1200x6":       [(0.0, 1), (1200.0, 6)],
    "orig+1300x8":       [(0.0, 1), (1300.0, 8)],
    "orig+800x4+1450x8": [(0.0, 1), (800.0, 4), (1450.0, 8)],
    "orig+1250x12":      [(0.0, 1), (1250.0, 12)],
}


def stations_of(cfg):
    out = []
    for rad, k in cfg:
        for j in range(k):
            a = 2 * math.pi * j / max(k, 1)
            out.append(np.array([rad * math.cos(a), rad * math.sin(a)]))
    return out


def order_nn(start, pts):
    """最近邻顺序 + 2-opt。pts: list of (key, point)."""
    rem = list(range(len(pts)))
    cur = np.asarray(start, float)
    order = []
    while rem:
        j = min(rem, key=lambda i: np.linalg.norm(pts[i][1] - cur))
        order.append(j)
        rem.remove(j)
        cur = pts[j][1]
    return order


def run(cfg, mode, n_src, seed, dirfrac):
    """mode: 'lazy' 立即解决 | 'batch' 全部扫完再按最近邻解决"""
    w = pb.World(n_src, seed, dirfrac)
    dog = pb.Dog(w)
    live = [int(c) for c in w.channels]
    pending = {}

    for st in stations_of(cfg):
        if dog.t > pb.MAX_TIME or not any(w.alive):
            break
        dog.move(st)
        for ch in [c for c in live if w.alive[w.ch2idx[c]]]:
            kind, b = dog.detect(ch)
            if kind == "near":
                dog.clear()
            elif kind == "direction":
                if mode == "lazy":
                    pb.resolve_channel(dog, ch, dog.p.copy(), b, "rollout")
                else:
                    pending.setdefault(ch, (dog.p.copy(), b))

    if mode == "batch":
        items = [(ch, v[0], v[1]) for ch, v in pending.items()]
        # 用首观测点近似目标位置排序（真实位置未知）
        pts = [(ch, op + 700 * np.array([math.cos(b), math.sin(b)]))
               for ch, op, b in items]
        for idx in order_nn(dog.p, pts):
            ch, op, b = items[idx]
            if not w.alive[w.ch2idx[ch]] or dog.t > pb.MAX_TIME:
                continue
            pb.resolve_channel(dog, ch, op, b, "rollout")

    # 补扫
    if any(w.alive):
        for rad, k in [(700.0, 5), (1300.0, 8), (1750.0, 10)]:
            if dog.t > pb.MAX_TIME or not any(w.alive):
                break
            for j in range(k):
                if dog.t > pb.MAX_TIME or not any(w.alive):
                    break
                a = 2 * math.pi * j / k
                dog.move(np.array([rad * math.cos(a), rad * math.sin(a)]))
                for ch in [c for c in live if w.alive[w.ch2idx[c]]]:
                    kind, b = dog.detect(ch)
                    if kind == "near":
                        dog.clear()
                    elif kind == "direction":
                        pb.resolve_channel(dog, ch, dog.p.copy(), b, "rollout")

    return dog.t, int(w.alive.sum())


def main():
    print("=" * 100)
    print("站位配置 x 解算顺序 扫描 (每格 30 局: 源数{10,13,16} x seeds1-10, 定向占比 0.3)")
    print("=" * 100)
    print(f"{'配置':>20} {'模式':>6} {'全清率':>7} {'均时(s)':>9} {'中位(s)':>9} {'P90(s)':>9}")
    rows = []
    for name, cfg in CONFIGS.items():
        for mode in ("lazy", "batch"):
            ts, unres = [], []
            for n_src in (10, 13, 16):
                for s in range(1, 11):
                    t, u = run(cfg, mode, n_src, s, 0.3)
                    ts.append(t)
                    unres.append(u)
            ts.sort()
            full = sum(1 for u in unres if u == 0) / len(unres)
            row = dict(cfg=name, mode=mode, full=full, mean=statistics.mean(ts),
                       med=statistics.median(ts), p90=ts[int(0.9 * (len(ts) - 1))])
            rows.append(row)
            print(f"{name:>20} {mode:>6} {full:>7.2f} {row['mean']:>9.0f} "
                  f"{row['med']:>9.0f} {row['p90']:>9.0f}")
    best = min(rows, key=lambda r: (-r["full"], r["mean"]))
    print(f"\n最优: {best['cfg']} / {best['mode']}  全清率={best['full']:.2f} 均时={best['mean']:.0f}s")
    with open("station_sweep.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("-> station_sweep.json")


if __name__ == "__main__":
    main()
