"""时间去向拆解：行进 / 检测 / 清除 / 盲扫兜底 各占多少。

复用 policy_bench 的世界与策略，仅加计时分账，用来判断
"投资哪一个模块最划算"。
"""
import math, statistics, json
import numpy as np
import policy_bench as pb


class Acct:
    def __init__(self):
        self.move = 0.0
        self.detect = 0.0
        self.clear = 0.0
        self.blind_move = 0.0
        self.blind_clear = 0.0
        self.in_blind = False


ACCT = Acct()


class DogA(pb.Dog):
    def move(self, q):
        q = np.asarray(q, float)
        dt = float(np.linalg.norm(q - self.p)) / pb.V
        if ACCT.in_blind:
            ACCT.blind_move += dt
        else:
            ACCT.move += dt
        self.t += dt
        self.p = q

    def detect(self, ch):
        ACCT.detect += pb.T_DETECT
        return super().detect(ch)

    def clear(self):
        if ACCT.in_blind:
            ACCT.blind_clear += pb.T_CLEAR
        else:
            ACCT.clear += pb.T_CLEAR
        return super().clear()


_orig_blind = pb.blind_ray_clear


def blind_wrapped(*a, **k):
    ACCT.in_blind = True
    try:
        return _orig_blind(*a, **k)
    finally:
        ACCT.in_blind = False


def main():
    pb.Dog = DogA
    pb.blind_ray_clear = blind_wrapped
    rows = []
    for pol in ("P1", "P2", "P3"):
        for dirfrac in (0.0, 0.3):
            agg = dict(move=0.0, detect=0.0, clear=0.0, blind_move=0.0, blind_clear=0.0, tot=0.0)
            n = 0
            for n_src in (10, 13, 16):
                for s in range(1, 11):
                    global ACCT
                    ACCT = Acct()
                    r = pb.run(pol, n_src, s, dirfrac)
                    agg["move"] += ACCT.move
                    agg["detect"] += ACCT.detect
                    agg["clear"] += ACCT.clear
                    agg["blind_move"] += ACCT.blind_move
                    agg["blind_clear"] += ACCT.blind_clear
                    agg["tot"] += r["time"]
                    n += 1
            for k in agg:
                agg[k] /= n
            rows.append(dict(policy=pol, dirfrac=dirfrac, **agg))

    print("=" * 96)
    print("时间去向拆解 (每局平均秒数, 30 局/格)")
    print("=" * 96)
    print(f"{'策略':>5} {'定向占比':>7} {'总时':>8} {'常规行进':>9} {'检测':>7} {'常规清除':>9} "
          f"{'盲扫行进':>9} {'盲扫清除':>9} {'盲扫占比':>8}")
    for r in rows:
        blind = r["blind_move"] + r["blind_clear"]
        print(f"{r['policy']:>5} {r['dirfrac']:>7.1f} {r['tot']:>8.0f} {r['move']:>9.0f} "
              f"{r['detect']:>7.0f} {r['clear']:>9.0f} {r['blind_move']:>9.0f} "
              f"{r['blind_clear']:>9.0f} {blind / max(r['tot'], 1):>8.2f}")
    with open("time_breakdown.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\n-> time_breakdown.json")


pb_acct_holder = [ACCT]

if __name__ == "__main__":
    main()
