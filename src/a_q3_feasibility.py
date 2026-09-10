#!/usr/bin/env python3
"""A题问题3 可行性硬核查：用附录3参数实际积分，看 C<0.15 到底需要多久、能否达到。

这是选A题的决定性风险点：若 D(C) 在低含水率崩塌，干燥永远完不成。
"""
import numpy as np
import pandas as pd

R0 = 0.02
C0, T0 = 2.55, 28.0
h, hm = 25.0, 8e-7

a1 = pd.read_excel("problem/A题/附件/附件1.xlsx")
a1.columns = ["t", "Tair", "Cair"]
t_a, Ta_a, Ca_a = a1["t"].values, a1["Tair"].values, a1["Cair"].values
TAIR_END, CAIR_END = Ta_a[-1], Ca_a[-1]


def air(t):
    if t <= t_a[-1]:
        return np.interp(t, t_a, Ta_a), np.interp(t, t_a, Ca_a)
    return TAIR_END, CAIR_END


def props3(C):
    rho = 650 + 128 * C
    cp = 1450 + 2736 * C / (C + 1)
    k = 0.21 + 0.38 * C / (C + 1)
    return rho, cp, k


def D3(C, T_K):
    Cs = np.maximum(C, 1e-4)
    return 2.4e-3 * np.exp(-0.45 / Cs) * np.exp(-3850.0 / T_K)


def simulate(nr=41, dt=2.0, t_end=400 * 3600, report_every=3600 * 6, verbose=True):
    """轴对称一维，有限体积 + 显式(自适应dt)。返回时间序列。"""
    r = np.linspace(0, R0, nr)
    dr = r[1] - r[0]
    T = np.full(nr, T0)
    C = np.full(nr, C0)
    t = 0.0
    hist = []
    next_rep = 0.0
    n_sub = 0

    while t < t_end:
        Tair_t, Cair_t = air(t)
        rho, cp, k = props3(C)
        alpha = k / (rho * cp)
        TK = T + 273.15
        D = D3(C, TK)

        # 稳定性：显式格式 dt <= dr^2/(2*max(alpha,D))
        dmax = max(alpha.max(), D.max())
        dt_use = min(dt, 0.35 * dr * dr / dmax)

        # --- 温度 ---
        Tn = T.copy()
        kf = 0.5 * (k[:-1] + k[1:])
        rf = 0.5 * (r[:-1] + r[1:])
        flux = kf * rf * (T[1:] - T[:-1]) / dr        # k*r*dT/dr at faces
        # 内部节点 i=1..nr-2
        Tn[1:-1] = T[1:-1] + dt_use / (rho[1:-1] * cp[1:-1]) * \
            (flux[1:] - flux[:-1]) / (dr * r[1:-1])
        # 中心 r=0：对称，用 2*d2T/dr2
        Tn[0] = T[0] + dt_use * alpha[0] * 2 * (T[1] - T[0]) * 2 / (dr * dr)
        # 表面：半控制体能量平衡  -k dT/dr = h(Ts - Tair)
        vol_s = r[-1] * dr / 2
        Tn[-1] = T[-1] + dt_use / (rho[-1] * cp[-1] * vol_s) * (
            h * (Tair_t - T[-1]) * r[-1] - flux[-1]
        )

        # --- 水分 ---
        Cn = C.copy()
        Df = 0.5 * (D[:-1] + D[1:])
        fluxC = Df * rf * (C[1:] - C[:-1]) / dr
        Cn[1:-1] = C[1:-1] + dt_use * (fluxC[1:] - fluxC[:-1]) / (dr * r[1:-1])
        Cn[0] = C[0] + dt_use * D[0] * 2 * (C[1] - C[0]) * 2 / (dr * dr)
        Cn[-1] = C[-1] + dt_use / vol_s * (
            hm * (Cair_t - C[-1]) * r[-1] - fluxC[-1]
        )
        Cn = np.maximum(Cn, 1e-6)

        T, C = Tn, Cn
        t += dt_use
        n_sub += 1

        if not np.all(np.isfinite(T)) or not np.all(np.isfinite(C)):
            print(f"  !! 发散于 t={t:.1f}s")
            return None, hist

        if t >= next_rep:
            hist.append((t, T.copy(), C.copy()))
            next_rep += report_every
            if verbose and len(hist) % 8 == 1:
                print(f"    t={t/3600:8.2f}h  Tc={T[0]:6.2f} Ts={T[-1]:6.2f}"
                      f"  Cc={C[0]:.5f} Cs={C[-1]:.5f} Cmax={C.max():.5f}"
                      f"  dt={dt_use:.3f}s")
        if C.max() < 0.15:
            print(f"\n  ★ 全场 C<0.15 达成于 t={t:.1f}s = {t/3600:.3f} h"
                  f"  (子步数 {n_sub})")
            return t, hist
    print(f"\n  ✗ 到 t_end={t_end/3600:.0f}h 仍未达成: Cmax={C.max():.6f}"
          f" (中心 {C[0]:.6f})  子步数 {n_sub}")
    return None, hist


if __name__ == "__main__":
    print("=" * 78)
    print("A题问题3 可行性核查：附录3参数，判定 C<0.15 是否可达")
    print("=" * 78)
    print(f"\n烘房条件: 附件1 末端 Tair={TAIR_END}C, Cair={CAIR_END}")
    print(f"平衡含水率下限 = Cair = {CAIR_END} < 0.15 目标  -> 热力学上可达\n")

    t_done, hist = simulate()

    if hist:
        print("\n  最终剖面 (t=%.2f h):" % (hist[-1][0] / 3600))
        _, Tf, Cf = hist[-1]
        rr = np.linspace(0, R0, len(Cf))
        for i in range(0, len(Cf), 5):
            print(f"    r={rr[i]*100:5.2f}cm  T={Tf[i]:6.2f}C  C={Cf[i]:.6f}")
