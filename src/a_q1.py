"""
CUMCM2026 A题 (药材烘干) —— 问题1 数值验证脚本
=================================================

模型：无限长圆柱轴对称一维 (r 方向) 热-质耦合(实际上是单向弱耦合：D 只依赖 C，
不依赖 T，故两方程在数学上解耦，各自独立求解)。

    热传导:  rho*cp*dT/dt = (1/r) d/dr( k * r * dT/dr )
    水分扩散: dC/dt        = (1/r) d/dr( D(C) * r * dC/dr ),  D(C)=7e-9*exp(-0.89/C)

    r=R:  -k dT/dr = h *(T_s - T_air(t))
          -D dC/dr = hm*(C_s - C_air(t))
    r=0:  dT/dr = dC/dr = 0  (对称)

离散：有限体积法(FVM，柱坐标下天然守恒，r=0 处无 1/r 奇点问题)
  节点  r_i = i*dr, i=0..N   (dr=0.1cm -> N=20 -> 21 个节点)
  控制体 V_i = (r_{i+1/2}^2 - r_{i-1/2}^2)/2   (每单位长度/2π)
  界面面积 A_{i+1/2} = r_{i+1/2}
  时间推进：
    - 热方程: Crank-Nicolson (θ=0.5)，线性三对角直接解，无条件稳定
    - 水分方程: 全隐式(θ=1) + Picard 迭代处理 D(C) 非线性 (界面 D 取相邻节点算术平均)

运行: python src/a_q1.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import solve_banded

# ---------------------------------------------------------------- 参数(附录2)
RHO = 820.0        # kg/m^3
CP = 2600.0        # J/(kg K)
K_TH = 0.36        # W/(m K)
H_CONV = 25.0      # W/(m^2 K)
HM = 8e-7          # m/s
D0 = 7e-9          # m^2/s
EA = 0.89          # kg/kg  (D = D0*exp(-EA/C))

R_OUT = 0.02       # m  (2 cm)
T_INIT = 28.0      # degC
C_INIT = 2.55      # kg/kg (干基)

BASE = Path(__file__).resolve().parents[1]
XLSX1 = BASE / "problem" / "A题" / "附件" / "附件1.xlsx"


def diffusivity(C: np.ndarray) -> np.ndarray:
    """D(C) = 7e-9 * exp(-0.89/C)，对极小/非正 C 做下限保护。"""
    Cs = np.maximum(C, 1e-6)
    return D0 * np.exp(-EA / Cs)


# ---------------------------------------------------------------- 烘房条件
def load_air():
    df = pd.read_excel(XLSX1, sheet_name=0, header=0)
    df.columns = ["t", "T", "C"]
    df = df.astype(float)
    t = df["t"].to_numpy()
    return t, df["T"].to_numpy(), df["C"].to_numpy()


T_AIR_T, T_AIR_V, C_AIR_V = load_air()


def t_air(t: float) -> float:
    return float(np.interp(t, T_AIR_T, T_AIR_V))


def c_air(t: float) -> float:
    return float(np.interp(t, T_AIR_T, C_AIR_V))


# ---------------------------------------------------------------- 网格
class Grid:
    def __init__(self, n_nodes: int):
        self.n = n_nodes
        self.dr = R_OUT / (n_nodes - 1)
        self.r = np.linspace(0.0, R_OUT, n_nodes)
        dr = self.dr
        # 界面半径 r_{i+1/2}, i=0..n-2  (内部界面)
        self.rf = self.r[:-1] + dr / 2.0
        # 控制体 "体积" (每单位长度 /2pi)
        rm = np.empty(n_nodes)   # r_{i-1/2}
        rp = np.empty(n_nodes)   # r_{i+1/2}
        rm[0] = 0.0
        rp[0] = dr / 2.0
        rm[1:-1] = self.r[1:-1] - dr / 2.0
        rp[1:-1] = self.r[1:-1] + dr / 2.0
        rm[-1] = R_OUT - dr / 2.0
        rp[-1] = R_OUT
        self.V = (rp ** 2 - rm ** 2) / 2.0
        self.A_out = R_OUT      # 外表面面积因子


def banded_solve(lower, diag, upper, rhs):
    """解三对角系统 (lower[i]*x[i-1] + diag[i]*x[i] + upper[i]*x[i+1] = rhs[i])."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


# ---------------------------------------------------------------- 热方程 CN
def step_heat(T, g: Grid, dt, t_old, t_new, theta=0.5):
    n, dr, V = g.n, g.dr, g.V
    Ta_o, Ta_n = t_air(t_old), t_air(t_new)
    cap = RHO * CP * V / dt

    # 传导导纳 G_{i+1/2} = k*A/dr
    G = K_TH * g.rf / dr                      # 长度 n-1
    lower = np.zeros(n); diag = np.zeros(n); upper = np.zeros(n)
    rhs = np.zeros(n)

    for i in range(n):
        gl = G[i - 1] if i > 0 else 0.0
        gr = G[i] if i < n - 1 else 0.0
        # 显式部分 (old)
        expl = 0.0
        if i > 0:
            expl += gl * (T[i - 1] - T[i])
        if i < n - 1:
            expl += gr * (T[i + 1] - T[i])
        if i == n - 1:
            expl += g.A_out * H_CONV * (Ta_o - T[i])
        rhs[i] = cap[i] * T[i] + (1 - theta) * expl
        # 隐式部分 (new)
        d = cap[i]
        if i > 0:
            d += theta * gl
            lower[i] = -theta * gl
        if i < n - 1:
            d += theta * gr
            upper[i] = -theta * gr
        if i == n - 1:
            d += theta * g.A_out * H_CONV
            rhs[i] += theta * g.A_out * H_CONV * Ta_n
        diag[i] = d
    return banded_solve(lower, diag, upper, rhs)


# ------------------------------------------------------- 水分方程 隐式+Picard
def step_moist(C, g: Grid, dt, t_new, tol=1e-12, itmax=60):
    n, dr, V = g.n, g.dr, g.V
    Ca = c_air(t_new)
    cap = V / dt
    Cnew = C.copy()
    for _ in range(itmax):
        Dn = diffusivity(Cnew)
        Df = 0.5 * (Dn[:-1] + Dn[1:])         # 界面扩散系数(算术平均)
        G = Df * g.rf / dr                    # 长度 n-1
        lower = np.zeros(n); diag = np.zeros(n); upper = np.zeros(n)
        rhs = cap * C
        for i in range(n):
            d = cap[i]
            if i > 0:
                d += G[i - 1]
                lower[i] = -G[i - 1]
            if i < n - 1:
                d += G[i]
                upper[i] = -G[i]
            if i == n - 1:
                d += g.A_out * HM
                rhs[i] += g.A_out * HM * Ca
            diag[i] = d
        Cit = banded_solve(lower, diag, upper, rhs)
        err = np.max(np.abs(Cit - Cnew))
        Cnew = Cit
        if err < tol:
            break
    return Cnew, err


# ---------------------------------------------------------------- 求解驱动
def solve(dt, t_end, n_nodes=21, save_times=()):
    g = Grid(n_nodes)
    T = np.full(n_nodes, T_INIT)
    C = np.full(n_nodes, C_INIT)
    nsteps = int(round(t_end / dt))
    assert abs(nsteps * dt - t_end) < 1e-9
    snaps = {}
    want = sorted(save_times)
    picard_max = 0
    mass0 = float(np.sum(g.V * C))
    flux_int = 0.0
    for s in range(1, nsteps + 1):
        t_old = (s - 1) * dt
        t_new = s * dt
        T = step_heat(T, g, dt, t_old, t_new)
        Cs_old = C[-1]
        C, err = step_moist(C, g, dt, t_new)
        picard_max = max(picard_max, err)
        # 表面质量通量 (隐式端点值)
        flux_int += g.A_out * HM * (C[-1] - c_air(t_new)) * dt
        for tw in want:
            if abs(t_new - tw) < 1e-9:
                snaps[tw] = (T.copy(), C.copy())
    mass1 = float(np.sum(g.V * C))
    bal = dict(mass0=mass0, mass1=mass1, lost=mass0 - mass1,
               flux_int=flux_int,
               rel_err=abs((mass0 - mass1) - flux_int) / max(mass0 - mass1, 1e-30))
    return g, T, C, snaps, picard_max, bal


def table(g, snaps, idx, times, key, fmt="{:.4f}"):
    cols = [f"r={g.r[i]*100:.1f}cm" for i in idx]
    rows = []
    for t in times:
        arr = snaps[t][key]
        rows.append([fmt.format(arr[i]) for i in idx])
    return pd.DataFrame(rows, index=[f"t={t}s" for t in times], columns=cols)


def main():
    times = [100, 300, 600, 900, 1200, 1500, 1800]
    n = 21
    g = Grid(n)
    idx = [np.argmin(np.abs(g.r - x)) for x in (0.0, 0.005, 0.010, 0.015, 0.020)]

    print("=" * 78)
    print("0) 网格与预诊断")
    print("=" * 78)
    print(f"节点数 n = {n}, dr = {g.dr*100:.4f} cm, r 节点(cm) = "
          f"{np.round(g.r*100,3).tolist()}")
    alpha = K_TH / (RHO * CP)
    D_init = float(diffusivity(np.array([C_INIT]))[0])
    D_air = float(diffusivity(np.array([0.0196]))[0])
    print(f"热扩散率 alpha = k/(rho*cp) = {alpha:.6e} m^2/s")
    print(f"热特征时间 rho*cp*R^2/k = R^2/alpha = {R_OUT**2/alpha:.2f} s")
    print(f"热 Biot 数 h*R/k = {H_CONV*R_OUT/K_TH:.4f}")
    print(f"D(C=2.55)  = 7e-9*exp(-0.89/2.55) = {D_init:.6e} m^2/s")
    print(f"   校验 exp(-0.89/2.55) = {np.exp(-EA/C_INIT):.6f}")
    print(f"水分特征时间 R^2/D(C0) = {R_OUT**2/D_init:.1f} s "
          f"= {R_OUT**2/D_init/3600:.2f} h")
    print(f"传质 Biot 数 hm*R/D(C0) = {HM*R_OUT/D_init:.4f}")
    print(f"D(C=0.0196)= {D_air:.6e} m^2/s  (若表面降到烘房水平, D 塌缩 "
          f"{D_init/D_air:.3e} 倍)")
    print(f"初始表面传质通量 hm*(C0-Cair0) = {HM*(C_INIT-0.01963):.6e} "
          f"kg/kg * m/s")
    print(f"  => 表面浓度梯度需求 |dC/dr| = flux/D = "
          f"{HM*(C_INIT-0.01963)/D_init:.4f} (kg/kg)/m "
          f"= {HM*(C_INIT-0.01963)/D_init/100:.6f} (kg/kg)/cm")

    print()
    print("=" * 78)
    print("1) 稳定性/收敛性检验: dt = 1 s vs dt = 0.25 s (n=21, t_end=1800s)")
    print("=" * 78)
    res = {}
    for dt in (1.0, 0.25):
        gg, T, C, sn, pmax, bal = solve(dt, 1800.0, n, times)
        res[dt] = (T, C, sn, pmax, bal)
        print(f"dt={dt:>5}s : Picard 最大残差 = {pmax:.3e} | "
              f"质量守恒相对误差 = {bal['rel_err']:.3e} | "
              f"1800s 内失水(体积积分) = {bal['lost']:.6e}")
    T1, C1 = res[1.0][0], res[1.0][1]
    T2, C2 = res[0.25][0], res[0.25][1]
    print(f"t=1800s 温度   最大差 |dt1 - dt0.25| = {np.max(np.abs(T1-T2)):.6e} degC")
    print(f"t=1800s 水分   最大差 |dt1 - dt0.25| = {np.max(np.abs(C1-C2)):.6e} kg/kg")
    # 全时刻最大差
    dTmax = max(np.max(np.abs(res[1.0][2][t][0] - res[0.25][2][t][0])) for t in times)
    dCmax = max(np.max(np.abs(res[1.0][2][t][1] - res[0.25][2][t][1])) for t in times)
    print(f"所有输出时刻: 温度最大差 = {dTmax:.6e} degC, "
          f"水分最大差 = {dCmax:.6e} kg/kg")

    # 空间收敛: dr=0.05cm (41节点), dt=0.25
    g41, T41, C41, sn41, pmax41, bal41 = solve(0.25, 1800.0, 41, times)
    sel41 = [np.argmin(np.abs(g41.r - x)) for x in (0.0, 0.005, 0.010, 0.015, 0.020)]
    print(f"空间检验 dr=0.05cm(41节点,dt=0.25s): "
          f"5 个采样点温度最大差 = "
          f"{np.max(np.abs(T41[sel41] - T2[idx])):.6e} degC, "
          f"水分最大差 = {np.max(np.abs(C41[sel41] - C2[idx])):.6e} kg/kg")

    snaps = {t: {0: res[0.25][2][t][0], 1: res[0.25][2][t][1]} for t in times}

    print()
    print("=" * 78)
    print("2) 温度场 T(r,t)  [degC]   (dt=0.25s, dr=0.1cm)")
    print("=" * 78)
    tabT = table(g, snaps, idx, times, 0)
    print(tabT.to_string())
    print()
    print("对照烘房空气温度 T_air(t):")
    print("  " + "  ".join(f"t={t}s: {t_air(t):.4f}" for t in times))

    print()
    print("=" * 78)
    print("3) 水分浓度场 C(r,t)  [kg/kg 干基]  (dt=0.25s, dr=0.1cm)")
    print("=" * 78)
    tabC = table(g, snaps, idx, times, 1)
    print(tabC.to_string())
    print()
    print("同一张表放大到 8 位小数(看清变化量级):")
    print(table(g, snaps, idx, times, 1, "{:.8f}").to_string())
    print()
    print("对照烘房空气水分浓度 C_air(t):")
    print("  " + "  ".join(f"t={t}s: {c_air(t):.5f}" for t in times))

    print()
    print("=" * 78)
    print("4) 关键诊断")
    print("=" * 78)
    Tc0, Ts0 = T_INIT, T_INIT
    T18 = snaps[1800][0]; C18 = snaps[1800][1]
    print(f"[温度] 1800s: 中心 {T18[idx[0]]:.4f} degC, 表面 {T18[idx[-1]]:.4f} degC, "
          f"烘房 {t_air(1800):.4f} degC")
    print(f"        与烘房温差: 中心 {t_air(1800)-T18[idx[0]]:.4f} K, "
          f"表面 {t_air(1800)-T18[idx[-1]]:.4f} K")
    print(f"        升温幅度: 中心 {T18[idx[0]]-Tc0:.4f} K, "
          f"表面 {T18[idx[-1]]-Ts0:.4f} K")
    print(f"        径向不均匀度 T_s - T_c = {T18[idx[-1]]-T18[idx[0]]:.4f} K")
    print(f"[水分] 1800s: 中心 {C18[idx[0]]:.6f}, 表面 {C18[idx[-1]]:.6f} kg/kg")
    print(f"        变化量: 中心 {C18[idx[0]]-C_INIT:+.6e}, "
          f"表面 {C18[idx[-1]]-C_INIT:+.6e} kg/kg")
    print(f"        相对变化: 中心 {(C18[idx[0]]/C_INIT-1)*100:+.6f} %, "
          f"表面 {(C18[idx[-1]]/C_INIT-1)*100:+.6f} %")
    Cbar = float(np.sum(g.V * C18) / np.sum(g.V))
    print(f"        截面平均含水率 1800s = {Cbar:.6f} kg/kg "
          f"(初始 {C_INIT}), 失水 {(1-Cbar/C_INIT)*100:.4f} %")
    print(f"        全场 C 极差 max-min = {C18.max()-C18.min():.6e} kg/kg")

    # 长时间跑到 14400s 看整体行为
    print()
    print("=" * 78)
    print("5) 补充: 跑满全烘干时段 t_end=14400s (dt=1s) 看水分能降到多少")
    print("=" * 78)
    tl = [1800, 3600, 7200, 10800, 14400]
    gl, Tl, Cl, snl, pml, ball = solve(1.0, 14400.0, n, tl)
    print(f"{'t(s)':>7} {'T_c':>9} {'T_s':>9} {'T_air':>9} "
          f"{'C_c':>12} {'C_s':>12} {'C_avg':>12} {'失水%':>8}")
    for t in tl:
        Tt, Ct = snl[t]
        cav = float(np.sum(gl.V * Ct) / np.sum(gl.V))
        print(f"{t:>7} {Tt[idx[0]]:>9.4f} {Tt[idx[-1]]:>9.4f} {t_air(t):>9.4f} "
              f"{Ct[idx[0]]:>12.6f} {Ct[idx[-1]]:>12.6f} {cav:>12.6f} "
              f"{(1-cav/C_INIT)*100:>8.4f}")
    print(f"质量守恒相对误差(14400s) = {ball['rel_err']:.3e}")

    # 附件2 对照：题目给出的半径收缩曲线
    a2 = pd.read_excel(BASE / "problem" / "A题" / "附件" / "附件2.xlsx", header=0)
    a2.columns = ["t", "R"]
    print()
    print("附件2 (实测半径收缩) 对照: t=0 R=2cm, t=1800s R="
          f"{a2['R'].iloc[1]}cm, t=14400s R="
          f"{float(np.interp(14400, a2['t'], a2['R'])):.4f}cm, "
          f"终值 R={a2['R'].iloc[-1]}cm")

    # 存盘
    out = BASE / "results"
    out.mkdir(exist_ok=True)
    tabT.to_csv(out / "a_q1_T.csv", encoding="utf-8-sig")
    tabC.to_csv(out / "a_q1_C.csv", encoding="utf-8-sig")
    print(f"\n表格已保存: {out/'a_q1_T.csv'}, {out/'a_q1_C.csv'}")


if __name__ == "__main__":
    main()
