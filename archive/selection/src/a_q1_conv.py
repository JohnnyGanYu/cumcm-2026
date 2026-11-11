"""A题问题1: 网格收敛性研究 + 与附件2半径收缩数据的自洽性交叉检验。"""
import numpy as np
import pandas as pd
from pathlib import Path
from a_q1 import solve, Grid, diffusivity, C_INIT, R_OUT, D0, EA, HM, t_air, c_air, BASE

TIMES = [1800]
SAMPLE = (0.0, 0.005, 0.010, 0.015, 0.020)

print("=" * 78)
print("A) 网格收敛研究 (dt=0.25s, t_end=1800s) —— 表面存在陡薄层, 检查 dr 是否够细")
print("=" * 78)
print("边界层厚度估计 sqrt(D*t): D(C=2.55)=%.4e -> delta=%.4f cm; "
      % (diffusivity(np.array([2.55]))[0],
         np.sqrt(diffusivity(np.array([2.55]))[0] * 1800) * 100))

rows = []
store = {}
for n in (21, 41, 81, 161, 321):
    g, T, C, sn, pm, bal = solve(0.25, 1800.0, n, TIMES)
    idx = [np.argmin(np.abs(g.r - x)) for x in SAMPLE]
    Cbar = float(np.sum(g.V * C) / np.sum(g.V))
    store[n] = (g, T, C, Cbar)
    rows.append(dict(n=n, dr_cm=g.dr * 100,
                     C_surf=C[-1], C_r15=C[idx[3]], C_r10=C[idx[2]],
                     C_center=C[0], C_avg=Cbar,
                     T_surf=T[-1], T_center=T[0],
                     mass_err=bal["rel_err"]))
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
print()
print("相邻网格差 (Richardson 观察):")
ns = [21, 41, 81, 161, 321]
for a, b in zip(ns[:-1], ns[1:]):
    Ca, Cb = store[a][2], store[b][2]
    ga, gb = store[a][0], store[b][0]
    ia = [np.argmin(np.abs(ga.r - x)) for x in SAMPLE]
    ib = [np.argmin(np.abs(gb.r - x)) for x in SAMPLE]
    print(f"  n={a:>4}->{b:>4}: |dC|max(5点)={np.max(np.abs(Ca[ia]-Cb[ib])):.3e}  "
          f"dC_surf={Cb[-1]-Ca[-1]:+.3e}  dC_avg={store[b][3]-store[a][3]:+.3e}  "
          f"|dT|max={np.max(np.abs(store[a][1][ia]-store[b][1][ib])):.3e}")

print()
print("=" * 78)
print("B) 时间步收敛 (n=161, 细网格下再验 dt)")
print("=" * 78)
prev = None
for dt in (2.0, 1.0, 0.5, 0.25):
    g, T, C, sn, pm, bal = solve(dt, 1800.0, 161, TIMES)
    idx = [np.argmin(np.abs(g.r - x)) for x in SAMPLE]
    Cbar = float(np.sum(g.V * C) / np.sum(g.V))
    msg = f"dt={dt:>5}s  C_surf={C[-1]:.8f}  C_avg={Cbar:.8f}  T_surf={T[-1]:.6f}"
    if prev is not None:
        msg += (f"  |dC|max vs 上一行={np.max(np.abs(C - prev[0])):.3e}"
                f"  |dT|max={np.max(np.abs(T - prev[1])):.3e}")
    print(msg)
    prev = (C.copy(), T.copy())

print()
print("=" * 78)
print("C) 与附件2 (半径收缩) 的自洽性交叉检验")
print("=" * 78)
a2 = pd.read_excel(BASE / "problem" / "A题" / "附件" / "附件2.xlsx", header=0)
a2.columns = ["t", "R"]
# 若 rho=820 是湿基总密度, 干基含水率 C0=2.55 =>
frac_dry = 1.0 / (1.0 + C_INIT)
rho_dry = 820.0 * frac_dry
rho_w0 = 820.0 - rho_dry
print(f"设 rho=820 kg/m3 为初始湿基总密度, C0=2.55 kg/kg(干基):")
print(f"  干物质密度 = {rho_dry:.3f} kg/m3, 初始水密度 = {rho_w0:.3f} kg/m3")
print(f"  初始水体积分数(按 1000 kg/m3) = {rho_w0/1000:.4f}")
print()
print("假设体积收缩全部来自失水(干物质骨架体积不变):")
print(f"{'t(s)':>7} {'R_exp(cm)':>10} {'V/V0':>8} {'水体积损失':>10} "
      f"{'ΔC_shrink':>10} {'C_shrink':>9} {'C_model':>9}")
g161, _, _, _ = store[161]
model_t = [1800]
for t in [1800]:
    Rexp = float(np.interp(t, a2["t"], a2["R"]))
    vr = (Rexp / 2.0) ** 2
    dvol_w = 1.0 - vr                       # 每单位初始体积失去的水体积
    dC = dvol_w * 1000.0 / rho_dry          # kg水/kg干
    print(f"{t:>7} {Rexp:>10.4f} {vr:>8.4f} {dvol_w:>10.4f} "
          f"{dC:>10.4f} {C_INIT-dC:>9.4f} {store[161][3]:>9.4f}")

# 全时段
print()
print("全时段对照 (模型 dt=1s, n=161):")
tl = [1800, 3600, 7200, 10800, 14400, 43200, 86400, 259200]
tl_run = [t for t in tl if t <= 14400]
g, T, C, sn, pm, bal = solve(1.0, 14400.0, 161, tl_run)
print(f"{'t(s)':>8} {'C_avg_model':>12} {'失水%_model':>12} "
      f"{'R_exp(cm)':>10} {'V/V0':>7} {'C_shrink':>9} {'失水%_shrink':>12}")
for t in tl_run:
    Tt, Ct = sn[t]
    cav = float(np.sum(g.V * Ct) / np.sum(g.V))
    Rexp = float(np.interp(t, a2["t"], a2["R"]))
    vr = (Rexp / 2.0) ** 2
    dC = (1 - vr) * 1000.0 / rho_dry
    print(f"{t:>8} {cav:>12.6f} {(1-cav/C_INIT)*100:>12.4f} "
          f"{Rexp:>10.4f} {vr:>7.4f} {C_INIT-dC:>9.4f} {dC/C_INIT*100:>12.4f}")
print()
Rend = float(a2["R"].iloc[-1])
vr = (Rend / 2.0) ** 2
dC = (1 - vr) * 1000.0 / rho_dry
print(f"附件2 终态 t=259200s R={Rend}cm -> V/V0={vr:.4f} -> "
      f"按收缩推算终态 C={C_INIT-dC:.4f} kg/kg (失水 {dC/C_INIT*100:.2f}%)")

print()
print("=" * 78)
print("D) 边界条件量纲/量级审查")
print("=" * 78)
Dc = float(diffusivity(np.array([C_INIT]))[0])
print(f"BC: -D dC/dr = hm*(C_s - C_air)")
print(f"  hm = {HM:.3e} m/s,  D(C0) = {Dc:.4e} m^2/s")
print(f"  传质 Biot = hm*R/D(C0) = {HM*R_OUT/Dc:.4f}  (O(1)~O(10): 内外阻力可比,"
      f" 表面浓度会显著低于中心, 但不会瞬间降到 C_air)")
print(f"  若按 Biot 定义所需表面梯度: (C0-C_air)*hm/D = "
      f"{(C_INIT-0.0196)*HM/Dc:.2f} (kg/kg)/m = "
      f"{(C_INIT-0.0196)*HM/Dc/100:.4f} (kg/kg)/cm")
print(f"  => 在 dr=0.1cm 网格上, 首个网格内 C 应下降约 "
      f"{(C_INIT-0.0196)*HM/Dc/100*0.1:.4f} kg/kg, 即 O(0.4) —— 陡但可解析,"
      f" 前提是 dr 足够小")
print()
print("单位一致性: hm [m/s] * ΔC [kg水/kg干] 给出 [kg水/(kg干·m^2·s)]·m ... "
      "与 D*dC/dr [m^2/s * (kg/kg)/m] = [(kg/kg)*m/s] 一致 -> BC 量纲自洽。")
print("注意: 该 BC 把烘房'水分浓度'(0.0196~0.0499 kg/kg, 空气含湿量)与药材"
      "干基含水率(2.55 kg/kg)当作同一标度直接相减, 物理上并不等价 —— "
      "严格做法应用平衡含水率 C_eq(T,RH) 而非空气含湿量。")
