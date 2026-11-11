#!/usr/bin/env python3
"""核实 A题 modeler 抛出的两个关键论断。

论断1（可能颠覆全题）：蒸发潜热 ~1100 W/m2 > 对流显热 ~550 W/m2，
    严格耦合会把表面压到湿球温度(31-40C)，药材到不了50C，与"预热平衡"矛盾。
论断2（推翻我之前的"失水108.86%矛盾"）：三路交叉验证收缩律，
    附录4密度式给 R_f=1.211cm，与实测 1.198cm 仅差 1.1%。
"""
import numpy as np
import pandas as pd

R0 = 0.02
C0 = 2.55
LAMBDA_V = 2.4e6      # J/kg 汽化潜热(50C附近约2382 kJ/kg)
h, hm = 25.0, 8e-7

print("=" * 78)
print("论断1：蒸发潜热 vs 对流显热 —— 表面能否达到50C？")
print("=" * 78)

# 干密度：rho=820 是初始湿基总密度
rho_wet0 = 820.0
rho_d_app2 = rho_wet0 / (1 + C0)
print(f"\n附录2: 湿基总密度 {rho_wet0} -> 干骨架密度 = {rho_wet0}/(1+{C0}) = {rho_d_app2:.2f} kg/m3")

a1 = pd.read_excel("problem/A题/附件/附件1.xlsx")
a1.columns = ["t", "Tair", "Cair"]
C_air_0, C_air_end = a1["Cair"].iloc[0], a1["Cair"].iloc[-1]

print("\n蒸发通量 m_dot = rho_d * hm * (C_s - C_air)   [kg/(m2·s)]")
print(f"{'C_s':>6} {'C_air':>8} {'m_dot':>12} {'潜热λ·m_dot':>14} {'显热h·ΔT(ΔT=22)':>18} {'比值':>8}")
print("-" * 78)
for C_s in (2.55, 2.0, 1.5, 1.0, 0.5, 0.15):
    m_dot = rho_d_app2 * hm * (C_s - C_air_0)
    q_lat = LAMBDA_V * m_dot
    q_sen = h * 22.0
    print(f"{C_s:>6.2f} {C_air_0:>8.4f} {m_dot:>12.3e} {q_lat:>14.1f} {q_sen:>18.1f} {q_lat/q_sen:>8.2f}")

print("\n>>> 论断1的数值确认：初期潜热负荷确实远超对流显热。")

# 湿球温度计算
print("\n" + "=" * 78)
print("湿球温度核算（决定表面温度上限）")
print("=" * 78)


def p_sat(Tc):
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def humidity_ratio(Tc, rh=1.0):
    ps = p_sat(Tc) * rh
    return 0.622 * ps / (101325 - ps)


def wet_bulb(Tdb, w):
    """由干球温度与含湿量迭代求湿球温度（绝热饱和近似）。"""
    cp_a, cp_v = 1006.0, 1860.0
    lo, hi = -10.0, Tdb
    for _ in range(200):
        Twb = 0.5 * (lo + hi)
        w_s = humidity_ratio(Twb, 1.0)
        lam = 2501000 - 2326 * Twb
        # 绝热饱和：h(Tdb,w) = h(Twb,w_s)
        lhs = cp_a * Tdb + w * (2501000 + cp_v * Tdb)
        rhs = cp_a * Twb + w_s * (2501000 + cp_v * Twb)
        if lhs > rhs:
            lo = Twb
        else:
            hi = Twb
    return Twb


for Tdb, w, tag in ((28.0, C_air_0, "烘干初期(附件1首)"),
                    (50.0, C_air_end, "恒温段(附件1末)"),
                    (50.0, 0.02, "参考 w=0.02"),
                    (50.0, 0.05, "参考 w=0.05")):
    Twb = wet_bulb(Tdb, w)
    ws = humidity_ratio(Tdb, 1.0)
    rh = w / ws * 100
    print(f"  {tag:22} Tdb={Tdb:5.1f}C w={w:.5f} (RH={rh:5.1f}%) -> Twb = {Twb:5.2f} C")

print("\n>>> 若严格耦合潜热，表面温度趋向湿球温度而非干球温度。")

# 关键：定量评估潜热对温度场的实际影响
print("\n" + "=" * 78)
print("定量：把潜热项加进表面能量平衡，稳态表面温度是多少？")
print("=" * 78)
print("表面稳态能量平衡: h*(T_air - T_s) = lambda_v * rho_d * hm * (C_s - C_air) + 内导热")
print("忽略内导热(保守上界估计)，解 T_s:")
for C_s in (2.55, 1.5, 0.5, 0.15, 0.05):
    m_dot = rho_d_app2 * hm * (C_s - C_air_end)
    q_lat = LAMBDA_V * m_dot
    dT_needed = q_lat / h
    T_s = 50.165 - dT_needed
    flag = "  <-- 低于环境温度很多" if dT_needed > 10 else ""
    print(f"  C_s={C_s:>5.2f}: 潜热={q_lat:8.1f} W/m2 需要ΔT={dT_needed:7.2f}K"
          f" -> T_s={T_s:7.2f} C{flag}")

print("""
>>> 关键判断：
    表面 C_s 会在极短时间内(<100s)从2.55降到很低(前面已验证1800s时=1.512)，
    潜热负荷随 C_s 下降而快速减小。所以"1100 W/m2"只在最初极短时间成立。
    需要看的是：随着表面干燥，潜热负荷衰减到什么水平。
""")

print("=" * 78)
print("论断2：三路交叉验证收缩律（推翻我之前的108.86%失水矛盾）")
print("=" * 78)
a2 = pd.read_excel("problem/A题/附件/附件2.xlsx")
a2.columns = ["t", "r_cm"]
Rf = a2["r_cm"].iloc[-1]
print(f"\n实测: R {a2['r_cm'].iloc[0]} -> {Rf} cm")

# 路线1：仅径向收缩 vs 各向同性
V_rad = (Rf / 2.0) ** 2
V_iso = (Rf / 2.0) ** 3
print(f"\n[路线1] 仅径向收缩 V_f/V_0 = (R_f/R_0)^2 = {V_rad:.4f}")
print(f"        各向同性     V_f/V_0 = (R_f/R_0)^3 = {V_iso:.4f}")

# 路线2：附录4 密度式 + 干物质守恒
rho_d0_app4 = (760 + 90 * C0) / (1 + C0)
rho_d_dry_app4 = 760.0 / 1.0     # C->0
V_ratio_app4 = rho_d0_app4 / rho_d_dry_app4
R_pred_app4 = 2.0 * np.sqrt(V_ratio_app4)
print(f"\n[路线2] 附录4: rho_d(C=2.55) = (760+90*2.55)/3.55 = {rho_d0_app4:.2f} kg/m3")
print(f"        rho_d(C->0) = 760/1 = {rho_d_dry_app4:.1f} kg/m3")
print(f"        干物质守恒 V_f/V_0 = {rho_d0_app4:.2f}/{rho_d_dry_app4:.1f} = {V_ratio_app4:.4f}")
print(f"        -> R_f = 2*sqrt({V_ratio_app4:.4f}) = {R_pred_app4:.4f} cm"
      f"   (实测 {Rf})  偏差 {abs(R_pred_app4-Rf)/Rf*100:.2f}%")

# 路线3：失水体积法
V_water_per_dry = 2.45 / 1000.0   # m3 水 / kg干物质 (失水2.45 kg/kg, 水密度1000)
V0_per_dry = 1.0 / rho_d0_app4
dV_ratio = V_water_per_dry / V0_per_dry
print(f"\n[路线3] 失水体积法: 单位干物质初始体积 = 1/{rho_d0_app4:.2f} = {V0_per_dry:.6f} m3/kg")
print(f"        失水2.45kg/kg 体积 = {V_water_per_dry:.6f} m3/kg")
print(f"        ΔV/V_0 = {dV_ratio:.4f} -> V_f/V_0 = {1-dV_ratio:.4f}")
R_pred_r3 = 2.0 * np.sqrt(max(1 - dV_ratio, 1e-9))
print(f"        -> R_f = {R_pred_r3:.4f} cm")

print(f"\n{'路线':<28}{'V_f/V_0':>10}{'R_f(cm)':>10}{'vs实测偏差':>12}")
print("-" * 62)
for tag, vr in (("实测(仅径向)", V_rad),
                ("实测(各向同性)", V_iso),
                ("附录4密度式+干物质守恒", V_ratio_app4),
                ("失水体积法", 1 - dV_ratio)):
    rf = 2.0 * np.sqrt(max(vr, 1e-9))
    print(f"{tag:<28}{vr:>10.4f}{rf:>10.4f}{abs(rf-Rf)/Rf*100:>11.2f}%")

print("""
>>> 结论：附录4密度式(路线2)给出 R_f=1.2114 cm，与实测 1.198 cm 仅差约1.1%。
    我此前用 "V比0.3588 反推失水108.86%" 得出"矛盾"的说法是【错误的】——
    错在假设了干骨架体积不变(只有水占的体积消失)，
    而附录4的密度式本身就含有骨架压实(rho_d 从278.7涨到760)。
    正确结论：收缩数据与附录4参数自洽，且明确否定各向同性收缩假设。
""")
