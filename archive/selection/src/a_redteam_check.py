#!/usr/bin/env python3
"""核实红队对A题的两条关键断言，避免误信。

断言1: 附录2密度820 与 附录3公式在C=2.55时给出976.4，"不一致19%"
断言2: 传质边界条件 -D dC/dr = hm(C_surf - C_air) 量纲/基准不匹配
"""
import numpy as np
import pandas as pd

R0 = 0.02
C0 = 2.55

print("=" * 78)
print("核实红队断言1：附录2/附录3 密度是否真的矛盾")
print("=" * 78)

rho_app2 = 820.0
rho_app3_at_C0 = 650 + 128 * C0
rho_app4_at_C0 = 760 + 90 * C0
print(f"  附录2 给定常数        rho = {rho_app2}")
print(f"  附录3 公式 @C=2.55    rho = 650+128*2.55 = {rho_app3_at_C0}")
print(f"  附录4 公式 @C=2.55    rho = 760+90*2.55  = {rho_app4_at_C0}")
print(f"\n  附录3 vs 附录2 相对差 = {(rho_app3_at_C0-rho_app2)/rho_app2*100:+.1f}%")
print(f"  附录4 vs 附录2 相对差 = {(rho_app4_at_C0-rho_app2)/rho_app2*100:+.1f}%")

# 关键：若 rho 是"湿基总密度"(含水)，则干骨架密度应一致
print("\n  检验：若 rho 为湿物料总密度，干骨架密度 rho_dry = rho/(1+C)")
for name, rr in (("附录2", rho_app2), ("附录3", rho_app3_at_C0), ("附录4", rho_app4_at_C0)):
    print(f"    {name}: rho_dry = {rr:.1f}/(1+2.55) = {rr/(1+C0):.2f} kg/m3")

print("\n  检验：附录3/4 公式在 C=0 (全干) 时的截距")
print(f"    附录3: rho(C=0) = 650 kg/m3   附录4: rho(C=0) = 760 kg/m3")
print("    -> 截距即干骨架表观密度。两附录截距不同(650 vs 760)，")
print("       说明附录3与附录4本身就描述不同情形(附录4含收缩，骨架被压实→密度更高)。")
print("    -> 附录4骨架密度高于附录3，与'收缩使体积减小、骨架致密'物理一致。")

# 用附录4验证收缩自洽性
print("\n" + "=" * 78)
print("核实：附录4 密度公式 + 附件2 收缩数据 是否自洽（骨架质量守恒）")
print("=" * 78)
a2 = pd.read_excel("problem/A题/附件/附件2.xlsx")
a2.columns = ["t", "r_cm"]
r_end_cm = a2["r_cm"].iloc[-1]
print(f"  附件2: R 从 {a2['r_cm'].iloc[0]} cm 收缩到 {r_end_cm} cm")

# 骨架质量守恒: V(t)*rho(t)/(1+C(t)) = const
def rho4(C):
    return 760 + 90 * C


V_ratio_obs = (r_end_cm / a2["r_cm"].iloc[0]) ** 2   # 半径平方比(长度方向按题设不变或等比?)
print(f"  观测体积比(仅径向收缩, 长度不变) = (r_end/r0)^2 = {V_ratio_obs:.4f}")
V_ratio_iso = (r_end_cm / a2["r_cm"].iloc[0]) ** 3   # 各向同性收缩
print(f"  观测体积比(各向同性收缩)        = (r_end/r0)^3 = {V_ratio_iso:.4f}")

m_dry_0 = rho4(C0) / (1 + C0)      # 单位体积干骨架
print(f"\n  单位初始体积的干骨架质量 = rho4(2.55)/(1+2.55) = {m_dry_0:.3f} kg/m3")
print("\n  反解: 给定末态 C_end, 骨架守恒要求的体积比")
for Cend in (0.30, 0.20, 0.15, 0.10, 0.05, 0.02):
    m_dry_1 = rho4(Cend) / (1 + Cend)
    V_ratio_need = m_dry_0 / m_dry_1
    r_need_rad = a2["r_cm"].iloc[0] * np.sqrt(V_ratio_need)
    r_need_iso = a2["r_cm"].iloc[0] * V_ratio_need ** (1 / 3)
    print(f"    C_end={Cend:5.2f}: V比={V_ratio_need:.4f} -> R(仅径向)={r_need_rad:.4f}cm"
          f"  R(各向同性)={r_need_iso:.4f}cm")
print(f"\n  实测末态 R = {r_end_cm} cm")
print("  -> 对比可判断哪种收缩假设与数据吻合。")

print("\n" + "=" * 78)
print("核实红队断言2：传质边界条件的浓度基准")
print("=" * 78)
a1 = pd.read_excel("problem/A题/附件/附件1.xlsx")
a1.columns = ["t", "Tair", "Cair"]
print(f"  附件1 烘房'水分浓度' 范围: {a1['Cair'].min():.5f} ~ {a1['Cair'].max():.5f} kg/kg")
print(f"  药材初始 干基含水率 C0 = {C0} kg/kg")
print(f"  两者比值 = {C0/a1['Cair'].max():.1f} 倍")

# 若 Cair 是空气绝对湿度(kg水/kg干空气)，在50C的饱和湿度是多少？
def p_sat(Tc):
    """Magnus公式, 饱和水蒸气压 Pa"""
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


for Tc in (28.0, 50.0):
    ps = p_sat(Tc)
    w_sat = 0.622 * ps / (101325 - ps)
    print(f"\n  T={Tc}C: 饱和蒸气压 {ps:.0f} Pa, 饱和绝对湿度 w_sat = {w_sat:.5f} kg/kg(干空气)")

w28 = 0.622 * p_sat(28) / (101325 - p_sat(28))
w50 = 0.622 * p_sat(50) / (101325 - p_sat(50))
print(f"\n  附件1 起始 Cair={a1['Cair'].iloc[0]:.5f} -> 若为绝对湿度, 相对湿度@28C = "
      f"{a1['Cair'].iloc[0]/w28*100:.1f}%")
print(f"  附件1 末端 Cair={a1['Cair'].iloc[-1]:.5f} -> 若为绝对湿度, 相对湿度@50C = "
      f"{a1['Cair'].iloc[-1]/w50*100:.1f}%")
print("\n  >>> 若上面两个相对湿度都落在 0~100% 的合理区间，")
print("      则'烘房水分浓度'确实可解释为空气绝对湿度(kg水/kg干空气)，")
print("      与药材干基含水率(kg水/kg干物质)是【不同基准】，红队断言2成立。")

# 量纲核查
print("\n  量纲核查: -D*dC/dr = hm*(C_s - C_air)")
D_typ = 4.94e-9
hm = 8e-7
print(f"    左侧: D*dC/dr ~ [m2/s]*[kg/kg]/[m] = {D_typ*1.0/1e-3:.3e} [m/s * kg/kg]")
print(f"    右侧: hm*dC    ~ [m/s]*[kg/kg]     = {hm*1.0:.3e} [m/s * kg/kg]")
print("    -> 两侧量纲一致(都是 m/s × kg/kg)，若两侧C同为干基则量纲无矛盾。")
print("    -> 但物理上 C_air(空气基准) 与 C_s(固体基准) 不同基准，")
print("       直接相减缺乏平衡关系支撑。这是建模假设，必须在论文中显式声明。")

# 定量影响评估：改变边界条件写法对结果影响多大
print("\n  定量影响: 表面传质 Biot 数")
for D_ in (4.94e-9, 1.35e-8, 8.0e-10):
    Bi_m = hm * R0 / D_
    print(f"    D={D_:.2e}: Bi_m = hm*R/D = {Bi_m:7.3f}  ", end="")
    if Bi_m > 10:
        print("(内部扩散控制, 表面系数影响小)")
    elif Bi_m < 0.1:
        print("(表面对流控制, 边界写法极敏感)")
    else:
        print("(过渡区, 两者都重要)")
