#!/usr/bin/env python3
"""A题 尺度自洽性体检：判断"1800s内中心水分不变"是物理真实还是数值假象。

关键：比较热扩散与水分扩散的特征时间尺度。
"""
import numpy as np

R = 0.02          # m, 半径2cm
L = 0.25          # m, 长25cm
C0 = 2.55         # kg/kg 干基含水率
T0, Tair = 28.0, 50.0

print("=" * 78)
print("A题 尺度自洽性体检")
print("=" * 78)

# ---------- 附录2 参数（问题1专用，常物性）----------
rho, cp, k = 820.0, 2600.0, 0.36
h, hm = 25.0, 8e-7


def D_app2(C):
    return 7e-9 * np.exp(-0.89 / C)


alpha = k / (rho * cp)
print("\n【附录2 参数（问题1）】")
print(f"  热扩散系数 alpha = k/(rho*cp) = {alpha:.6e} m2/s")
print(f"  热特征时间 tau_T = R^2/alpha  = {R**2/alpha:.1f} s = {R**2/alpha/60:.2f} min")
print(f"  Biot(热) = h*R/k = {h*R/k:.4f}   ", end="")
print("(<0.1 集中参数适用)" if h * R / k < 0.1 else "(>0.1 必须解PDE，内部有梯度)")

for C in (C0, 1.0, 0.5, 0.15, 0.05):
    D = D_app2(C)
    print(f"  D(C={C:>5.2f}) = {D:.6e} m2/s   tau_C = R^2/D = {R**2/D:.4e} s"
          f" = {R**2/D/3600:.2f} h")

D0 = D_app2(C0)
print(f"\n  水分/热 时间尺度比 tau_C/tau_T = {(R**2/D0)/(R**2/alpha):.1f} 倍")
print(f"  Biot(质) = hm*R/D(C0) = {hm*R/D0:.4f}   ", end="")
print("(>>1 表面传质受内部扩散控制)" if hm * R / D0 > 10 else "")

# 1800s 内水分扩散渗透深度
for t in (100, 600, 1800, 14400):
    delta = np.sqrt(D0 * t)
    print(f"  t={t:>6}s  水分渗透深度 sqrt(D*t) = {delta*1000:.4f} mm"
          f"  (占半径 {delta/R*100:.2f}%)")

print("\n  >>> 结论: 附录2的D在C=2.55时约", f"{D0:.2e}", "m2/s,")
print("      tau_C =", f"{R**2/D0/3600:.1f}", "h >> 1800s,")
print("      所以问题1的1800s内中心水分几乎不变是【物理真实】，不是数值假象。")
print("      问题1的'预热阶段'本质是热传导主导，水分只在表层薄层变化。")

# ---------- 附录3 参数（问题2/3，变物性）----------
print("\n" + "=" * 78)
print("【附录3 参数（问题2/3）】")


def props_app3(C):
    rho_ = 650 + 128 * C
    cp_ = 1450 + 2736 * C / (C + 1)
    k_ = 0.21 + 0.38 * C / (C + 1)
    return rho_, cp_, k_


def D_app3(C, T_K):
    return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T_K)


for C in (C0, 1.0, 0.5, 0.15):
    for Tc in (28.0, 50.0):
        TK = Tc + 273.15
        D = D_app3(C, TK)
        r_, c_, k_ = props_app3(C)
        al = k_ / (r_ * c_)
        print(f"  C={C:>5.2f} T={Tc:>5.1f}C: D={D:.4e} m2/s  tau_C={R**2/D/3600:8.2f} h"
              f" | rho={r_:6.1f} cp={c_:6.1f} k={k_:.4f} alpha={al:.3e}"
              f" tau_T={R**2/al/60:6.2f} min")

print("\n  >>> 问题3要求各处C<0.15。看C=0.15时的tau_C：")
TK50 = 50.0 + 273.15
D_end = D_app3(0.15, TK50)
print(f"      D(0.15, 50C) = {D_end:.4e} m2/s, tau_C = {R**2/D_end/3600:.2f} h")
D_start = D_app3(C0, TK50)
print(f"      D(2.55, 50C) = {D_start:.4e} m2/s, tau_C = {R**2/D_start/3600:.2f} h")
print(f"      D 变化倍数 = {D_start/D_end:.1f}x  -> 强非线性，干燥后期极慢")

# ---------- 附录4 参数（问题4，含收缩）----------
print("\n" + "=" * 78)
print("【附录4 参数（问题4，含尺寸收缩）】")


def props_app4(C):
    return 760 + 90 * C, 1850 + 2150 * C / (C + 1), 0.12 + 0.20 * C / (C + 1)


def D_app4(C, T_K):
    return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / T_K)


for C in (C0, 1.0, 0.5, 0.15):
    TK = TK50
    D = D_app4(C, TK)
    print(f"  C={C:>5.2f} T=50C: D={D:.4e} m2/s  tau_C={R**2/D/3600:8.2f} h")

# 收缩：附件2 半径 2.000 -> 1.198 cm
r_end = 0.01198
print(f"\n  收缩: R 2.000cm -> 1.198cm, 体积比 = {(r_end/R)**2:.4f}")
print(f"  若固体骨架质量守恒, 干基含水率从 {C0} 降到 C_end:")
print("    m_dry 不变; V*rho = m_dry*(1+C)")
for Cend in (0.15, 0.05):
    rho0, _, _ = props_app4(C0)
    rho1, _, _ = props_app4(Cend)
    V_ratio = ((1 + Cend) / rho1) / ((1 + C0) / rho0)
    r_pred = R * np.sqrt(V_ratio)
    print(f"    C_end={Cend}: 预测体积比={V_ratio:.4f} -> R={r_pred*100:.4f} cm"
          f"  (实测 1.198 cm)")

print("\n  >>> 附件2 半径在 t=241200s(67.0h) 后恒定 1.198cm")
print("      这既可能是'干燥完成骨架不再收缩'，也可能是数据在此截断。")
print("      两种解读会给出不同的问题4答案，论文必须明确说明所采用的解读及依据。")
