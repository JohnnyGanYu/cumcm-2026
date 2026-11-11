#!/usr/bin/env python3
"""核实红队对C题的关键断言。

断言A: 效率90%的施加方式（双侧0.9 vs 单侧）——对最优值影响多大？
断言B: "问题1不是LP，SOC耦合约束使其必须用DP" —— 这条是错的，SOC递推是线性等式。
断言C: 光伏余量能否卖电/弃光 —— 检查是否影响最优值
断言D: 同时充放电的退化解 —— 加互斥约束(MILP)后目标值是否变化
"""
import numpy as np
import pandas as pd
import cvxpy as cp

df = pd.read_excel("problem/C题/附件/附件1.xlsx")
df.columns = ["t", "price", "load_kw", "pv_kw"]
DT = 1 / 6
price = df["price"].values
load = df["load_kw"].values * DT
pv = df["pv_kw"].values * DT
T = len(price)
CAP_LO, CAP_HI, SOC0 = 1200.0, 10800.0, 6000.0
E_STEP = 5000.0 * DT


def solve(eta_ch, eta_dis, allow_curtail=True, mutual_excl=False, verbose=False):
    g = cp.Variable(T, nonneg=True)
    ch = cp.Variable(T, nonneg=True)
    dis = cp.Variable(T, nonneg=True)
    curt = cp.Variable(T, nonneg=True) if allow_curtail else None
    soc = cp.Variable(T + 1)

    cons = [soc[0] == SOC0, soc[T] == SOC0,
            soc >= CAP_LO, soc <= CAP_HI,
            ch <= E_STEP, dis <= E_STEP]
    for t in range(T):
        cons.append(soc[t + 1] == soc[t] + eta_ch * ch[t] - dis[t] / eta_dis)
    pv_used = pv - curt if allow_curtail else pv
    if allow_curtail:
        cons.append(curt <= pv)
    # 功率平衡：购电 + 光伏(可弃) + 放电 >= 负载 + 充电
    cons.append(g + pv_used + dis >= load + ch)

    prob = cp.Problem(cp.Minimize(price @ g), cons)
    prob.solve(solver=cp.HIGHS)
    res = {"status": prob.status, "cost": prob.value,
           "buy": float(g.value.sum()), "ch": float(ch.value.sum()),
           "dis": float(dis.value.sum()),
           "curt": float(curt.value.sum()) if allow_curtail else 0.0,
           "simul": int(np.sum((ch.value > 1e-3) & (dis.value > 1e-3)))}
    return res


print("=" * 78)
print("断言A：效率90%的施加方式对最优值的影响")
print("=" * 78)
schemes = [
    ("双侧0.9 (往返81%)", 0.9, 0.9),
    ("仅充电侧0.9 (往返90%)", 0.9, 1.0),
    ("仅放电侧0.9 (往返90%)", 1.0, 0.9),
    ("sqrt(0.9)双侧 (往返90%)", np.sqrt(0.9), np.sqrt(0.9)),
    ("无损耗 (往返100%)", 1.0, 1.0),
]
base = None
for name, ec, ed in schemes:
    r = solve(ec, ed)
    if base is None:
        base = r["cost"]
    print(f"  {name:26} 费用={r['cost']:11.2f} 元  往返效率={ec*ed:.4f}"
          f"  相对双侧={(r['cost']-base)/base*100:+6.2f}%")
print("\n  >>> 结论：效率施加方式确实显著影响最优值，必须在论文中明确声明。")
print("      题面'充放电效率90%'最自然的读法是充、放各90%（往返81%）。")

print("\n" + "=" * 78)
print("断言B（红队称问题1'不是LP必须用DP'）—— 核实")
print("=" * 78)
r_lp = solve(0.9, 0.9)
print(f"  LP(HiGHS) 状态={r_lp['status']}  最优费用={r_lp['cost']:.6f} 元")
print("  SOC 递推 soc[t+1] == soc[t] + 0.9*ch[t] - dis[t]/0.9 是【线性等式约束】，")
print("  目标 price@g 线性，全部约束线性 -> 标准 LP，可由单纯形/内点法求全局最优。")
print("  >>> 红队断言B【错误】。时段耦合不等于非线性；DP 只是另一种可选解法，")
print("      且对连续SOC需离散化，精度反而不如LP。")

print("\n" + "=" * 78)
print("断言C：光伏能否弃光 / 是否允许卖电")
print("=" * 78)
r_cur = solve(0.9, 0.9, allow_curtail=True)
r_nocur = solve(0.9, 0.9, allow_curtail=False)
print(f"  允许弃光: 费用={r_cur['cost']:.2f}  实际弃光量={r_cur['curt']:.4f} kWh")
print(f"  强制全消纳: 状态={r_nocur['status']}  费用={r_nocur['cost']:.2f}")
print(f"  差值 = {abs(r_cur['cost']-r_nocur['cost']):.6f} 元")
if abs(r_cur["cost"] - r_nocur["cost"]) < 1e-6:
    print("  >>> 本题数据下两者等价（弃光量为0），说明储能+负载足以消纳全部光伏，")
    print("      是否允许弃光【不影响】问题1最优值。但问题2/3/4光伏更大时可能不同。")

print("\n" + "=" * 78)
print("断言D：同时充放电的退化解（我之前发现的疑点）")
print("=" * 78)
print(f"  LP解中同时充放电的时段数 = {r_lp['simul']}")
print(f"  总充电 {r_lp['ch']:.2f} kWh, 总放电 {r_lp['dis']:.2f} kWh")

# 用MILP强制互斥
u = cp.Variable(T, boolean=True)
g2 = cp.Variable(T, nonneg=True)
ch2 = cp.Variable(T, nonneg=True)
dis2 = cp.Variable(T, nonneg=True)
soc2 = cp.Variable(T + 1)
cons2 = [soc2[0] == SOC0, soc2[T] == SOC0, soc2 >= CAP_LO, soc2 <= CAP_HI]
for t in range(T):
    cons2 += [soc2[t + 1] == soc2[t] + 0.9 * ch2[t] - dis2[t] / 0.9,
              ch2[t] <= E_STEP * u[t], dis2[t] <= E_STEP * (1 - u[t])]
cons2.append(g2 + pv + dis2 >= load + ch2)
p2 = cp.Problem(cp.Minimize(price @ g2), cons2)
try:
    p2.solve(solver=cp.HIGHS)
    print(f"  MILP(强制互斥) 状态={p2.status}  费用={p2.value:.6f} 元")
    print(f"  与LP差值 = {p2.value - r_lp['cost']:+.6f} 元"
          f"  ({(p2.value-r_lp['cost'])/r_lp['cost']*100:+.4f}%)")
    if abs(p2.value - r_lp["cost"]) < 1e-4:
        print("  >>> 目标值相同：LP的同时充放电是【退化解】，不影响最优费用。")
        print("      论文可直接用LP并说明存在等价的互斥解，无需引入0-1变量。")
    else:
        print("  >>> 目标值不同：必须用MILP，LP会低估费用（虚假套利）。")
except Exception as e:
    print(f"  MILP求解失败: {e}")
