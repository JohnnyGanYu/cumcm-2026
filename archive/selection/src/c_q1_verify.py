"""问题1 独立复核: scipy.linprog 交叉验证 + 解可行性逐条审计 + 弃光必要性实验."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
import cvxpy as cp

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "problem" / "C题" / "附件" / "附件1.xlsx"
CSV = ROOT / "results" / "q1_schedule.csv"

DT = 1 / 6
T = 144
SOC_LO, SOC_HI, SOC_INIT = 1200.0, 10800.0, 6000.0
PMAX, ETA = 5000.0, 0.90
E_STEP = PMAX * DT

df = pd.read_excel(XL, sheet_name="Sheet1")
price = df["电价"].to_numpy(float)
load_e = df["小区负载"].to_numpy(float) * DT
pv_e = df["光伏发电预测功率"].to_numpy(float) * DT
net_e = load_e - pv_e

# ============ 1. scipy.linprog 独立建模 (变量 x = [g, ch, dis], SOC 消元为累积和) ============
# SOC_t = 6000 + sum_{k<t}(eta*ch_k - dis_k/eta)
# 约束: SOC_LO <= SOC_t <= SOC_HI for t=1..143 ; SOC_144 == 6000
n = 3 * T
c = np.concatenate([price, np.zeros(T), np.zeros(T)])

# 平衡: -(g + pv + dis) + load + ch <= 0  ->  -g + ch - dis <= pv - load = -net
A_bal = sparse.hstack([-sparse.eye(T), sparse.eye(T), -sparse.eye(T)])
b_bal = -net_e

L = sparse.csr_matrix(np.tril(np.ones((T, T)), -1))   # 前缀和(不含当期), 行 t 对应 SOC_t, t=0..143
S = sparse.hstack([sparse.csr_matrix((T, T)), ETA * L, -(1 / ETA) * L])  # SOC_t - 6000
A_ub = sparse.vstack([A_bal, S, -S])
b_ub = np.concatenate([b_bal,
                       np.full(T, SOC_HI - SOC_INIT),
                       np.full(T, SOC_INIT - SOC_LO)])
# 终端等式 SOC_144 = 6000 -> eta*sum(ch) - sum(dis)/eta = 0
A_eq = sparse.hstack([sparse.csr_matrix((1, T)),
                      sparse.csr_matrix(np.full((1, T), ETA)),
                      sparse.csr_matrix(np.full((1, T), -1 / ETA))])
b_eq = np.array([0.0])
bounds = [(0, None)] * T + [(0, E_STEP)] * T + [(0, E_STEP)] * T

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
print("=" * 78)
print("1. scipy.optimize.linprog (HiGHS) 独立复核")
print("=" * 78)
print("status:", res.status, res.message.strip())
print(f"目标值 = {res.fun:.6f} 元")
g2, ch2, dis2 = res.x[:T], res.x[T:2 * T], res.x[2 * T:]
print(f"购电量 = {g2.sum():.6f} kWh, 充 = {ch2.sum():.4f}, 放 = {dis2.sum():.4f}")

# 与 cvxpy 结果比较
sched = pd.read_csv(CSV)
g1 = sched["购电kWh"].to_numpy(float)
obj1 = float(np.dot(price, g1))
print(f"cvxpy 解重算目标 = {obj1:.6f} 元;  两者相对差 = {abs(obj1-res.fun)/max(1,abs(res.fun)):.3e}")
print(f"逐时段购电量最大绝对差 = {np.abs(g1-g2).max():.6f} kWh  (LP 最优解可能非唯一, 目标值一致才是判据)")

# ============ 2. cvxpy 解的逐条约束审计 ============
print()
print("=" * 78)
print("2. cvxpy 最优解逐条约束审计 (独立于求解器重算)")
print("=" * 78)
ch1 = sched["充电kWh"].to_numpy(float)
dis1 = sched["放电kWh"].to_numpy(float)
soc = np.empty(T + 1)
soc[0] = SOC_INIT
for t in range(T):
    soc[t + 1] = soc[t] + ETA * ch1[t] - dis1[t] / ETA
bal_slack = g1 + pv_e + dis1 - load_e - ch1
checks = {
    "非负 g>=0": g1.min() >= -1e-7,
    "非负 ch>=0": ch1.min() >= -1e-7,
    "非负 dis>=0": dis1.min() >= -1e-7,
    "ch <= 833.3333": ch1.max() <= E_STEP + 1e-6,
    "dis <= 833.3333": dis1.max() <= E_STEP + 1e-6,
    "供电平衡 >= 0": bal_slack.min() >= -1e-6,
    "SOC >= 1200": soc.min() >= SOC_LO - 1e-6,
    "SOC <= 10800": soc.max() <= SOC_HI + 1e-6,
    "SOC_0 = 6000": abs(soc[0] - SOC_INIT) < 1e-6,
    "SOC_144 = 6000": abs(soc[T] - SOC_INIT) < 1e-6,
    "SOC <= 铭牌12000": soc.max() <= 12000 + 1e-6,
    "无同时充放": int(np.sum((ch1 > 1e-6) & (dis1 > 1e-6))) == 0,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print(f"  SOC 独立递推 min={soc.min():.4f} max={soc.max():.4f} 末值={soc[T]:.6f}")
print(f"  平衡松弛 (弃光) 合计 = {bal_slack.sum():.8f} kWh -> 最优解{'不弃光' if bal_slack.sum() < 1e-6 else '有弃光'}")
print(f"  全部检查: {'ALL PASS' if all(checks.values()) else 'HAS FAILURE'}")

# ============ 3. 弃光必要性实验 ============
print()
print("=" * 78)
print("3. 弃光(净负载为负)必要性实验")
print("=" * 78)
neg = np.where(net_e < 0)[0]
print(f"净负载为负时段: {len(neg)} 个 (索引 {neg.min()}..{neg.max()}), 过剩合计 {(-net_e[neg]).sum():.2f} kWh")
print(f"过剩时段单时段最大过剩 {(-net_e[neg]).max():.2f} kWh << 单时段充电上限 {E_STEP:.2f} kWh")


def solve(*, storage=True, curtail=True, ch_cap=E_STEP, tag=""):
    g = cp.Variable(T, nonneg=True)
    ch = cp.Variable(T, nonneg=True)
    dis = cp.Variable(T, nonneg=True)
    s = cp.Variable(T + 1)
    cons = [s[1:] == s[:-1] + ETA * ch - dis / ETA,
            s >= SOC_LO, s <= SOC_HI, s[0] == SOC_INIT, s[T] == SOC_INIT,
            ch <= ch_cap, dis <= E_STEP]
    cons.append(g + pv_e + dis >= load_e + ch if curtail else g + pv_e + dis == load_e + ch)
    if not storage:
        cons += [ch == 0, dis == 0]
    p = cp.Problem(cp.Minimize(price @ g), cons)
    try:
        p.solve(solver=cp.HIGHS)
    except cp.error.SolverError as e:  # noqa: BLE001
        print(f"  {tag}: SolverError {e}")
        return None
    slack = None
    if g.value is not None:
        slack = float(np.sum(g.value + pv_e + np.asarray(dis.value).ravel() - load_e - np.asarray(ch.value).ravel()))
    print(f"  {tag}: status={p.status}, obj={'None' if p.value is None else f'{p.value:.4f}'}"
          + (f", 弃光={slack:.4f} kWh" if slack is not None else ""))
    return p.status


print("(a) 允许弃光 + 有储能           ", end="")
solve(tag="")
print("(b) 禁止弃光(等式) + 有储能      ", end="")
solve(curtail=False, tag="")
print("(c) 禁止弃光(等式) + 禁用储能    ", end="")
solve(curtail=False, storage=False, tag="")
print("(d) 禁止弃光 + 储能充电上限压到 200 kWh/时段 ", end="")
solve(curtail=False, ch_cap=200.0, tag="")
print("(e) 允许弃光 + 储能充电上限压到 200 kWh/时段 ", end="")
solve(ch_cap=200.0, tag="")
print()
print("结论: 本题数据下过剩电量小(单时段<=352.92 kWh)且储能余量充足, 储能可全额吸收过剩,")
print("      因此 '允许弃光(>=)' 与 '强制消纳(=)' 最优值相同、最优解不弃光; 但一旦充电能力受限,")
print("      等式形式会变不可行, 所以模型应写成不等式(留弃光自由度)才鲁棒。")
