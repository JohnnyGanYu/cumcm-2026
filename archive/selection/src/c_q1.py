"""CUMCM2026 C题 问题1: 微网日前购电计划 —— 线性规划建模与求解.

模型 (主约定 A):
  索引 t = 0..143, 每个时段长 dt = 10 min = 1/6 h.
  附件1 第 t 行的时间标签是该时段的 *结束* 时刻 (第0行 00:10 => 时段 00:00-00:10).

  决策变量 (全部单位 kWh, 均 >= 0):
    g_t   : 该时段从外网购电量
    ch_t  : 该时段储能从母线吸收的电量 (母线侧计量)
    dis_t : 该时段储能向母线输出的电量 (母线侧计量)

  功率(电量)平衡 (允许弃光, 不允许倒送):
    g_t + pv_t*dt + dis_t >= load_t*dt + ch_t

  储能状态:
    SOC_{t+1} = SOC_t + eta*ch_t - dis_t/eta      (约定 A, 效率双侧施加, 往返效率 eta^2 = 0.81)
    SOC_0 = SOC_144 = 6000
    1200 <= SOC_t <= 10800   (t = 0..144)

  功率上限 (母线侧):
    0 <= ch_t <= Pmax*dt, 0 <= dis_t <= Pmax*dt, Pmax = 5000 kW

  目标:  min sum_t price_t * g_t

同时求解的对照 / 敏感性场景:
  A_no_storage : 禁用储能 (ch = dis = 0)
  B_single_eta : SOC_{t+1} = SOC_t + ch_t - dis_t/eta (效率单侧施加, 往返 0.9)
  A_no_curtail : 平衡式改为等式 (必须全额消纳光伏, 不许弃光, 不许倒送)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "problem" / "C题" / "附件" / "附件1.xlsx"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ---------------- 参数 ----------------
DT = 1.0 / 6.0          # h
T = 144
CAP_MAX = 12000.0       # kWh 铭牌容量
SOC_LO, SOC_HI = 1200.0, 10800.0
SOC_INIT = 6000.0
PMAX = 5000.0           # kW
ETA = 0.90
E_STEP = PMAX * DT      # 单时段最大充/放电量 kWh = 833.333...

# ---------------- 数据 ----------------
df = pd.read_excel(XL, sheet_name="Sheet1")
price = df["电价"].to_numpy(float)
load_kw = df["小区负载"].to_numpy(float)
pv_kw = df["光伏发电预测功率"].to_numpy(float)
assert len(df) == T, len(df)

load_e = load_kw * DT     # kWh
pv_e = pv_kw * DT         # kWh
net_e = load_e - pv_e     # 净负载电量 kWh


def solve(name, *, storage=True, eta_mode="both", curtail=True, verbose=False):
    g = cp.Variable(T, nonneg=True, name="g")
    ch = cp.Variable(T, nonneg=True, name="ch")
    dis = cp.Variable(T, nonneg=True, name="dis")
    soc = cp.Variable(T + 1, name="soc")

    cons = []
    if curtail:
        bal = g + pv_e + dis >= load_e + ch
    else:
        bal = g + pv_e + dis == load_e + ch
    cons.append(bal)

    if eta_mode == "both":       # 约定 A: 充放各计一次 eta
        soc_dyn = soc[1:] == soc[:-1] + ETA * ch - dis / ETA
    elif eta_mode == "single":   # 约定 B: 只在放电侧计 eta
        soc_dyn = soc[1:] == soc[:-1] + ch - dis / ETA
    else:
        raise ValueError(eta_mode)
    cons.append(soc_dyn)

    c_lo = soc >= SOC_LO
    c_hi = soc <= SOC_HI
    cons += [c_lo, c_hi, soc[0] == SOC_INIT, soc[T] == SOC_INIT]

    c_ch = ch <= E_STEP
    c_dis = dis <= E_STEP
    cons += [c_ch, c_dis]

    if not storage:
        cons += [ch == 0, dis == 0]

    prob = cp.Problem(cp.Minimize(price @ g), cons)
    prob.solve(solver=cp.HIGHS, verbose=verbose)
    return dict(
        name=name, prob=prob, status=prob.status,
        obj=prob.value,
        g=None if g.value is None else np.asarray(g.value).ravel(),
        ch=None if ch.value is None else np.asarray(ch.value).ravel(),
        dis=None if dis.value is None else np.asarray(dis.value).ravel(),
        soc=None if soc.value is None else np.asarray(soc.value).ravel(),
        cons=dict(bal=bal, soc_lo=c_lo, soc_hi=c_hi, ch_cap=c_ch, dis_cap=c_dis),
    )


def hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def slot_label(i):
    return f"{hhmm(i*10)}-{hhmm((i+1)*10)}"


# =========================================================
print("=" * 78)
print("0. 数据与派生量核对")
print("=" * 78)
print(f"时段数 T={T}, dt={DT:.6f} h, 单时段充/放电量上限 = {E_STEP:.4f} kWh")
print(f"电价  min={price.min():.4f} max={price.max():.4f} 元/kWh")
print(f"负载  min={load_kw.min():.2f} max={load_kw.max():.2f} kW; 全天用电 {load_e.sum():.2f} kWh")
print(f"光伏  min={pv_kw.min():.2f} max={pv_kw.max():.2f} kW; 全天发电 {pv_e.sum():.2f} kWh")
neg = np.where(net_e < -1e-9)[0]
print(f"净负载为负(光伏过剩)的时段数 = {len(neg)}")
if len(neg):
    print(f"  区间: {slot_label(neg[0])} .. {slot_label(neg[-1])} (索引 {neg[0]}..{neg[-1]}, 连续={np.all(np.diff(neg)==1)})")
    print(f"  过剩电量合计 = {(-net_e[neg]).sum():.2f} kWh, 单时段最大过剩 = {(-net_e[neg]).max():.2f} kWh")
print(f"净负载为正时段的用电需求合计 = {net_e[net_e>0].sum():.2f} kWh")

print()
print("=" * 78)
print("1. 主模型 A (允许弃光, 效率双侧 eta=0.9, 往返 0.81)")
print("=" * 78)
A = solve("A_main")
print(f"求解器 = HiGHS, status = {A['status']}")
if A["status"] not in ("optimal", "optimal_inaccurate"):
    raise SystemExit(f"主模型不可行/异常: {A['status']}")
print(f"最优目标值(全天购电费) = {A['obj']:.4f} 元")
print(f"全天总购电量 = {A['g'].sum():.4f} kWh")
print(f"平均购电单价 = {A['obj']/A['g'].sum():.4f} 元/kWh")
print(f"全天充电量合计 = {A['ch'].sum():.4f} kWh, 放电量合计 = {A['dis'].sum():.4f} kWh")
print(f"储能循环损耗 = {A['ch'].sum() - A['dis'].sum():.4f} kWh")
# 弃光量
surplus = A["g"] + pv_e + A["dis"] - load_e - A["ch"]
print(f"平衡式松弛(=弃光量)合计 = {surplus.sum():.4f} kWh, 最大单时段 {surplus.max():.4f} kWh")
print(f"同时充放电时段数 (ch>1e-6 且 dis>1e-6) = {int(np.sum((A['ch']>1e-6)&(A['dis']>1e-6)))}")

# ---------------- 表1 ----------------
print()
print("=" * 78)
print("2. 表1 —— 指定时段购电量 (主模型 A)")
print("=" * 78)
tab1_starts = [600, 720, 840, 960, 1080, 1200]   # 10:00 12:00 14:00 16:00 18:00 20:00
rows1 = []
for m in tab1_starts:
    i = m // 10
    rows1.append(dict(时段=slot_label(i), 行标签=str(df.loc[i, "时间"]),
                      电价=round(price[i], 4),
                      负载kWh=round(load_e[i], 4), 光伏kWh=round(pv_e[i], 4),
                      购电量kWh=round(float(A["g"][i]), 4),
                      充电kWh=round(float(A["ch"][i]), 4),
                      放电kWh=round(float(A["dis"][i]), 4)))
t1 = pd.DataFrame(rows1)
print(t1.to_string(index=False))
print(f"\n全天购电量 = {A['g'].sum():.4f} kWh")
print(f"全天购电费 = {A['obj']:.4f} 元")

# ---------------- 表2 ----------------
print()
print("=" * 78)
print("3. 表2 —— 储能 4 小时区间充/放电合计 (主模型 A)")
print("=" * 78)
rows2 = []
for k in range(6):
    lo, hi = k * 24, (k + 1) * 24          # 24 个 10min 时段 = 4 h
    rows2.append(dict(区间=f"{hhmm(k*240)}-{hhmm((k+1)*240)}",
                      充电量kWh=round(float(A["ch"][lo:hi].sum()), 4),
                      放电量kWh=round(float(A["dis"][lo:hi].sum()), 4),
                      区间末SOC=round(float(A["soc"][hi]), 4)))
t2 = pd.DataFrame(rows2)
print(t2.to_string(index=False))
print(f"\n0:00 储电量  = {A['soc'][0]:.4f} kWh")
print(f"24:00 储电量 = {A['soc'][T]:.4f} kWh")
print(f"充电合计 {A['ch'].sum():.4f} kWh / 放电合计 {A['dis'].sum():.4f} kWh")
print(f"SOC 轨迹 min={A['soc'].min():.4f} max={A['soc'].max():.4f} kWh")

# ---------------- 校验 5: 紧约束 / 套利行为 ----------------
print()
print("=" * 78)
print("4. 约束活跃性 (active/紧约束) 检查")
print("=" * 78)
tol = 1e-6
soc_at_lo = np.where(A["soc"] <= SOC_LO + 1e-4)[0]
soc_at_hi = np.where(A["soc"] >= SOC_HI - 1e-4)[0]
ch_at_cap = np.where(A["ch"] >= E_STEP - 1e-4)[0]
dis_at_cap = np.where(A["dis"] >= E_STEP - 1e-4)[0]
bal_tight = np.where(surplus <= 1e-6)[0]
print(f"SOC 触下限 1200 的节点数 = {len(soc_at_lo)}  节点索引前后: {soc_at_lo[:5]}...{soc_at_lo[-5:] if len(soc_at_lo) else ''}")
print(f"SOC 触上限 10800 的节点数 = {len(soc_at_hi)}  节点索引前后: {soc_at_hi[:5]}...{soc_at_hi[-5:] if len(soc_at_hi) else ''}")
print(f"充电到功率上限的时段数 = {len(ch_at_cap)}  {ch_at_cap}")
print(f"放电到功率上限的时段数 = {len(dis_at_cap)}  {dis_at_cap}")
print(f"平衡式取等(无弃光)的时段数 = {len(bal_tight)} / {T}")
print(f"铭牌容量 12000 kWh 是否成为紧约束: SOC 最大 {A['soc'].max():.2f} kWh -> {'否 (被 10800 上限先约束)' if A['soc'].max() < CAP_MAX - 1e-6 else '是'}")

# 对偶
try:
    dual_bal = np.asarray(A["cons"]["bal"].dual_value).ravel()
    print(f"平衡式对偶变量(影子价格 元/kWh) min={dual_bal.min():.4f} max={dual_bal.max():.4f}, 非零时段数={int(np.sum(np.abs(dual_bal)>1e-8))}")
except Exception as e:  # noqa: BLE001
    print("对偶变量不可用:", e)

print()
print("--- 峰谷套利行为: 充/放电时段的电价分布 ---")
ch_idx = np.where(A["ch"] > 1e-6)[0]
dis_idx = np.where(A["dis"] > 1e-6)[0]
if len(ch_idx):
    w = A["ch"][ch_idx]
    print(f"充电时段数={len(ch_idx)}, 电价 min={price[ch_idx].min():.4f} max={price[ch_idx].max():.4f}, "
          f"电量加权均价={float(np.dot(price[ch_idx], w)/w.sum()):.4f}")
    print(f"  充电时段(索引): {ch_idx.tolist()}")
if len(dis_idx):
    w = A["dis"][dis_idx]
    print(f"放电时段数={len(dis_idx)}, 电价 min={price[dis_idx].min():.4f} max={price[dis_idx].max():.4f}, "
          f"电量加权均价={float(np.dot(price[dis_idx], w)/w.sum()):.4f}")
    print(f"  放电时段(索引): {dis_idx.tolist()}")
print(f"套利判据: 放电均价 {'>' if len(dis_idx) and len(ch_idx) and np.dot(price[dis_idx],A['dis'][dis_idx])/A['dis'][dis_idx].sum() > np.dot(price[ch_idx],A['ch'][ch_idx])/A['ch'][ch_idx].sum() else '<='} 充电均价")

print()
print("每小时汇总 (电价均值 / 负载 / 光伏 / 购电 / 充电 / 放电 / 小时末SOC):")
hdr = f"{'hour':>5} {'price':>8} {'load':>10} {'pv':>10} {'buy':>10} {'chg':>9} {'dis':>9} {'SOC':>9}"
print(hdr)
for h in range(24):
    lo, hi = h * 6, (h + 1) * 6
    print(f"{h:>5} {price[lo:hi].mean():>8.4f} {load_e[lo:hi].sum():>10.2f} {pv_e[lo:hi].sum():>10.2f} "
          f"{A['g'][lo:hi].sum():>10.2f} {A['ch'][lo:hi].sum():>9.2f} {A['dis'][lo:hi].sum():>9.2f} {A['soc'][hi]:>9.2f}")

# ---------------- 场景对照 ----------------
print()
print("=" * 78)
print("5. 对照场景")
print("=" * 78)
NS = solve("A_no_storage", storage=False)
print(f"[禁用储能] status={NS['status']}, 购电费={NS['obj']:.4f} 元, 购电量={NS['g'].sum():.4f} kWh")
manual = float(np.dot(price, np.maximum(net_e, 0.0)))
print(f"  解析核对 sum(price*max(net,0)) = {manual:.4f} 元 (差 {abs(manual-NS['obj']):.2e})")
saving = NS["obj"] - A["obj"]
print(f"  => 储能经济价值 = {saving:.4f} 元/天 ({saving/NS['obj']*100:.3f}% of 无储能成本)")

B = solve("B_single_eta", eta_mode="single")
print(f"[约定B 单侧效率, 往返0.9] status={B['status']}, 购电费={B['obj']:.4f} 元, 购电量={B['g'].sum():.4f} kWh, "
      f"充={B['ch'].sum():.2f} 放={B['dis'].sum():.2f}")
print(f"  与主模型 A 差 = {A['obj']-B['obj']:.4f} 元 (A 更贵, 因往返效率 0.81 < 0.9)")

NC = solve("A_no_curtail", curtail=False)
print(f"[强制全额消纳(等式,禁弃光禁倒送)] status={NC['status']}", end="")
if NC["obj"] is not None:
    print(f", 购电费={NC['obj']:.4f} 元")
else:
    print(" -> 不可行, 说明光伏过剩必须允许弃光或倒送")

# ---------------- 落盘 ----------------
detail = pd.DataFrame(dict(
    idx=np.arange(T), 时段=[slot_label(i) for i in range(T)],
    电价=price, 负载kW=load_kw, 光伏kW=pv_kw,
    负载kWh=load_e, 光伏kWh=pv_e, 净负载kWh=net_e,
    购电kWh=A["g"], 充电kWh=A["ch"], 放电kWh=A["dis"],
    弃光kWh=surplus, 时段末SOC=A["soc"][1:],
))
detail.to_csv(OUT / "q1_schedule.csv", index=False, encoding="utf-8-sig")
t1.to_csv(OUT / "q1_table1.csv", index=False, encoding="utf-8-sig")
t2.to_csv(OUT / "q1_table2.csv", index=False, encoding="utf-8-sig")

summary = dict(
    solver="HiGHS (via cvxpy)",
    dt_h=DT, T=T, eta=ETA, e_step_kwh=E_STEP,
    soc_bounds=[SOC_LO, SOC_HI], soc_init=SOC_INIT, cap_nameplate=CAP_MAX,
    main=dict(status=A["status"], cost_yuan=float(A["obj"]), buy_kwh=float(A["g"].sum()),
              charge_kwh=float(A["ch"].sum()), discharge_kwh=float(A["dis"].sum()),
              curtail_kwh=float(surplus.sum()),
              soc_min=float(A["soc"].min()), soc_max=float(A["soc"].max())),
    no_storage=dict(status=NS["status"], cost_yuan=float(NS["obj"]), buy_kwh=float(NS["g"].sum())),
    storage_value_yuan=float(saving),
    single_eta=dict(status=B["status"], cost_yuan=float(B["obj"])),
    no_curtail=dict(status=NC["status"], cost_yuan=None if NC["obj"] is None else float(NC["obj"])),
    table1=rows1, table2=rows2,
    pv_surplus_slots=int(len(neg)), pv_surplus_kwh=float((-net_e[neg]).sum()) if len(neg) else 0.0,
)
(OUT / "q1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("已写出:", OUT / "q1_schedule.csv", OUT / "q1_table1.csv", OUT / "q1_table2.csv", OUT / "q1_summary.json", sep="\n  ")
