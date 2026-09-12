"""Summarize every pre-registered final pair; fail closed on missing evidence."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]


def aggregate(rows):
    times = [r['mean_clear_time_s'] for r in rows]
    cleared = sum(r['cleared'] for r in rows)
    total = sum(r['total_jammers'] for r in rows)
    return {
        'episodes':len(rows), 'cleared':cleared, 'total_sources':total,
        'full_clear_episodes':sum(r['cleared']==r['total_jammers'] for r in rows),
        'clear_rate':cleared/total,
        'arithmetic_mean_s':statistics.fmean(times),
        'pooled_mean_s':sum(r['total_virtual_time_s'] for r in rows)/cleared,
        'median_s':statistics.median(times),
        'p95_s':sorted(times)[math.ceil(.95*len(times))-1],
        'worst_s':max(times), 'within_450_episodes':sum(t<=450 for t in times),
        'mean_move_distance_m':statistics.fmean(r['move_distance_m'] for r in rows),
        'mean_measure_calls':statistics.fmean(r['measure_calls'] for r in rows),
        'mean_clear_calls':statistics.fmean(r['clear_calls'] for r in rows),
        'max_wall_s':max(r['wall_s'] for r in rows),
    }


def main():
    plan = json.loads((ROOT/'results/q4_450_final_plan.json').read_text(encoding='utf-8'))
    raw = json.loads((ROOT/'results/q4_450_final_validation.json').read_text(encoding='utf-8'))
    expected = {(s,d,v) for s in plan['seeds'] for d in plan['directional_counts'] for v in plan['arms']}
    rows = raw['records']
    raw['source_sha256'] = {name.replace('\\','/'): sha for name,sha in raw['source_sha256'].items()}
    assert len(rows)==len(expected) and {(r['seed'],r['directional'],r['variant']) for r in rows}==expected
    assert raw['source_unchanged']
    for name, sha in raw['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha, name
    for name, sha in plan['sha256'].items():
        assert raw['source_sha256'][name]==sha
    assert all('error' not in r and r['mean_clear_time_s'] is not None for r in rows)
    assert all(r['runtime_profile']==plan['profile'] for r in rows if r['variant']=='candidate')
    summary = {arm:aggregate([r for r in rows if r['variant']==arm]) for arm in plan['arms']}
    strata = {str(d):{arm:aggregate([r for r in rows if r['variant']==arm and r['directional']==d])
                     for arm in plan['arms']} for d in plan['directional_counts']}
    pairs = {(r['seed'],r['directional'],r['variant']):r for r in rows}
    rng = random.Random(450)
    reductions=[]
    for _ in range(2000):
        # Q4 strata reuse layouts at a given seed. Resample whole seed clusters
        # to preserve that dependence, as well as the within-case paired arms.
        sampled_seeds = rng.choices(plan['seeds'], k=len(plan['seeds']))
        keys=[(s,d) for d in plan['directional_counts'] for s in sampled_seeds]
        a=statistics.fmean(pairs[s,d,'baseline']['mean_clear_time_s'] for s,d in keys)
        b=statistics.fmean(pairs[s,d,'candidate']['mean_clear_time_s'] for s,d in keys)
        reductions.append(1-b/a)
    reductions.sort()
    reduction=1-summary['candidate']['arithmetic_mean_s']/summary['baseline']['arithmetic_mean_s']
    all_clear = summary['candidate']['full_clear_episodes']==plan['episodes_per_arm']
    result=dict(profile=plan['profile'], summary=summary, strata=strata,
                relative_reduction=reduction, paired_seed_cluster_bootstrap_95pct=[reductions[49],reductions[1949]],
                all_clear_gate_passed=all_clear,
                target_450_met=all_clear and summary['candidate']['arithmetic_mean_s']<=450,
                source_sha256=raw['source_sha256'],
                interpretation='Offline simulator validation; not official test results. All pre-registered episodes included.')
    (ROOT/'results/q4_450_final_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    b,c = summary['baseline'],summary['candidate']
    status = '达到' if result['target_450_met'] else '尚未达到'
    lines = [
        '# 问题四优化结果（2026-09-12）', '',
        f"当前交付版本 **{plan['profile']}** 在独立 48 局配对测试中，逐局平均定位清除时间从 "
        f"**{b['arithmetic_mean_s']:.2f} s** 降到 **{c['arithmetic_mean_s']:.2f} s**，下降 **{reduction:.2%}**。",
        f"清除 **{c['cleared']}/{c['total_sources']}** 个干扰源，**{c['full_clear_episodes']}/48** 局全清。"
        f"本次**{status}逐局均值不超过 450 s 的目标**。", '',
        '## 独立配对结果', '',
        '| 场景 | 局数 | 原版均值/s | 新版均值/s | 原版清除 | 新版清除 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for d, group in strata.items():
        a,n=group['baseline'],group['candidate']
        lines.append(f"| {d} 个定向源 | {n['episodes']} | {a['arithmetic_mean_s']:.2f} | {n['arithmetic_mean_s']:.2f} "
                     f"| {a['cleared']}/{a['total_sources']} | {n['cleared']}/{n['total_sources']} |")
    lines.extend([
        f"| 合计 | 48 | {b['arithmetic_mean_s']:.2f} | {c['arithmetic_mean_s']:.2f} | {b['cleared']}/{b['total_sources']} | {c['cleared']}/{c['total_sources']} |", '',
        '单局指标始终为该局总虚拟时间 ÷ 已清除个数；上表对 48 个单局指标取算术平均。',
        f"另一个口径为合计总时间 ÷ 合计清除数：原版 **{b['pooled_mean_s']:.2f} s**，新版 **{c['pooled_mean_s']:.2f} s**。两者不混用。", '',
        '此处“原版”指同样包含近场可行域修复的 `DogStrategy`；该对照并非远端 `main` 的完整原始快照。', '',
        f"新版 **{c['within_450_episodes']}/48** 局不超过 450 s；中位数 **{c['median_s']:.2f} s**，"
        f"P95 **{c['p95_s']:.2f} s**，最差 **{c['worst_s']:.2f} s**。不能据均值声称每局均能达到 450 s。", '',
        f"按种子为簇、保留三档场景相关性的 2000 次配对 bootstrap，降幅的 95% 区间为 "
        f"{reductions[49]:.2%}–{reductions[1949]:.2%}。", '',
        '## 实际修改', '',
        '1. 在真实运行入口接入 `Q4FastStrategy`。问题三继续使用 `DogStrategy`；问题四保留原版回退入口。',
        '2. 将 950 m × 6 的内环、1900 m × 14 的外环和目标估计位置放进同一开放巡回，站位访问后只消耗一次。',
        '3. 修正旧方位角的坐标参照：从产生方位角的观测点构造位移，必要时先在当前位置补测；允许合法的区域外归航。',
        '4. 用全部历史示向度、1800 m 场地约束和 1500 m 接收距离上界裁剪定位区域；保留读数舍入容差。',
        '5. 保留既有检测预算、归航重试、射线盲清除与异常退出逻辑。', '',
        '6. 同步本地已有的近场可行域修复：无信号观测不能把同一格点内的所有定向假设误删；覆盖模板采用同一保守规则。全向路径不受该修复影响。', '',
        f"同批案例的平均移动距离由 **{b['mean_move_distance_m']:.0f} m** 降至 **{c['mean_move_distance_m']:.0f} m**；"
        f"平均检测次数由 **{b['mean_measure_calls']:.1f}** 降至 **{c['mean_measure_calls']:.1f}**。", '',
        '开发中评估了稀疏站位、内环站位迁移、停靠后普扫、先搜索后统一清除、朝向增益选点等方案。'
        '较激进方案曾出现接近或低于 450 s、但漏清目标的结果，未作为当前交付默认配置。'
        '18 站版本在第一次留出检验中失败后，该数据块转作诊断；修正后的 20 站版本重新使用未见种子进行本表评估。', '',
        '## 验证与复现', '',
        '- 最终测试种子：261212101–261212116；每个种子分别生成 5、10、16 个定向源场景。前两档总源数按原生成器取 10–16，16 定向档总数为 16。',
        '- 策略不知道种子、源位置、真实朝向、实际接收半径或本局总数；真值只在评估器运行结束后用于计算清除率。',
        '- 原版和新版使用同一 `arena.py` 和同一物理常数；全部 96 条结果保留，无丢弃失败案例。运行前登记种子与核心源码哈希，运行期间检查逐局文件中列出的 5 个源码文件。该清单不包含 `feasible.py`；PR 交付阶段补记完整运行时哈希并重放候选，未将补记内容冒充预先登记。',
        '- 基础回归 205 项通过；Q4、HTTP 等价性、真实入口落盘与打包测试 10 项通过；来源与配置验证 10 项通过；提交前增加近场修复回归 2 项，共 227 项。',
        '- 该结果来自离线仿真器；HTTP 回环在 seed=7007、5 定向源的一例中与进程内结果一致。离线客户端将真实剩余预算固定为 1200 秒，以避免主机负载改变虚拟时间比较，因此本表不验证官方墙钟时限。本次没有进行官方测试，也没有消耗正式测试机会。', '',
        '```powershell',
        '.venv/Scripts/python.exe -m src.b_q4_450_eval --seeds 261212101 261212102 261212103 261212104 261212105 261212106 261212107 261212108 261212109 261212110 261212111 261212112 261212113 261212114 261212115 261212116 --directional 5 10 16 --variants baseline candidate --workers 6 --timeout 240 --out results/q4_450_final_validation.json',
        '.venv/Scripts/python.exe -m src.b_q4_450_report',
        '```', '',
        '登记文件：`results/q4_450_final_plan.json`；逐局结果：`results/q4_450_final_validation.json`；'
        '汇总与源码哈希：`results/q4_450_final_summary.json`。', '',
        '提交核验与开发证据索引：`results/q4_450_evidence.md`。', '',
        '## Windows 使用', '',
        '新版代码已同步到 `dist_windows/`，并重新生成根目录 `CUMCM2026B_robot_windows.zip`。问题四命令：', '',
        '```powershell',
        'python run_official.py --robot-id 你的参赛队号 --directional',
        '```', '',
        '与原版对照时在同一命令末尾加 `--legacy-q4`。运行日志的 `summary.strategy_profile` 记录新版版本号。', '',
        '固定双环仍是有界搜索，并不构成所有合法布局必然全清的数学证明。独立样本全清和真实官方表现是不同层次的证据。',
    ])
    report='\n'.join(lines)+'\n'
    (ROOT/'paper/q4_450_optimization.md').write_text(report,encoding='utf-8',newline='\n')
    (ROOT/'dist_windows/Q4_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
