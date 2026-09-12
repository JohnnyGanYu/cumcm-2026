"""Validate complete paired Q4 evidence and write a reproducible report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import re
import statistics


def metrics(rows):
    times = sorted(r['mean_clear_time_s'] for r in rows)
    cleared = sum(r['cleared'] for r in rows)
    return dict(episodes=len(rows), cleared=cleared,
                total=sum(r['total_jammers'] for r in rows),
                full_clear=sum(r['cleared']==r['total_jammers'] for r in rows),
                mean_s=statistics.mean(times), median_s=statistics.median(times),
                pooled_s=sum(r['total_virtual_time_s'] for r in rows)/cleared,
                p95_s=times[min(len(times)-1, int(.95*(len(times)-1)))],
                worst_s=max(times), under_410=sum(t<=410 for t in times),
                move_m=statistics.mean(r['move_distance_m'] for r in rows),
                measures=statistics.mean(r['measure_calls'] for r in rows),
                clears=statistics.mean(r['clear_calls'] for r in rows),
                tail_s=statistics.mean(r['tail_s'] for r in rows),
                max_wall_s=max(r['wall_s'] for r in rows))


def paired(rows):
    groups = {}
    for row in rows:
        if row.get('error') or row.get('mean_clear_time_s') is None:
            raise ValueError('Errors or undefined episode metrics must be investigated')
        key = (row['seed'], row['count'], row['directional'])
        pair = groups.setdefault(key, {})
        if row['variant'] in pair:
            raise ValueError(f'Duplicate episode arm: {key}')
        pair[row['variant']] = row
    if any(set(p) != {'baseline','candidate'} for p in groups.values()):
        raise ValueError('Every case must have exactly baseline and candidate arms')
    if any(p['baseline']['total_jammers'] != p['candidate']['total_jammers']
           for p in groups.values()):
        raise ValueError('Unpaired source populations')
    return groups


def summarize(rows):
    pairs = paired(rows)
    base = metrics([p['baseline'] for p in pairs.values()])
    cand = metrics([p['candidate'] for p in pairs.values()])
    losses = [dict(seed=k[0], count=k[1], directional=k[2],
                   baseline=p['baseline']['cleared'], candidate=p['candidate']['cleared'])
              for k,p in pairs.items() if p['candidate']['cleared']<p['baseline']['cleared']]
    by_seed = {}
    for k,p in pairs.items():
        by_seed.setdefault(k[0], []).append(p['candidate']['mean_clear_time_s']-
                                          p['baseline']['mean_clear_time_s'])
    rng = random.Random(410)
    keys = sorted(by_seed)
    samples = sorted(statistics.mean(d for s in rng.choices(keys,k=len(keys))
                                    for d in by_seed[s]) for _ in range(4000))
    return dict(baseline=base, candidate=cand,
                mean_reduction_pct=100*(1-cand['mean_s']/base['mean_s']),
                pooled_reduction_pct=100*(1-cand['pooled_s']/base['pooled_s']),
                paired_time_delta_ci95_s=[samples[100],samples[3899]],
                candidate_clear_regressions=losses,
                paired_time_wins=sum(p['candidate']['mean_clear_time_s']<
                                     p['baseline']['mean_clear_time_s'] for p in pairs.values()))


def markdown(result):
    b,c=result['baseline'],result['candidate']
    regression=Path('results/q4_410_regression_final.txt')
    test_text=regression.read_text(encoding='utf-8',errors='replace') if regression.exists() else ''
    passed=re.search(r'(\d+) passed',test_text)
    test_status=(f"{passed.group(1)} 项回归通过" if passed and ' failed' not in test_text
                 else '完整回归状态请查 results/q4_410_regression_final.txt')
    lines=[
        '# 问题四优化报告：410 秒目标', '',
        '交付版本：`q4-fast-v3`。基线为本次修改前的 `q4-fast-v2` 精确文件快照，与截图中的运行版本一致。', '',
        f"独立 {c['episodes']} 局配对测试中，合计平均定位清除时间由 **{b['pooled_s']:.2f} s** 降为 **{c['pooled_s']:.2f} s**，降低 **{result['pooled_reduction_pct']:.2f}%**。新版清除 **{c['cleared']}/{c['total']}** 个源，**{c['full_clear']}/{c['episodes']}** 局全清。", '',
        f"**410 秒目标尚未达成，合计口径仍高 {c['pooled_s']-410:.2f} s。** 以下均为本地离线结果，本次未连接官方模拟器、未消耗正式测试机会，不能视为新版官方成绩。", '',
        '## 独立结果', '',
        '| 口径 | 原版 | 新版 |', '|---|---:|---:|',
        f"| 总虚拟时间 / 总清除数 | {b['pooled_s']:.2f} s | {c['pooled_s']:.2f} s |",
        f"| 各局平均时间的算术均值 | {b['mean_s']:.2f} s | {c['mean_s']:.2f} s |",
        f"| 中位数 | {b['median_s']:.2f} s | {c['median_s']:.2f} s |",
        f"| P95 | {b['p95_s']:.2f} s | {c['p95_s']:.2f} s |",
        f"| 最差单局 | {b['worst_s']:.2f} s | {c['worst_s']:.2f} s |",
        f"| 不超过 410 s 的局数 | {b['under_410']}/{b['episodes']} | {c['under_410']}/{c['episodes']} |",
        f"| 清除数 | {b['cleared']}/{b['total']} | {c['cleared']}/{c['total']} |",
        f"| 平均移动距离 | {b['move_m']:.0f} m | {c['move_m']:.0f} m |",
        f"| 平均检测次数 | {b['measures']:.2f} | {c['measures']:.2f} |",
        f"| 平均清除调用 | {b['clears']:.2f} | {c['clears']:.2f} |", '',
        f"新版逐局时间改善 {result['paired_time_wins']}/{c['episodes']} 局；清除数退化 {len(result['candidate_clear_regressions'])} 局。按种子为簇做 4000 次配对 bootstrap，平均时间差的 95% 区间为 [{result['paired_time_delta_ci95_s'][0]:.2f}, {result['paired_time_delta_ci95_s'][1]:.2f}] s。负值表示新版更快。", '',
        '### 按源数与定向数分层', '',
        '| 总源数 | 定向数 | 局数 | 原版均值/s | 新版均值/s | 新版合计均值/s | 新版清除 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for scenario,s in result['scenarios'].items():
        n,d=map(int,scenario.split(':')); old,new=s['baseline'],s['candidate']
        lines.append(f"| {'随机10–16' if n==0 else n} | {d} | {new['episodes']} | {old['mean_s']:.2f} | {new['mean_s']:.2f} | {new['pooled_s']:.2f} | {new['cleared']}/{new['total']} |")
    lines += ['',
        '10 源等稀疏场景仍要完成未知频道搜索，清除数较少使固定搜索成本难以摊薄。不能只引用 16 源或开发集成绩来声称达成 410 秒。', '',
        '## 修改内容', '',
        '1. 多起点开放巡回：比较不同首站的最近邻和 2-opt 路线，再作 Or-opt；把 Q4 的优化规模门槛从 24 提到 64，避免开局因目标与站位总数较多而直接退回最近邻。',
        '2. 长条定位区域就地清除：测向失败后，沿定位区域直径方向构造包围矩形，最多布置 48 个清除点；每格半对角不超过 19.9 m。区域过大则保留旧螺旋与后续救援。',
        '3. 射线兜底保留原铺点，只过滤距离任一历史示向度半平面超过 20 m、必然无法成功的清除点；仍使用独立射线预算。',
        '4. 固定布局改为内环 1000 m × 7、外环 1870 m × 12，外环旋转 15°，加原点共 20 站。默认不启用站位迁移。',
        '5. 已确认 16 个不同源频道时停止搜索未观测频道，已观测但未清除的源仍会重试；在保守区域允许时选择最近的可保证清除位置。',
        '6. 官方入口补写实际 `strategy_config`，运行日志同时记录策略版本、布局和新增参数，便于复核上机运行的确切版本。', '',
        '所有策略决策只使用公开规则和接口观测；不使用测试种子、真实位置、真实朝向、实际接收半径或本局总源数。`arena.py`、`strategy.py`、`q3_fast.py` 与修改前哈希一致。', '',
        '## 证据与边界', '',
        '- 截图三局池化 486.87 s、逐局均值 498.30 s；动作计时与本地模型逐项一致。最后一次清除后的空搜仅 0、213.99、144.30 s，无法单独解释全部差距。见 [官方日志计时审计](../results/q4_410_official_audit.json)。',
        '- 开发集 21 局最终布局均值 387.78 s，清除 314/314；这批案例平均源数较高，不能代替独立验证。旧候选及开发源码保存在 `results/q4_410_development_sources.zip`，避免后续生产更新改变历史实验含义。',
        '- 额外 10000 案例、160000 个全定向源的固定站位可见性检查：旧布局 8 个源不可见，最终布局 0 个。更快的 19 站布局新增 4 个遗漏，已否决。这里只评价是否至少能收到一次信号，不是完整策略清除率，也不是连续区域覆盖证明。见 [几何检查](../results/q4_410_fixed_layout_10000.json)。',
        '- 最后另比较 200/300/450 m 站位迁移，在同一 21 案例中分别比最终版本慢 0.26%/2.81%/0.93%，均未纳入；记录在 `results/q4_410_relocation_development.json`。',
        f'- {test_status}，包括新增几何与路径测试、Q3 回归、HTTP/进程内等价、真实入口落盘及近场定向约束。4 个开发案例的生产重放与冻结候选在虚拟时间和动作计数上逐项相同。',
        '- 两臂使用同一个本地仿真器；运行前保存种子和核心运行时哈希，运行期间逐条核对源码未变。所有计划案例均保留，没有删去失败或慢案例。源码与计划随逐局 JSON 保存。',
        '- 离线客户端的真实剩余预算固定为 1200 s，用于保持配对虚拟时间不受主机负载影响；本地源位置、朝向、接收半径及误差场分布仍是模型假设。官方真实表现与墙钟限制需上机复测。', '',
        '## 文件与复现', '',
        '- 生产实现：`src/b_sim/q4_fast.py`。',
        '- 固定源数独立测试：`results/q4_410_holdout_fixed.json`（261912201–261912220，各 10:5、10:10、13:5、13:13、16:5、16:16）。',
        '- 随机源数独立测试：`results/q4_410_holdout_random.json`（261913201–261913220，各 0:5、0:10）。',
        '- 汇总：`results/q4_410_final_summary.json`；评测器：`src/b_q4_410_eval.py`；回归：`results/q4_410_regression_final.txt`。',
        '- 源码、Windows 副本和根目录 `CUMCM2026B_robot_windows.zip` 同步交付。修改前的完整 v2 交付包备份：`results/q4_410_baseline/CUMCM2026B_robot_windows_v2.zip`。', '',
        '```powershell',
        'python -m src.b_q4_410_eval --seeds 261912201 261912202 --scenarios 10:5 10:10 13:5 13:13 16:5 16:16 --variants baseline candidate --workers 4 --out results/q4_410_replay_new.json',
        'python -m src.b_q4_410_report results/q4_410_holdout_fixed.json results/q4_410_holdout_random.json --out results/q4_410_final_summary.json --markdown paper/q4_410_optimization.md',
        'python -m src.pack_dist_windows --check',
        '```', '',
        'Windows 包运行问题四：', '', '```powershell',
        'python run_official.py --robot-id 你的参赛队号 --directional', '```', '',
        '运行后检查日志 `summary.strategy_profile` 为 `q4-fast-v3`。`--legacy-q4` 仍回退到更早的 DogStrategy；若需精确回退截图中的 v2，请使用上面的备份压缩包。', '',
    ]
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('inputs',nargs='+')
    ap.add_argument('--out',default='results/q4_410_final_summary.json')
    ap.add_argument('--markdown')
    args=ap.parse_args()
    rows=[]
    evidence=[]
    for name in args.inputs:
        data=json.loads(Path(name).read_text(encoding='utf-8'))
        if not data['source_unchanged']:
            raise ValueError(f'Source changed during {name}')
        if data['completed_records'] != data['plan']['expected_records']:
            raise ValueError(f'Incomplete planned evaluation: {name}')
        if len(data['records']) != data['completed_records']:
            raise ValueError(f'Incorrect record count: {name}')
        rows.extend(data['records'])
        evidence.append(dict(path=name,plan=data['plan']))
    result=summarize(rows)
    result['scenarios']={f'{n}:{d}':summarize([r for r in rows
                                           if (r['count'],r['directional'])==(n,d)])
                         for n,d in sorted({(r['count'],r['directional']) for r in rows})}
    result['evidence']=evidence
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.markdown:
        Path(args.markdown).write_text(markdown(result),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','scenarios')},indent=2))


if __name__=='__main__':
    main()
