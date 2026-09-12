"""Validate complete paired Q3 evidence and write a reproducible report."""
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
                worst_s=max(times), under_200=sum(t<=200 for t in times),
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
    rng = random.Random(200)
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


def validate_input(data):
    if not data['source_unchanged']:
        raise ValueError('Evaluation source changed')
    plan = data['plan']
    expected = {(seed, n, d, variant)
                for scenario in plan['scenarios']
                for n, d in [tuple(map(int, scenario.split(':')))]
                for seed in plan['seeds'] for variant in plan['variants']}
    rows = data['records']
    actual = [(r['seed'], r['count'], r['directional'], r['variant']) for r in rows]
    if (len(expected) != plan['expected_records'] or set(actual) != expected
            or len(actual) != len(expected) or data['completed_records'] != len(actual)):
        raise ValueError('Missing, duplicate, or unplanned evaluation cases')
    if any(r['directional'] != 0 for r in rows):
        raise ValueError('Directional sources do not belong in the Q3 evaluation')
    paired(rows)
    return rows


def markdown(result):
    main = result['scenarios']['0:0']
    b, c = main['baseline'], main['candidate']
    all_c = result['candidate']
    selection = json.loads(Path('results/q3_200_selection.json').read_text(encoding='utf-8'))
    test_file = Path('results/q3_200_regression_final.txt')
    test_text = test_file.read_text(encoding='utf-8', errors='replace') if test_file.exists() else ''
    match = re.search(r'(\d+) passed', test_text)
    test_status = f'{match.group(1)} 项回归通过' if match and ' failed' not in test_text else '回归状态待更新'
    reached = c['pooled_s'] <= 200 and c['mean_s'] <= 200
    lines = [
        '# 问题三优化报告：200 秒目标', '',
        '交付版本：`q3-fast-v2`。对照是本次修改前的 `q3-fast-v1` 精确快照；问题四保持 `q4-fast-v3`。', '',
        f"独立随机源数 {c['episodes']} 局中，合计平均定位清除时间由 **{b['pooled_s']:.2f} 秒** 降至 **{c['pooled_s']:.2f} 秒**，变化 **{main['pooled_reduction_pct']:.2f}%**（正值表示提速）；各局均值由 **{b['mean_s']:.2f} 秒** 变为 **{c['mean_s']:.2f} 秒**。", '',
        ('**该离线样本达到 200 秒目标。**' if reached else '**200 秒目标尚未达到。**') +
        ' 本次未连接官方模拟器，不能把离线结果当作新版官方成绩，也不能保证每一局达到目标。', '',
        f"连同固定 10/13/16 源各 20 局压力测试，共完成 **{all_c['episodes']} 局配对**；新版清除 **{all_c['cleared']}/{all_c['total']}** 个源，清除数退化 **{len(result['candidate_clear_regressions'])} 局**。", '',
        '## 独立随机 100 局：主要比较', '',
        '| 指标 | 修改前 | 修改后 |', '|---|---:|---:|',
    ]
    for label, key, unit in [
            ('总虚拟时间 / 总清除数', 'pooled_s', ' 秒'),
            ('逐局平均时间的算术均值', 'mean_s', ' 秒'),
            ('中位数', 'median_s', ' 秒'), ('P95', 'p95_s', ' 秒'),
            ('最慢单局', 'worst_s', ' 秒'), ('平均移动距离', 'move_m', ' 米'),
            ('平均检测次数', 'measures', ''), ('平均清除调用', 'clears', '')]:
        lines.append(f'| {label} | {b[key]:.2f}{unit} | {c[key]:.2f}{unit} |')
    lines.extend([
        f"| ≤200 秒的局数 | {b['under_200']}/{b['episodes']} | {c['under_200']}/{c['episodes']} |",
        f"| 清除数 | {b['cleared']}/{b['total']} | {c['cleared']}/{c['total']} |", '',
        f"配对中新版更快 {main['paired_time_wins']}/{c['episodes']} 局。4000 次按种子配对 bootstrap 的逐局平均时间差 95% 区间为 [{main['paired_time_delta_ci95_s'][0]:.2f}, {main['paired_time_delta_ci95_s'][1]:.2f}] 秒，负值表示新版更快。", '',
        '## 按源数分层', '',
        '| 场景 | 局数 | 原版均值/s | 新版均值/s | 原版合计均值/s | 新版合计均值/s | 新版清除数 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for scenario, group in result['scenarios'].items():
        n = int(scenario.split(':')[0]); old, new = group['baseline'], group['candidate']
        lines.append(f"| {'随机10–16源' if n == 0 else str(n)+'源'} | {new['episodes']} | {old['mean_s']:.2f} | {new['mean_s']:.2f} | {old['pooled_s']:.2f} | {new['pooled_s']:.2f} | {new['cleared']}/{new['total']} |")
    lines.extend(['',
        '固定源数样本只作压力测试，不能用源数较多的低均值替代随机主样本。搜遍同样区域的开销在源数少时更难摊薄。', '',
        '## 修改与依据', '',
        *[f'{i}. {change}' for i, change in enumerate(selection['changes'], 1)], '',
        '策略只使用公开规则和每频道实际接口观测，不读取测试种子、真实位置、接收半径或实际总源数；计时和物理模型未修改。共享清除几何函数、Q4、基础策略和可行域模块保留原实现。', '',
        '## 对截图与局限的说明', '',
        '- 截图三局合计评分 209.01 秒，逐局均值 215.90 秒。11 源慢局原点只发现 2 源，另两局分别发现 12/8 源；慢局无失败清除、无尾部空搜。主要额外开销来自搜索和移动。',
        '- 本地随机分布、观测误差场与官方案例不同，不能直接把截图的 209.01 秒与新版离线数字相减，也不能按离线降幅预测官方已达标。',
        '- 调参和独立验证分开，最终源数随机种子为 262009001–262009100；固定 10/13/16 源分别为 262009201–220、262009301–320、262009401–420。种子提前查重并保存计划；全部计划案例、慢局和错误均留存。',
        '- 每局离线客户端提供固定 1200 秒剩余预算，使虚拟时间比较不被本机负载改变；真实官方墙钟预算还需正式复测。',
        f'- {test_status}，结果见 `results/q3_200_regression_final.txt`；另有独立的 Q4 行为一致性和发行包检查。', '',
        '## 文件与复现', '',
        '- 生产：`src/b_sim/q3_fast.py`；Windows 副本：`dist_windows/b_sim/q3_fast.py`。',
        '- 原版本完整备份：`results/q3_200_baseline/CUMCM2026B_robot_windows.zip`（Q3v1 + Q4v3）。`--legacy-q3` 仍指向更早的 DogStrategy，不能用作本轮精确基线。',
        '- 全部最终逐局记录：`results/q3_200_holdout_*.json`；汇总：`results/q3_200_final_summary.json`。',
        '- 开发试验及否决原因：`paper/q3_200_development.md`；开发源码冻结包：`results/q3_200_development_sources.zip`。', '',
        '```powershell',
        'python -m src.b_q3_200_eval --seeds 262009001 262009002 --scenarios 0:0 --variants baseline candidate --workers 4 --out results/q3_200_replay_new.json',
        'python -m src.pack_dist_windows --check',
        '```', '',
        'Windows 包运行问题三：', '', '```powershell',
        'python run_official.py --robot-id 你的参赛队号', '```', '',
        '检查运行日志 `summary.strategy_profile` 为 `q3-fast-v2`。参赛队号需使用有效队号；截图中的队号错误不属于定位策略失败。', '',
    ])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', default='results/q3_200_final_summary.json')
    ap.add_argument('--markdown')
    args = ap.parse_args()
    rows, evidence = [], []
    for name in args.inputs:
        data = json.loads(Path(name).read_text(encoding='utf-8'))
        rows.extend(validate_input(data))
        evidence.append(dict(path=name, plan=data['plan']))
    result = summarize(rows)
    result['scenarios'] = {f'{n}:0': summarize([r for r in rows if r['count'] == n])
                           for n in sorted({r['count'] for r in rows})}
    result['evidence'] = evidence
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.markdown:
        Path(args.markdown).write_text(markdown(result), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence'}, indent=2))


if __name__ == '__main__':
    main()

