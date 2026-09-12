"""Validate registered Q3 evidence and generate the delivery report."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]


def aggregate(rows):
    times=[r['mean_clear_time_s'] for r in rows]
    cleared=sum(r['cleared'] for r in rows)
    total=sum(r['total_jammers'] for r in rows)
    return dict(episodes=len(rows),cleared=cleared,total_sources=total,
                full_clear_episodes=sum(r['cleared']==r['total_jammers'] for r in rows),
                arithmetic_mean_s=statistics.fmean(times),
                pooled_mean_s=sum(r['total_virtual_time_s'] for r in rows)/cleared,
                median_s=statistics.median(times),p95_s=sorted(times)[math.ceil(.95*len(times))-1],
                worst_s=max(times),within_220_episodes=sum(t<=220 for t in times),
                mean_move_distance_m=statistics.fmean(r['move_distance_m'] for r in rows),
                mean_measure_calls=statistics.fmean(r['measure_calls'] for r in rows),
                mean_clear_calls=statistics.fmean(r['clear_calls'] for r in rows),
                max_wall_s=max(r['wall_s'] for r in rows))


def _finite_nonnegative(value, field):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0):
        raise ValueError('Expected a finite nonnegative number: '+field)
    return value


def _integer(value, field, minimum=0, maximum=None):
    # bool is an int subclass, but is not a valid experiment count or seed.
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError('Invalid integer: '+field)
    return value


def _config_json(value):
    if not isinstance(value, dict):
        raise ValueError('Strategy configuration must be a complete dictionary')
    # Canonical JSON distinguishes true, 1 and 1.0; ordinary dict equality does
    # not. No omitted keys, coercion, rounding or nonfinite values are accepted.
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False,
                          separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid strategy configuration') from exc


def validate_cohort(plan, cohort, raw):
    """Check all raw cases before computing any summary; never drop failures.

    ``plan['arm_configs']`` must map every registered arm to the full
    ``asdict(strategy.config)`` obtained after constructing that arm.
    ``cohort`` supplies unique seeds, count (None or 10..16), and result path.
    This pure validation boundary also supports synthetic corruption checks.
    """
    arms=plan['arms']
    seeds=cohort['seeds']
    count=cohort['count']
    if (not isinstance(arms,list) or not arms
            or any(not isinstance(a,str) or not a for a in arms)
            or len(set(arms))!=len(arms)):
        raise ValueError('Registered arms must be nonempty and unique')
    if (not isinstance(seeds,list) or not seeds
            or any(type(s) is not int for s in seeds)
            or len(set(seeds))!=len(seeds)):
        raise ValueError('Registered seeds must be nonempty and unique integers')
    if count is not None:
        _integer(count,'registered count',10,16)
    configs=plan.get('arm_configs')
    if not isinstance(configs,dict) or set(configs)!=set(arms):
        raise ValueError('Register exactly one full configuration for every arm')
    expected_configs={arm:_config_json(configs[arm]) for arm in arms}
    expected={(s,a,count) for s in seeds for a in arms}
    rows=raw.get('records')
    if not isinstance(rows,list) or len(rows)!=len(expected):
        raise ValueError('Missing, duplicate or unexpected cases: '+cohort['results'])
    for key in ('expected_records','completed_records'):
        if type(raw.get(key)) is not int or raw[key]!=len(expected):
            raise ValueError('Invalid completion metadata: '+key)
    if (raw.get('source_unchanged') is not True
            or raw.get('source_sha256')!=plan['source_sha256']):
        raise ValueError('Source freeze mismatch: '+cohort['results'])
    required={'seed','arm','count','config','runtime_profile','total_jammers',
              'directional_jammers','cleared','missed_channels','cleared_ratio',
              'total_virtual_time_s','mean_clear_time_s','move_distance_m',
              'wall_s','measure_calls','clear_calls'}
    actual=set()
    totals={}
    for r in rows:
        if not isinstance(r,dict) or not required.issubset(r):
            raise ValueError('Missing required case fields; retain the raw record')
        if 'error' in r or r['mean_clear_time_s'] is None:
            raise ValueError('Evaluation contains an error; retain the raw failure and do not report a passing gate.')
        if (type(r['seed']) is not int or not isinstance(r['arm'],str)
                or (r['count'] is not None and type(r['count']) is not int)):
            raise ValueError('Invalid case identity')
        identity=(r['seed'],r['arm'],r['count'])
        if identity not in expected or identity in actual:
            raise ValueError('Duplicate or unexpected case: '+repr(identity))
        actual.add(identity)
        if _config_json(r['config'])!=expected_configs[r['arm']]:
            raise ValueError('Strategy configuration differs from registration: '+repr(identity))
        if r['arm']==plan['candidate'] and r['runtime_profile']!=plan['profile']:
            raise ValueError('Wrong candidate profile')
        total=_integer(r['total_jammers'],'total_jammers',10,16)
        cleared=_integer(r['cleared'],'cleared',0,total)
        _integer(r['directional_jammers'],'directional_jammers',0,0)
        if cleared==0:
            raise ValueError('Zero cleared sources cannot yield a valid mean or passing gate')
        if count is not None and total!=count:
            raise ValueError('Generated source count differs from fixed cohort count')
        if r['seed'] in totals and totals[r['seed']]!=total:
            raise ValueError('Paired arms have different source counts')
        totals[r['seed']]=total
        missed=r['missed_channels']
        if not isinstance(missed,list):
            raise ValueError('missed_channels must be a list')
        for channel in missed:
            _integer(channel,'missed channel',1,20)
        if len(set(missed))!=len(missed) or len(missed)!=total-cleared:
            raise ValueError('Missed channels disagree with source and clearance counts')
        _integer(r['measure_calls'],'measure_calls')
        _integer(r['clear_calls'],'clear_calls',cleared)
        for field in ('total_virtual_time_s','mean_clear_time_s',
                      'move_distance_m','wall_s','cleared_ratio'):
            _finite_nonnegative(r[field],field)
        if not math.isclose(r['mean_clear_time_s'],r['total_virtual_time_s']/cleared,
                            rel_tol=1e-10,abs_tol=1e-9):
            raise ValueError('Per-case mean must equal total virtual time / cleared')
        if not math.isclose(r['cleared_ratio'],cleared/total,
                            rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Clearance ratio disagrees with source counts')
    if actual!=expected:
        raise ValueError('Missing registered cases: '+cohort['results'])
    return rows


def load_cohort(plan,cohort):
    raw=json.loads((ROOT/cohort['results']).read_text(encoding='utf-8'))
    return validate_cohort(plan,cohort,raw)


def main():
    plan=json.loads((ROOT/'results/q3_220_final_plan.json').read_text(encoding='utf-8'))
    for path,sha in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Current runtime differs from frozen evaluation: '+path)
    cohorts={c['name']:load_cohort(plan,c) for c in plan['cohorts']}
    summary={name:{a:aggregate([r for r in rows if r['arm']==a]) for a in plan['arms']}
             for name,rows in cohorts.items()}
    baseline=summary['random']['baseline']
    candidate=summary['random'][plan['candidate']]
    pairs={r['seed']:{a:next(x for x in cohorts['random'] if x['seed']==r['seed'] and x['arm']==a)
                      for a in plan['arms']} for r in cohorts['random']}
    rng=random.Random(220)
    improvements=[]
    for _ in range(2000):
        selected=rng.choices(list(pairs),k=len(pairs))
        old=statistics.fmean(pairs[s]['baseline']['mean_clear_time_s'] for s in selected)
        new=statistics.fmean(pairs[s][plan['candidate']]['mean_clear_time_s'] for s in selected)
        improvements.append(1-new/old)
    improvements.sort()
    improvement=1-candidate['arithmetic_mean_s']/baseline['arithmetic_mean_s']
    full=all(s[plan['candidate']]['full_clear_episodes']==s[plan['candidate']]['episodes']
             for s in summary.values())
    result=dict(profile=plan['profile'],summary=summary,relative_reduction=improvement,
                paired_bootstrap_95pct=[improvements[49],improvements[1949]],
                all_cohorts_full_clear=full,
                target_220_met=full and candidate['arithmetic_mean_s']<=220,
                source_sha256=plan['source_sha256'],offline_only=True)
    (ROOT/'results/q3_220_final_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    lines=['# 问题三优化结果（2026-09-12）','',
           f"当前版本 **{plan['profile']}** 在未参与调参的 {candidate['episodes']} 个随机源数案例中，逐局平均定位清除时间的算术均值从 "
           f"**{baseline['arithmetic_mean_s']:.2f} s** 降为 **{candidate['arithmetic_mean_s']:.2f} s**，降低 **{improvement:.2%}**。",
           f"新版清除 **{candidate['cleared']}/{candidate['total_sources']}** 个源，**{candidate['full_clear_episodes']}/{candidate['episodes']}** 局全清。"
           + ('综合 220 秒目标已在本次离线样本中达到。' if result['target_220_met'] else '**本次尚未达到综合 220 秒目标。**'),'',
           '## 独立结果','',
           '| 场景 | 每臂局数 | 对照均值/s | 新版均值/s | 对照清除 | 新版清除 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,arms in summary.items():
        b,c=arms['baseline'],arms[plan['candidate']]
        label='随机 10–16 个源（主指标）' if name=='random' else name
        lines.append(f"| {label} | {c['episodes']} | {b['arithmetic_mean_s']:.2f} | {c['arithmetic_mean_s']:.2f} | {b['cleared']}/{b['total_sources']} | {c['cleared']}/{c['total_sources']} |")
    lines.extend(['',
        '单局平均时间严格定义为该局总虚拟时间 ÷ 该局成功清除数；主指标对随机源数案例的这些值取算术平均。固定源数压力测试单独列出，不混入主指标。',
        f"池化口径（合计总时间 ÷ 合计清除数）为对照 **{baseline['pooled_mean_s']:.2f} s**、新版 **{candidate['pooled_mean_s']:.2f} s**，不与主指标混用。",'',
        f"新版 **{candidate['within_220_episodes']}/{candidate['episodes']}** 局不超过 220 s；中位数 **{candidate['median_s']:.2f} s**，P95 **{candidate['p95_s']:.2f} s**，最差 **{candidate['worst_s']:.2f} s**。",
        f"2000 次按随机案例种子配对 bootstrap 的均值降幅 95% 区间：**{improvements[49]:.2%}–{improvements[1949]:.2%}**。",'',
        '## 修改内容',''])
    lines.extend(f'{i}. {change}' for i,change in enumerate(plan['changes'],1))
    lines.extend(['',f"主评估中，平均移动距离由 **{baseline['mean_move_distance_m']:.0f} m** 降至 **{candidate['mean_move_distance_m']:.0f} m**；平均检测次数由 **{baseline['mean_measure_calls']:.2f}** 变为 **{candidate['mean_measure_calls']:.2f}**。",'',
        '## 验证、复现与使用','',
        '所有种子、配置和运行依赖哈希在最终评估前登记于 `results/q3_220_final_plan.json`，运行期间持续校验。全部案例保留，包括错误与漏清；脚本在资料不全、源码变化或配置不一致时拒绝生成合格结论。',
        '对照为本项目的 `DogStrategy`，与新版共用原物理常数、案例生成器和离线协议边界。策略只能使用协议观测及题面公开范围，不读取种子、源位置、实际源数或真实接收半径。',
        '固定源数 10/13/16 的三组压力测试仅用于评估，实际源数不传入策略。主评估保持生成器默认的随机源数分布。',
        '220 秒判定门槛为随机组主指标不超过 220 s，且四组中的新版案例全部清除；不要求每局或每个固定源数组的均值均小于 220 s。单次清除的几何保证限于已获得的保守定位区域，继承的离散搜索覆盖不构成所有合法布局全清的证明。',
        '两臂离线客户端均固定返回 1200 秒真实剩余预算，以避免主机负载改变虚拟时间比较。数据不代表官方成绩，也不能据此承诺所有合法布局都全清或每局都在 220 秒内。本次没有连接官方测试服务。','',
        '开发中比较了归航偏角/逼近比例、共享接收半径约束、额外检测、未来覆盖预扣、站位迁移/固定环/缓存、栅格精度及多起点路线。只将开发阶段表现稳定的机制列入默认交付；失败方案及逐局数据保留在 `results/q3_220_*development.json` 等开发记录中。','',
        '在仓库根目录使用已安装 numpy、pytest、hypothesis 的 Python 解释器，运行：','',
        '```powershell','python -m src.b_q3_220_report','python -m src.pack_dist_windows --check','python -m src.build_windows_zip --check','```','',
        '完整评估命令、源码哈希和环境信息见登记文件；回归测试结果见 `results/q3_220_regression.txt`。','',
        'Windows 包已同步并重新打包。在 `dist_windows` 目录运行：','',
        '```powershell','python run_official.py --robot-id 你的参赛队号','```','',
        '问题三默认启用新版，添加 `--legacy-q3` 可回退到对照策略。问题四仍通过 `--directional` 使用原 `q4-fast-v2`，其核心源码未改动。运行日志记录策略版本和实际配置。',''])
    report='\n'.join(lines)
    for path in ('paper/q3_220_optimization.md','dist_windows/Q3_OPTIMIZATION_REPORT.md'):
        (ROOT/path).write_text(report,encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
