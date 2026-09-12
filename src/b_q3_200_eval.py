"""Paired Q3 evaluation against the exact pre-change q3-fast-v1 snapshot.

Never connects to an external service. Truth is used by the evaluator only.
Every scheduled case, including errors and incomplete clears, is retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / 'results/q3_200_baseline/q3_fast.py'


def baseline_class():
    spec = importlib.util.spec_from_file_location('src.b_sim._q3_200_baseline', SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Q3FastStrategy


def episode(seed, directional, count, variant):
    from src.b_eval.dev_measure import InProcessClient
    from src.b_sim.arena import Arena, generate_case
    from src.b_sim.strategy import StrategyConfig
    from src.b_sim.q3_fast import Q3FastStrategy
    if variant == 'baseline':
        cls = baseline_class()
    elif variant == 'candidate':
        cls = Q3FastStrategy
    else:
        module, name = variant.split(':')
        cls = getattr(importlib.import_module(module), name)
    cfg = StrategyConfig(directional_probe=False)
    arena = Arena(generate_case(seed, count=count or None, directional_count=directional))
    client = InProcessClient(arena)
    dog = cls(client, cfg)
    started = time.monotonic()
    summary = dog.run()
    success_times = [e['virtual_time_s'] for e in arena.robot.trace
                     if e.get('clear_result') == 'success']
    return dict(seed=seed, directional=directional, count=count, variant=variant,
                **arena.stats(), wall_s=time.monotonic()-started,
                last_clear_s=max(success_times, default=0),
                tail_s=arena.robot.virtual_time_s-max(success_times, default=0),
                cover_visits=dog.cover_visits, events=dog.events,
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                runtime_profile=summary.get('strategy_profile'),
                search_layout=summary.get('search_layout'), config=asdict(dog.config))


def aggregate(records):
    groups = {}
    for variant in sorted({r['variant'] for r in records}):
        scheduled = [r for r in records if r['variant'] == variant]
        valid = [r for r in scheduled if 'error' not in r]
        values = [r['mean_clear_time_s'] for r in valid if r['mean_clear_time_s'] is not None]
        cleared = sum(r['cleared'] for r in valid)
        total = sum(r['total_jammers'] for r in valid)
        groups[variant] = dict(
            scheduled=len(scheduled), errors=len(scheduled)-len(valid),
            zero_clear_episodes=sum(r['cleared'] == 0 for r in valid),
            mean_s=statistics.mean(values) if len(values) == len(valid) and values else None,
            pooled_s=sum(r['total_virtual_time_s'] for r in valid)/cleared if cleared else None,
            median_s=statistics.median(values) if values else None,
            worst_s=max(values, default=None), under_200=sum(v <= 200 for v in values),
            cleared=cleared, total=total,
            full_clear_episodes=sum(r['cleared'] == r['total_jammers'] for r in valid),
            mean_move_m=statistics.mean(r['move_distance_m'] for r in valid) if valid else None,
            mean_measures=statistics.mean(r['measure_calls'] for r in valid) if valid else None,
            mean_tail_s=statistics.mean(r['tail_s'] for r in valid) if valid else None)
    return groups


def source_hashes(extra=()):
    files = [p for p in (ROOT/'src/b_sim').glob('*.py') if not p.name.startswith('test')]
    files += [SNAPSHOT, Path(__file__), ROOT/'src/b_eval/dev_measure.py']
    files += [ROOT / (v.split(':')[0].replace('.', '/') + '.py')
              for v in extra if ':' in v]
    files += [ROOT/'src/run_official.py']
    for variant in extra:
        if ':' not in variant:
            continue
        module, name = variant.split(':')
        cls = getattr(importlib.import_module(module), name)
        files += [Path(inspect.getfile(base)).resolve() for base in cls.__mro__
                  if base.__module__.startswith('src.')]
        files += [ROOT/(module.replace('.', '/')+'.py')
                  for module in getattr(cls, 'source_dependencies', ())]
    return {str(p.relative_to(ROOT)).replace('\\', '/'):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[7001, 7002])
    ap.add_argument('--scenarios', nargs='+', default=['0:0'],
                    help='total:directional; total=0 uses generator random 10..16')
    ap.add_argument('--variants', nargs='+', default=['baseline', 'candidate'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=240)
    ap.add_argument('--out', default='results/q3_200_development.json')
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    scenarios = [tuple(map(int, s.split(':'))) for s in args.scenarios]
    if any(d != 0 or n not in (0, 10, 11, 12, 13, 14, 15, 16) for n, d in scenarios):
        ap.error('Q3 requires zero directional sources and total=0 or 10..16')
    if args.worker:
        n, d = scenarios[0]
        print(json.dumps(episode(args.seeds[0], d, n, args.variants[0])))
        return
    tasks = [(s, d, n, v) for n, d in scenarios for s in args.seeds for v in args.variants]
    if len(tasks) != len(set(tasks)):
        ap.error('Duplicate scheduled cases are not allowed')
    initial = source_hashes(args.variants)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        ap.error(f'Refusing to overwrite existing evidence: {out}')
    plan = dict(seeds=args.seeds, scenarios=args.scenarios, variants=args.variants,
                expected_records=len(tasks), source_sha256=initial,
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                python=sys.version, timeout_s=args.timeout)
    out.with_suffix('.plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    records = []
    def work(task):
        s, d, n, v = task
        command = [sys.executable, '-m', 'src.b_q3_200_eval', '--worker',
                   '--seeds', str(s), '--scenarios', f'{n}:{d}', '--variants', v]
        try:
            run = subprocess.run(command, capture_output=True, text=True,
                                 timeout=args.timeout, check=True, cwd=ROOT)
            return json.loads(run.stdout)
        except Exception as exc:
            return dict(seed=s, directional=d, count=n, variant=v,
                        error=str(exc), stderr=getattr(exc, 'stderr', None))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({k:row.get(k) for k in (
                'seed', 'count', 'directional', 'variant', 'cleared',
                'total_jammers', 'mean_clear_time_s', 'error')}), flush=True)
            unchanged = source_hashes(args.variants) == initial
            payload = dict(plan=plan, source_unchanged=unchanged,
                           completed_records=len(records), summary=aggregate(records),
                           records=sorted(records, key=lambda r:(r['count'], r['directional'],
                                                               r['seed'], r['variant'])))
            out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            if not unchanged:
                raise RuntimeError('Evaluation source changed during this run')
    print(json.dumps(aggregate(records), indent=2))
    if any('error' in r for r in records):
        raise SystemExit('Evaluation contains errors; do not report it as a complete validation')


if __name__ == '__main__':
    main()
