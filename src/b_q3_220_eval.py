"""Bounded, source-audited offline Q3 paired evaluation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time


def episode(seed, arm, count=None):
    from src.b_sim.arena import Arena, generate_case
    from src.b_sim.strategy import DogStrategy, StrategyConfig
    from src.b_q3_220_client import InProcessClient
    module, _, name = arm.partition(':')
    cls = DogStrategy if arm == 'baseline' else getattr(importlib.import_module(module), name)
    arena = Arena(generate_case(seed, directional_count=0, count=count))
    strategy = cls(InProcessClient(arena), StrategyConfig(directional_probe=False))
    start = time.monotonic()
    summary = strategy.run()
    stats = arena.stats()
    return dict(seed=seed, arm=arm, count=count, **stats,
                wall_s=time.monotonic()-start,
                runtime_profile=summary.get('strategy_profile', cls.__name__),
                config=asdict(strategy.config),
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                events=strategy.events)


def source_paths(arms):
    from src.pack_dist_windows import RUNTIME_MODULES
    paths = {Path('src/b_sim')/n for n in RUNTIME_MODULES}
    paths.update(map(Path, ['src/b_q3_220_eval.py','src/b_q3_220_client.py',
                           'src/b_q3_220_report.py','src/run_official.py',
                           'src/pack_dist_windows.py','src/build_windows_zip.py']))
    for arm in arms:
        if arm != 'baseline':
            module, name = arm.split(':')
            cls = getattr(importlib.import_module(module),name)
            paths.add(Path(module.replace('.', '/')+'.py'))
            for base in cls.__mro__:
                if base.__module__.startswith('src.'):
                    paths.add(Path(inspect.getfile(base)).resolve().relative_to(Path.cwd()))
            for module in getattr(cls,'source_dependencies',()):
                paths.add(Path(module.replace('.', '/')+'.py'))
    return sorted(paths)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--arms', nargs='+', default=['baseline'])
    p.add_argument('--count', type=int)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--out', default='results/q3_220_development.json')
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0],args.arms[0],args.count),ensure_ascii=True))
        return
    paths = source_paths(args.arms)
    def hashes():
        return {q.as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
    initial = hashes()
    tasks = [(s,a) for s in args.seeds for a in args.arms]
    records = []
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    def run(task):
        seed,arm=task
        command=[sys.executable,'-m','src.b_q3_220_eval','--worker','--seeds',str(seed),'--arms',arm]
        if args.count is not None:
            command.extend(['--count',str(args.count)])
        try:
            result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=args.timeout)
            return json.loads(result.stdout)
        except Exception as exc:
            return dict(seed=seed,arm=arm,count=args.count,error=str(exc),stderr=getattr(exc,'stderr',None))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,t) for t in tasks]):
            row=future.result()
            records.append(row)
            unchanged=hashes()==initial
            payload=dict(records=sorted(records,key=lambda r:(r['seed'],r['arm'])),
                         expected_records=len(tasks),completed_records=len(records),
                         source_sha256=initial,source_unchanged=unchanged,
                         client_real_budget_s=1200,offline_only=True)
            out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
            print(json.dumps({k:row.get(k) for k in ('seed','arm','cleared','total_jammers','mean_clear_time_s','wall_s','error')}),flush=True)
            if not unchanged:
                raise RuntimeError('Evaluation source changed; do not accept these results.')


if __name__=='__main__':
    main()
