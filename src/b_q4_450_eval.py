"""Bounded offline Q4 paired experiments (never contacts the official server)."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def episode(seed, directional, variant):
    from src.b_eval.dev_measure import InProcessClient
    from src.b_sim.arena import Arena, generate_case
    from src.b_sim.strategy import DogStrategy, StrategyConfig
    from src.b_sim.q4_fast import Q4FastStrategy
    from src.b_q4_450_experiments import ExperimentalQ4Strategy
    cfg = StrategyConfig(directional_probe=directional > 0)
    arena = Arena(generate_case(seed, directional_count=directional, count=16 if directional == 16 else None))
    strategy_class = DogStrategy if variant == 'baseline' else (Q4FastStrategy if variant == 'candidate' else ExperimentalQ4Strategy)
    strategy = strategy_class(InProcessClient(arena), cfg)
    if variant == 'geometry':
        strategy.fast_search = False
    if variant == 'ring20':
        strategy.inner_count, strategy.outer_count = 6, 14
    if variant == 'ring22':
        strategy.inner_count, strategy.outer_count = 8, 14
    if variant.startswith('joint'):
        strategy.joint_rings = True
        strategy.inner_count, strategy.outer_count = 6, 14
    if variant in ('joint18','joint18pig'):
        strategy.inner_count, strategy.outer_count = 6, 12
    if variant == 'joint16':
        strategy.inner_count, strategy.outer_count = 4, 12
    if variant == 'joint14':
        strategy.inner_count, strategy.outer_count = 4, 10
    if variant == 'joint12':
        strategy.inner_count, strategy.outer_count = 4, 8
    if variant.startswith('relocate'):
        strategy.inner_count, strategy.outer_count = 6, 14
        strategy.joint_rings = True
        strategy.relocate_inner = True
    if variant == 'relocate18':
        strategy.outer_count = 12
    if variant.startswith('beam'):
        strategy.inner_count, strategy.outer_count = 6, 14
        strategy.joint_rings = True
        strategy.beam_homing = True
    if variant == 'beam18':
        strategy.outer_count = 12
    if variant.startswith('r18'):
        strategy.inner_count, strategy.outer_count = 6, 12
        strategy.joint_rings = True
        strategy.relocate_inner = True
        if variant in ('r18i850','r18i900','r18i800'):
            strategy.inner_radius = float(variant[-3:])
        if variant == 'r18o1850':
            strategy.outer_radius = 1850
        if variant == 'r18steer':
            cfg.steer_toward_next = True
        if variant == 'r18r600':
            strategy.relocate_radius = 600
        if variant.startswith('r18prune'):
            strategy.prune_gain = int(variant.removeprefix('r18prune'))
    if variant.startswith('opport'):
        strategy.inner_count, strategy.outer_count = 4, 12
        strategy.joint_rings = True
        strategy.scan_after_clear = True
        strategy.prune_gain = 0
        if variant == 'opport_defer':
            cfg.cover_defer = True
        if variant == 'opport18':
            strategy.inner_count = 6
    if variant.startswith('survey'):
        strategy.inner_count, strategy.outer_count = 6, 12
        strategy.joint_rings = True
        strategy.survey_first = True
        if variant == 'surveylocal':
            strategy.survey_local = 400
    if variant.endswith('pig'):
        cfg.piggyback_coverage = True
        cfg.piggyback_min_gain_cells = 80
        cfg.piggyback_max_per_stop = 20
    started = time.monotonic()
    summary = strategy.run()
    stats = arena.stats()
    successes = [e['virtual_time_s'] for e in arena.robot.trace if e.get('clear_result') == 'success']
    return dict(seed=seed, directional=directional, variant=variant, **stats,
                wall_s=time.monotonic()-started, cover_visits=strategy.cover_visits,
                last_clear_s=max(successes, default=0), events=strategy.events,
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                runtime_profile=summary.get('strategy_profile',variant),
                search_layout=summary.get('search_layout'),
                config=asdict(cfg))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[7001,7002])
    ap.add_argument('--directional', nargs='+', type=int, default=[5])
    ap.add_argument('--variants', nargs='+', default=['baseline','geometry'])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--out', default='results/q4_450_development.json')
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0],args.directional[0],args.variants[0]), ensure_ascii=True))
        return
    tasks = [(s,d,v) for d in args.directional for s in args.seeds for v in args.variants]
    source_files = list(map(Path, ['src/b_sim/strategy.py','src/b_sim/q4_fast.py',
                                  'src/b_sim/arena.py','src/b_q4_450_experiments.py',
                                  'src/b_q4_450_eval.py']))
    initial_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    records = []
    def work(task):
        s,d,v = task
        command = [sys.executable,'-m','src.b_q4_450_eval','--worker','--seeds',str(s),'--directional',str(d),'--variants',v]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=True)
            return json.loads(run.stdout)
        except Exception as exc:
            return dict(seed=s,directional=d,variant=v,error=str(exc),stderr=getattr(exc,'stderr',None))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({k:row.get(k) for k in ('seed','directional','variant','cleared','total_jammers','mean_clear_time_s','wall_s','error')}),flush=True)
            current_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
            payload = {'records':sorted(records,key=lambda r:(r['directional'],r['seed'],r['variant'])),
                       'source_sha256':initial_hashes,'source_unchanged':current_hashes==initial_hashes,
                       'expected_records':len(tasks),'completed_records':len(records)}
            out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            if current_hashes != initial_hashes:
                raise RuntimeError('Runtime source changed during evaluation; results cannot be accepted.')


if __name__ == '__main__':
    main()
