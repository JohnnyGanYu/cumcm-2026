"""Bounded Q3 grid/cover development around the frozen fast geometry strategy."""
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

from src.b_q3_geometry_experiment import Q3GeometryStrategy


class Grid50Strategy(Q3GeometryStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.feasible_cell_step_m=50


class Grid100Strategy(Q3GeometryStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.feasible_cell_step_m=100


class Cover6Strategy(Q3GeometryStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.cover_max_stations=6


class Cover8Strategy(Q3GeometryStrategy):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.cover_max_stations=8


VARIANTS={'fast':Q3GeometryStrategy,'grid50':Grid50Strategy,'grid100':Grid100Strategy,
          'cap6':Cover6Strategy,'cap8':Cover8Strategy}


def episode(seed,variant):
    from src.b_sim.arena import Arena, generate_case
    from src.b_sim.strategy import StrategyConfig
    from src.b_q3_220_client import InProcessClient
    arena=Arena(generate_case(seed,directional_count=0))
    strategy=VARIANTS[variant](InProcessClient(arena),StrategyConfig(directional_probe=False))
    start=time.monotonic()
    strategy.run()
    return dict(seed=seed,variant=variant,**arena.stats(),wall_s=time.monotonic()-start,
                cover_visits=strategy.cover_visits,
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                config=asdict(strategy.config))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--seeds',nargs='+',type=int,default=[7001,7002,7003,7004])
    p.add_argument('--variants',nargs='+',default=list(VARIANTS))
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--timeout',type=float,default=120)
    p.add_argument('--out',default='results/q3_220_grid_pilot.json')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0],args.variants[0]),ensure_ascii=True))
        return
    paths=list(map(Path, ['src/b_q3_grid_experiment.py','src/b_q3_geometry_experiment.py',
                        'src/b_q3_220_client.py','src/b_sim/strategy.py','src/b_sim/arena.py',
                        'src/b_sim/feasible.py','src/b_sim/q4_fast.py','src/b_sim/route.py']))
    def hashes():
        return {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    initial=hashes()
    tasks=[(s,v) for s in args.seeds for v in args.variants]
    def run(task):
        seed,variant=task
        command=[sys.executable,'-m','src.b_q3_grid_experiment','--worker','--seeds',str(seed),'--variants',variant]
        try:
            result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=args.timeout)
            return json.loads(result.stdout)
        except Exception as exc:
            return dict(seed=seed,variant=variant,error=str(exc),stderr=getattr(exc,'stderr',None))
    records=[]
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,t) for t in tasks]):
            row=future.result()
            records.append(row)
            unchanged=initial==hashes()
            payload=dict(records=sorted(records,key=lambda r:(r['seed'],r['variant'])),
                         source_sha256=initial,source_unchanged=unchanged,
                         expected_records=len(tasks),completed_records=len(records),offline_only=True)
            out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
            print(json.dumps({k:row.get(k) for k in ('seed','variant','cleared','total_jammers','mean_clear_time_s','wall_s','error')}),flush=True)
            if not unchanged:
                raise RuntimeError('Source changed during experiment')


if __name__=='__main__':main()
