"""Joint target/cover layout cost experiment; conservative coverage is unchanged."""
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

from src.b_q3_combined_experiment import FastClear
from src.b_sim.arena import SPEED_MPS,MEASURE_SECONDS,CHANNEL_SWITCH_SECONDS
from src.b_sim.feasible import _cells,_coverage_table,_ring_layouts,guaranteed_reach,cover_stations
from src.b_sim.route import nearest_neighbour,two_opt,order_route,path_length


class JointCover(FastClear):
    include_seeds=False
    source_dependencies=('src.b_q3_combined_experiment','src.b_q3_clearance_experiment',
                         'src.b_q3_geometry_experiment','src.b_q3_refine_experiment',
                         'src.b_q3_multistart_experiment')

    def _cover_plan(self):
        cfg=self.config
        if not cfg.adaptive_cover:
            return []
        pending=tuple(sorted(st.channel for st in self.channels.values() if self._search_target(st)))
        if not pending:
            self._cover_cache=None
            return []
        cache=self._cover_cache
        if cache is not None and cache[0]==pending and cache[1] and not cfg.cover_replan_always:
            return list(cache[1])
        original=super()._cover_plan()
        if not original or not cfg.cover_ring_layouts:
            return original
        sets=[self.channels[ch].feasible for ch in pending if self.channels[ch].feasible is not None]
        anchors=[st.estimate or (self._seed_waypoint(st) if self.include_seeds else None)
                 for st in self.channels.values() if st.active]
        anchors=[p for p in anchors if p is not None]
        if not anchors:
            return original
        # The original plan remains eligible even if its bounded greedy search
        # is incomplete. Every new layout must cover all actual remaining cells.
        # No hypothetical observation is applied to any live feasible set.
        plans=[list(original)]
        greedy=cover_stations(sets,self.pos,cell_step=cfg.feasible_cell_step_m,
                             max_stations=cfg.cover_max_stations,min_gain=cfg.cover_min_gain_cells,
                             travel_weight_m=cfg.cover_travel_weight_m,
                             gain_tolerance=cfg.cover_gain_tolerance,anchors=anchors)
        if greedy and all(not fs.uncovered_by(greedy) for fs in sets):
            plans.append(greedy)
        need=set().union(*(set(fs.alive) for fs in sets))
        step=cfg.feasible_cell_step_m
        reach=guaranteed_reach(step)
        table=_coverage_table(step,reach)
        points=_cells(step)
        # Equivalent to feasible._layout_plan's conservative set-cover check,
        # while postponing expensive full route refinement until shortlisting.
        for layout in _ring_layouts(step,reach):
            remaining=set(need)
            kept=[]
            for si in layout:
                if remaining.intersection(table[si]):
                    kept.append(points[si])
                    remaining.difference_update(table[si])
            if not remaining and kept:
                plans.append(kept)
        unique={tuple(sorted(plan)):plan for plan in plans}
        probe_cost=len(sets)*(MEASURE_SECONDS+CHANNEL_SWITCH_SECONDS)
        def score(plan,full=False):
            joint=anchors+plan
            route=order_route(self.pos,joint) if full else two_opt(self.pos,nearest_neighbour(self.pos,joint))
            return path_length(self.pos,route)/SPEED_MPS+len(plan)*probe_cost
        shortlist=sorted(unique.values(),key=score)[:8]
        if original not in shortlist:
            shortlist.append(original)
        chosen=min(shortlist,key=lambda plan:score(plan,True))
        self._cover_cache=(pending,list(chosen))
        return list(chosen)


class JointCoverSeeds(JointCover):
    include_seeds=True


VARIANTS={'fastclear':FastClear,'joint':JointCover,'jointseeds':JointCoverSeeds}


def episode(seed,variant):
    from src.b_sim.arena import Arena,generate_case
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
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--timeout',type=float,default=120)
    p.add_argument('--out',default='results/q3_220_joint_cover_pilot.json')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0],args.variants[0]),ensure_ascii=True))
        return
    paths=list(map(Path,['src/b_q3_joint_cover_experiment.py','src/b_q3_geometry_experiment.py',
                        'src/b_q3_combined_experiment.py','src/b_q3_clearance_experiment.py',
                        'src/b_q3_refine_experiment.py','src/b_q3_multistart_experiment.py',
                        'src/b_q3_220_client.py','src/b_sim/strategy.py','src/b_sim/arena.py',
                        'src/b_sim/feasible.py','src/b_sim/q4_fast.py','src/b_sim/route.py']))
    def hashes():
        return {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    initial=hashes()
    tasks=[(s,v) for s in args.seeds for v in args.variants]
    def run(task):
        seed,variant=task
        command=[sys.executable,'-m','src.b_q3_joint_cover_experiment','--worker','--seeds',str(seed),'--variants',variant]
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
