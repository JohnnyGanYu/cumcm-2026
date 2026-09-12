"""Offline development of Q3 homing; no private world access in the policy."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from src.b_sim.geometry import centroid, clip_polygon, polygon_diameter
from src.b_sim.q4_fast import bounded_region
from src.b_sim.strategy import DogStrategy, StrategyConfig


class Q3GeometryStrategy(DogStrategy):
    bounded = True
    anchored = True
    fast = True
    aggressive_clear = 0
    one_centroid = False
    fraction = .8
    lateral = .08
    angle = 12.
    dogleg_fraction = .65
    dogleg_max = 700.
    shared_range = False
    near_clear = False

    def __init__(self, client, config=None):
        super().__init__(client,config)
        self.polygons = {}

    def _probe_channel_at(self, point, state):
        super()._probe_channel_at(point, state)
        if not self.bounded or state.cleared or len(state.bearings) < (1 if self.one_centroid else 2):
            return
        poly = bounded_region(state.bearings)
        if self.shared_range:
            for px,py in state.silent_points:
                for bearing in state.bearings:
                    qx,qy=bearing.x,bearing.y
                    # |source-P|>R>=|source-Q|: a conservative closed half-plane.
                    poly=clip_polygon(poly,(px-qx,py-qy,(px*px+py*py-qx*qx-qy*qy)/2))
        if poly:
            self.polygons[state.channel]=poly
            state.estimate = centroid(poly)
            state.diameter = polygon_diameter(poly)[0]
            state.cover_radius = max(math.dist(state.estimate, p) for p in poly)

    def _dogleg(self, state):
        if not self.anchored:
            return super()._dogleg(state)
        last = state.bearings[-1]
        if math.dist(self.pos, (last.x,last.y)) > 30:
            before = len(state.bearings)
            self._probe_channel_at(self.pos,state)
            if state.cleared or len(state.bearings)>before:
                return True
        last=state.bearings[-1]
        centre=state.feasible.centroid() if state.feasible else None
        expected=math.dist((last.x,last.y),centre) if centre else 600
        length=max(100,min(self.dogleg_max,expected*self.dogleg_fraction))
        for offset in (self.angle,-self.angle,0.):
            angle=math.radians(last.svd_deg+offset)
            p=(last.x+length*math.cos(angle),last.y+length*math.sin(angle))
            before=len(state.bearings)
            self._probe_channel_at(p,state)
            if state.cleared or len(state.bearings)>before:
                return True
            if self._stop(): break
        return False

    def _step_closer(self,state,distance):
        if not self.fast: return super()._step_closer(state,distance)
        ex,ey=state.estimate
        ux,uy=(ex-self.pos[0])/max(distance,1e-9),(ey-self.pos[1])/max(distance,1e-9)
        advance=min(distance*self.fraction,max(0,distance-15))
        lateral=min(80,advance*self.lateral)
        self._probe_channel_at((self.pos[0]+ux*advance-uy*lateral,self.pos[1]+uy*advance+ux*lateral),state)
        return True

    def _can_clear_directly(self,state):
        return (state.diameter<=self.aggressive_clear) or super()._can_clear_directly(state)

    def _clear_at(self,point,channel):
        polygon=self.polygons.get(channel,[])
        if self.near_clear and polygon and point==self.channels[channel].estimate:
            _,pa,pb=polygon_diameter(polygon)
            centres=[point,((pa[0]+pb[0])/2,(pa[1]+pb[1])/2)]
            candidates=[]
            for cx,cy in centres:
                if max(math.dist((cx,cy),p) for p in polygon)>19.99:
                    continue
                lo,hi=0.,1.
                for _ in range(40):
                    t=(lo+hi)/2
                    p=(cx+t*(self.pos[0]-cx),cy+t*(self.pos[1]-cy))
                    if max(math.dist(p,v) for v in polygon)<=19.99: lo=t
                    else: hi=t
                candidates.append((cx+lo*(self.pos[0]-cx),cy+lo*(self.pos[1]-cy)))
            if candidates: point=min(candidates,key=lambda p:math.dist(p,self.pos))
        return super()._clear_at(point,channel)


def episode(seed,variant):
    from src.b_sim.arena import Arena, generate_case
    from src.b_eval.dev_measure import InProcessClient
    arena=Arena(generate_case(seed,directional_count=0))
    cfg=StrategyConfig(directional_probe=False)
    strategy=(DogStrategy if variant=='baseline' else Q3GeometryStrategy)(InProcessClient(arena),cfg)
    if variant=='anchor': strategy.bounded=False; strategy.fast=False
    if variant=='bounded': strategy.anchored=False; strategy.fast=False
    if variant=='anchorbounded': strategy.fast=False
    if variant=='fastonly': strategy.bounded=False; strategy.anchored=False
    if variant.startswith('clear'): strategy.aggressive_clear=float(variant.removeprefix('clear'))
    if variant.startswith('centroid'): strategy.one_centroid=True
    if variant.endswith('95'): strategy.fraction=.95; strategy.lateral=.04
    if variant=='straight': strategy.lateral=0.; strategy.fraction=.95
    if variant=='angle6': strategy.angle=6.
    if variant.startswith('shared'): strategy.shared_range=True
    if variant.endswith('near'): strategy.near_clear=True
    if variant=='sharedcentroid': strategy.one_centroid=True
    if variant in ('clear75','clear40'): cfg.direct_clear_diameter_m=float(variant.removeprefix('clear')); strategy.aggressive_clear=0
    if variant in ('long6','long12'):
        strategy.dogleg_fraction=.95; strategy.dogleg_max=1000.; strategy.angle=float(variant.removeprefix('long'))
    if variant.startswith('pig'):
        cfg.piggyback_min_baseline_m=float(variant.removeprefix('pig'))
    started=time.monotonic()
    summary=strategy.run()
    return dict(seed=seed,variant=variant,**arena.stats(),wall_s=time.monotonic()-started,
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                events=strategy.events,cover_visits=strategy.cover_visits,config=asdict(cfg))


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--seeds',nargs='+',type=int,default=[7001,7002])
    ap.add_argument('--variants',nargs='+',default=['baseline','fast'])
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--timeout',type=float,default=120)
    ap.add_argument('--out',default='results/q3_220_geometry_initial.json')
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0],args.variants[0]),ensure_ascii=True));return
    paths=[Path(p) for p in ('src/b_q3_geometry_experiment.py','src/b_sim/strategy.py','src/b_sim/q4_fast.py','src/b_sim/arena.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    records=[]
    tasks=[(s,v) for s in args.seeds for v in args.variants]
    def work(task):
        seed,variant=task
        command=[sys.executable,'-m','src.b_q3_geometry_experiment','--worker','--seeds',str(seed),'--variants',variant]
        try:
            r=subprocess.run(command,capture_output=True,text=True,check=True,timeout=args.timeout)
            return json.loads(r.stdout)
        except Exception as e:
            return dict(seed=seed,variant=variant,error=str(e),stderr=getattr(e,'stderr',None))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();records.append(row)
            print(json.dumps({k:row.get(k) for k in ('seed','variant','cleared','total_jammers','mean_clear_time_s','wall_s','error')}),flush=True)
            current={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            Path(args.out).write_text(json.dumps(dict(records=sorted(records,key=lambda r:(r['seed'],r['variant'])),source_sha256=hashes,source_unchanged=current==hashes,expected_records=len(tasks),completed_records=len(records)),ensure_ascii=False,indent=2),encoding='utf-8')
            if current!=hashes: raise RuntimeError('Source changed during experiment')


if __name__=='__main__': main()
