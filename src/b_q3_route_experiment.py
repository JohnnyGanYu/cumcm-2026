"""Bounded Q3 route/coverage development experiments; no hidden truth in strategies."""
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

from src.b_sim.strategy import DogStrategy, StrategyConfig
from src.b_sim.feasible import cover_route
from src.b_sim.route import order_route, path_length


class RouteExperiment(DogStrategy):
    clear_scan_gain = 0
    seed_anchors = False
    relocate_radius = 0
    fixed_ring_count = 0
    fixed_ring_radius = 1050
    fixed_ring_phase_search = True

    def _visit_cover_station(self, station):
        if self.relocate_radius:
            nearby = [st for st in self.channels.values() if st.active and st.estimate is not None
                      and math.dist(station, st.estimate) <= self.relocate_radius]
            if nearby:
                state = min(nearby, key=lambda st: math.dist(station,st.estimate))
                if self._home_channel(state):
                    self._invalidate_cover_plan()
                    return super()._visit_cover_station(self.pos)
                state.exhausted = True
        result = super()._visit_cover_station(station)
        if self.fixed_ring_count and hasattr(self, '_fixed_stations'):
            self._fixed_stations = [p for p in self._fixed_stations if p != station]
        return result

    def _clear_at(self, point, channel):
        ok = super()._clear_at(point, channel)
        if ok and self.clear_scan_gain:
            for state in self._scan_order(list(self.channels.values())):
                if self._stop():
                    break
                if state.possible and not state.bearings and state.feasible is not None:
                    if state.feasible.coverage_gain(*self.pos) >= self.clear_scan_gain:
                        self._probe_channel_at(self.pos, state)
            self._invalidate_cover_plan()
        return ok

    def _cover_plan(self):
        if self.fixed_ring_count:
            pending = [st for st in self.channels.values() if self._search_target(st)]
            if not pending:
                return []
            if not hasattr(self, '_fixed_stations'):
                targets = [st.estimate or self._seed_waypoint(st) for st in self.channels.values() if st.active]
                targets = [p for p in targets if p is not None]
                def layout(phase):
                    return [(self.fixed_ring_radius*math.cos(phase+2*math.pi*i/self.fixed_ring_count),
                             self.fixed_ring_radius*math.sin(phase+2*math.pi*i/self.fixed_ring_count))
                            for i in range(self.fixed_ring_count)]
                plans = [layout(2*math.pi*j/(12*self.fixed_ring_count))
                         for j in range(12 if self.fixed_ring_phase_search else 1)]
                self._fixed_stations = min(plans, key=lambda plan:path_length(self.pos,order_route(self.pos,targets+plan)))
            self._fixed_stations = [p for p in self._fixed_stations
                                    if any(st.feasible is not None and st.feasible.coverage_gain(*p)>0 for st in pending)]
            if self._fixed_stations:
                return list(self._fixed_stations)
            return super()._cover_plan()
        if not self.seed_anchors:
            return super()._cover_plan()
        cfg = self.config
        pending = tuple(sorted(st.channel for st in self.channels.values() if self._search_target(st)))
        if not pending:
            return []
        cache = self._cover_cache
        if cache is not None and cache[0] == pending and cache[1]:
            return list(cache[1])
        sets = [self.channels[ch].feasible for ch in pending if self.channels[ch].feasible is not None]
        anchors = [st.estimate or self._seed_waypoint(st) for st in self.channels.values() if st.active]
        anchors = [p for p in anchors if p is not None]
        stations = cover_route(sets, self.pos, cell_step=cfg.feasible_cell_step_m,
                               max_stations=cfg.cover_max_stations,
                               min_gain=1, travel_weight_m=cfg.cover_travel_weight_m,
                               gain_tolerance=cfg.cover_gain_tolerance,
                               use_ring_layouts=cfg.cover_ring_layouts, anchors=anchors)
        self._cover_cache = pending, list(stations)
        return list(stations)


def make(client, variant):
    cfg = StrategyConfig()
    if variant == 'baseline':
        return DogStrategy(client, cfg)
    strategy = RouteExperiment(client, cfg)
    if variant.startswith('scan'):
        strategy.clear_scan_gain = int(variant[4:])
    if variant.startswith('pig'):
        cfg.piggyback_coverage = True
        cfg.piggyback_min_gain_cells = int(variant[3:])
        cfg.piggyback_max_per_stop = 20
    if variant.startswith('tol'):
        cfg.cover_gain_tolerance = int(variant[3:]) / 100
    if variant == 'anchors':
        strategy.seed_anchors = True
    if variant == 'steer':
        cfg.steer_toward_next = True
    if variant == 'fine50':
        cfg.feasible_cell_step_m = 50
    if variant == 'defer':
        cfg.cover_defer = True
    if variant.startswith('relocate'):
        strategy.relocate_radius = int(variant[8:])
    if variant.startswith('fixed'):
        strategy.fixed_ring_count = int(variant.removeprefix('fixed').removesuffix('plain'))
        strategy.fixed_ring_radius = 1050 if strategy.fixed_ring_count == 8 else 1000
        strategy.fixed_ring_phase_search = not variant.endswith('plain')
    return strategy


def episode(seed, variant):
    from src.b_eval.dev_measure import InProcessClient
    from src.b_sim.arena import Arena, generate_case
    arena = Arena(generate_case(seed, directional_count=0))
    strategy = make(InProcessClient(arena), variant)
    started = time.monotonic()
    strategy.run()
    return dict(seed=seed, variant=variant, directional=0, **arena.stats(),
                wall_s=time.monotonic()-started, cover_visits=strategy.cover_visits,
                missed_channels=[j.channel for j in arena.case.jammers if not j.cleared],
                config=asdict(strategy.config))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[7001,7002,7003,7004])
    ap.add_argument('--variants', nargs='+', default=['baseline','scan80','scan150','pig150','tol80','anchors'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--out', default='results/q3_220_route_small.json')
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(episode(args.seeds[0],args.variants[0]), ensure_ascii=True))
        return
    tasks = [(s,v) for s in args.seeds for v in args.variants]
    source_files = list(map(Path, ['src/b_sim/strategy.py','src/b_sim/arena.py',
                                  'src/b_q3_route_experiment.py']))
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    records = []
    def work(task):
        s,v = task
        command = [sys.executable,'-m','src.b_q3_route_experiment','--worker','--seeds',str(s),'--variants',v]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=True)
            return json.loads(run.stdout)
        except Exception as exc:
            return dict(seed=s,variant=v,error=str(exc),stderr=getattr(exc,'stderr',None))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({k:row.get(k) for k in ('seed','variant','cleared','total_jammers','mean_clear_time_s','wall_s','error')}),flush=True)
            current = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
            payload = {'records':sorted(records,key=lambda r:(r['seed'],r['variant'])),
                       'source_sha256':hashes,'source_unchanged':current==hashes,
                       'expected_records':len(tasks),'completed_records':len(records)}
            out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            if current != hashes:
                raise RuntimeError('Runtime source changed during evaluation.')


if __name__ == '__main__':
    main()
