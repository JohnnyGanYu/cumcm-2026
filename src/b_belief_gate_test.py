"""验证 5.节的门禁建议：diameter<=80 才允许清除。真实闭环 A/B。"""
import json, math, statistics, threading, time, sys
sys.path.insert(0, '.')
from src.b_sim.arena import Arena, generate_case
from src.b_sim.client import RobotClient
from src.b_sim.server import Session, make_server
from src.b_sim.strategy import DogStrategy, StrategyConfig

class GatedStrategy(DogStrategy):
    def __init__(self, client, config=None, gate_m=80.0):
        super().__init__(client, config); self.gate_m = gate_m; self.gate_blocks = 0
    def _home_channel(self, state):
        orig = self.config.clear_distance_m
        return super()._home_channel(state)
    def _clear_at(self, point, channel):
        st = self.channels.get(channel)
        # 只在"围绕点估计清除"时门禁；射线盲扫/盲清除不受影响（它们不依赖估计精度）
        import sys as _s
        caller = _s._getframe(1).f_code.co_name
        if (caller in ('_home_channel','_step_closer') and st is not None
                and math.isfinite(st.diameter) and st.diameter > self.gate_m
                and st.homing_measures < self.config.max_homing_measures - 1):
            self.gate_blocks += 1
            return False
        return super()._clear_at(point, channel)

def run(seed, directional, variant, gate_m=80.0):
    case = generate_case(seed, directional_count=directional)
    arena = Arena(case); sess = Session('g', arena, real_limit_s=1200.0)
    h = make_server(sess, port=0); port = h.server_address[1]
    threading.Thread(target=h.serve_forever, daemon=True).start()
    cfg = StrategyConfig(directional_probe=directional>0)
    t0=time.monotonic(); gb=0
    try:
        c = RobotClient('g', base_url=f'http://127.0.0.1:{port}')
        s = DogStrategy(c,cfg) if variant=='baseline' else GatedStrategy(c,cfg,gate_m)
        s.run()
        gb = getattr(s,'gate_blocks',0)
    finally:
        h.shutdown(); h.server_close()
    st = arena.stats()
    return dict(seed=seed, variant=variant, total=st['total_jammers'], cleared=st['cleared'],
                ratio=st['cleared_ratio'], virt=st['total_virtual_time_s'],
                meanT=st['mean_clear_time_s'], clears=st['clear_calls'],
                meas=st['measure_calls'], gate_blocks=gb, wall=round(time.monotonic()-t0,2))

rows=[]
for i in range(25):
    seed = 70000+i
    for v in ('baseline','gated'):
        r = run(seed, 5, v); rows.append(r)
    b,g = rows[-2],rows[-1]
    print(f"seed={seed} base {b['cleared']}/{b['total']} clears={b['clears']} virt={b['virt']:.0f} | "
          f"gated {g['cleared']}/{g['total']} clears={g['clears']} virt={g['virt']:.0f} blocks={g['gate_blocks']}", flush=True)

def agg(v):
    rs=[r for r in rows if r['variant']==v]
    mt=[r['meanT'] for r in rs if r['meanT']]
    return dict(variant=v, n=len(rs),
        full=round(sum(1 for r in rs if r['ratio']>=1-1e-9)/len(rs),4),
        ratio=round(statistics.fmean(r['ratio'] for r in rs),4),
        missed=sum(r['total']-r['cleared'] for r in rs),
        meanT=round(statistics.fmean(mt),1),
        virt=round(statistics.fmean(r['virt'] for r in rs),1),
        clears=round(statistics.fmean(r['clears'] for r in rs),2),
        meas=round(statistics.fmean(r['meas'] for r in rs),1),
        blocks=round(statistics.fmean(r['gate_blocks'] for r in rs),2))
out={'baseline':agg('baseline'),'gated':agg('gated')}
print(json.dumps(out,ensure_ascii=False,indent=2))
json.dump({'summary':out,'rows':rows}, open('results/b_belief_gate_test.json','w'), ensure_ascii=False, indent=2)
