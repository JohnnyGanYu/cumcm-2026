"""定位第一次归航失败的确切代码出口。

`b_abandon_reasons` 显示 events 为空 => 首次归航失败不走任何 self.events 分支，
说明它是通过 `_dogleg` / `_step_closer` 返回 False 而静默退出的。

`_home_channel` 的静默失败出口只有三处：
  E1  state.estimate is None 且 `_dogleg(state)` 返回 False
  E2  `_step_closer(state, distance)` 返回 False
  E3  距离 <= clear_distance 时 `_clear_at` 失败且 `_spiral` 也返回 False

而 `_dogleg` 返回 False 的原因是：所有候选方位的 dog-leg 落点都越界
（`math.hypot(*candidate) > ARENA_RADIUS`），或归航预算耗尽。
`_step_closer` 返回 False 的原因是：两个横向候选点都越界。

=> 若主因是"落点越界"，那么尾部源就是**边缘源**：机器狗站在环上(R=1250)朝外侧
   目标做 dog-leg，380 m 的基线把它推到 1800 m 区域外，于是所有候选被拒、归航
   直接放弃，只能等下一轮。这与"尾部源 mean_dist_from_origin=1303 m"一致。
   修法不是重试，而是**把 dog-leg / step_closer 的候选点裁剪回区域内**而不是丢弃。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import threading

from .b_sim.arena import ARENA_RADIUS, Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig


class TracedStrategy(DogStrategy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.exits = collections.Counter()
        self.dogleg_reject = collections.Counter()
        self.step_reject = collections.Counter()
        self.first_fail_info: list[dict] = []
        self._depth_ch = None

    def _home_channel(self, state):
        top = self._depth_ch is None
        if top:
            self._depth_ch = state.channel
        r = super()._home_channel(state)
        if top:
            self._depth_ch = None
            if not r and not state.cleared:
                self.exits["home_failed"] += 1
        return r

    def _dogleg(self, state):
        cfg = self.config
        last = state.bearings[-1] if state.bearings else None
        r = super()._dogleg(state)
        if not r and last is not None:
            # 复算候选点，判断是否全部越界
            out = 0
            tot = 0
            for offset in (cfg.dogleg_offset_deg, cfg.dogleg_offset_deg * 0.5, 0.0):
                for sign in (1.0, -1.0) if offset else (1.0,):
                    ang = math.radians(last.svd_deg + sign * offset)
                    cx = self.pos[0] + cfg.dogleg_length_m * math.cos(ang)
                    cy = self.pos[1] + cfg.dogleg_length_m * math.sin(ang)
                    tot += 1
                    if math.hypot(cx, cy) > ARENA_RADIUS:
                        out += 1
            self.dogleg_reject["all_out_of_bounds" if out == tot else "other"] += 1
            self.first_fail_info.append({
                "where": "dogleg", "channel": state.channel,
                "pos_r": round(math.hypot(*self.pos)),
                "cands_out_of_bounds": out, "cands_total": tot,
                "homing_measures": state.homing_measures,
                "consecutive_silent": state.consecutive_silent,
            })
        return r

    def _step_closer(self, state, distance):
        cfg = self.config
        est = state.estimate
        r = super()._step_closer(state, distance)
        if not r and est is not None:
            ex, ey = est
            dx, dy = ex - self.pos[0], ey - self.pos[1]
            norm = math.hypot(dx, dy) or 1.0
            ux, uy = dx / norm, dy / norm
            advance = max(cfg.clear_distance_m * 0.8, distance * cfg.approach_fraction)
            advance = (min(advance, distance - cfg.clear_distance_m * 0.35)
                       if distance > cfg.clear_distance_m else advance)
            lateral = min(distance * cfg.lateral_fraction, 220.0)
            out = 0
            for sign in (1.0, -1.0):
                cx = self.pos[0] + ux * advance - uy * lateral * sign
                cy = self.pos[1] + uy * advance + ux * lateral * sign
                if math.hypot(cx, cy) > ARENA_RADIUS * 1.02:
                    out += 1
            self.step_reject["all_out_of_bounds" if out == 2 else "other"] += 1
            self.first_fail_info.append({
                "where": "step_closer", "channel": state.channel,
                "pos_r": round(math.hypot(*self.pos)),
                "est_r": round(math.hypot(ex, ey)),
                "cands_out_of_bounds": out, "cands_total": 2,
                "distance": round(distance),
                "consecutive_silent": state.consecutive_silent,
            })
        return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=81000)
    ap.add_argument("--directional", type=int, default=5)
    args = ap.parse_args()

    dl = collections.Counter()
    sc = collections.Counter()
    allinfo = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        case = generate_case(seed, directional_count=args.directional)
        arena = Arena(case)
        session = Session("dog-x", arena, real_limit_s=1200.0)
        httpd = make_server(session, port=0)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            client = RobotClient("dog-x", base_url="http://127.0.0.1:" + str(port))
            strat = TracedStrategy(client, StrategyConfig(
                directional_probe=args.directional > 0))
            strat.run()
            dl.update(strat.dogleg_reject)
            sc.update(strat.step_reject)
            allinfo += strat.first_fail_info
            print(f"seed={seed} dogleg_fail={sum(strat.dogleg_reject.values())} "
                  f"step_fail={sum(strat.step_reject.values())} "
                  f"home_failed={strat.exits['home_failed']}", flush=True)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def summ(where):
        g = [x for x in allinfo if x["where"] == where]
        if not g:
            return {}
        oob = [x for x in g if x["cands_out_of_bounds"] == x["cands_total"]]
        import statistics as st
        return {
            "n": len(g),
            "all_candidates_out_of_bounds": len(oob),
            "pct_all_oob": round(len(oob) / len(g) * 100, 1),
            "mean_pos_r": round(st.fmean(x["pos_r"] for x in g)),
            "mean_consecutive_silent": round(
                st.fmean(x["consecutive_silent"] for x in g), 1),
        }

    report = {
        "episodes": args.episodes,
        "dogleg_reject": dict(dl),
        "step_reject": dict(sc),
        "dogleg_detail": summ("dogleg"),
        "step_closer_detail": summ("step_closer"),
        "arena_radius": ARENA_RADIUS,
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    with open("results/b_exitpath.json", "w", encoding="utf-8") as h:
        json.dump({"report": report, "samples": allinfo[:200]}, h,
                  ensure_ascii=False, indent=2)
    print("\nwrote results/b_exitpath.json")


if __name__ == "__main__":
    main()
