"""统计首次归航失败的原因分布，判断"能否让第一次就成功"。

已测得的关键事实（`b_tail_diag`，8 局）：
  尾部源的第一次归航在 t≈2000-3500 s 失败
  之后要等 p50=6809 s 才有第二次归航
  第二次归航只花 p50=248 s 就清除成功
=> 浪费的不是归航，是**等待重试**。

本脚本回答：第一次为什么失败？失败原因决定修法：
  lost signal        -> 归航路上走出扇区/接收半径；换路径或放宽容忍度
  measure budget     -> 归航检测额度不够
  clear budget       -> 围绕错误估计打满 clear
若绝大多数是 lost signal，则"就地重试"没用（信号确实没了），
必须改成"记账 + 等走到更近的环站位时再归航"，即按几何而非按时间调度。
"""

from __future__ import annotations

import argparse
import collections
import json
import threading

from .b_sim.arena import Arena, generate_case
from .b_sim.client import RobotClient
from .b_sim.server import Session, make_server
from .b_sim.strategy import DogStrategy, StrategyConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=81000)
    ap.add_argument("--directional", type=int, default=5)
    args = ap.parse_args()

    reasons = collections.Counter()
    per_episode = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        case = generate_case(seed, directional_count=args.directional)
        arena = Arena(case)
        session = Session("dog-ab", arena, real_limit_s=1200.0)
        httpd = make_server(session, port=0)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            client = RobotClient("dog-ab", base_url="http://127.0.0.1:" + str(port))
            strat = DogStrategy(client, StrategyConfig(
                directional_probe=args.directional > 0))
            strat.run()
            local = collections.Counter()
            for e in strat.events:
                key = e.split(":", 1)[1].strip() if ":" in e else e
                reasons[key] += 1
                local[key] += 1
            per_episode.append({"seed": seed, "events": dict(local)})
            print(f"seed={seed} " + " | ".join(f"{k}={v}" for k, v in local.most_common(6)),
                  flush=True)
        finally:
            httpd.shutdown()
            httpd.server_close()

    total = sum(reasons.values())
    report = {
        "episodes": args.episodes,
        "total_events": total,
        "reasons": {k: {"n": v, "pct": round(v / total * 100, 1)}
                    for k, v in reasons.most_common()},
    }
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    with open("results/b_abandon_reasons.json", "w", encoding="utf-8") as h:
        json.dump({"report": report, "per_episode": per_episode}, h,
                  ensure_ascii=False, indent=2)
    print("\nwrote results/b_abandon_reasons.json")


if __name__ == "__main__":
    main()
