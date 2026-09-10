#!/usr/bin/env python3
"""定位"清除成功后仍继续清除"的调用来源。打桩看是谁在成功后又调 _clear_at。"""
import math, traceback
from src.b_sim.rehearse import run_episode
from src.b_sim.strategy import DogStrategy, StrategyConfig

orig_clear = DogStrategy._clear_at
records = []

def clear_at(self, point, channel):
    st = self.channels[channel]
    was = st.cleared
    # 取调用栈里 b_sim 内的函数名
    stack = [f.name for f in traceback.extract_stack()[:-1] if "strategy.py" in f.filename]
    r = orig_clear(self, point, channel)
    records.append({
        "ch": channel, "was_cleared": was, "ok": r,
        "caller": stack[-1] if stack else "?",
        "chain": "->".join(stack[-3:]),
    })
    return r

DogStrategy._clear_at = clear_at
ep = run_episode(7001, directional=5, config=StrategyConfig(directional_probe=True),
                 real_limit_s=1200.0)
print(f"局: {ep['cleared']}/{ep['total_jammers']} 清除调用={ep['clear_calls']}")

waste = [r for r in records if r["was_cleared"]]
print(f"\n总 _clear_at 调用 {len(records)}, 其中【已清除后仍调用】{len(waste)} 次")
import collections
print("\n浪费调用的来源链:")
for chain, n in collections.Counter(r["chain"] for r in waste).most_common():
    print(f"  {n:>3}x  {chain}")
print("\n全部调用来源:")
for chain, n in collections.Counter(r["chain"] for r in records).most_common():
    print(f"  {n:>3}x  {chain}")
