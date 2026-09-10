#!/usr/bin/env python3
"""追踪 seed 7001 ch11 的实际执行路径：射线盲扫到底有没有被调用。

前三次修改都失败，不再猜参数。给关键方法打桩记录真实调用顺序。
"""
import math

from src.b_sim.rehearse import run_episode
from src.b_sim.strategy import DogStrategy, StrategyConfig

TRACE = []
CH = 11

_orig_ray = DogStrategy._ray_blind_clear
_orig_final = DogStrategy._final_attempt
_orig_home = DogStrategy._home_channel
_orig_clear = DogStrategy._clear_at


def ray(self, state):
    if state.channel == CH:
        TRACE.append(f"_ray_blind_clear ENTER bearings={len(state.bearings)} "
                     f"clear_attempts={state.clear_attempts}")
    r = _orig_ray(self, state)
    if state.channel == CH:
        TRACE.append(f"_ray_blind_clear EXIT -> {r} clear_attempts={state.clear_attempts}")
    return r


def final(self, state):
    if state.channel == CH:
        TRACE.append(f"_final_attempt ENTER estimate={state.estimate} "
                     f"bearings={len(state.bearings)} clear_attempts={state.clear_attempts}")
    r = _orig_final(self, state)
    if state.channel == CH:
        TRACE.append(f"_final_attempt EXIT -> {r}")
    return r


def home(self, state):
    if state.channel == CH:
        TRACE.append(f"_home_channel ENTER silent={state.consecutive_silent} "
                     f"homing={state.homing_measures} clear_attempts={state.clear_attempts}")
    r = _orig_home(self, state)
    if state.channel == CH:
        TRACE.append(f"_home_channel EXIT -> {r}")
    return r


def clear_at(self, point, channel):
    r = _orig_clear(self, point, channel)
    if channel == CH:
        TRACE.append(f"  _clear_at ({point[0]:.0f},{point[1]:.0f}) -> {r} "
                     f"attempts={self.channels[channel].clear_attempts}")
    return r


DogStrategy._ray_blind_clear = ray
DogStrategy._final_attempt = final
DogStrategy._home_channel = home
DogStrategy._clear_at = clear_at


ep = run_episode(7001, directional=5, count=None,
                 config=StrategyConfig(directional_probe=True), real_limit_s=1200.0)
print(f"seed 7001: cleared={ep['cleared']}/{ep['total_jammers']} missed={ep['missed_channels']}")
print(f"总 clear 调用={ep['clear_calls']}  measure={ep['measure_calls']}")
print()
print(f"=== ch{CH} 执行路径 ({len(TRACE)} 条) ===")
for i, line in enumerate(TRACE):
    print(f"{i:3d} {line}")

print()
ray_calls = [t for t in TRACE if "_ray_blind_clear ENTER" in t]
print(f"射线盲扫被调用次数: {len(ray_calls)}")
if not ray_calls:
    print("=> 射线盲扫从未被调用！说明退出路径不是我以为的那条。")
else:
    clears = [t for t in TRACE if "_clear_at" in t]
    print(f"ch{CH} 的 clear 调用共 {len(clears)} 次")
