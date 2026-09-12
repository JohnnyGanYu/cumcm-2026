"""Numerical safety checks for the Q4 rescue development geometry."""
import math
import random

from .b_sim.q4_fast import region_cover_points
from .b_sim.geometry import Bearing, wedge_half_planes
from .b_sim.q4_fast import bounded_region
from .b_sim.q4_fast import Q4FastStrategy as FullRescueStrategy
from .b_eval.dev_measure import InProcessClient
from .b_sim.arena import Arena, Case, Jammer
from .b_sim.strategy import StrategyConfig
import pytest


def test_region_cover_covers_rotated_enclosing_rectangles():
    rng=random.Random(9410)
    accepted=0
    for _ in range(120):
        angle=rng.uniform(-math.pi,math.pi)
        ux,uy=math.cos(angle),math.sin(angle)
        x,y=rng.uniform(-1000,1000),rng.uniform(-1000,1000)
        length,width=rng.uniform(1,900),rng.uniform(1,50)
        def point(u,v):
            return (x+ux*u-uy*v,y+uy*u+ux*v)
        polygon=[point(u,v) for u,v in ((0,0),(length,0),(length,width),(0,width))]
        centres=region_cover_points(polygon)
        if not centres:
            continue
        accepted+=1
        assert len(centres)<=48
        for i in range(21):
            for j in range(9):
                target=point(length*i/20,width*j/8)
                assert min(math.dist(target,c) for c in centres)<=19.9+1e-6
    assert accepted>=50


def test_wedge_filter_never_rejects_a_clear_that_could_succeed():
    rng=random.Random(42001)
    for _ in range(1000):
        source=(rng.uniform(-1000,1000),rng.uniform(-1000,1000))
        bearing_angle=rng.uniform(-180,180)
        distance=rng.uniform(20,1499)
        observer=(source[0]-distance*math.cos(math.radians(bearing_angle)),
                  source[1]-distance*math.sin(math.radians(bearing_angle)))
        bearing=Bearing(*observer,round(bearing_angle+rng.uniform(-1,1),2))
        a=rng.uniform(-math.pi,math.pi)
        radius=rng.uniform(0,20)
        clear=(source[0]+radius*math.cos(a),source[1]+radius*math.sin(a))
        assert all(nx*clear[0]+ny*clear[1]-c<=20+1e-6
                   for nx,ny,c in wedge_half_planes(bearing,1.005))


def test_grid_budget_preserves_old_fallback_for_very_large_region():
    assert region_cover_points([(-900,-900),(900,-900),(900,900),(-900,900)])==[]


def test_observed_source_inside_covered_region():
    rng=random.Random(99410)
    accepted=0
    for _ in range(200):
        source=(rng.uniform(-1100,1100),rng.uniform(-1100,1100))
        bearings=[]
        for i in range(2):
            angle=rng.uniform(-math.pi,math.pi)
            distance=rng.uniform(100,1400)
            observer=(source[0]-distance*math.cos(angle),source[1]-distance*math.sin(angle))
            bearings.append(Bearing(*observer,round(math.degrees(angle)+rng.uniform(-1,1),2)))
        polygon=bounded_region(bearings)
        assert polygon
        points=region_cover_points(polygon)
        if points:
            accepted+=1
            assert min(math.dist(source,p) for p in points)<=19.9+1e-6
    assert accepted>150


@pytest.mark.parametrize('polygon', [[],[(2.,3.)],[(2.,3.),(2.,3.)],
    [(0.,0.),(600.,0.)],[(0.,0.),(600.,1e-10),(600.,2e-10)],
    [(1799.,0.),(1799.,1e-8),(1799.0000001,1e-8)]])
def test_empty_thin_and_degenerate_regions(polygon):
    points=region_cover_points(polygon)
    assert len(points)<=48
    for p in points:
        assert all(math.isfinite(value) for value in p)
    for vertex in polygon:
        assert min(math.dist(vertex,p) for p in points)<=19.9+1e-6


def test_exact_grid_budget_boundary():
    span=19.9*math.sqrt(2)
    # A one-ulp round-off at exactly 48 cells may conservatively choose the
    # fallback; stay just inside the mathematical boundary for this check.
    assert len(region_cover_points([(0.,0.),((48-1e-9)*span,0.)]))==48
    assert region_cover_points([(0.,0.),(48*span+.01,0.)])==[]


def make_dog(jammer,virtual_budget=324000):
    arena=Arena(Case([jammer],'rescue-geometry-test',9410))
    client=InProcessClient(arena)
    client.enter()
    dog=FullRescueStrategy(client,StrategyConfig(directional_probe=True,
                                                virtual_time_budget_s=virtual_budget))
    return dog,arena


def test_near_response_clears_immediately_without_homing():
    dog,arena=make_dog(Jammer(1,103.,0.,1000.,180.))
    state=dog.channels[1]
    dog._probe_channel_at((100.,0.),state)
    assert state.cleared
    assert arena.robot.measure_calls==1
    assert arena.robot.clear_calls==1
    assert not state.bearings
    assert not state.ray_swept


def test_grid_rescue_obeys_virtual_stop_before_call():
    dog,arena=make_dog(Jammer(1,100.,0.,1000.),virtual_budget=0)
    state=dog.channels[1]
    state.add_bearing(0.,0.,0.)
    state.add_bearing(0.,10.,math.degrees(math.atan2(-10.,100.)))
    assert not dog._spiral(state)
    assert arena.robot.clear_calls==0
    assert dog._stopped


def test_grid_rescue_stops_after_one_successful_clear():
    dog,arena=make_dog(Jammer(1,600.,0.,1000.,180.))
    state=dog.channels[1]
    state.add_bearing(0.,0.,0.)
    state.add_bearing(300.,10.,math.degrees(math.atan2(-10.,300.)))
    state.estimate=(800.,0.)
    assert dog._spiral(state)
    assert state.cleared
    assert arena.robot.clear_calls<=48
    assert arena.robot.clear_success==1


def test_ray_rescue_keeps_single_bearing_fallback():
    dog,arena=make_dog(Jammer(1,640.,0.,1000.,180.))
    state=dog.channels[1]
    state.add_bearing(0.,0.,0.)
    assert dog._ray_blind_clear(state)
    assert state.cleared
    assert state.ray_swept
    assert arena.robot.clear_calls==20
    calls=arena.robot.clear_calls
    assert not dog._ray_blind_clear(state)
    assert arena.robot.clear_calls==calls


def test_ray_rescue_obeys_independent_call_budget():
    dog,arena=make_dog(Jammer(1,640.,0.,1000.,180.))
    dog.config.ray_clear_max_attempts=2
    state=dog.channels[1]
    state.add_bearing(0.,0.,0.)
    assert not dog._ray_blind_clear(state)
    assert arena.robot.clear_calls==2


def test_ray_rescue_obeys_virtual_stop_before_call():
    dog,arena=make_dog(Jammer(1,640.,0.,1000.,180.),virtual_budget=0)
    state=dog.channels[1]
    state.add_bearing(0.,0.,0.)
    assert not dog._ray_blind_clear(state)
    assert arena.robot.clear_calls==0
