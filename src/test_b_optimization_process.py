"""Spawned benchmark workers must use the immutable parent source snapshot."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

from .b_optimization_benchmark import frozen_versions, initialize_worker, run_episode


def _worker_snapshot_probe(seed):
    from . import b_optimization_benchmark as benchmark
    bundles = benchmark._WORKER_BUNDLES
    paths = {arm: {name: module.__file__ for name, module in modules.items()}
             for arm, modules in bundles.items()}
    pair = benchmark.worker_pair(seed, 0,
                                 {"certified_silence": False, "bounded_homing": False}, "local")
    return {"pid": os.getpid(), "paths": paths, "pair": pair}


def test_spawned_worker_uses_disk_snapshot_and_preserves_generated_case(monkeypatch):
    with frozen_versions("83a3064") as (bundles, metadata):
        root = Path(bundles["baseline"]["strategy"].__file__).parent.parent.resolve()
        package_names = {arm: bundle["package_name"] for arm, bundle in bundles.items()}
        overrides = {"certified_silence": False, "bounded_homing": False}
        parent_result = run_episode(bundles, 930000, 0, "candidate", overrides)
        assert parent_result["finished"] and not parent_result["exception"]

        def parent_only_run(self):
            raise AssertionError("a process worker must not inherit mutable parent strategy classes")

        # This deliberately changes only the parent's in-memory class. Child
        # imports must still resolve to the frozen files, never these objects.
        monkeypatch.setattr(bundles["candidate"]["strategy"].DogStrategy, "run", parent_only_run)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(str(root), package_names)) as pool:
            child = pool.submit(_worker_snapshot_probe, 930000).result(timeout=60)
        assert child["pid"] != os.getpid()
        for modules in child["paths"].values():
            assert all(Path(filename).resolve().is_relative_to(root) for filename in modules.values())
        baseline, candidate = child["pair"]["baseline"], child["pair"]["candidate"]
        assert baseline["case_sha256"] == candidate["case_sha256"] == parent_result["case_sha256"]
        assert baseline["total_jammers"] == candidate["total_jammers"]
        assert candidate["trace_sha256"] == parent_result["trace_sha256"]
        assert candidate["config"] == parent_result["config"]
        assert candidate["certification_audit"] == parent_result["certification_audit"]
        assert candidate["finished"] and not candidate["exception"] and not candidate["aborted_events"]
        assert metadata["baseline_commit"].startswith("83a3064")
