"""Offline arena client for paired Q4 evaluation.

The protocol boundary is copied unchanged from the local development harness;
controller and evaluation-framework imports are deliberately unnecessary here.
No official service is contacted.
"""
from __future__ import annotations
import time
from src.b_sim.arena import Arena
from src.b_sim.client import ClearResult, MeasureResult, RobotClient

ROBOT_ID = "dev-measure"
REAL_BUDGET_S = 1200.0


class InProcessClient(RobotClient):
    """Legacy client boundary bound to an in-process arena (no HTTP hop)."""

    def __init__(self, arena: Arena) -> None:
        super().__init__(ROBOT_ID)
        self.arena = arena

    def _act(self, path: str, point, channel) -> tuple[dict, float]:
        normalized = None if point is None else (float(point[0]), float(point[1]))
        self._seq += 1
        if path == "/enter":
            body = self.arena.enter()
        elif path == "/measure":
            body = self.arena.measure(normalized[0], normalized[1], channel)
        elif path == "/clear":
            body = self.arena.clear(normalized[0], normalized[1], channel)
        else:
            body = self.arena.exit()
        virtual = self.arena.robot.virtual_time_s
        self.virtual_time_s = virtual
        self.log.append(
            {
                "path": path,
                "request_id": f"{path.lstrip('/')}-{self._seq}",
                "request": {"position": normalized, "channel": channel},
                "http_status": 200,
                "response": {"accepted": True, "virtual_time_s": virtual, **body},
            }
        )
        return body, virtual

    def enter(self) -> dict:
        body, _ = self._act("/enter", None, None)
        self._enter_monotonic = time.monotonic()
        return body

    def measure(self, x: float, y: float, channel: int) -> MeasureResult:
        body, virtual = self._act("/measure", (x, y), channel)
        return MeasureResult(body["measure_result"], body.get("svd_deg"), virtual)

    def clear(self, x: float, y: float, channel: int) -> ClearResult:
        body, virtual = self._act("/clear", (x, y), channel)
        return ClearResult(body["clear_result"], virtual)

    def exit(self) -> dict:
        body, _ = self._act("/exit", None, None)
        return body

    def real_budget_left_s(self) -> float:
        return REAL_BUDGET_S
