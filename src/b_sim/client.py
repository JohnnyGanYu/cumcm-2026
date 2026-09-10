"""机器狗侧 HTTP 客户端：串行动作、唯一 request_id、网络重试复用原请求。

设计要点（对齐附件2 第12节编程注意事项）：
- 每个新动作生成新 request_id；仅在网络异常重试时复用原请求内容与原 ID；
- 同时检查 HTTP 状态与 accepted；accepted=false 时 virtual_time_s 不代表当前虚拟时刻；
- 使用 /enter 返回的 remaining_real_duration_s 控制现实时间预算，不假定 1200 秒；
- 自行记录完整指令与响应序列（模拟器不提供该功能）。
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# 协议允许的坐标绝对值上限（米）；abs(x)/abs(y) 超过该值越界拒发，
# 恰为 MAX_COORD_ABS 视为合法。与 server.MAX_COORD_ABS 保持一致（R5.3）。
MAX_COORD_ABS = 2_000_000.0


class ProtocolError(RuntimeError):
    """动作未被接受，或响应结构不符合预期。"""


class TestEnded(RuntimeError):
    """接口已关闭或测试已结束（连接失败、超时退出）。"""


@dataclass
class MeasureResult:
    result: str  # "no_signal" | "near" | "direction"
    svd_deg: float | None
    virtual_time_s: float

    @property
    def has_direction(self) -> bool:
        return self.result == "direction"


@dataclass
class ClearResult:
    result: str  # "success" | "no_target_in_range"
    virtual_time_s: float

    @property
    def ok(self) -> bool:
        return self.result == "success"


@dataclass
class RobotClient:
    robot_id: str
    base_url: str = "http://127.0.0.1:2026"
    timeout_s: float = 5.0
    max_network_retries: int = 3
    arena_id: str = "default"
    log: list[dict] = field(default_factory=list)
    _seq: int = 0
    virtual_time_s: float = 0.0
    remaining_real_s: float = 0.0
    _enter_monotonic: float | None = None

    # ---- 动作 ----

    def enter(self) -> dict:
        body = self._post("/enter", self._base_payload("enter"))
        self.remaining_real_s = float(body.get("remaining_real_duration_s", 0.0))
        self._enter_monotonic = time.monotonic()
        return body

    def measure(self, x: float, y: float, channel: int) -> MeasureResult:
        self._check_coord(x, y)
        payload = self._base_payload("measure")
        payload["position"] = {"x": float(x), "y": float(y)}
        payload["channel"] = int(channel)
        body = self._post("/measure", payload)
        return MeasureResult(
            result=body["measure_result"],
            svd_deg=body.get("svd_deg"),
            virtual_time_s=float(body["virtual_time_s"]),
        )

    def clear(self, x: float, y: float, channel: int) -> ClearResult:
        self._check_coord(x, y)
        payload = self._base_payload("clear")
        payload["position"] = {"x": float(x), "y": float(y)}
        payload["channel"] = int(channel)
        body = self._post("/clear", payload)
        return ClearResult(result=body["clear_result"], virtual_time_s=float(body["virtual_time_s"]))

    def exit(self) -> dict:
        return self._post("/exit", self._base_payload("exit"))

    # ---- 现实时间预算 ----

    def real_elapsed_s(self) -> float:
        if self._enter_monotonic is None:
            return 0.0
        return time.monotonic() - self._enter_monotonic

    def real_budget_left_s(self) -> float:
        return max(0.0, self.remaining_real_s - self.real_elapsed_s())

    # ---- 内部 ----

    @staticmethod
    def _check_coord(x: float, y: float) -> None:
        """发送前坐标越界拦截（R5.3）。

        `abs(x)` 或 `abs(y)` 超过 MAX_COORD_ABS（含 NaN/inf 等非有限值）即在发送前
        抛 `ProtocolError`；不生成 request_id、不发送 HTTP 请求、不记账、不推进虚拟时刻。
        恰为 MAX_COORD_ABS（2_000_000 m）视为合法而放行。
        """
        for axis, value in (("x", x), ("y", y)):
            fv = float(value)
            if not math.isfinite(fv) or abs(fv) > MAX_COORD_ABS:
                raise ProtocolError(
                    f"坐标越界：position.{axis}={value} 超出 ±{MAX_COORD_ABS:g} m，发送前拒绝"
                )

    def _base_payload(self, tag: str) -> dict:
        self._seq += 1
        return {
            "arena_id": self.arena_id,
            "robot_id": self.robot_id,
            "request_id": f"{tag}-{self._seq}",
        }

    def _post(self, path: str, payload: dict) -> dict:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.max_network_retries + 1):
            request = urllib.request.Request(
                self.base_url + path,
                data=raw,  # 重试时复用完全相同的请求体与 request_id
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    status = response.status
                break
            except urllib.error.HTTPError as exc:  # 有 JSON 体的错误
                body = json.loads(exc.read().decode("utf-8") or "{}")
                status = exc.code
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt == self.max_network_retries:
                    raise TestEnded(f"{path} 连接失败（接口可能未开放或测试已结束）: {exc}") from exc
                time.sleep(0.2 * attempt)
        else:  # pragma: no cover
            raise TestEnded(f"{path} 重试耗尽: {last_error}")

        self.log.append({"path": path, "request": payload, "http_status": status, "response": body})
        if status != 200 or body.get("accepted") is not True:
            raise ProtocolError(f"{path} 未执行 http={status} body={body}")
        self.virtual_time_s = float(body["virtual_time_s"])
        return body

    def dump_log(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.log, handle, ensure_ascii=False, indent=2)
