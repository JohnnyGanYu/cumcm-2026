"""本地 mock 模拟器：按附件2 协议提供 /enter /measure /clear /exit。

用途：离线无限次演练机器狗策略。官方模拟器为最终判据。
只监听 127.0.0.1，与官方一致；不实现登录、加密日志和界面。

已实现的协议细节：路径精确匹配、仅 POST、Content-Type 校验、请求体大小上限、
重复键拒绝、未知字段 accepted=false、arena_id/robot_id 校验、request_id 幂等
（同 ID 同内容重放原响应，同 ID 异内容 409）、channel 整数值域、坐标有限性与
绝对值上限、accepted=false 时 virtual_time_s 为 0。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .arena import (
    CHANNEL_MAX,
    CHANNEL_MIN,
    MAX_REAL_SECONDS,
    Arena,
    ArenaRejected,
    generate_case,
)

MAX_BODY_BYTES = 65536
MAX_COORD_ABS = 2_000_000.0
VALID_PATHS = {"/enter", "/measure", "/clear", "/exit"}
ALLOWED_FIELDS = {
    "/enter": {"arena_id", "robot_id", "request_id"},
    "/exit": {"arena_id", "robot_id", "request_id"},
    "/measure": {"arena_id", "robot_id", "request_id", "position", "channel"},
    "/clear": {"arena_id", "robot_id", "request_id", "position", "channel"},
}


class BadRequest(Exception):
    """对应 HTTP 400。"""


class Rejected(Exception):
    """对应 HTTP 200 + accepted=false。"""


class Conflict(Exception):
    """对应 HTTP 409。"""


def _no_duplicate_keys(pairs):
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise BadRequest(f"重复键 {key}")
        seen[key] = value
    return seen


class Session:
    """一局测试的会话状态：案例、竞技场、幂等表、现实时间限制。"""

    def __init__(self, robot_id: str, arena: Arena, *, real_limit_s: float = MAX_REAL_SECONDS) -> None:
        self.robot_id = robot_id
        self.arena = arena
        self.real_limit_s = real_limit_s
        self.idempotent: dict[str, tuple[str, str, dict]] = {}
        self.enter_monotonic: float | None = None
        self.lock = threading.Lock()

    def remaining_real_s(self) -> float:
        if self.enter_monotonic is None:
            return self.real_limit_s
        return max(0.0, self.real_limit_s - (time.monotonic() - self.enter_monotonic))


class Handler(BaseHTTPRequestHandler):
    server_version = "cumcm-b-mock/1.0"
    protocol_version = "HTTP/1.1"
    session: Session  # 由 make_server 注入到类属性

    def log_message(self, format, *args):  # noqa: A002,N802 - 抑制默认 stderr 噪声
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def do_GET(self):  # noqa: N802
        self._send(405, {"accepted": False, "real_timestamp_ms": _now_ms(), "virtual_time_s": 0})

    def do_POST(self):  # noqa: N802
        path = self.path
        if path not in VALID_PATHS:
            self._send(404, self._envelope(False))
            return
        ctype = self.headers.get("Content-Type", "")
        base, *params = [p.strip() for p in ctype.split(";")]
        if base.lower() != "application/json" or any(
            p.lower() not in {"charset=utf-8", ""} for p in params
        ):
            self._send(415, self._envelope(False))
            return
        encoding = (self.headers.get("Content-Encoding") or "identity").strip().lower()
        if encoding != "identity":
            self._send(415, self._envelope(False))
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._send(413, self._envelope(False))
            return
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
            if not isinstance(payload, dict):
                raise BadRequest("请求体必须是 JSON 对象")
            status, body = self._dispatch(path, payload)
        except BadRequest:
            status, body = 400, self._envelope(False)
        except Conflict:
            status, body = 409, self._envelope(False)
        except Rejected:
            status, body = 200, self._envelope(False)
        except json.JSONDecodeError:
            status, body = 400, self._envelope(False)
        self._send(status, body)

    # ---- 业务分派 ----

    def _dispatch(self, path: str, payload: dict) -> tuple[int, dict]:
        session = self.session
        unknown = set(payload) - ALLOWED_FIELDS[path]
        arena_id = payload.get("arena_id")
        robot_id = payload.get("robot_id")
        request_id = payload.get("request_id")

        arena_id_s = _require_identifier("arena_id", arena_id)
        robot_id_s = _require_identifier("robot_id", robot_id)
        request_id_s = _require_identifier("request_id", request_id)
        if len(robot_id_s.encode()) > 64 or len(request_id_s.encode()) > 128:
            raise BadRequest("标识符超长")

        pos_x = pos_y = 0.0
        channel = 0
        if path in {"/measure", "/clear"}:
            position = payload.get("position")
            channel = payload.get("channel")
            if not isinstance(position, dict):
                raise BadRequest("缺少 position")
            if set(position) - {"x", "y"}:
                unknown.add("position.*")
            for axis in ("x", "y"):
                value = position.get(axis)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise BadRequest(f"position.{axis} 必须是数值")
                if not math.isfinite(float(value)) or abs(float(value)) > MAX_COORD_ABS:
                    raise BadRequest(f"position.{axis} 越界")
            if isinstance(channel, bool) or not isinstance(channel, (int, float)):
                raise BadRequest("channel 必须是数值")
            if float(channel) != int(channel):
                raise BadRequest("channel 必须是整数")
            channel = int(channel)
            if not CHANNEL_MIN <= channel <= CHANNEL_MAX:
                raise BadRequest("channel 越界")
            pos_x, pos_y = float(position["x"]), float(position["y"])
            action_key = (path, json.dumps({"x": pos_x, "y": pos_y, "channel": channel}, sort_keys=True))
        else:
            action_key = (path, "")

        # 未知字段、arena_id/robot_id 不匹配：accepted=false，不占用 request_id
        if unknown:
            raise Rejected(f"未知字段 {sorted(unknown)}")
        if arena_id_s != "default":
            raise Rejected("arena_id 必须是 default")
        if robot_id_s != session.robot_id:
            raise Rejected("robot_id 与登录参赛队号不一致")

        with session.lock:
            prior = session.idempotent.get(request_id_s)
            if prior is not None:
                prior_path, prior_key, prior_body = prior
                if (prior_path, prior_key) != action_key:
                    raise Conflict("同一 request_id 对应不同动作")
                return 200, prior_body

            if path != "/enter" and session.enter_monotonic is not None and session.remaining_real_s() <= 0:
                session.arena.robot.finished = True
                session.arena.robot.exit_reason = "real_time_exhausted"
                raise Rejected("程序运行时间已用尽")

            try:
                if path == "/enter":
                    extra = session.arena.enter()
                    session.enter_monotonic = time.monotonic()
                    extra["remaining_real_duration_s"] = int(session.remaining_real_s())
                elif path == "/measure":
                    extra = session.arena.measure(pos_x, pos_y, channel)
                elif path == "/clear":
                    extra = session.arena.clear(pos_x, pos_y, channel)
                else:
                    extra = session.arena.exit()
            except ArenaRejected as exc:
                raise Rejected(str(exc)) from exc

            body = self._envelope(True)
            body.update(extra)
            session.idempotent[request_id_s] = (action_key[0], action_key[1], body)
            return 200, body

    # ---- 响应 ----

    def _envelope(self, accepted: bool) -> dict:
        virtual = self.session.arena.robot.virtual_time_s if accepted else 0
        return {"accepted": accepted, "real_timestamp_ms": _now_ms(), "virtual_time_s": virtual}

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _require_identifier(name: str, value: object) -> str:
    """校验并收窄标识符字段类型。"""
    if value is None:
        raise BadRequest(f"缺少 {name}")
    if not isinstance(value, str) or not value:
        raise BadRequest(f"{name} 必须是非空字符串")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise BadRequest(f"{name} 含控制字符")
    return value


def _now_ms() -> int:
    return int(time.time() * 1000)


def make_server(
    session: Session,
    *,
    host: str = "127.0.0.1",
    port: int = 2026,
    verbose: bool = False,
) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"session": session})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.verbose = verbose  # type: ignore[attr-defined]
    return httpd


def main() -> None:
    parser = argparse.ArgumentParser(description="B 题本地 mock 模拟器")
    parser.add_argument("--robot-id", required=True, help="机器狗标识（对应参赛队号）")
    parser.add_argument("--port", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=None, help="干扰源总数，默认 10-16 随机")
    parser.add_argument("--directional", type=int, default=0, help="定向干扰源个数（问题4 用）")
    parser.add_argument("--real-limit", type=float, default=MAX_REAL_SECONDS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    case = generate_case(args.seed, count=args.count, directional_count=args.directional)
    session = Session(args.robot_id, Arena(case), real_limit_s=args.real_limit)
    httpd = make_server(session, port=args.port, verbose=args.verbose)
    print(f"mock simulator on http://127.0.0.1:{args.port}  jammers={case.total} "
          f"directional={case.directional_count} seed={args.seed}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n" + json.dumps(session.arena.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
