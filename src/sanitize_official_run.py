"""官方演练/正式测试产物脱敏，用于安全入库。

为什么需要：官方客户端产出的 .jlog 内嵌真实参赛队号与 server_ticket，
控制台日志含队号与本机 Windows 用户名。竞赛要求提交物全程匿名，
仓库也不应存放服务端票据。

策略：文本层面正则替换 + JSON 字段置空，不解析私有二进制结构，
因此对未知格式也安全（只做替换，不重写长度字段）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# 需要置空的敏感 JSON 字段（值可能是 base64 票据或指纹）
# 需要置空的敏感 JSON 字段（值可能是 base64 票据、包裹密钥或指纹）。
# 这份清单必须与 supporting/README.md 第 2 节列出的字段一致 —— 曾经不一致：
# 文档声称会清除 wrapped_dek_b64（RSA-OAEP 包裹的数据加密密钥），脚本里却漏了它，
# 于是「只有脱敏件可入库」这条硬约束在最敏感的那一项上是空的。
REDACT_KEYS = (
    "server_ticket_b64",
    "authorization_nonce_b64",
    "package_signing_public_key_b64",
    "device_digest",
    "signature_b64",
    "payload_signature_b64",
    "wrapped_dek_b64",
    # 无 _b64 后缀的写法，官方字段命名若有差异也能覆盖
    "wrapped_dek",
    "server_ticket",
    "authorization_nonce",
)


# 复核用的独立清单：无论 REDACT_KEYS 怎么改，这些字段一旦带非零值就必须报错。
# 与 supporting/README.md 第 2 节的声称一一对应。
MUST_BE_ZEROED = (
    "server_ticket_b64",
    "authorization_nonce_b64",
    "package_signing_public_key_b64",
    "device_digest",
    "signature_b64",
    "payload_signature_b64",
    "wrapped_dek_b64",
)

def sanitize_bytes(raw: bytes, team_no: str) -> tuple[bytes, list[str]]:
    """对原始字节做脱敏，返回 (结果, 命中项说明)。

    只替换等长或更短的内容，避免破坏可能存在的长度前缀。
    """
    hits: list[str] = []
    out = raw

    if team_no:
        tn = team_no.encode()
        if tn in out:
            # 用同长度占位，尽量不改变字节偏移
            placeholder = b"T" * len(tn)
            out = out.replace(tn, placeholder)
            hits.append(f"team_no({team_no}) -> {placeholder.decode()}")

    # Windows 用户目录
    def _win_user(m: re.Match[bytes]) -> bytes:
        name = m.group(1)
        return b"C:\\Users\\" + b"U" * len(name)

    new = re.sub(rb"C:\\Users\\([A-Za-z0-9._-]+)", _win_user, out)
    if new != out:
        hits.append("windows user dir -> C:\\Users\\U...")
        out = new

    # JSON 票据字段：保持字符串长度，避免破坏外层长度声明
    for key in REDACT_KEYS:
        pat = rb'"' + key.encode() + rb'"\s*:\s*"([^"]*)"'

        def _redact(m: re.Match[bytes], k: str = key) -> bytes:
            val = m.group(1)
            filler = b"0" * len(val)
            return b'"' + k.encode() + b'":"' + filler + b'"'

        new = re.sub(pat, _redact, out)
        if new != out:
            hits.append(f"{key} -> zeroed")
            out = new

    return out, hits


def main() -> int:
    ap = argparse.ArgumentParser(description="官方测试产物脱敏")
    ap.add_argument("inputs", nargs="+", help="待脱敏文件")
    ap.add_argument("--team-no", required=True, help="要抹掉的真实参赛队号")
    ap.add_argument("--suffix", default=".sanitized", help="输出文件后缀标记")
    args = ap.parse_args()

    for path_s in args.inputs:
        src = Path(path_s)
        if not src.is_file():
            print(f"跳过（不存在）: {src}")
            continue
        raw = src.read_bytes()
        out, hits = sanitize_bytes(raw, args.team_no)
        dst = src.with_name(src.stem + args.suffix + src.suffix)
        dst.write_bytes(out)
        print(f"{src.name} -> {dst.name}  ({len(out)} bytes)")
        for h in hits:
            print(f"    redacted: {h}")
        if not hits:
            print("    (无敏感命中)")

        # 复核：确认队号与用户名确实不在输出里
        leftovers = []
        if args.team_no.encode() in out:
            leftovers.append("team_no")
        if re.search(rb"C:\\Users\\(?!U+\b)[A-Za-z0-9._-]+", out):
            leftovers.append("windows_user")
        # 密钥/票据字段的落地复核。
        # 这里刻意用独立的 MUST_BE_ZEROED 而不是 REDACT_KEYS —— 若复用后者，
        # 从 REDACT_KEYS 摘掉一个字段会同时使它退出复核范围，变成自证清白。
        for key in MUST_BE_ZEROED:
            pat = rb'"' + key.encode() + rb'"\s*:\s*"([^"]*)"'
            for m in re.finditer(pat, out):
                val = m.group(1)
                if val and set(val) != {ord("0")}:
                    leftovers.append(f"{key}(未置零)")
                    break
        if leftovers:
            print(f"    !! 仍残留: {leftovers}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
