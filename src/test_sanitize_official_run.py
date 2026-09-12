"""脱敏脚本的回归。

「只有脱敏件可入库」是匿名硬约束的最后一道闸，而脚本写于 9-10，
日志格式此后加过 paper_table1_row / strategy_class / strategy_config / crash。
这里同时验两件事：敏感值确实被抹掉，且新增字段没被脱敏破坏。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "src" / "sanitize_official_run.py"
TEAM = "202612345678"


def _sample() -> dict:
    """构造一份含真队号与服务端机密的日志，字段布局照当前 run_official 输出。"""
    return {
        "robot_id": TEAM,
        "directional": True,
        "test_kind": "formal",
        "case_code": "CASE-7",
        "paper_table1_row": {
            "测试案例编码": "CASE-7",
            "清除干扰源个数": 13,
            "平均定位清除时间_s": 439.7718,
            "程序运行时间_s": 5.12,
        },
        "summary": {
            "cleared_count": 13,
            "virtual_time_s": 5717.03,
            "strategy_profile": "q4-fast-v3",
            "search_layout": {"outer_radius_m": 1870.0},
        },
        "strategy_class": "Q4FastStrategy",
        "crash": None,
        "protocol_log": [
            {
                "path": "/enter",
                "request": {"robot_id": TEAM},
                "response": {
                    "accepted": True,
                    "server_ticket_b64": "c2VjcmV0LXRpY2tldA==",
                    "wrapped_dek_b64": "d3JhcHBlZC1kZWs=",
                },
            },
            {"path": "/measure", "request": {"robot_id": TEAM}, "response": {"bearing": 12.5}},
        ],
    }


def _run(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "official_log_x.json"
    src.write_text(json.dumps(_sample(), ensure_ascii=False, indent=2), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(src), "--team-no", TEAM],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = src.with_suffix(".sanitized.json")
    assert out.exists(), "脱敏件未生成"
    return out, out.read_text(encoding="utf-8")


def test_team_number_is_fully_removed(tmp_path):
    _, text = _run(tmp_path)
    assert TEAM not in text, "真队号仍有残留"
    assert "T" * len(TEAM) in text, "未写入等长占位"


def test_server_secrets_are_zeroed(tmp_path):
    _, text = _run(tmp_path)
    assert "c2VjcmV0LXRpY2tldA==" not in text, "server_ticket_b64 未清除"
    assert "d3JhcHBlZC1kZWs=" not in text, "wrapped_dek_b64 未清除"


def test_scoring_fields_survive_sanitization(tmp_path):
    """脱敏不得动成绩字段 —— 它们是表 1 与论文成绩的唯一来源。"""
    out, _ = _run(tmp_path)
    doc = json.loads(out.read_text(encoding="utf-8"))
    row = doc["paper_table1_row"]
    assert row["清除干扰源个数"] == 13
    assert abs(row["平均定位清除时间_s"] - 439.7718) < 1e-6
    assert doc["summary"]["virtual_time_s"] == 5717.03
    assert doc["summary"]["strategy_profile"] == "q4-fast-v3"
    assert doc["summary"]["search_layout"]["outer_radius_m"] == 1870.0
    assert doc["strategy_class"] == "Q4FastStrategy"
    assert len(doc["protocol_log"]) == 2, "协议记录条数被改变"


def test_redact_keys_match_supporting_readme():
    """脚本的 REDACT_KEYS 必须覆盖 supporting/README.md 声称清除的每个字段。

    这两处曾经不一致：文档写了 wrapped_dek_b64，脚本漏实现，
    最敏感的包裹密钥反而是唯一没被清掉的。
    """
    sys.path.insert(0, str(ROOT / "src"))
    import sanitize_official_run as mod

    doc = (ROOT / "supporting" / "README.md").read_text(encoding="utf-8")
    claimed = [
        "server_ticket_b64", "authorization_nonce_b64",
        "package_signing_public_key_b64", "device_digest",
        "signature_b64", "payload_signature_b64", "wrapped_dek_b64",
    ]
    for key in claimed:
        assert key in doc, f"{key} 不在 supporting/README.md 的声称清单里，请同步文档"
        assert key in mod.REDACT_KEYS, (
            f"supporting/README.md 声称清除 {key}，但 REDACT_KEYS 没有它 —— "
            "文档承诺了脚本没做的事"
        )


def test_verification_list_is_independent_of_redact_list():
    """复核清单必须独立于替换清单，否则摘掉字段等于同时关掉对它的检查。

    实测过：复核若遍历 REDACT_KEYS，把 wrapped_dek_b64 摘掉后脚本仍返回 0，
    密钥原文照样写进"脱敏件"。改用独立的 MUST_BE_ZEROED 后返回 1 并指名字段。
    """
    sys.path.insert(0, str(ROOT / "src"))
    import sanitize_official_run as mod

    assert hasattr(mod, "MUST_BE_ZEROED"), "缺少独立的复核清单"
    assert mod.MUST_BE_ZEROED is not mod.REDACT_KEYS, "复核清单不能就是替换清单"
    for key in mod.MUST_BE_ZEROED:
        assert key in mod.REDACT_KEYS, (
            f"{key} 要复核却不在替换清单里，会导致永远报残留"
        )


def test_missing_redact_key_is_caught(tmp_path):
    """摘掉一个替换字段后，复核必须报错退出 —— 锁住上一条的实际行为。"""
    text = (ROOT / "src" / "sanitize_official_run.py").read_text(encoding="utf-8")
    broken = text.replace('    "wrapped_dek_b64",\n', "", 1)
    assert broken != text, "样本脚本里找不到 wrapped_dek_b64 行"
    script = tmp_path / "broken.py"
    script.write_text(broken, encoding="utf-8")

    log = tmp_path / "log.json"
    log.write_text(json.dumps(_sample(), ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), str(log), "--team-no", TEAM],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, "漏掉密钥字段却通过了复核"
    assert "wrapped_dek_b64" in proc.stdout, "未指名出问题的字段"
