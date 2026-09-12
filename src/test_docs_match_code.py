"""文档声称的生产参数与代码实际值的一致性回归。

这类脱节已经真实发生三次：RACEDAY 第 7 节抄了 DogStrategy 基类默认值、
论文 5.3/5.6 节停在 1750 m 与 950x6/1900x14、配图停在 1250x12。
参数改了而文档没跟上时，评审看到的是自相矛盾的材料。

这里只钉「当前交付配置」这一类断言：文档里凡声明交付值的地方，
必须与策略类的类属性一致。历史对照值不在断言范围内。
"""
from __future__ import annotations

from pathlib import Path

from src.b_sim import q3_fast, q4_fast
from src.b_sim.q3_fast import Q3FastStrategy
from src.b_sim.q4_fast import Q4FastStrategy

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_raceday_states_current_delivery_params():
    """RACEDAY 第 7 节必须写交付类的值，不是 DogStrategy 基类默认值。"""
    text = _read("RACEDAY.md")
    assert f"{Q3FastStrategy.ring_radius:.1f}" in text or "1150" in text
    assert f"{Q4FastStrategy.outer_radius:.1f}" in text
    assert f"{Q4FastStrategy.inner_radius:.1f}" in text
    assert q3_fast.PROFILE in text
    assert q4_fast.PROFILE in text


def test_raceday_does_not_present_legacy_ring_as_current():
    """1100 m / 8 站是 --legacy-q3 才生效的基类默认值。"""
    text = _read("RACEDAY.md")
    if "probe_ring_radius" in text:
        idx = text.index("probe_ring_radius")
        window = text[max(0, idx - 400):idx + 400]
        assert "legacy" in window, (
            "RACEDAY 提到 probe_ring_radius 时必须说明它只在 --legacy-q3 回退时生效"
        )


def test_paper_declares_delivered_outer_radius():
    """论文正文出现的交付外环值必须是 1870，且给出几何下界依据。"""
    text = _read("paper/main.md")
    assert f"{Q4FastStrategy.outer_radius:.0f}" in text
    assert "1863.5" in text, "外环下界 R_A/cos(pi/12) 的数值应在论文中给出"


def test_readme_release_md5_matches_zip():
    """README/RACEDAY 记的 MD5 必须是当前 zip 的实际值。"""
    import hashlib

    zip_path = ROOT / "CUMCM2026B_robot_windows.zip"
    if not zip_path.exists():
        return
    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    for name in ("README.md", "RACEDAY.md"):
        text = _read(name)
        assert digest in text, (
            f"{name} 未记录当前交付包 MD5 {digest}；"
            "改完 dist_windows 要重跑 build_windows_zip 并更新文档与 Release"
        )
