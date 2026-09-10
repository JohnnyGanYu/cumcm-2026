#!/usr/bin/env python3
"""补跑 gpt-5.6-terra 失败的 3 个 modeler 角色，换用其它家族模型。"""
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from multi_review import OUT, one_robust  # noqa: E402

# 换家族：C题已由 claude-opus-5 做过 modeler，这里给 A/B/C 各配一个非claude非gpt家族
FILL = [
    ("A", "modeler", "minimax-m2.5"),
    ("B", "modeler", "minimax-m2.5"),
    ("A", "modeler", "glm-5"),
]


def main():
    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    # 丢弃 gpt-5.6-terra 的失败记录
    kept = [r for r in existing
            if not (r["model"] == "gpt-5.6-terra" and not r["ok"])]
    print(f"保留 {len(kept)} 条有效记录，补跑 {len(FILL)} 个")

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(one_robust, p, r, m): (p, r, m) for p, r, m in FILL}
        for f in as_completed(futs):
            res = f.result()
            kept.append(res)
            tag = "OK " if res["ok"] else "ERR"
            if res.get("degraded"):
                tag = "DEG"
            n = len(res.get("content", "") or "")
            print(f"[{tag}] {res['problem']}/{res['role']}/{res['model']} "
                  f"{res['elapsed_s']}s {n}", flush=True)
            OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=1))

    ok = sum(1 for r in kept if r["ok"] and len(r.get("content", "") or "") >= 1500)
    print(f"\n最终有效 {ok}/{len(kept)} -> {OUT}")


if __name__ == "__main__":
    main()
