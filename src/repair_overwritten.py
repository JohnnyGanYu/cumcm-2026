#!/usr/bin/env python3
"""修复被旧进程空回复覆写的 4 个槽：A/redteam, B/redteam, A/critic, C/critic。

原因：第一次运行(无重试机制)的进程在收尾时用空回复覆写了 JSON。
"""
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from multi_review import OUT, one_robust  # noqa: E402

MISSING = [
    ("A", "redteam", "glm-5"),
    ("B", "redteam", "glm-5"),
    ("A", "critic", "deepseek-3.2"),
    ("C", "critic", "deepseek-3.2"),
]


def main():
    data = json.loads(OUT.read_text())
    # 丢掉所有无效条目
    kept = [r for r in data if r["ok"] and len(r.get("content", "") or "") >= 1500]
    print(f"保留有效 {len(kept)} 份，重跑 {len(MISSING)} 个被覆写的槽\n")

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(one_robust, p, r, m): (p, r, m) for p, r, m in MISSING}
        for f in as_completed(futs):
            res = f.result()
            n = len(res.get("content", "") or "")
            ok = res["ok"] and n >= 1500
            print(f"[{'OK ' if ok else 'ERR'}] {res['problem']}/{res['role']}/{res['model']} "
                  f"{res['elapsed_s']}s {n if res['ok'] else res.get('error','')}", flush=True)
            if ok:
                kept.append(res)

    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=1))
    print(f"\n=== 最终 {len(kept)} 份 ===")
    for x in sorted(kept, key=lambda y: (y["problem"], y["role"])):
        print(f"  {x['problem']}/{x['role']:8}/{x['model']:16} {len(x['content']):6}")


if __name__ == "__main__":
    main()
