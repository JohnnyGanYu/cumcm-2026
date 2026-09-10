#!/usr/bin/env python3
"""只补 B题 modeler 一个槽。换家族(minimax-m2.5，速度快)，缩短输出要求避免超时。"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from multi_review import OUT, PROBLEMS, client  # noqa: E402

SYS = """你是资深数学建模竞赛(CUMCM)国奖级选手。针对这道题给出你独立的完整建模路线。

必须覆盖四问，每问包含：
1) 数学模型：决策变量、目标函数、约束、关键公式（写具体公式和数字，不要只说方法名）
2) 求解算法与关键实现细节
3) 该问最容易做错的具体位置
最后给出：本题拉分点在哪一问、什么做法能拿高分。

【重要执行约束】
你没有任何可用工具，无法联网搜索，也无法调用函数。禁止输出任何工具调用、函数调用、
搜索请求或 <function_calls> 之类标记。禁止说"我先搜索一下""让我查一下"。
题目所需信息已全部在用户消息中给出，直接基于这些信息和你的专业知识作答。
立即开始输出正式分析内容，不要有前置说明。控制在4000字以内，重点写问题3和问题4的策略设计。"""

CANDIDATES = ["minimax-m2.5", "glm-5", "deepseek-3.2"]


def main():
    data = json.loads(OUT.read_text())
    # 移除 B/modeler 的所有失败记录
    kept = [r for r in data
            if not (r["problem"] == "B" and r["role"] == "modeler"
                    and not (r["ok"] and len(r.get("content", "") or "") >= 1500))]
    print(f"保留 {len(kept)} 条")

    for model in CANDIDATES:
        print(f"\n尝试 {model} ...", flush=True)
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": PROBLEMS["B"]}],
                max_tokens=7000,
                timeout=420.0,
            )
            txt = r.choices[0].message.content or ""
            el = round(time.time() - t0, 1)
            print(f"  返回 {len(txt)} chars / {el}s")
            if len(txt) >= 1500 and "function_calls" not in txt:
                kept.append({"problem": "B", "role": "modeler", "model": model,
                             "ok": True, "elapsed_s": el,
                             "tokens": r.usage.total_tokens if r.usage else None,
                             "content": txt, "attempts": 1})
                OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=1))
                print(f"  ✓ 已写入 {OUT}")
                break
            print("  内容不合格，换下一个模型")
        except Exception as e:
            print(f"  失败 {type(e).__name__}: {str(e)[:120]}")

    final = json.loads(OUT.read_text())
    ok = [x for x in final if x["ok"] and len(x.get("content", "") or "") >= 1500]
    print(f"\n最终有效 {len(ok)} 份")
    for x in sorted(ok, key=lambda y: (y["problem"], y["role"])):
        print(f"  {x['problem']}/{x['role']:8}/{x['model']:16} {len(x['content']):6}")


if __name__ == "__main__":
    main()
