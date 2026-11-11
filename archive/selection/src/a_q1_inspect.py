"""Inspect A题 attachments for problem 1."""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "problem" / "A题" / "附件"

for name in ["附件1.xlsx", "附件2.xlsx"]:
    p = BASE / name
    xl = pd.ExcelFile(p)
    print("=" * 70)
    print(name, "sheets:", xl.sheet_names)
    for sh in xl.sheet_names:
        df = pd.read_excel(p, sheet_name=sh, header=None)
        print(f"-- sheet {sh!r} shape={df.shape}")
        print(df.head(8).to_string())
        if df.shape[0] > 8:
            print("...tail...")
            print(df.tail(4).to_string())
