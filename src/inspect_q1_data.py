"""Inspect CUMCM2026 C题 附件1 (problem 1 input data)."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "problem" / "C题" / "附件" / "附件1.xlsx"

xls = pd.ExcelFile(XL)
print("sheets:", xls.sheet_names)
for sh in xls.sheet_names:
    df = pd.read_excel(XL, sheet_name=sh)
    print("=" * 70)
    print("sheet:", sh, "shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(5).to_string())
    print("...")
    print(df.tail(3).to_string())
    print("dtypes:\n", df.dtypes)
    num = df.select_dtypes(include=[np.number])
    if len(num.columns):
        print("describe:\n", num.describe().to_string())
    print("nulls:\n", df.isna().sum().to_string())
