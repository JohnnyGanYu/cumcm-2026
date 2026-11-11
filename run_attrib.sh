#!/bin/bash
# 归因诊断批量跑：Q3/Q4 四个定向档并行，日志落 /tmp。
# 用法：先在仓库根目录激活环境（pip install -r requirements.txt），再 bash run_attrib.sh
cd "$(dirname "$0")" || exit 1
P=${PYTHON:-python}
$P -m src.b_attrib_diag --selftest --directional 0 --seed-start 320000 > /tmp/st_q3.log 2>&1
$P -m src.b_attrib_diag --episodes 60 --seed-start 320000 --directional 0  --label q3     > /tmp/attrib_q3.log    2>&1 &
$P -m src.b_attrib_diag --episodes 60 --seed-start 330000 --directional 5  --label q4_d5  > /tmp/attrib_q4d5.log  2>&1 &
$P -m src.b_attrib_diag --episodes 60 --seed-start 340000 --directional 10 --label q4_d10 > /tmp/attrib_q4d10.log 2>&1 &
$P -m src.b_attrib_diag --episodes 60 --seed-start 350000 --directional 14 --label q4_d14 > /tmp/attrib_q4d14.log 2>&1 &
wait
echo ALL_DONE
