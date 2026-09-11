#!/bin/bash
cd /home/opsuser/.hermes/profiles/instance2/work/cumcm-2026
P=/home/opsuser/.venvs/cumcm-2026/bin/python
$P -m src.b_attrib_diag --selftest --directional 0 --seed-start 320000 > /tmp/st_q3.log 2>&1
$P -m src.b_attrib_diag --episodes 60 --seed-start 320000 --directional 0  --label q3     > /tmp/attrib_q3.log    2>&1 &
$P -m src.b_attrib_diag --episodes 60 --seed-start 330000 --directional 5  --label q4_d5  > /tmp/attrib_q4d5.log  2>&1 &
$P -m src.b_attrib_diag --episodes 60 --seed-start 340000 --directional 10 --label q4_d10 > /tmp/attrib_q4d10.log 2>&1 &
$P -m src.b_attrib_diag --episodes 60 --seed-start 350000 --directional 14 --label q4_d14 > /tmp/attrib_q4d14.log 2>&1 &
wait
echo ALL_DONE
