#!/usr/bin/env python3
"""纠正红队对B题问题1反例理由的错误说法。

红队称：钝角三角形，以最长边(直径)为直径的圆无法覆盖对角顶点。
事实：恰好相反。钝角三角形的最小外接圆 **就是** 以最长边为直径的圆。
      真正的反例来自 **锐角** 三角形。
"""
import numpy as np

print("=" * 78)
print("三角形：以最长边(=直径d)为直径的圆 是否覆盖第三个顶点？")
print("=" * 78)
print(f"{'类型':<12}{'角度(度)':<28}{'最长边d':>10}{'MEC半径R':>11}"
      f"{'R/(d/2)':>10}{'直径圆覆盖?':>12}")
print("-" * 78)


def analyze_triangle(A, B, C, label):
    A, B, C = map(np.asarray, (A, B, C))
    pts = [A, B, C]
    # 边长
    sides = [np.linalg.norm(B - C), np.linalg.norm(A - C), np.linalg.norm(A - B)]
    # 角度（余弦定理）：angs[i] 是顶点 i 处的角
    angs = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        v1, v2 = pts[j] - pts[i], pts[k] - pts[i]
        ca = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angs.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
    d = max(sides)                    # 直径 = 最长边
    imax = int(np.argmax(angs))       # 最大角所在顶点
    # 最长边的两个端点 = 除最大角顶点外的两点
    others = [i for i in range(3) if i != imax]
    P, Q = pts[others[0]], pts[others[1]]
    mid = (P + Q) / 2
    r_diam = d / 2
    dist_third = np.linalg.norm(pts[imax] - mid)
    covered = dist_third <= r_diam + 1e-12

    # 最小外接圆半径：钝角/直角 -> d/2；锐角 -> 外接圆半径 abc/(4K)
    a, b, c = sides
    K = 0.5 * abs(np.cross(B - A, C - A))
    R_circum = a * b * c / (4 * K)
    R_mec = r_diam if max(angs) >= 90 else R_circum

    astr = ", ".join(f"{x:.1f}" for x in angs)
    print(f"{label:<12}{astr:<28}{d:>10.4f}{R_mec:>11.4f}"
          f"{R_mec/r_diam:>10.5f}{('是' if covered else '否'):>12}")
    return covered, R_mec / r_diam, max(angs)


# 钝角三角形
analyze_triangle((0, 0), (10, 0), (3, 1.2), "钝角")
analyze_triangle((0, 0), (10, 0), (1, 0.5), "强钝角")
# 直角
analyze_triangle((0, 0), (10, 0), (0, 6), "直角")
# 锐角
analyze_triangle((0, 0), (10, 0), (5, 7), "锐角")
analyze_triangle((0, 0), (10, 0), (5, 8.66), "正三角形")
analyze_triangle((0, 0), (10, 0), (5, 5.1), "近直角锐角")

print("-" * 78)
print("""
结论（纠正红队说法）：
  * 钝角三角形：最大角 >= 90°，最小外接圆恰为"以最长边为直径的圆"，
    钝角顶点落在该圆【内部】-> 直径圆【能】覆盖。R/(d/2) = 1.0
  * 直角三角形：斜边中点即外心，第三顶点恰在圆【上】-> 临界情形，仍覆盖。
  * 锐角三角形：三个角都 < 90°，最小外接圆由三点共圆决定，
    R = abc/(4K) > d/2 -> 直径圆【不能】覆盖。★这才是反例来源★
  * 极值：正三角形 R/(d/2) = 2/sqrt(3) = 1.15470，即 Jung 定理上界。

所以 B 题问题 1 第二问的正确论证是：
  定位区域(凸多边形)的顶点集中若存在锐角三角形，则最小外接圆半径 > d/2，
  以直径为直径的圆无法覆盖定位区域。答案是【不能】。
  红队所述"钝角三角形是反例"方向恰好说反了。
""")

# 与实际交会定位反例对照
print("=" * 78)
print("与实际交会定位反例对照（四检测点，真实源(600,500)）")
print("=" * 78)
verts = [
    (606.083478438, 512.945994088),
    (585.905086197, 496.653126302),
    (613.640861228, 503.551854993),
    (594.372036131, 486.849581926),
]
import itertools
maxang_list = []
for tri in itertools.combinations(range(4), 3):
    pts = [np.array(verts[t]) for t in tri]
    angs = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        v1, v2 = pts[j] - pts[i], pts[k] - pts[i]
        ca = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angs.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
    maxang_list.append((tri, max(angs)))
    kind = "锐角(MEC由三点定)" if max(angs) < 90 else "非锐角"
    print(f"  顶点{tri} 最大角 {max(angs):.4f}°  {kind}")

n_acute = sum(1 for _, m in maxang_list if m < 90)
print(f"\n  4个顶点三角形中有 {n_acute} 个是锐角三角形")
print("  -> 存在锐角三角形，故 MEC 半径 > d/2，直径圆不覆盖。与数值结果一致。")
