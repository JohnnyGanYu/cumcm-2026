"""生成问题分析章节的 5 张流程图（图 2-1 ~ 2-5），黑白线框风格，中文标注。

字体：SimHei / Noto Sans CJK SC / 系统默认，按可用性回退。
- b_flow_q1.png  图 2-1 问题一分析流程
- b_flow_q2.png  图 2-2 问题二分析流程
- b_flow_q3.png  图 2-3 问题三分析流程
- b_flow_q4.png  图 2-4 问题四分析流程
- b_flow_roadmap.png 图 2-5 全文技术路线总图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# ---- 中文字体回退链：SimHei -> Noto Sans CJK SC -> 任意含 CJK 的字体 ----
CANDIDATES = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
chosen = None
for c in CANDIDATES:
    try:
        if any(c == f.name for f in font_manager.fontManager.ttflist):
            chosen = c
            break
    except Exception:
        continue
if chosen is None:
    # 兜底：扫描含 CJK 的字体
    for f in font_manager.fontManager.ttflist:
        if 'CJK' in f.name or 'Hei' in f.name or '黑' in f.name:
            chosen = f.name
            break
if chosen is None:
    chosen = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print(f'流程图中文字体: {chosen}')

OUT = '/home/opsuser/.hermes/profiles/instance2/work/cumcm-2026/figures'


def box(ax, x, y, w, h, text, fs=8.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", fc="white",
                                ec="black", lw=1.1))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, linespacing=1.28)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=13, lw=1.1, color='black'))


def setup(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')


# ---------- 图 2-1 问题一 ----------
fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
setup(ax)
box(ax, 5.0, 9.1, 5.8, 1.6, '观测数据（检测点，示向度）\n误差 ±1°', fs=8)
box(ax, 2.5, 6.9, 4.8, 1.5, '可行域\n= 半平面交')
box(ax, 7.5, 6.9, 4.8, 1.5, '凸包\n（极点集）')
box(ax, 2.5, 3.4, 5.0, 1.5, '直径 = 极点对最大距离\n（旋转卡壳，O(n log n)）')
box(ax, 7.5, 3.4, 5.0, 1.5, '最小外接圆（Welzl）\nr_min ≤ d/√3（Jung 定理）')
box(ax, 5.0, 0.9, 8.8, 1.3, '覆盖判定：以直径为直径的圆能否覆盖？\n一般不能（正三角形反例）')
arrow(ax, 5, 8.3, 2.5, 7.65)
arrow(ax, 5, 8.3, 7.5, 7.65)
arrow(ax, 2.5, 6.15, 2.5, 4.15)
arrow(ax, 7.5, 6.15, 7.5, 4.15)
arrow(ax, 2.5, 2.65, 4.3, 1.55)
arrow(ax, 7.5, 2.65, 5.8, 1.55)
arrow(ax, 4.3, 1.55, 7.5, 2.35)  # 反例引回覆盖判定
fig.tight_layout()
fig.savefig(f'{OUT}/b_flow_q1.png', bbox_inches='tight')
plt.close(fig)

# ---------- 图 2-2 问题二 ----------
fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
setup(ax)
box(ax, 5.0, 9.0, 6.2, 1.2, '第一观测点 S1\n示向度 θ1 ± 1°')
box(ax, 3.0, 6.4, 5.0, 1.7, '误差传播\n面积随距离增大，\n交会角 → 0° 或 180° 时发散')
box(ax, 8.0, 6.4, 5.0, 1.7, 'S2 可行域\n四类硬约束\n（区域内/接收稳健/交会角/基线）')
box(ax, 5.5, 3.2, 6.8, 1.6, '选点准则：最坏面积最小化\n兼顾基线—移动耗时权衡')
box(ax, 5.5, 0.9, 7.6, 1.6, '候选区域（≈38800 m²）推荐点 (480,880)\n蒙特卡洛：无界比例 0、真值包含率 1.000')
arrow(ax, 5, 8.4, 3, 7.3)
arrow(ax, 5, 8.4, 8, 7.3)
arrow(ax, 3, 5.5, 4.8, 4.0)
arrow(ax, 8, 5.5, 6.4, 4.0)
arrow(ax, 5.5, 2.4, 5.5, 1.55)
fig.tight_layout()
fig.savefig(f'{OUT}/b_flow_q2.png', bbox_inches='tight')
plt.close(fig)

# ---------- 图 2-3 问题三 ----------
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
setup(ax)
box(ax, 5.0, 9.2, 6.6, 1.2, '问题三：全向源，数量未知\nN ∈ [10,16]，评分 M = T/Nc')
box(ax, 2.6, 6.6, 4.6, 1.6, '原点普扫\n发现 1000 m 内源\n（约 49% 的源）')
box(ax, 7.4, 6.6, 4.8, 1.6, '覆盖设计\n环 R=1150 m × 6 站位\n最坏间隙 < 1000 m')
box(ax, 3.0, 3.6, 5.4, 1.7, '定位与清除\ndog-leg + 归航、就近优先\n射线盲扫兜底')
box(ax, 7.8, 3.6, 4.8, 1.7, '自适应覆盖兜底\n环扫穷尽后按沉默\n重排站位')
box(ax, 5.0, 0.9, 7.0, 1.3, '终止：已发现源全清\n空频道按沉默判定')
arrow(ax, 5, 8.6, 2.6, 7.4)
arrow(ax, 5, 8.6, 7.4, 7.4)
arrow(ax, 2.6, 5.8, 2.8, 4.4)
arrow(ax, 7.4, 5.8, 7.6, 4.4)
arrow(ax, 3.0, 2.75, 4.5, 1.55)
arrow(ax, 7.8, 2.75, 5.6, 1.55)
fig.tight_layout()
fig.savefig(f'{OUT}/b_flow_q3.png', bbox_inches='tight')
plt.close(fig)

# ---------- 图 2-4 问题四 ----------
fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
setup(ax)
box(ax, 5.0, 9.2, 6.8, 1.6, '问题四：问题三 + 定向源\n仅 90° 半扇区可见，方向未知')
box(ax, 2.6, 6.7, 4.8, 1.7, '双环布局\n内环 1000 m×7\n外环 1870 m×12（旋转 15°）')
box(ax, 7.6, 6.7, 4.8, 1.7, '外环半径几何下界\nRo ≥ RA/cos(π/12)\n≈ 1863.5 m，留 6.5 m 余量')
box(ax, 3.0, 3.5, 5.2, 1.7, '站位与目标联合巡回\n开放路径 TSP\n最近邻 + 2-opt')
box(ax, 7.8, 3.5, 4.8, 1.7, '外环方位采样\n对朝外辐射源\n多角度观测')
box(ax, 5.0, 0.9, 6.8, 1.3, '沉默 ≠ 空（定向源）\n不提前判空；清除预算护栏')
arrow(ax, 5, 8.4, 2.6, 7.5)
arrow(ax, 5, 8.4, 7.6, 7.5)
arrow(ax, 2.6, 5.85, 3.0, 4.35)
arrow(ax, 7.6, 5.85, 7.6, 4.35)
arrow(ax, 3.0, 2.65, 4.5, 1.55)
arrow(ax, 7.8, 2.65, 5.5, 1.55)
fig.tight_layout()
fig.savefig(f'{OUT}/b_flow_q4.png', bbox_inches='tight')
plt.close(fig)

# ---------- 图 2-5 全文技术路线 ----------
fig, ax = plt.subplots(figsize=(4.4, 3.1), dpi=200)
setup(ax)
box(ax, 5.0, 9.3, 6.6, 1.2, '问题分解\n（几何 → 选点 → 覆盖 → 定向）')
box(ax, 2.0, 6.5, 3.8, 1.7, '问题一：几何\n凸包 + Jung 上界')
box(ax, 5.0, 6.5, 3.8, 1.7, '问题二：选点\n误差传播 +\n四约束候选区域')
box(ax, 8.0, 6.5, 3.8, 1.7, '问题三、四：策略\n覆盖—交会—逼近\n三阶段')
box(ax, 3.0, 3.7, 4.8, 1.7, '验证\n配对 A/B 检验 +\n独立留出集种子块')
box(ax, 7.0, 3.7, 4.8, 1.7, '压力与稳健性\n六档边界矩阵 +\n外环压力样本')
box(ax, 5.0, 0.9, 6.8, 1.3, '官方模拟器实测\n（演练 + 正式局）与交付冻结')
arrow(ax, 5, 8.7, 2.0, 7.35)
arrow(ax, 5, 8.7, 5.0, 7.35)
arrow(ax, 5, 8.7, 8.0, 7.35)
arrow(ax, 2.5, 5.65, 3.0, 4.55)
arrow(ax, 7.5, 5.65, 7.0, 4.55)
arrow(ax, 3.0, 2.85, 4.5, 1.55)
arrow(ax, 7.0, 2.85, 5.6, 1.55)
fig.tight_layout()
fig.savefig(f'{OUT}/b_flow_roadmap.png', bbox_inches='tight')
plt.close(fig)

print('5 张流程图已生成（中文标注）')
