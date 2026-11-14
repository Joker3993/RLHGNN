"""
CD (Critical Difference) 图绘制 — Friedman + Nemenyi 统计显著性分析
=================================================================
基于 Demsar (2006) 方法：对多个方法在多个数据集上的表现做排名，
用 Friedman 检验判断是否存在显著差异，再用 Nemenyi 事后检验
通过 CD 图可视化方法间的统计显著性聚类。

【v2 优化】参考学术论文截图风格，重构排版层级与防重叠算法：
  - 垂直层级：CD 标尺 (顶部) → 排名轴 → 显著性分组粗线 → 引线 → 方法名 (底部)
  - 标签错开算法：贪心分配 y 层级 + 估算文本宽度，杜绝标签互相覆盖
  - 边缘对齐：根据 x 位置自动选择 ha='left'/'center'/'right'，防止越界裁切
  - 平局处理：相同 rank 的方法自动错开到不同 y 层级
  - 学术风格：Liberation Serif (Times 兼容) 字体，纯黑白配色

使用方法：
    1. 修改下方的数据区域（datasets, methods_all, accuracy_data, f1_data）
    2. 运行脚本：python cd_diagram.py
    3. 生成的图片保存在当前目录
"""

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.font_manager as fm

matplotlib.use('Agg')  # 无头模式，服务器/远程环境不弹窗
import matplotlib.pyplot as plt
from itertools import combinations

# ====================== 全局样式设置 ======================
# 注册 Liberation Serif (Times New Roman 兼容) 用于学术论文风格
for _fp in [
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]:
    try:
        fm.fontManager.addfont(_fp)
    except Exception:
        pass

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学符号与正文配套
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# ====================== 【修改这里】你的数据 ======================
datasets = ['BPI12C', 'BPI12CW', 'BPI13I', 'BPI13P', 'BPI13CP', 'BPI2020P', 'P2P', 'OTC']
methods_all = ['MiDA', 'MiTFM', 'SuTraN', 'JARVIS', 'HiGPP', 'SGAP', 'MHG-Predictor', 'RLHGNN(ours)']

accuracy_data = np.array([
    [0.795, 0.826, 0.700, 0.510, 0.522, 0.823, 0.789, 0.765],
    [0.794, 0.827, 0.736, 0.507, 0.520, 0.831, 0.794, 0.767],
    [0.784, 0.743, 0.511, 0.498, 0.508, 0.773, 0.684, 0.784],
    [0.769, 0.822, 0.679, 0.471, 0.488, 0.625, 0.740, 0.740],
    [0.793, 0.835, 0.738, 0.508, 0.528, 0.831, 0.779, 0.751],
    [0.728, 0.771, 0.580, 0.369, 0.416, 0.843, 0.803, 0.758],
    [0.786, 0.833, 0.710, 0.533, 0.529, 0.830, 0.796, 0.752],
    [0.781, 0.833, 0.745, 0.543, 0.531, 0.839, 0.793, 0.774],
])

f1_data = np.array([
    [0.655, 0.691, 0.405, 0.423, 0.424, 0.417, 0.771, 0.590],
    [0.643, 0.711, 0.416, 0.454, 0.475, 0.455, 0.787, 0.582],
    [0.620, 0.461, 0.189, 0.372, 0.334, 0.406, 0.524, 0.546],
    [0.627, 0.690, 0.384, 0.387, 0.416, 0.356, 0.580, 0.575],
    [0.654, 0.713, 0.412, 0.452, 0.489, 0.479, 0.787, 0.567],
    [0.529, 0.616, 0.186, 0.233, 0.227, 0.476, 0.795, 0.714],
    [0.641, 0.698, 0.363, 0.488, 0.490, 0.438, 0.792, 0.720],
    [0.626, 0.720, 0.385, 0.483, 0.493, 0.465, 0.792, 0.569],
])


# ====================== 统计检验 ======================

def friedman_test(data):
    """
    Friedman 检验
    data: shape (n_methods, n_datasets)
    返回: (chi2, p_value, mean_ranks)
    """
    n_methods, n_datasets = data.shape
    ranks = np.zeros_like(data)
    for d in range(n_datasets):
        ranks[:, d] = n_methods + 1 - stats.rankdata(data[:, d], method='average')

    mean_ranks = np.mean(ranks, axis=1)
    # Friedman 统计量（Demsar 2006 式 1，使用平均排名 R_j）：
    #   chi2_F = 12N/(k(k+1)) * [ sum_j(R_j^2) - k(k+1)^2/4 ]
    # 修正项为 k(k+1)^2/4（无 N 分母），与经典式
    #   chi2_F = 12/(Nk(k+1)) * sum_j(R_j^2) - 3N(k+1)  （R_j 为排名总和）
    # 完全等价；除以 N 会系统性高估统计量（零假设下给出 3(k+1)(N-1) 而非 0）。
    chi2 = (12 * n_datasets / (n_methods * (n_methods + 1))) * (
            np.sum(mean_ranks ** 2) - (n_methods * (n_methods + 1) ** 2) / 4
    )
    p_value = 1 - stats.chi2.cdf(chi2, n_methods - 1)
    return chi2, p_value, mean_ranks


def nemenyi_cd(n_methods, n_datasets, alpha=0.05):
    """
    计算 Nemenyi 检验的 Critical Difference (CD)
    """
    q_critical = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
        0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
               7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
    }
    q_alpha = q_critical[alpha].get(n_methods, 2.949)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    return cd, q_alpha


def pairwise_significance(mean_ranks, n_methods, n_datasets, alpha=0.05):
    """
    成对 Nemenyi 显著性检验（无传递性）。
    对每一对方法 (i, j)，计算 z = |R_i - R_j| / sqrt(k(k+1)/(6N))，
    用 Bonferroni 校正：p_adj = min(1, p * C(k,2))。
    返回 list of (i, j, significant: bool)。
    与分组连线不同，这里只标注两两之间的直接显著性，避免传递性扩散。
    """
    from scipy.stats import norm
    cd, _ = nemenyi_cd(n_methods, n_datasets, alpha=alpha)
    # 标准差
    se = np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))
    n_pairs = n_methods * (n_methods - 1) // 2

    results = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            z = diff / se
            p = 2 * (1 - norm.cdf(z))
            p_adj = min(1.0, p * n_pairs)  # Bonferroni
            # 也用 CD 直接判断（等价于未校正的 Nemenyi 临界值）
            sig_by_cd = diff > cd - 1e-10
            results.append((i, j, p_adj < alpha, sig_by_cd, diff, cd))
    return results


# ====================== CD 图绘制（学术风格版） ======================

def _measure_label_extents(ax, names, fontsize=10.5, fontweight='normal'):
    """
    用 matplotlib 渲染器实测每个标签在数据坐标下的宽度与高度（rank 单位）。
    必须在 xlim 与 ylim 都确定后调用，否则数据坐标转换会失真。
    """
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()

    widths = []
    heights = []
    for name in names:
        t = ax.text(0, 0, name, fontsize=fontsize, fontweight=fontweight,
                    transform=ax.transData)
        ax.figure.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer)
        # 宽度
        p0 = inv.transform((bbox.x0, 0))
        p1 = inv.transform((bbox.x1, 0))
        widths.append(abs(p1[0] - p0[0]))
        # 高度
        p2 = inv.transform((0, bbox.y0))
        p3 = inv.transform((0, bbox.y1))
        heights.append(abs(p3[1] - p2[1]))
        t.remove()
    return widths, heights


def _assign_stagger_levels(ax, x_positions, names, fontsize=10.5):
    """
    贪心分配 y 层级避免水平重叠。
    使用 matplotlib 渲染器实测文本宽度，杜绝估算误差。
    按 x 排序后顺序处理，每个标签找最低（最靠近 0）的不冲突层级。
    返回 (levels, widths) 两个列表。
    """
    n = len(x_positions)
    widths, _ = _measure_label_extents(ax, names, fontsize=fontsize)

    # 按 x 排序，记录原始索引
    order = sorted(range(n), key=lambda i: x_positions[i])

    levels = [0] * n
    # level -> list of (x_left, x_right) 占用区间
    level_occ = {}
    min_gap = 0.05  # 标签间最小水平间隙（rank 单位）

    for idx in order:
        x = x_positions[idx]
        w = widths[idx]
        x_left = x - w / 2
        x_right = x + w / 2

        lvl = 0
        while True:
            occ = level_occ.get(lvl, [])
            conflict = False
            for xl, xr in occ:
                if not (x_right + min_gap <= xl or x_left - min_gap >= xr):
                    conflict = True
                    break
            if not conflict:
                break
            lvl += 1

        levels[idx] = lvl
        level_occ.setdefault(lvl, []).append((x_left, x_right))

    return levels, widths


def _assign_columns(x_positions, n_cols=2):
    """
    把方法按 x 位置分到左右两列，参考图风格。
    按 x 升序排好后，依次分配到左列、右列、左列、右列...（zig-zag）
    返回 columns: list of lists，每个内层 list 是 (sorted_idx, x) 的列表，
    以及 col_of: dict sorted_idx -> column_index (0=left, 1=right)。
    """
    n = len(x_positions)
    order = sorted(range(n), key=lambda i: x_positions[i])
    columns = [[], []]
    col_of = {}
    for k, idx in enumerate(order):
        col = k % n_cols
        columns[col].append(idx)
        col_of[idx] = col
    return columns, col_of


def draw_cd_diagram(mean_ranks, methods_list, cd, save_path,
                    figsize=None, fig_caption=None,
                    show_pairwise=False):
    """
    学术风格 CD 图（L 型折线引线版，参考论文截图）：
    垂直层级（顶 → 底）：
        CD 标尺  →  排名轴 (rank 1 在右, rank N 在左)  →
        Nemenyi 显著性分组粗黑线（传递性连通分量）  →
        [可选] 成对显著性连线（细线，无传递性）  →
        L 型折线引线（先垂直下落，再水平延伸至标签）→
        方法名（左右两列分布）

    ───────────────────────────────────────────────────────────────
    【可调参数 - 自己微调时关注这里】
    ───────────────────────────────────────────────────────────────
    show_pairwise : 是否画"成对显著性细线"
                    - False (默认): 只画标准 CD 图（粗分组线），符合 Demsar 2006 规范
                    - True: 额外画细线连接 |rank diff| > CD 的方法对，
                      用于在传递性把所有方法连成一大组时仍能区分真正显著的差异对

    代码内可调样式常量（在本函数体内搜索即可定位）：
      ★ THICK_BAR_WIDTH  : 粗分组线宽度（默认 4.5）
      ★ THIN_BAR_WIDTH   : 成对细线宽度（默认 1.4）
      ★ THIN_BAR_ALPHA   : 成对细线透明度（默认 0.85）
    ───────────────────────────────────────────────────────────────
    """
    n = len(methods_list)
    order = np.argsort(mean_ranks)
    sr = mean_ranks[order]
    sm = [methods_list[i] for i in order]

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  ★★★ 线条样式常量 — 自己微调时改这里 ★★★                          ║
    # ╚══════════════════════════════════════════════════════════════╝
    THICK_BAR_WIDTH = 3  # ★ 粗分组线宽度（参考图约 3-5；越大越粗）
    THIN_BAR_WIDTH  = 2  # ★ 成对显著性细线宽度（只在 show_pairwise=True 时生效）
    THIN_BAR_ALPHA  = 0.85  # ★ 成对显著性细线透明度（0-1）
    # ╔══════════════════════════════════════════════════════════════╗
    # ║  其他可调样式                                                  ║
    # ╚══════════════════════════════════════════════════════════════╝
    CONNECTOR_LINE_WIDTH = 0.9   # L 型折线引线宽度
    AXIS_LINE_WIDTH      = 1.0   # 排名轴主线宽度
    CD_BAR_LINE_WIDTH    = 1.4   # CD 标尺线宽度

    # ---- 计算显著性分组（BFS 连通分量）----
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(sr[i] - sr[j]) < cd - 1e-10:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    groups = []
    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            group = []
            while queue:
                cur = queue.pop(0)
                group.append(cur)
                for nb in adj[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(group) > 1:
                groups.append(sorted(group))

    # ---- 计算成对显著性差异（diff > CD），用于画细线 ----
    pairwise_sig_diff = []  # list of (i, j) where |sr[i]-sr[j]| > CD
    for i in range(n):
        for j in range(i + 1, n):
            if abs(sr[i] - sr[j]) > cd + 1e-10:
                pairwise_sig_diff.append((i, j))

    # ---- 计算每个方法的 x 位置 ----
    # rank 1 (best) -> x = n (右端)，rank N (worst) -> x = 1 (左端)
    x_positions = (n + 1) - sr.copy()

    # ---- 左右分列：按 x 顺序 zig-zag ----
    columns, col_of = _assign_columns(x_positions, n_cols=2)
    n_left = len(columns[0])
    n_right = len(columns[1])
    max_rows = max(n_left, n_right)

    # ---- 创建画布 ----
    fig_w = 10.0 if figsize is None else figsize[0]
    # 画布高度按行数决定，确保标签有充足垂直空间
    fig_h = 3.0 + max_rows * 0.38 if figsize is None else figsize[1]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    # x 范围
    x_min, x_max = 1.0, float(n)
    # 两侧留出充足空间放标签
    x_margin = 2.5
    ax.set_xlim(x_min - x_margin, x_max + x_margin)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # 临时 ylim，先 draw 一次稳定 axes 像素宽度（用于测文本宽度）
    ax.set_ylim(-5.0, 2.5)
    fig.canvas.draw()

    # 实测标签宽度
    label_widths, _ = _measure_label_extents(ax, sm, fontsize=10.5)
    max_label_w = max(label_widths) if label_widths else 1.5

    # ---- 垂直布局 ----
    cd_y = 1.6
    axis_y = 0.6
    group_y = 0.30          # Nemenyi 粗分组线
    pairwise_y_base = 0.10  # 成对显著性细线起始 y
    pairwise_y_step = -0.13  # 多条细线之间的间距

    # 标签列的 y 坐标（每行一个 y）
    label_y_top = -0.2
    label_y_step = -0.42
    # 每列各 row 的 y
    label_y_for = [None] * n
    for col_idx, col_methods in enumerate(columns):
        for row_idx, m_idx in enumerate(col_methods):
            label_y_for[m_idx] = label_y_top + row_idx * label_y_step

    bottom_y = label_y_top + (max_rows - 1) * label_y_step - 0.4
    ax.set_ylim(bottom_y, cd_y + 0.6)
    fig.canvas.draw()

    # ylim 确定后测高度（用于引线避让文字）
    _, label_heights = _measure_label_extents(ax, sm, fontsize=10.5)

    # 标签 x 坐标：左列在最左，右列在最右
    # 左列右边缘 x = x_min - x_margin + 0.3 + max_label_w
    # 右列左边缘 x = x_max + x_margin - 0.3 - max_label_w
    left_col_x_right = x_min - x_margin + 0.4 + max_label_w
    right_col_x_left = x_max + x_margin - 0.4 - max_label_w

    # 标签 x 中心坐标和对齐方式
    label_x_for = [None] * n
    label_ha_for = [None] * n
    for m_idx in range(n):
        if col_of[m_idx] == 0:  # 左列，右对齐到 left_col_x_right
            label_x_for[m_idx] = left_col_x_right
            label_ha_for[m_idx] = 'right'
        else:  # 右列，左对齐到 right_col_x_left
            label_x_for[m_idx] = right_col_x_left
            label_ha_for[m_idx] = 'left'

    # ---- 1. CD 标尺（最顶部）----
    cd_right = x_max + 0.5
    cd_left = cd_right - cd
    if cd_left < x_min - 1.5:
        cd_left = x_min - 0.5
        cd_right = cd_left + cd

    ax.hlines(cd_y, cd_left, cd_right, colors='black', linewidth=CD_BAR_LINE_WIDTH, zorder=3)
    ax.vlines(cd_left, cd_y - 0.08, cd_y + 0.08, colors='black', linewidth=CD_BAR_LINE_WIDTH, zorder=3)
    ax.vlines(cd_right, cd_y - 0.08, cd_y + 0.08, colors='black', linewidth=CD_BAR_LINE_WIDTH, zorder=3)
    ax.text((cd_left + cd_right) / 2, cd_y + 0.12, 'CD',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    # ---- 2. 排名轴 ----
    ax.hlines(axis_y, x_min - 0.3, x_max + 0.3, colors='black', linewidth=AXIS_LINE_WIDTH, zorder=2)
    for rk in range(1, n + 1):
        rank_label = n + 1 - rk  # x=rk 对应 rank=n+1-rk
        ax.vlines(rk, axis_y - 0.04, axis_y + 0.10,
                  colors='black', linewidth=0.9, zorder=2)
        ax.text(rk, axis_y + 0.13, str(rank_label),
                ha='center', va='bottom', fontsize=10, color='black')

    # ---- 3. Nemenyi 显著性分组粗黑线 ----
    # ★★★ 粗分组线宽度由 THICK_BAR_WIDTH 控制 ★★★
    group_line_y = {}
    occupied_x = []
    for gidx, group in enumerate(groups):
        gx_min = min(x_positions[idx] for idx in group)
        gx_max = max(x_positions[idx] for idx in group)
        y_try = group_y
        while any(not (gx_max + 0.05 <= xl or gx_min - 0.05 >= xr)
                 for xl, xr, yy in occupied_x if abs(yy - y_try) < 0.05):
            y_try -= 0.15
        group_line_y[gidx] = y_try
        occupied_x.append((gx_min, gx_max, y_try))
        ax.plot([gx_min, gx_max], [y_try, y_try],
                color='black', linewidth=THICK_BAR_WIDTH, zorder=4, solid_capstyle='butt')

    # ---- 3.5 成对显著性差异细线（仅当 show_pairwise 且存在显著差异对时）----
    # 当 Nemenyi 传递性把所有方法连成一大组时，这些细线能指出
    # 实际存在显著差异（|rank diff| > CD）的方法对
    pairwise_line_y_for = {}  # (i,j) -> y
    if show_pairwise and pairwise_sig_diff:
        # 只画"非冗余"的显著对：相邻方法若都显著不同，则跳过更远的方法对
        # 即：对每个方法 i，只画它与最近但显著不同的方法 j 的连线
        # 这样能避免大量交叉的长线段
        non_redundant_pairs = []
        # 按 rank 排序的方法索引
        sorted_idx_by_rank = sorted(range(n), key=lambda k: sr[k])
        for pos_i, i in enumerate(sorted_idx_by_rank):
            # 向右找第一个显著不同的方法
            for pos_j in range(pos_i + 1, len(sorted_idx_by_rank)):
                j = sorted_idx_by_rank[pos_j]
                if abs(sr[i] - sr[j]) > cd + 1e-10:
                    non_redundant_pairs.append((i, j))
                    break  # 只连最近的一个

        # 按 rank 差异从小到大排序，差异小的画在更靠近粗线的位置（上方）
        non_redundant_pairs_sorted = sorted(
            non_redundant_pairs,
            key=lambda p: abs(sr[p[0]] - sr[p[1]])
        )

        occupied_pw_x = []
        for pidx, (i, j) in enumerate(non_redundant_pairs_sorted):
            xi = x_positions[i]
            xj = x_positions[j]
            # 找一个不冲突的 y（从 pairwise_y_base 开始向下错开）
            y_try = pairwise_y_base
            while any(not (max(xi, xj) + 0.05 <= xl or min(xi, xj) - 0.05 >= xr)
                     for xl, xr, yy in occupied_pw_x if abs(yy - y_try) < 0.02):
                y_try -= pairwise_y_step
            pairwise_line_y_for[(i, j)] = y_try
            occupied_pw_x.append((min(xi, xj), max(xi, xj), y_try))
            # ★★★ 成对细线宽度/透明度由 THIN_BAR_WIDTH / THIN_BAR_ALPHA 控制 ★★★
            ax.plot([xi, xj], [y_try, y_try],
                    color='black', linewidth=THIN_BAR_WIDTH, zorder=3,
                    solid_capstyle='butt', alpha=THIN_BAR_ALPHA)

    # ---- 4. L 型折线引线 + 标签 ----
    # 计算每个标签的水平边界（用于检测垂直段是否穿过文字）
    label_x_left_for = [None] * n
    label_x_right_for = [None] * n
    for i in range(n):
        w = label_widths[i]
        if label_ha_for[i] == 'right':
            label_x_left_for[i] = label_x_for[i] - w
            label_x_right_for[i] = label_x_for[i]
        else:  # left
            label_x_left_for[i] = label_x_for[i]
            label_x_right_for[i] = label_x_for[i] + w

    # 智能计算每个方法引线垂直段的最高 y（避免穿过其他标签的文字）
    def _vertical_top_for(method_idx):
        x_conn = x_positions[method_idx]
        my_y = label_y_for[method_idx]
        stop_y = None

        for j in range(n):
            if j == method_idx:
                continue
            # 只检查 y 比我高（在上方）的标签：
            # 垂直引线最低点为当前标签中线，位于下方的标签文字必然低于此线，
            # 不可能被引线穿过，因此只需考虑 label_y > my_y 的标签。
            if label_y_for[j] <= my_y:
                continue
            # 检查垂直段 x 是否落在 j 标签水平区间内
            if label_x_left_for[j] - 0.02 <= x_conn <= label_x_right_for[j] + 0.02:
                # j 在我上方且会穿过，垂直段止于 j 文字下沿下方一点
                j_label_y_top = label_y_for[j]
                j_label_y_bottom = j_label_y_top - label_heights[j]
                candidate_top = j_label_y_bottom - 0.03
                if stop_y is None or candidate_top < stop_y:
                    stop_y = candidate_top

        if stop_y is not None:
            return stop_y
        # 否则延伸到分组线/轴
        # 优先检查成对显著性细线（更靠近标签）
        if show_pairwise:
            for (ii, jj), y in pairwise_line_y_for.items():
                if method_idx in (ii, jj):
                    return y
        for gidx, group in enumerate(groups):
            if method_idx in group:
                return group_line_y[gidx]
        return axis_y - 0.04

    for i, (x, name) in enumerate(zip(x_positions, sm)):
        is_ours = 'ours' in name.lower()
        fontweight = 'bold' if is_ours else 'normal'

        # 参考图风格：轴上无标记圆点（移除 markers）

        # 标签 y 和 x
        label_y = label_y_for[i]
        label_x = label_x_for[i]

        # 垂直段最高点
        v_top = _vertical_top_for(i)
        # 垂直段最低点：标签文字中线附近
        v_bottom = label_y - label_heights[i] / 2

        # 绘制 L 型折线：从 (x, v_top) 垂直下到 (x, v_bottom)，再水平到 (label_x, v_bottom)
        # ★ L 型引线宽度由 CONNECTOR_LINE_WIDTH 控制 ★
        ax.plot([x, x, label_x],
                [v_top, v_bottom, v_bottom],
                color='black', linewidth=CONNECTOR_LINE_WIDTH, zorder=1,
                solid_capstyle='butt', solid_joinstyle='miter')

        # 标签
        ax.text(label_x, label_y, name, ha=label_ha_for[i], va='center',
                fontsize=10.5, fontweight=fontweight, color='black',
                zorder=7)

    # ---- 图注 ----
    if fig_caption:
        fig.text(0.5, 0.01, fig_caption, ha='center', va='bottom',
                 fontsize=9.5, style='italic', color='black')

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close()
    print(f"  [OK] saved: {save_path}")


# ====================== 主程序 ======================

def run_analysis(data_matrix, methods, metric_name, save_path):
    print(f"\n{'=' * 60}")
    print(f"[{metric_name}] Friedman + Nemenyi analysis")
    print(f"{'=' * 60}")

    n_methods, n_datasets = data_matrix.shape
    chi2, p_value, mean_ranks = friedman_test(data_matrix)
    cd, q_alpha = nemenyi_cd(n_methods, n_datasets, alpha=0.05)

    print(f"  datasets: {n_datasets}, methods: {n_methods}")
    print(f"  Friedman chi^2 = {chi2:.4f}, p = {p_value:.6f}")
    print(f"  CD(alpha=0.05) = {cd:.4f} (q_alpha={q_alpha:.3f})")

    print(f"\n  --- mean ranks (lower is better) ---")
    sorted_idx = np.argsort(mean_ranks)
    for i in sorted_idx:
        print(f"    {methods[i]:20s}: {mean_ranks[i]:.3f}")

    if p_value < 0.05:
        print(f"\n  [SIG] Friedman test significant (p={p_value:.6f} < 0.05)")
    else:
        print(f"\n  [NS]  Friedman test not significant (p={p_value:.6f} >= 0.05)")

    # 打印成对显著性差异（用于控制台参考）
    pairwise = pairwise_significance(mean_ranks, n_methods, n_datasets, alpha=0.05)
    sig_pairs = [(i, j, diff, cd) for (i, j, sig_bonf, sig_cd, diff, cd) in pairwise
                 if sig_cd]
    if sig_pairs:
        print(f"\n  --- pairwise significant differences (|rank diff| > CD={cd:.3f}) ---")
        for i, j, diff, _ in sig_pairs:
            print(f"    {methods[i]:20s} vs {methods[j]:20s}: diff={diff:.3f}")

    # 绘制 CD 图
    # 标准学术 CD 图风格：只画 Nemenyi 粗分组线（Demsar 2006 规范）
    # 如需展示成对显著性细线，把 show_pairwise 改为 True
    caption = (
        rf"Fig. Comparison of {metric_name} with the Nemenyi test "
        rf"(at $p \leq 0.05$). "
        rf"Groups of approaches that are not significantly different "
        rf"are connected by the thick bar."
    )

    draw_cd_diagram(mean_ranks, methods, cd, save_path=save_path,
                    fig_caption=caption, show_pairwise=False)

    return mean_ranks, cd


if __name__ == '__main__':
    # === Accuracy CD 图 ===
    run_analysis(accuracy_data, methods_all, 'Accuracy',
                 save_path='cd_diagram_accuracy.png')

    # === F1-score CD 图 ===
    run_analysis(f1_data, methods_all, 'F1-score',
                 save_path='cd_diagram_f1.png')

    print("\n[Done] all diagrams generated.")
