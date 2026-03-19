import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

# =========================
# 中文字体
# =========================
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams["axes.unicode_minus"] = False

# =========================
# 数据
# 顺序：
# 有效性 / 泛化性 / 局部性 / 流畅性 / 一致性
# =========================
categories = ["有效性", "泛化性", "局部性", "流畅性", "一致性"]
titles = [
    "(a) 批次大小：200",
    "(b) 批次大小：100",
    "(c) 批次大小：50",
    "(d) 批次大小：10"
]

blue_data = {
    "200": [100, 92, 96, 72, 42],
    "100": [100, 90, 90, 60, 38],
    "50":  [100, 84, 80, 58, 32],
    "10":  [100, 78, 80, 60, 28],
}

red_data = {
    "200": [68, 62, 50, 48, 8],
    "100": [62, 60, 46, 38, 6],
    "50":  [56, 50, 40, 32, 4],
    "10":  [48, 40, 40, 15, 3],
}

def draw_radar(ax, vals_blue, vals_red, categories, title):
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    vals_blue = vals_blue + vals_blue[:1]
    vals_red = vals_red + vals_red[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 隐藏默认分类标签，后面手动画
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # 半径刻度
    ax.set_ylim(0, 100)
    yticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=14)
    ax.set_rlabel_position(0)

    # 关闭默认网格和边框
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # 填充
    ax.fill(angles, vals_blue, color="#8ab7f0", alpha=0.35, zorder=1)
    ax.fill(angles, vals_red, color="#f07f7f", alpha=0.45, zorder=2)

    # 轮廓
    ax.plot(angles, vals_blue, color="#7fb0ec", linewidth=3, zorder=3)
    ax.plot(angles, vals_red, color="#f07f7f", linewidth=3, zorder=4)

    # 手动画网格线
    grid_color = "#cfd6e6"
    for r in yticks[1:]:
        rs = [r] * len(angles)
        ax.plot(angles, rs, color=grid_color, linewidth=1.2, alpha=0.9, zorder=5)

    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 100], color=grid_color, linewidth=1.2, alpha=0.9, zorder=5)

    # 手动画类别标签：把半径设得比100更大一点，例如 112
    label_radius = 114
    for ang, lab in zip(angles[:-1], categories):
        ax.text(
            ang, label_radius, lab,
            fontproperties=font_prop,
            fontsize=18,
            ha="center",
            va="center"
        )

    # 标题
    ax.set_title(title, y=-0.28, fontproperties=font_prop, fontsize=20)

# =========================
# 作图
# =========================
fig, axes = plt.subplots(1, 4, figsize=(20, 6), subplot_kw=dict(polar=True))

for ax, key, title in zip(axes, ["200", "100", "50", "10"], titles):
    draw_radar(ax, blue_data[key], red_data[key], categories, title)

plt.tight_layout()
plt.savefig("radar_batchsize_cn_fixed2.png", dpi=300, bbox_inches="tight")
plt.show()
