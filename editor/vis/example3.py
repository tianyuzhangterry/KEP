import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm

# =========================
# 中文字体
# =========================
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams["axes.unicode_minus"] = False

# =========================
# 数据：LLaVA-1.5 / E-IC
# =========================
methods = ["FT-V", "FT-L", "IKE", "SERAC", "MEND", "TP", "LTE", "VisEdit", "LPEdit"]
metrics = ["可靠性", "文本泛化性", "多模态泛化性", "文本局部性", "多模态局部性", "平均值"]

data = np.array([
    [52.02, 50.85, 46.23, 98.30, 91.12, 67.70],
    [51.91, 50.25, 46.98, 97.59, 93.78, 68.10],
    [92.49, 86.66, 78.46, 74.88, 64.11, 79.32],
    [41.23, 39.98, 40.99,100.00,  7.29, 45.90],
    [91.90, 92.43, 90.81, 89.46, 85.44, 90.01],
    [57.86, 56.23, 54.28, 63.35, 87.04, 63.75],
    [92.04, 90.77, 89.78, 84.38, 87.21, 88.84],
    [95.06, 94.19, 93.12,100.00, 94.74, 95.42],
    [95.23, 94.50, 93.43,100.00, 97.41, 96.11],
])

# =========================
# 颜色范围：按局部数据拉伸
# 为了增强 90~100 的分辨率，不用 0~100，而用数据有效区间
# =========================
vmin = 0   # 你也可以改成 np.percentile(data, 5)
vmax = 100

# PowerNorm 会增强高值区的颜色分辨率
norm = PowerNorm(gamma=2.5, vmin=vmin, vmax=vmax)

# =========================
# 作图
# =========================
fig, ax = plt.subplots(figsize=(9.5, 6.2))

im = ax.imshow(
    data,
    cmap="YlOrRd",     # 红黄色系，差异更明显
    norm=norm,
    aspect="auto"
)

# 坐标轴标签
ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(metrics, fontproperties=font_prop, fontsize=13)
ax.set_yticks(np.arange(len(methods)))
ax.set_yticklabels(methods, fontproperties=font_prop, fontsize=13)

# 标题
ax.set_title("LLaVA-1.5 在 E-IC 数据集上的表现", fontproperties=font_prop, fontsize=18, pad=14)

# 网格线
ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(methods), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
ax.tick_params(which="minor", bottom=False, left=False)

# 单元格数值
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        text_color = "white" if val >= 80 else "black"
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            fontsize=10.5,
            color=text_color,
            fontproperties=font_prop
        )

# 高亮 LPEdit 一行
highlight_row = methods.index("LPEdit")
rect = Rectangle(
    (-0.5, highlight_row - 0.5),
    len(metrics), 1,
    fill=False, edgecolor="#1f77b4", linewidth=2.8
)
ax.add_patch(rect)

# 颜色条
cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
cbar.ax.set_ylabel("指标值", fontproperties=font_prop, fontsize=12)
for t in cbar.ax.get_yticklabels():
    t.set_fontproperties(font_prop)

plt.tight_layout()
plt.savefig("llava15_eic_heatmap_enhanced.png", dpi=300, bbox_inches="tight")
plt.show()
