import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import numpy as np

# 中文字体
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = fm.FontProperties(fname=font_path)

matplotlib.rcParams["axes.unicode_minus"] = False

# -----------------------------
# Data
# -----------------------------
data = {
    "LLaMA3": {
        "有效性": {
            "Counterfact": [83.3, 61.2, 63.2, 64.4, 65.7, 67.8, 99.6],
            "Zsre":        [30.5, 32.1, 0.9, 2.0, 34.6, 1.2, 95.3],
        },
        "泛化性": {
            "Counterfact": [67.8, 62.4, 61.2, 61.4, 64.7, 61.0, 90.7],
            "Zsre":        [30.2, 31.4, 1.1, 1.8, 31.3, 1.0, 90.3],
        },
        "局部性": {
            "Counterfact": [46.6, 47.1, 45.4, 49.4, 51.6, 45.3, 87.2],
            "Zsre":        [15.5, 14.7, 0.5, 0.7, 18.5, 0.6, 33.0],
        },
    },
    "GPT2-XL": {
        "有效性": {
            "Counterfact": [63.5, 42.7, 50.8, 54.6, 94.7, 51.5, 96.2],
            "Zsre":        [37.1, 25.0, 0.0, 47.5, 79.2, 38.6, 86.7],
        },
        "泛化性": {
            "Counterfact": [42.2, 35.9, 50.8, 51.2, 85.8, 50.0, 89.1],
            "Zsre":        [33.3, 22.4, 0.0, 43.6, 71.4, 41.5, 79.3],
        },
        "局部性": {
            "Counterfact": [57.1, 63.1, 49.2, 52.7, 60.5, 52.1, 78.4],
            "Zsre":        [10.4, 12.7, 0.0, 14.3, 26.4, 13.8, 26.1],
        },
    }
}

methods = ["FT-L", "FT-W", "MEND", "ROME", "MEMIT", "SERAC", "ECE (WI)"]
datasets = ["Counterfact数据集", "ZsRE数据集"]
metrics = ["有效性", "泛化性", "局部性"]
models = ["LLaMA3", "GPT2-XL"]

colors = [
    "#e6b8af",  # FT-L
    "#f4cccc",  # FT-W
    "#d9d2e9",  # MEND
    "#cfe2f3",  # ROME
    "#b6d7a8",  # MEMIT
    "#d9ead3",  # SERAC
    "#6d9eeb",  # ECE (WI)
]

fig, axes = plt.subplots(2, 3, figsize=(14, 6))

x = np.arange(len(datasets))
n_methods = len(methods)
bar_width = 0.10

for row, model in enumerate(models):
    for col, metric in enumerate(metrics):
        ax = axes[row, col]

        for i, method in enumerate(methods):
            vals = [
                data[model][metric]["Counterfact"][i],
                data[model][metric]["Zsre"][i],
            ]
            ax.bar(
                x + (i - (n_methods - 1) / 2) * bar_width,
                vals,
                width=bar_width,
                color=colors[i],
                edgecolor="none",
                label=method if (row == 0 and col == 0) else None
            )

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11, fontproperties=font_prop)

        if row == 0:
            ax.set_title(metric, fontsize=18, pad=10, fontproperties=font_prop)

        if metric == "有效性":
            ax.set_ylim(0, 110)
            ax.set_yticks([20, 40, 60, 80, 100])
        elif metric == "泛化性":
            ax.set_ylim(0, 110)
            ax.set_yticks([20, 40, 60, 80, 100])
        elif metric == "局部性":
            ax.set_ylim(0, 90)
            ax.set_yticks([10, 30, 50, 70, 90])

        ax.grid(axis="y", linestyle="-", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=11)

# 左侧模型标签
fig.text(0.04, 0.76, "(a) LLaMA3", fontsize=15, va="center", fontproperties=font_prop)
fig.text(0.04, 0.32, "(b) GPT2-XL", fontsize=15, va="center", fontproperties=font_prop)

# 图例
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, 0.02),
    frameon=True,
    fontsize=11,
    handlelength=2.8,
    columnspacing=1.5,
    handletextpad=0.5,
    prop=font_prop
)

# caption
caption = "在 Counterfact 和 ZsRE 数据集上，LLaMA3 和 GPT2-XL 模型进行 2000 次连续编辑时多种知识编辑方法的性能对比"
fig.text(0.5, -0.04, caption, ha="center", fontsize=12, fontproperties=font_prop)

plt.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.22, wspace=0.28, hspace=0.42)

plt.savefig("editing_performance.png", dpi=300, bbox_inches="tight")
plt.show()
