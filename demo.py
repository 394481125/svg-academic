import svg_academic as sat
import numpy as np

# 1. 一键创建并注册自定义期刊
sat.journal_manager.add_custom_journal(
    journal_name="our_journal",  # 自定义期刊名称
    colors=['#FFC107', '#26A69A', '#EF5350'],  # 专属调色板
    theme_config={  # 主题样式配置
        "axes.linewidth": 1.0,
        "grid.alpha": 0.2,
        "lines.markersize": 6
    },
    width=5.0,  # 图表宽度（英寸）
    height=4.5  # 图表高度（英寸）
)

# 2. 使用自定义期刊绘图
x = np.linspace(0, 15, 200)
y1 = np.sin(x) * np.exp(-0.1 * x)
y2 = np.cos(x) * np.exp(-0.1 * x)

with sat.plotter.use_journal('our_journal').create_fig() as p:
    p.ax.plot(x, y1, label="A", linewidth=2.5)  # 自动使用自定义配色
    p.ax.plot(x, y2, label="B", linewidth=2.5, linestyle="--")

    p.set_labels(x_label="T", y_label="A", title="Title")
    p.optimize_ax(hide_spines=["top", "right"])
    p.ax.legend()
    p.save_all_formats("custom_journal_demo")