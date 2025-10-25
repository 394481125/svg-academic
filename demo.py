import svg_academic as sat
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备示例数据
x = np.linspace(0, 15, 200)
y1 = np.sin(x) * np.exp(-0.1 * x)  # 衰减正弦曲线1
y2 = np.cos(x) * np.exp(-0.1 * x)  # 衰减正弦曲线2

# 2. 完全自定义绘图配置（不使用内置期刊预设）
with sat.plotter.create_fig() as p:  # 创建画布
    p.ax.plot(x, y1, label="A", color="#2ECC71", linewidth=2.5)
    p.ax.plot(x, y2, label="B", color="#3498DB", linewidth=2.5, linestyle="--")

    # 自定义图表元素
    p.set_labels(
        x_label="T",
        y_label="A",
        title="Sample"
    )
    p.optimize_ax(hide_spines=["top", "right"])  # 隐藏顶部和右侧边框
    p.ax.legend(frameon=True, loc="upper right")  # 显示图例
    p.ax.grid(alpha=0.3)  # 添加网格线

    # 3. 保存为SVG、PNG、PDF三种格式
    # 注意：save_all_formats会自动处理三种格式，无需额外配置
    p.save_all_formats("fully_custom_line_plot")
