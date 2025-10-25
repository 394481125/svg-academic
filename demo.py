import svg_academic as sat

# 准备数据：{组名: (均值, 误差)}
bar_data = {'Group A': (1.5, 0.2), 'Group B': (2.8, 0.4), 'Group C': (2.1, 0.3)}

# 调用模板绘图（自动应用期刊样式）
sat.plot_bar_with_error(
    bar_data,
    x_label="A",
    y_label="B",
    title="C",
    journal="nature",  # 应用Nature期刊格式
    save_path="bar_plot_with_error"  # 保存路径（自动生成三种格式）
)