# svg-academic: 学术图表绘制工具包

svg-academic 是一个专为科研人员设计的学术图表绘制工具包，支持多种期刊格式适配、自定义样式配置及多格式导出，帮助你快速生成符合学术出版标准的高质量图表。


## 核心功能

- **丰富的图表模板**：涵盖基础统计图、高级学术图表及机器学习/临床专用图表
- **期刊格式适配**：内置 Nature、Science、IEEE、Elsevier、Springer 等顶级期刊的格式配置
- **自定义样式系统**：支持自定义调色板、主题、尺寸及期刊格式（提供一键配置简化流程）
- **多格式导出**：一键保存为 SVG（AI 兼容）、PDF、PNG 格式
- **学术优化细节**：自动优化坐标轴、字体及布局，符合学术出版规范


## 依赖安装
部分高级图表需要额外依赖，可通过以下命令安装：
```bash
pip install scipy pandas seaborn adjustText
```


## 快速开始

### 一键创建自定义期刊配置
快速整合调色板、主题和尺寸，简化自定义流程：
```python
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
```

### 基础示例：完全自定义绘图（仅使用保存为SVG、PNG、PDF三种格式功能）
不依赖内置期刊配置，手动设置所有样式：
```python
import svg_academic as sat
import numpy as np

# 1. 准备示例数据
x = np.linspace(0, 15, 200)
y1 = np.sin(x) * np.exp(-0.1 * x)  # 衰减正弦曲线1
y2 = np.cos(x) * np.exp(-0.1 * x)  # 衰减正弦曲线2

# 2. 创建画布并绘图
with sat.plotter.create_fig() as p:  # 初始化画布
    p.ax.plot(x, y1, label="A", color="#2ECC71", linewidth=2.5)
    p.ax.plot(x, y2, label="B", color="#3498DB", linewidth=2.5, linestyle="--")
    
    # 自定义图表元素
    p.set_labels(x_label="T", y_label="A", title="Sample")  # 设置标签
    p.optimize_ax(hide_spines=["top", "right"])  # 隐藏顶部和右侧边框
    p.ax.legend(frameon=True, loc="upper right")  # 显示图例
    p.ax.grid(alpha=0.3)  # 添加网格线
    
    # 3. 保存为SVG、PNG、PDF三种格式
    p.save_all_formats("fully_custom_line_plot")  # 生成: .svg + .png + .pdf
```


### 基础示例：使用内置期刊配置
直接应用顶级期刊的预设样式（自动适配配色、尺寸和格式）：
```python
import svg_academic as sat
import numpy as np

# 1. 准备示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) + np.cos(x)

# 2. 应用期刊配置绘图
with sat.plotter.use_journal("nature").create_fig() as p:     #  指定期刊（自动加载其样式） 创建画布
                
    p.ax.plot(x, y1, label="A", color="#2ECC71", linewidth=2.5)
    p.ax.plot(x, y2, label="B", color="#3498DB", linewidth=2.5, linestyle="--")
    
    # 自定义图表元素
    p.set_labels(x_label="T", y_label="A", title="Sample")  # 设置标签
    p.optimize_ax(hide_spines=["top", "right"])  # 隐藏顶部和右侧边框
    p.ax.legend(frameon=True, loc="upper right")  # 显示图例
    p.ax.grid(alpha=0.3)  # 添加网格线
    
    # 3. 保存为SVG、PNG、PDF三种格式
    p.save_all_formats("fully_custom_line_plot")  # 生成: .svg + .png + .pdf
```


### 基础示例：使用内置图表模板
通过模板快速绘制专业图表（以带误差棒的柱状图为例）：
```python
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
```


## 内置期刊配置详情
工具包内置5种顶级期刊的预设配置，整合了专属的尺寸、配色和样式：

| 期刊名称   | 关联主题   | 关联尺寸          | 尺寸（宽×高，英寸） | 配色方案（HEX）                                                                 | 主题特殊配置                                                                 |
|------------|------------|-------------------|---------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `nature`   | `nature`   | `nature_single`   | 3.5 × 2.6           | `#0066CC`, `#DC3912`, `#FF9900`, `#109618`, `#990099`                           | 坐标轴线条宽度：0.8px；网格透明度：0.3                                        |
| `science`  | `science`  | `science_single`  | 2.3 × 2.0           | `#E63946`, `#F4A261`, `#2A9D8F`, `#264653`, `#A8DADC`                           | 坐标轴线条宽度：1.0px；网格透明度：0.2                                        |
| `ieee`     | `ieee`     | `ieee_double`     | 7.16 × 4.0          | `#007ACC`, `#D9534F`, `#5CB85C`, `#F0AD4E`, `#5BC0DE`                           | 线条宽度：2.0px；网格透明度：0.4                                              |
| `elsevier` | `simple`   | `elsevier_single` | 3.54 × 2.36         | `#222222`, `#666666`, `#999999`, `#CCCCCC`, `#EEEEEE`（灰度）                    | 坐标轴线条宽度：0.6px；网格透明度：0.5                                        |
| `springer` | `nature`   | `springer_single` | 3.3 × 2.5           | 同`nature`配色（`#0066CC`, `#DC3912`等）                                        | 同`nature`主题配置（坐标轴线条宽度0.8px，网格透明度0.3）                      |


## 支持的图表类型

### 基础模板
- 带误差棒的柱状图 (`plot_bar_with_error`)
- 带回归线的散点图 (`plot_scatter_with_regression`)


### 中级模板
- 学术风格热图 (`plot_heatmap`)
- 组合分布 plot（小提琴图+箱线图+散点）(`plot_distribution`)
- 堆叠柱状图 (`plot_stacked_bar`)
- 相关性矩阵图 (`plot_correlogram`)
- 漏斗图（适用于流程转化率展示）(`plot_funnel`)
- 意大利面条图（展示多组时间序列趋势）(`plot_spaghetti`)
- 小提琴箱线组合图（增强数据分布展示）(`plot_violin_box`)
- 气泡图（通过气泡大小展示第三维度数据）(`plot_bubble`)
- 带阴影区间的线图（展示误差范围）(`plot_line_with_shaded`)
- 雷达图（多指标数据对比）(`plot_radar_chart`)


### 高级学术图表
- 火山图（常用于差异表达分析）(`plot_volcano`)
- 桑基图（展示流量/关系流向）(`plot_sankey`)
- 云雨图（高效展示数据分布特征）(`plot_raincloud`)
- PCA 散点图（主成分分析可视化）(`plot_pca`)
- PCA 双标图（展示变量与样本关系）(`plot_pca_biplot`)
- 森林图（荟萃分析/效应量展示）(`plot_forest`)
- Upset 图（集合交集可视化）(`plot_upset`)
- Bland-Altman 图（一致性分析）(`plot_bland_altman`)
- 聚类热图（带层次聚类的热图）(`plot_clustermap`)
- 曼哈顿图（全基因组关联分析）(`plot_manhattan`)
- 基因表达点图（单细胞数据分析常用）(`plot_gene_dot_plot`)
- MA 图（差异表达数据展示）(`plot_ma`)
- 序列标志图（核酸/氨基酸序列保守性）(`plot_sequence_logo`)
- Circos 类似图（环形数据关联展示）(`plot_circos_like`)
- 配对哑铃图（前后对比/差值展示）(`plot_paired_dumbbell`)
- 脊线图（多组分布趋势对比）(`plot_ridge_plot`)
- 华夫图（占比可视化，类似饼图的替代方案）(`plot_waffle`)
- 树状图（层级数据占比展示）(`plot_treemap`)
- 平行坐标图（多维度数据分布）(`plot_parallel_coordinates`)
- 日历热图（时间序列数据分布）(`plot_calendar_heatmap`)
- 甘特图（项目时间线/实验流程展示）(`plot_gantt`)
- 柱状线图双轴（双指标对比，共享X轴）(`plot_bar_line_dual`)
- 密度热图（二维数据分布密度）(`plot_density_heatmap`)
- 多面板组合图（子图排版工具）(`create_multi_panel_figure`)


### 机器学习与统计图表
- ROC 曲线（分类模型性能评估）(`plot_roc_curve`)
- 精确率-召回率曲线（不平衡数据评估）(`plot_precision_recall_curve`)
- 混淆矩阵（分类结果可视化）(`plot_confusion_matrix`)
- 特征重要性图（模型解释性）(`plot_feature_importance`)
- 树状图（聚类结果展示）(`plot_dendrogram`)
- 生存曲线（生存分析可视化）(`plot_survival_curve`)



## 扩展：自定义中间对象名称
如需指定调色板、主题或尺寸的名称，可通过参数自定义：
```python
sat.journal_manager.add_custom_journal(
    journal_name="our_journal",
    palette_name="lab_colors",  # 自定义调色板名称
    theme_name="lab_style",     # 自定义主题名称
    size_name="lab_6x45",       # 自定义尺寸名称
    colors=['#1E88E5', '#FFC107'],  # 其他参数同上
    theme_config={"axes.linewidth": 1.0},
    width=6.0,
    height=4.5
)
```


### 分步自定义期刊配置（详细版）
如需更精细的控制，可分步创建配置：
```python
# 1. 添加自定义调色板
sat.color_manager.add_custom_palette(
    name="lab_palette",
    colors=['#1E88E5', '#FFC107', '#26A69A', '#EF5350']
)

# 2. 创建自定义主题（关联调色板）
sat.theme_manager.add_custom_theme(
    name="lab_theme",
    palette_name="lab_palette",
    config={
        "axes.linewidth": 1.0,
        "grid.alpha": 0.2,
        "lines.markersize": 6
    }
)

# 3. 定义自定义尺寸
sat.size_manager.add_custom_size(
    name="lab_size",
    width=6.0,
    height=4.5
)

# 4. 注册为期刊配置
sat.journal_manager.journal_presets["lab_journal"] = {
    "theme": "lab_theme",
    "size": "lab_size"
}

# 使用自定义期刊
sat.plot_scatter_with_regression(
    x_data, y_data,
    journal="lab_journal",
    save_path="step_by_step_custom_demo"
)
```


### 临时覆盖配置
绘图时临时修改期刊的主题或尺寸：
```python
with sat.plotter.use_journal("nature") \
        .use_theme("dark_mode")  # 临时替换为暗色主题
        .use_size("poster_large"):  # 临时使用海报尺寸
    # 绘图逻辑（示例）
    x = np.linspace(0, 10, 100)
    p.ax.plot(x, np.sin(x))
    p.save_all_formats("temporary_override_demo")
```


## 高级用法

### 自定义绘图方法
基于内置工具链扩展专属图表：
```python
import numpy as np

def plot_custom_chart(data, journal="nature", save_path=None):
    # 加载期刊配置并创建画布
    with sat.plotter.use_journal(journal).create_fig() as p:
        ax = p.ax  # 获取坐标轴对象
        
        # 自定义绘图逻辑（示例：带误差线的散点）
        x = np.arange(len(data))
        y = [item[0] for item in data]
        y_err = [item[1] for item in data]
        ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=5)
        
        # 学术优化
        p.set_labels("X轴", "Y轴", "自定义图表")
        p.optimize_ax(hide_spines=["top"])  # 隐藏顶部边框
        
        # 保存图表
        if save_path:
            p.save_all_formats(save_path)

# 使用自定义方法
custom_data = [(2.3, 0.4), (3.1, 0.3), (1.8, 0.5)]
plot_custom_chart(custom_data, journal="science", save_path="custom_chart_demo")
```


### 坐标轴格式化
使用 `format_axis` 函数美化刻度显示：
```python
from svg_academic.utils import format_axis

with sat.plotter.create_fig() as p:
    p.ax.plot([1, 2, 3], [1000, 2000, 3000])
    format_axis(p.ax, axis='y', style='sci')  # Y轴使用科学计数法
    format_axis(p.ax, axis='x', style='percent')  # X轴使用百分比格式
    p.save_all_formats("axis_format_demo")
```


## 全局配置
设置全局默认参数（字体、分辨率等）：
```python
sat.set_global_defaults(
    base_fontsize=12,  # 全局字体大小
    dpi=600,           # 导出分辨率（默认300dpi）
    transparent=True   # 透明背景（默认不透明）
)
```


## 输出格式
所有图表默认支持三种格式导出，无需额外配置：
- **SVG**：矢量图，兼容 Adobe Illustrator 等工具编辑
- **PDF**：适合学术期刊出版
- **PNG**：300dpi 位图，适合演示文稿

指定 `save_path="filename"` 后，工具会自动生成 `filename.svg`、`filename.pdf` 和 `filename.png`。

> 注：保存函数支持通过 `close_fig` 参数控制是否关闭画布（默认关闭），批量生成图表时可设置 `close_fig=False` 提升效率。