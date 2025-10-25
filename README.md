# svg-academic: 学术图表绘制工具包

svg-academic 是一个专为科研人员设计的学术图表绘制工具包，支持多种期刊格式适配、自定义样式配置及多格式导出，帮助你快速生成符合学术出版标准的高质量图表。


## 核心功能

- **丰富的图表模板**：涵盖基础统计图、高级学术图表及机器学习/临床专用图表
- **期刊格式适配**：内置 Nature、Science、IEEE 等顶级期刊的格式配置
- **自定义样式系统**：支持自定义调色板、主题、尺寸及期刊格式
- **多格式导出**：一键保存为 SVG（AI 兼容）、PDF、PNG 格式
- **学术优化细节**：自动优化坐标轴、字体及布局，符合学术出版规范


## 安装指南

```bash
pip install svg-academic
```

额外依赖（部分高级图表需要）：
```bash
pip install scipy pandas seaborn adjustText
```


## 快速开始

### 基础示例：带误差棒的柱状图

```python
import svg_academic as sat

# 准备数据：{组名: (均值, 误差)}
bar_data = {'Group A': (1.5, 0.2), 'Group B': (2.8, 0.4), 'Group C': (2.1, 0.3)}

# 绘制图表
sat.plot_bar_with_error(
    bar_data,
    x_label="实验组",
    y_label="测量值",
    title="带误差棒的柱状图示例",
    journal="nature",  # 应用Nature期刊格式
    save_path="bar_plot"  # 保存为 bar_plot.svg/png/pdf
)
```


## 内置期刊配置详情

工具包内置了5种顶级期刊的预设配置，每种配置整合了专属的尺寸、配色和样式参数，具体如下：

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

### 高级学术图表
- 火山图 (`plot_volcano`)
- 桑基图 (`plot_sankey`)
- 云雨图 (`plot_raincloud`)
- 相关性矩阵图 (`plot_correlogram`)
- Upset 图 (`plot_upset`)
- Bland-Altman 图 (`plot_bland_altman`)
- PCA 图 (`plot_pca`)
- 森林图 (`plot_forest`)
- 堆叠柱状图 (`plot_stacked_bar`)
- 聚类热图 (`plot_clustermap`)

### 顶级期刊专用图表
- 曼哈顿图 (`plot_manhattan`)
- 基因表达点图 (`plot_gene_dot_plot`)
- MA 图 (`plot_ma`)
- 序列标志图 (`plot_sequence_logo`)
- Circos 类似图 (`plot_circos_like`)

### 机器学习与统计图表
- ROC 曲线 (`plot_roc_curve`)
- 精确率-召回率曲线 (`plot_precision_recall_curve`)
- 混淆矩阵 (`plot_confusion_matrix`)
- 特征重要性图 (`plot_feature_importance`)
- 树状图 (`plot_dendrogram`)

### 临床与流行病学图表
- 生存曲线


## 自定义配置

### 1. 自定义期刊格式
整合主题和尺寸，创建专属期刊配置：

```python
# 1. 添加自定义调色板
sat.color_manager.add_custom_palette(
    name="lab_palette",
    colors=['#1E88E5', '#FFC107', '#26A69A', '#EF5350']
)

# 2. 创建自定义主题
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
    journal="lab_journal",  # 应用自定义期刊
    save_path="custom_journal_plot"
)
```


### 2. 临时覆盖配置
在绘图时临时修改配置：

```python
with sat.plotter.use_journal("nature") \
        .use_theme("dark_mode")  # 临时使用暗色主题
        .use_size("poster_large"):  # 临时使用海报尺寸
    # 绘图逻辑
    pass
```


## 高级用法

### 自定义绘图方法
基于内置工具链扩展自定义图表：

```python
import numpy as np

def plot_custom_chart(data, journal="nature", save_path=None):
    # 加载期刊配置并创建画布
    with sat.plotter.use_journal(journal).create_fig() as p:
        # 获取坐标轴对象
        ax = p.ax
        
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
plot_custom_chart(custom_data, journal="science", save_path="custom_plot")
```


### 坐标轴格式化
使用 `format_axis` 函数美化坐标轴：

```python
from svg_academic.utils import format_axis

with sat.plotter.create_fig() as p:
    p.ax.plot([1, 2, 3], [1000, 2000, 3000])
    format_axis(p.ax, axis='y', style='sci')  # Y轴使用科学计数法
    format_axis(p.ax, axis='x', style='percent')  # X轴使用百分比格式
```


## 全局配置
设置全局默认参数（字体、分辨率等）：

```python
sat.set_global_defaults(
    base_fontsize=12,  # 全局字体大小
    dpi=600,           # 导出分辨率
    transparent=True   # 透明背景
)
```


## 输出格式
所有图表默认支持三种格式导出：
- SVG：矢量图，兼容 Adobe Illustrator 编辑
- PDF：适合学术出版
- PNG：300dpi 位图，适合演示文稿

只需指定 `save_path="filename"`，工具会自动生成 `filename.svg`、`filename.pdf` 和 `filename.png`。
