# svg-academic: 学术图表绘制工具包

svg-academic 是一个专为科研人员设计的学术图表绘制工具包，支持多种期刊格式适配、自定义样式配置及多格式导出，帮助你快速生成符合学术出版标准的高质量图表。

## 核心功能

- **丰富的图表模板**：涵盖基础统计图、高级学术图表及机器学习/临床专用图表
- **期刊格式适配**：内置 Nature、Science、IEEE、Elsevier、Springer 等顶级期刊的格式配置
- **自定义样式系统**：支持自定义调色板、主题、尺寸及期刊格式（提供一键配置简化流程）
- **多格式导出**：一键保存为 SVG（AI 兼容）、PDF、PNG 格式
- **学术优化细节**：自动优化坐标轴、字体及布局，符合学术出版规范

## 安装指南

```bash
pip install svg-academic
```

## 快速开始

### 一键使用内置期刊配置

```python
import svg_academic as sat
import numpy as np

# 生成示例数据
x = np.linspace(0, 15, 200)
y1 = np.sin(x) * np.exp(-0.1 * x)
y2 = np.cos(x) * np.exp(-0.1 * x)

# 使用Nature期刊格式绘图
with sat.plotter.use_journal('nature').create_fig() as p:
    p.ax.plot(x, y1, label="Series A", linewidth=2.5)
    p.ax.plot(x, y2, label="Series B", linewidth=2.5, linestyle="--")
    
    p.set_labels(x_label="Time (s)", y_label="Amplitude", title="Damped Oscillations")
    p.optimize_ax(hide_spines=["top", "right"])
    p.ax.legend()
    p.save_all_formats("nature_demo")  # 自动保存为SVG、PDF、PNG
```

### 一键创建自定义期刊配置

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

## 内置期刊配置详情

工具包内置5种顶级期刊的预设配置，整合了专属的尺寸、配色和样式：

| 期刊名称 | 关联主题 | 关联尺寸 | 尺寸（宽×高，英寸） | 配色方案（HEX） | 主题特殊配置 |
|--|--|--|--|--|--|
| `nature` | `nature` | `nature_single` | 3.5 × 2.6 | `#0066CC`, `#DC3912`, `#FF9900`, `#109618`, `#990099` | 坐标轴线条宽度：0.8px；网格透明度：0.3 |
| `science` | `science` | `science_single` | 2.3 × 2.0 | `#E63946`, `#F4A261`, `#2A9D8F`, `#264653`, `#A8DADC` | 坐标轴线条宽度：1.0px；网格透明度：0.2 |
| `ieee` | `ieee` | `ieee_double` | 7.16 × 4.0 | `#007ACC`, `#D9534F`, `#5CB85C`, `#F0AD4E`, `#5BC0DE` | 线条宽度：2.0px；网格透明度：0.4 |
| `elsevier` | `simple` | `elsevier_single` | 3.54 × 2.36 | `#222222`, `#666666`, `#999999`, `#CCCCCC`, `#EEEEEE`（灰度） | 坐标轴线条宽度：0.6px；网格透明度：0.5 |
| `springer` | `nature` | `springer_single` | 3.3 × 2.5 | 同`nature`配色（`#0066CC`, `#DC3912`等） | 同`nature`主题配置（坐标轴线条宽度0.8px，网格透明度0.3） |

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
- ROC曲线 (`plot_roc_curve`)
- 精确率-召回率曲线 (`plot_precision_recall_curve`)
- 混淆矩阵 (`plot_confusion_matrix`)
- 特征重要性图 (`plot_feature_importance`)
- 树状图 (`plot_dendrogram`)

### 临床与流行病学图表
- 生存曲线 (`plot_survival_curve`)

## 高级用法

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

## API 参考

### JournalManager 类

```python
class JournalManager:
    def __init__(self, theme_manager, size_manager):
        # 初始化期刊管理器
        
    def load_journal_config(self, journal_name, **kwargs):
        """加载指定期刊的配置，返回宽度和高度"""
        
    def add_custom_journal(self, journal_name, colors, theme_config, width, height,
                          palette_name=None, theme_name=None, size_name=None):
        """一键添加自定义期刊配置"""
```

### 图表保存函数

```python
def save_fig(fig, save_path="output.svg", pad_inches=0.1, transparent=None, dpi=None, close_fig=True):
    """保存图表为指定格式，支持SVG、PDF、PNG"""
```

### 坐标轴格式化工具

```python
def format_axis(ax, axis='y', style='sci', scilimits=(0, 0)):
    """格式化坐标轴刻度，支持科学计数法、百分比等格式"""
```

## 许可证

本项目采用 MIT 许可证，详情参见 LICENSE 文件。

## 贡献指南

欢迎通过 GitHub 提交 Issue 或 Pull Request 参与项目开发。提交前请确保代码通过所有测试，并遵循项目的代码风格规范。