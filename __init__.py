# -*- coding: utf-8 -*-
"""
svg-academic: 学术图表绘制工具包
===============================
一个专注于生成符合学术期刊规范的SVG图表工具库，支持多种图表类型、期刊格式适配和高度自定义，
输出文件兼容Adobe Illustrator等矢量编辑软件。
"""

__version__ = "0.1.0"
__author__ = "科研绘图工具组"
__license__ = "MIT"

# 导出核心管理器（用于自定义配置）
from .managers import (
    color_manager,  # 调色板管理器
    theme_manager,  # 主题管理器
    size_manager,  # 尺寸管理器
    journal_manager  # 期刊配置管理器
)

# 导出核心绘图工具
from .plotter import plotter  # 链式调用绘图核心

# 导出全局配置函数
from .config import set_global_defaults  # 设置全局默认参数

# 导出常用图表模板（基础图表）
from .templates import (
    plot_bar_with_error,  # 带误差棒的柱状图
    plot_scatter_with_regression,  # 带回归线的散点图
)

# 导出中级图表模板
from .templates import (
    plot_heatmap,  # 学术风格热图
    plot_distribution,  # 组合分布 plot（小提琴+箱线+散点）
    plot_stacked_bar,  # 堆叠柱状图
    plot_correlogram,  # 相关性矩阵图
)

# 导出高级学术图表模板
from .templates import (
    plot_volcano,  # 火山图
    plot_sankey,  # 桑基图
    plot_raincloud,  # 云雨图
    plot_pca,  # PCA图
    plot_manhattan,  # 曼哈顿图
    plot_forest,  # 森林图
    plot_upset,  # Upset图
    plot_bland_altman,  # Bland-Altman图
    plot_clustermap,  # 聚类热图
    plot_gene_dot_plot,  # 基因表达点图
    plot_ma,  # MA图
    plot_sequence_logo,  # 序列标志图
    plot_circos_like,  # Circos类似图
)

# 导出机器学习与统计图表模板
from .templates import (
    plot_roc_curve,  # ROC曲线
    plot_precision_recall_curve,  # 精确率-召回率曲线
    plot_confusion_matrix,  # 混淆矩阵
    plot_feature_importance,  # 特征重要性图
    plot_dendrogram,  # 树状图
    plot_survival_curve,  # 生存曲线
)

# 导出工具函数
from .utils import format_axis

# 定义公共API（from svg_academic import * 时会导入这些对象）
__all__ = [
    # 版本与元数据
    "__version__", "__author__", "__license__",

    # 管理器
    "color_manager", "theme_manager", "size_manager", "journal_manager",

    # 核心工具
    "plotter", "set_global_defaults",

    # 工具函数
    "format_axis",

    # 图表模板（按类别排序）
    # 基础图表
    "plot_bar_with_error", "plot_scatter_with_regression",

    # 中级图表
    "plot_heatmap", "plot_distribution", "plot_stacked_bar", "plot_correlogram",

    # 高级学术图表
    "plot_volcano", "plot_sankey", "plot_raincloud", "plot_pca",
    "plot_manhattan", "plot_forest", "plot_upset", "plot_bland_altman",
    "plot_clustermap", "plot_gene_dot_plot", "plot_ma",
    "plot_sequence_logo", "plot_circos_like",

    # 机器学习与统计图表
    "plot_roc_curve", "plot_precision_recall_curve",
    "plot_confusion_matrix", "plot_feature_importance",
    "plot_dendrogram", "plot_survival_curve",
]
