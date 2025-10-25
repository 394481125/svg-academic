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
    color_manager,  # 调色板管理器（内置学术期刊常用配色）
    theme_manager,  # 主题管理器（预设不同期刊格式主题）
    size_manager,  # 尺寸管理器（控制图表尺寸与比例规范）
    journal_manager  # 期刊配置管理器（存储常见期刊格式参数）
)

# 导出核心绘图工具
from .plotter import plotter  # 链式调用绘图核心（支持自定义图表构建）

# 导出全局配置函数
from .config import set_global_defaults  # 设置全局默认参数（统一图表风格）

# 导出常用图表模板（基础图表）
from .templates import (
    plot_bar_with_error,  # 带误差棒的柱状图（支持均值±标准差/标准误）
    plot_scatter_with_regression,  # 带回归线的散点图（含置信区间）
)

# 导出中级图表模板
from .templates import (
    plot_heatmap,  # 学术风格热图（支持聚类与注释）
    plot_distribution,  # 组合分布图表（小提琴图+箱线图+散点图）
    plot_stacked_bar,  # 堆叠柱状图（支持多组数据对比）
    plot_correlogram,  # 相关性矩阵图（含显著性标记）
    plot_funnel,  # 漏斗图（适用于流程转化率展示）
    plot_spaghetti,  # 意大利面条图（展示多组时间序列趋势）
    plot_violin_box,  # 小提琴箱线组合图（增强数据分布展示）
    plot_bubble,  # 气泡图（通过气泡大小展示第三维度数据）
    plot_line_with_shaded,  # 带阴影区间的线图（展示误差范围）
    plot_radar_chart,  # 雷达图（多指标数据对比）
)

# 导出高级学术图表模板
from .templates import (
    plot_volcano,  # 火山图（常用于差异表达分析）
    plot_sankey,  # 桑基图（展示流量/关系流向）
    plot_raincloud,  # 云雨图（高效展示数据分布特征）
    plot_pca,  # PCA散点图（主成分分析可视化）
    plot_pca_biplot,  # PCA双标图（展示变量与样本关系）
    plot_manhattan,  # 曼哈顿图（全基因组关联分析）
    plot_forest,  # 森林图（荟萃分析/效应量展示）
    plot_upset,  # Upset图（集合交集可视化）
    plot_bland_altman,  # Bland-Altman图（一致性分析）
    plot_clustermap,  # 聚类热图（带层次聚类的热图）
    plot_gene_dot_plot,  # 基因表达点图（单细胞数据分析常用）
    plot_ma,  # MA图（差异表达数据展示）
    plot_sequence_logo,  # 序列标志图（核酸/氨基酸序列保守性）
    plot_circos_like,  # 类Circos图（环形数据关联展示）
    plot_paired_dumbbell,  # 配对哑铃图（前后对比/差值展示）
    plot_ridge_plot,  # 脊线图（多组分布趋势对比）
    plot_waffle,  # 华夫图（占比可视化，类似饼图的替代方案）
    plot_treemap,  # 树状图（层级数据占比展示）
    plot_parallel_coordinates,  # 平行坐标图（多维度数据分布）
    plot_calendar_heatmap,  # 日历热图（时间序列数据分布）
    plot_gantt,  # 甘特图（项目时间线/实验流程展示）
    plot_bar_line_dual,  # 柱状线图双轴（双指标对比，共享X轴）
    plot_density_heatmap,  # 密度热图（二维数据分布密度）
    create_multi_panel_figure,  # 多面板组合图（子图排版工具）
)

# 导出机器学习与统计图表模板
from .templates import (
    plot_roc_curve,  # ROC曲线（分类模型性能评估）
    plot_precision_recall_curve,  # 精确率-召回率曲线（不平衡数据评估）
    plot_confusion_matrix,  # 混淆矩阵（分类结果可视化）
    plot_feature_importance,  # 特征重要性图（模型解释性）
    plot_dendrogram,  # 树状图（聚类结果展示）
    plot_survival_curve,  # 生存曲线（生存分析可视化）
)

# 导出工具函数
from .utils import (
    format_axis,  # 坐标轴格式化（刻度、标签样式调整）
    optimize_academic_ax,  # 学术坐标轴优化（去除冗余元素，符合期刊规范）
    set_academic_labels,  # 学术标签设置（字体、大小、斜体等规范）
    add_panel_label,  # 子图面板标签（A/B/C...标注，符合多图排版规范）
    AcademicAnnotation  # 学术标注工具（显著性星号、连接线等）
)

# 定义公共API（from svg_academic import * 时会导入这些对象）
__all__ = [
    # 版本与元数据
    "__version__", "__author__", "__license__",

    # 管理器
    "color_manager", "theme_manager", "size_manager", "journal_manager",

    # 核心工具
    "plotter", "set_global_defaults",

    # 工具函数
    "format_axis", "optimize_academic_ax", "set_academic_labels",
    "add_panel_label", "AcademicAnnotation",

    # 图表模板（按类别排序）
    # 基础图表
    "plot_bar_with_error", "plot_scatter_with_regression",

    # 中级图表
    "plot_heatmap", "plot_distribution", "plot_stacked_bar", "plot_correlogram",
    "plot_funnel", "plot_spaghetti", "plot_violin_box", "plot_bubble",
    "plot_line_with_shaded", "plot_radar_chart",

    # 高级学术图表
    "plot_volcano", "plot_sankey", "plot_raincloud", "plot_pca", "plot_pca_biplot",
    "plot_manhattan", "plot_forest", "plot_upset", "plot_bland_altman",
    "plot_clustermap", "plot_gene_dot_plot", "plot_ma", "plot_sequence_logo",
    "plot_circos_like", "plot_paired_dumbbell", "plot_ridge_plot", "plot_waffle",
    "plot_treemap", "plot_parallel_coordinates", "plot_calendar_heatmap",
    "plot_gantt", "plot_bar_line_dual", "plot_density_heatmap",
    "create_multi_panel_figure",

    # 机器学习与统计图表
    "plot_roc_curve", "plot_precision_recall_curve", "plot_confusion_matrix",
    "plot_feature_importance", "plot_dendrogram", "plot_survival_curve",
]
