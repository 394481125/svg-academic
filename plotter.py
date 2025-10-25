# svg_academic/plotter.py
import matplotlib.pyplot as plt
import numpy as np

from .managers import journal_manager, size_manager, theme_manager
from .config import save_fig
from .utils import AcademicAnnotation, optimize_academic_ax, set_academic_labels, format_axis, add_panel_label


class AcademicPlot:
    """
    一键学术绘图核心类：通过链式调用整合配置、绘图、优化与保存。
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """重置内部状态，为下一次绘图做准备。"""
        self.journal = None
        self.theme = "nature"
        self.size = "nature_single"
        self.transparent = False
        self.fig = None
        self.ax = None
        self.axes = None
        print("\n--- Plotter已重置，准备新图 ---")

    def use_journal(self, name):
        """指定期刊（自动匹配主题+尺寸）。"""
        self.journal = name
        return self

    def use_theme(self, name):
        """自定义主题（覆盖期刊默认）。"""
        self.theme = name
        return self

    def use_size(self, name):
        """自定义尺寸（覆盖期刊默认）。"""
        self.size = name
        return self

    def set_transparent(self, transparent=True):
        """设置透明背景。"""
        self.transparent = transparent
        return self

    def create_fig(self, subplot=(1, 1), **kwargs):
        """
        创建画布/子图。
        【修复】智能处理 'polar' 参数，将其转换为正确的 'subplot_kw' 字典，
        以避免将无效参数传递给 Figure 对象。
        """
        if self.journal:
            width, height = journal_manager.load_journal_config(
                self.journal, transparent=self.transparent
            )
        else:
            # theme_manager.apply_theme(self.theme, transparent=self.transparent)
            width, height = size_manager.get_size(self.size)

        # --- FIX START ---
        # Intercept the 'polar' argument. If it's True, remove it from kwargs
        # and create the correct subplot_kw dictionary for matplotlib.
        if kwargs.pop('polar', False):
            # Ensure subplot_kw is a dictionary if it already exists
            if 'subplot_kw' not in kwargs:
                kwargs['subplot_kw'] = {}
            kwargs['subplot_kw']['projection'] = 'polar'
        # --- FIX END ---

        self.fig, self.axes = plt.subplots(subplot[0], subplot[1], figsize=(width, height), **kwargs)
        if subplot == (1, 1):
            self.ax = self.axes
        else:
            self.axes = self.axes.flatten() if hasattr(self.axes, 'flatten') else self.axes
            self.ax = self.axes[0]
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self._reset()

    def get_ax(self, index=0):
        """获取指定的子图坐标轴。"""
        if self.axes is None: raise RuntimeError("请先调用 create_fig() 创建画布。")
        return self.axes[index] if isinstance(self.axes, (list, np.ndarray)) else self.ax

    def annotate(self, ax=None):
        """获取指定坐标轴的标注工具。"""
        return AcademicAnnotation(ax or self.ax)

    def optimize_ax(self, ax=None, **kwargs):
        """
        【修复】优化指定坐标轴（或所有子图），重写逻辑以避免NumPy数组的布尔值歧义。
        """
        target_axes = []
        if ax:
            # Case 1: 用户指定了一个子图
            target_axes = [ax]
        elif self.axes is not None:
            # Case 2: 用户没有指定，优化所有存在的子图
            if hasattr(self.axes, '__iter__'):
                # 如果 self.axes 是列表或数组
                target_axes = self.axes
            else:
                # 如果 self.axes 是单个子图对象
                target_axes = [self.axes]

        for axis in target_axes:
            if axis:  # 确保子图对象存在
                optimize_academic_ax(axis, **kwargs)

        return self

    def set_labels(self, x_label="", y_label="", title=None, ax=None):
        """设置标签。"""
        set_academic_labels(ax or self.ax, x_label, y_label, title)
        return self

    def set_axis_format(self, axis='y', style='sci', ax=None, **kwargs):
        """链式方法：格式化坐标轴刻度。"""
        format_axis(ax or self.ax, axis=axis, style=style, **kwargs)
        return self

    def add_panel_label(self, label, ax=None, **kwargs):
        """链式方法：添加面板标签 (A, B, C...)。"""
        add_panel_label(ax or self.ax, label, **kwargs)
        return self

    def save(self, save_path="output.svg", **kwargs):
        """
        保存图表为单一格式并自动重置状态。
        """
        if not self.fig:
            raise RuntimeError("请先调用 create_fig() 创建画布。")
        save_fig(self.fig, save_path, transparent=self.transparent, close_fig=True, **kwargs)
        self._reset()

    def save_all_formats(self, base_save_path, **kwargs):
        """
        保存图表为SVG, PNG, 和 PDF三种格式并自动重置状态。
        """
        if not self.fig:
            raise RuntimeError("请先调用 create_fig() 创建画布。")

        base_name = base_save_path.rsplit('.', 1)[0]

        save_fig(self.fig, f"{base_name}.svg", transparent=self.transparent, close_fig=False, **kwargs)
        save_fig(self.fig, f"{base_name}.png", transparent=self.transparent, close_fig=False, **kwargs)
        save_fig(self.fig, f"{base_name}.pdf", transparent=self.transparent, close_fig=True, **kwargs)

        self._reset()


plotter = AcademicPlot()