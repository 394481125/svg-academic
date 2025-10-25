# svg_academic/utils.py
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


class AcademicAnnotation:
    """学术图表标注工具：统计显著性、箭头、文本等。"""

    def __init__(self, ax):
        self.ax = ax

    def add_significance(self, x1, x2, y, text="*", line_height_ratio=0.05, text_offset_ratio=0.01, **kwargs):
        """添加统计显著性标记。"""
        y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        line_y = y + y_range * line_height_ratio
        text_y = line_y + y_range * text_offset_ratio

        line_props = {'color': 'black', 'linewidth': 1.0, **kwargs}
        self.ax.plot([x1, x1, x2, x2], [y, line_y, line_y, y], **line_props)
        self.ax.text((x1 + x2) / 2, text_y, text, ha="center", va="bottom", fontweight="bold")


def optimize_academic_ax(ax, hide_spines=["top", "right"], grid=True):
    """学术图表坐标轴通用优化。"""
    for spine in hide_spines:
        ax.spines[spine].set_visible(False)

    ax.minorticks_on()
    if grid:
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=plt.rcParams.get('grid.alpha', 0.5) / 2)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=plt.rcParams.get('grid.alpha', 0.5))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)


def set_academic_labels(ax, x_label, y_label, title=None):
    """一键设置坐标轴标签与标题。"""
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=15)


def format_axis(ax, axis='y', style='sci', scilimits=(0, 0)):
    """
    【新功能】格式化坐标轴刻度。
    :param ax: 坐标轴对象。
    :param axis: 'x', 'y', 或 'both'。
    :param style: 'sci' (科学计数法), 'percent' (百分比), 或自定义函数。
    :param scilimits: 科学计数法的阈值。
    """
    if style == 'sci':
        formatter = plt.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(scilimits)
    elif style == 'percent':
        formatter = FuncFormatter(lambda y, _: f'{y:.0%}')
    elif callable(style):
        formatter = style
    else:
        raise ValueError(f"不支持的格式化类型: '{style}'")

    if axis in ['y', 'both']: ax.yaxis.set_major_formatter(formatter)
    if axis in ['x', 'both']: ax.xaxis.set_major_formatter(formatter)


def add_panel_label(ax, label, x=-0.1, y=1.1, fontsize_scale=1.2, **kwargs):
    """
    【新功能】在子图左上角添加 'A', 'B', 'C' 等面板标签。
    坐标是相对于坐标轴的比例（0,0是左下角，1,1是右上角）。
    """
    base_fontsize = plt.rcParams['font.size']
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=base_fontsize * fontsize_scale,
            fontweight='bold',
            va='top',
            ha='right',
            **kwargs)