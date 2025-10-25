# svg_academic/config.py
import matplotlib.pyplot as plt

# 全局默认值，可以通过 set_global_defaults 进行修改
_GLOBAL_DEFAULTS = {
    "font_family": ["Arial", "sans-serif"],
    "base_fontsize": 10,
    "transparent": False,
    "dpi": 300
}


def set_global_defaults(**kwargs):
    """
    设置项目的全局默认值，避免在每次绘图时重复配置。
    """
    _GLOBAL_DEFAULTS.update(kwargs)
    print(f"全局默认值已更新: {_GLOBAL_DEFAULTS}")


def init_svg_ai_compatible(
        font_family=None,
        base_fontsize=None,
        transparent=None,
        is_dark_mode=False
):
    """
    初始化Matplotlib以生成与Adobe Illustrator兼容的SVG。
    """
    plt.switch_backend('svg')
    font_family = font_family or _GLOBAL_DEFAULTS["font_family"]
    base_fontsize = base_fontsize or _GLOBAL_DEFAULTS["base_fontsize"]
    transparent = transparent if transparent is not None else _GLOBAL_DEFAULTS["transparent"]

    base_config = {
        'font.family': font_family,
        'svg.fonttype': 'none',
        'text.usetex': False,
        'axes.unicode_minus': False,
        'font.size': base_fontsize,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'legend.fontsize': base_fontsize * 0.9,
        'xtick.labelsize': base_fontsize * 0.9,
        'ytick.labelsize': base_fontsize * 0.9,
        'savefig.format': 'svg',
        'figure.subplot.wspace': 0.35,
        'figure.subplot.hspace': 0.35
    }
    if transparent:
        base_config.update({
            'savefig.facecolor': 'none', 'savefig.edgecolor': 'none', 'axes.facecolor': 'none'
        })
    if is_dark_mode:
        base_config.update({
            "axes.labelcolor": "white", "axes.edgecolor": "white", "xtick.color": "white",
            "ytick.color": "white", "text.color": "white", "grid.color": "#555555"
        })
    plt.rcParams.update(base_config)


def save_fig(fig, save_path="output.svg", pad_inches=0.1, transparent=None, dpi=None, close_fig=True):
    """
    【修改】以多种格式导出学术图表，并增加是否关闭画布的选项。
    """
    transparent = transparent if transparent is not None else _GLOBAL_DEFAULTS["transparent"]
    dpi = dpi or _GLOBAL_DEFAULTS["dpi"]
    fmt = save_path.split(".")[-1].lower()
    if fmt not in ["svg", "pdf", "png"]:
        raise ValueError(f"不支持的格式 '{fmt}'！请使用 'svg', 'pdf', 或 'png'。")

    # 应用紧凑布局，仅在首次保存时调用，避免重复
    if not hasattr(fig, '_layout_applied'):
        fig.tight_layout(pad=pad_inches)
        fig._layout_applied = True

    fig.savefig(save_path, format=fmt, bbox_inches="tight", dpi=dpi, transparent=transparent)

    # 【修改】仅在需要时关闭画布
    if close_fig:
        plt.close(fig)

    print(f"图表已保存至：{save_path} ({fmt.upper()}格式, AI兼容)")