# svg_academic/managers.py
import matplotlib.pyplot as plt
from collections import defaultdict
from .config import init_svg_ai_compatible


class ColorManager:
    """【新功能】颜色管理器，独立于主题之外，方便用户自定义和调用调色板。"""

    def __init__(self):
        self.palettes = {
            "nature": ['#0066CC', '#DC3912', '#FF9900', '#109618', '#990099'],
            "science": ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#A8DADC'],
            "ieee": ['#007ACC', '#D9534F', '#5CB85C', '#F0AD4E', '#5BC0DE'],
            "pastel": ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF'],
            "dark_mode": ['#29B6F6', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC'],
            "grayscale": ['#222222', '#666666', '#999999', '#CCCCCC', '#EEEEEE'],
        }

    def get_palette(self, name="nature"):
        """获取一个预定义的调色板。"""
        if name not in self.palettes:
            raise ValueError(f"调色板 '{name}' 不存在！可选: {list(self.palettes.keys())}")
        return self.palettes[name]

    def add_custom_palette(self, name, colors):
        """【用户自定义】添加一个新的自定义调色板。"""
        if not isinstance(colors, list):
            raise TypeError("颜色必须是一个列表，例如 ['#RRGGBB', ...]")
        self.palettes[name] = colors
        print(f"自定义调色板 '{name}' 已添加。")


class ThemeManager:
    """【优化】学术主题管理器 (原StyleManager)，管理matplotlib的rcparams配置。"""

    def __init__(self, color_manager):
        self.color_manager = color_manager
        self.themes = {
            "nature": {"palette": "nature", "config": {"axes.linewidth": 0.8, "grid.alpha": 0.3}},
            "science": {"palette": "science", "config": {"axes.linewidth": 1.0, "grid.alpha": 0.2}},
            "ieee": {"palette": "ieee", "config": {"lines.linewidth": 2.0, "grid.alpha": 0.4}},
            "simple": {"palette": "grayscale", "config": {"axes.linewidth": 0.6, "grid.alpha": 0.5}},
            "dark_mode": {
                "palette": "dark_mode",
                "config": {"axes.facecolor": "#212121", "figure.facecolor": "#212121", "grid.alpha": 0.1}
            },
        }

    def apply_theme(self, name="nature", base_fontsize=None, transparent=None):
        """应用一个完整的主题（颜色+配置）。"""
        if name not in self.themes:
            raise ValueError(f"主题 '{name}' 不存在！可选: {list(self.themes.keys())}")

        theme = self.themes[name]
        is_dark = name == 'dark_mode'
        init_svg_ai_compatible(base_fontsize=base_fontsize, transparent=transparent, is_dark_mode=is_dark)

        colors = self.color_manager.get_palette(theme["palette"])
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
        plt.rcParams.update(theme["config"])
        print(f"已应用主题 '{name}'。")

    def add_custom_theme(self, name, palette_name, config=None):
        """【用户自定义】添加一个新的自定义主题。"""
        if name in self.themes:
            raise ValueError(f"主题 '{name}' 已存在。")
        if palette_name not in self.color_manager.palettes:
            raise ValueError(f"调色板 '{palette_name}' 不存在。请先使用 color_manager.add_custom_palette 添加。")
        self.themes[name] = {"palette": palette_name, "config": config or {}}
        print(f"自定义主题 '{name}' 已添加。")


class SizeManager:
    """学术图表尺寸管理器 (单位：英寸)"""

    def __init__(self):
        self.size_presets = {
            "nature_single": (3.5, 2.6), "nature_double": (7.2, 5.4),
            "science_single": (2.3, 2.0), "science_double": (4.7, 3.5),
            "ieee_single": (3.5, 2.1), "ieee_double": (7.16, 4.0),
            "elsevier_single": (3.54, 2.36), "elsevier_double": (7.48, 4.72),
            "springer_single": (3.3, 2.5), "springer_double": (6.9, 5.2),
            "poster_large": (10.0, 7.5), "ppt_standard": (8.0, 6.0)
        }

    def get_size(self, size_name="nature_single"):
        if size_name not in self.size_presets:
            raise ValueError(f"尺寸 '{size_name}' 不存在！可选：\n{list(self.size_presets.keys())}")
        return self.size_presets[size_name]

    def add_custom_size(self, name, width, height):
        if name in self.size_presets:
            raise ValueError(f"尺寸名 '{name}' 已存在。")
        self.size_presets[name] = (width, height)
        print(f"自定义尺寸 '{name}' ({width}x{height}英寸) 已添加。")


class JournalManager:
    """期刊绘图一键管理器：整合主题与尺寸"""

    def __init__(self, theme_manager, size_manager):
        self._theme_manager = theme_manager
        self._size_manager = size_manager
        self.journal_presets = {
            "nature": {"theme": "nature", "size": "nature_single"},
            "science": {"theme": "science", "size": "science_single"},
            "ieee": {"theme": "ieee", "size": "ieee_double"},
            "elsevier": {"theme": "simple", "size": "elsevier_single"},
            "springer": {"theme": "nature", "size": "springer_single"},
        }

    def load_journal_config(self, journal_name, theme=None, size=None, **kwargs):
        """一键加载期刊配置。"""
        if journal_name not in self.journal_presets:
            raise ValueError(f"期刊预设 '{journal_name}' 不存在！可选：{list(self.journal_presets.keys())}")

        defaults = self.journal_presets[journal_name]
        use_theme = theme or defaults["theme"]
        use_size = size or defaults["size"]

        self._theme_manager.apply_theme(use_theme, **kwargs)
        width, height = self._size_manager.get_size(use_size)

        print(f"已加载期刊 '{journal_name}' 配置：")
        print(f"  - 主题: {use_theme} | 尺寸: {use_size} ({width}x{height}英寸)")
        return width, height


# --- 实例化所有管理器，供项目内其他模块导入使用 ---
color_manager = ColorManager()
theme_manager = ThemeManager(color_manager)
size_manager = SizeManager()
journal_manager = JournalManager(theme_manager, size_manager)