# svg_academic/templates.py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.sankey import Sankey
import matplotlib.pyplot as plt
from .plotter import plotter
from .utils import add_panel_label
from .managers import color_manager


# --- 基础模板 ---

def plot_bar_with_error(data_dict, x_label="", y_label="", title="", journal="nature", save_path=None):
    """
    模板：绘制带误差棒的柱状图。
    """
    labels = list(data_dict.keys())
    means = [val[0] for val in data_dict.values()]
    errors = [val[1] for val in data_dict.values()]
    x_pos = np.arange(len(labels))

    with plotter.use_journal(journal).create_fig() as p:
        p.ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.8, capsize=5, edgecolor='black')
        p.ax.set_xticks(x_pos)
        p.ax.set_xticklabels(labels)
        p.set_labels(x_label, y_label, title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_scatter_with_regression(x_data, y_data, x_label="", y_label="", title="", journal="science", save_path=None):
    """
    模板：绘制带回归线的散点图。
    """
    with plotter.use_journal(journal).create_fig() as p:
        p.ax.scatter(x_data, y_data, alpha=0.6, label="Data")
        try:
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(x_data, y_data)
            reg_line = slope * np.array(x_data) + intercept
            p.ax.plot(x_data, reg_line, color='red', linestyle='--', label=f'Fit: R²={r_value **2:.2f}')
        except ImportError:
            print("警告：scipy未安装，无法计算回归线。请运行 'pip install scipy'")
        p.set_labels(x_label, y_label, title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


# --- 中级模板 ---

def plot_heatmap(data, x_labels=None, y_labels=None, title="", cmap="coolwarm", save_path=None, journal="nature"):
    """
    模板：绘制学术风格的热图。
    """
    with plotter.use_journal(journal).create_fig() as p:
        sns.heatmap(data, xticklabels=x_labels or [], yticklabels=y_labels or [],
                    cmap=cmap, ax=p.ax, cbar_kws={'label': 'Value'})
        p.set_labels("", "", title)
        plt.setp(p.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(p.ax.get_yticklabels(), rotation=0)
        if save_path: p.save_all_formats(save_path)


def plot_distribution(data, x_labels=None, y_label="Value", title="", journal="science", save_path=None):
    """
    模板：绘制组合小提琴图、箱线图和散点图。
    """
    with plotter.use_journal(journal).create_fig() as p:
        sns.violinplot(data=data, ax=p.ax, inner=None, color=".8", linewidth=0)
        sns.boxplot(data=data, ax=p.ax, width=0.2, boxprops={'facecolor': 'None', 'zorder': 2})
        sns.stripplot(data=data, ax=p.ax, size=3, zorder=1, color=".3")
        if x_labels: p.ax.set_xticklabels(x_labels)
        p.set_labels("Group", y_label, title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


# --- 高级模板 (第一批) ---

def plot_volcano(
        df, logfc_col, pval_col, gene_col=None,
        logfc_thresh=1.0, pval_thresh=0.05,
        genes_to_label=None, save_path=None
):
    """
    模板：绘制火山图。
    """
    with plotter.use_journal("science").use_size("science_double").create_fig() as p:
        df['-log10(pvalue)'] = -np.log10(df[pval_col])
        up = (df[logfc_col] > logfc_thresh) & (df[pval_col] < pval_thresh)
        down = (df[logfc_col] < -logfc_thresh) & (df[pval_col] < pval_thresh)

        p.ax.scatter(df[logfc_col], df['-log10(pvalue)'], c='grey', alpha=0.5, s=10, label='Not Significant')
        p.ax.scatter(df.loc[up, logfc_col], df.loc[up, '-log10(pvalue)'], c='#E63946', alpha=0.7, s=20,
                     label='Up-regulated')
        p.ax.scatter(df.loc[down, logfc_col], df.loc[down, '-log10(pvalue)'], c='#0066CC', alpha=0.7, s=20,
                     label='Down-regulated')

        p.ax.axhline(y=-np.log10(pval_thresh), color='k', linestyle='--', linewidth=0.8)
        p.ax.axvline(x=logfc_thresh, color='k', linestyle='--', linewidth=0.8)
        p.ax.axvline(x=-logfc_thresh, color='k', linestyle='--', linewidth=0.8)

        if genes_to_label and gene_col:
            try:
                from adjustText import adjust_text
                texts = [p.ax.text(row[logfc_col], row['-log10(pvalue)'], row[gene_col], fontsize=8)
                         for i, row in df[df[gene_col].isin(genes_to_label)].iterrows()]
                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
            except ImportError:
                print("警告：为获得更好的标签效果，请安装 'pip install adjustText'")
                for i, row in df[df[gene_col].isin(genes_to_label)].iterrows():
                    p.ax.text(row[logfc_col], row['-log10(pvalue)'], row[gene_col], fontsize=8)

        p.set_labels(f'log₂ Fold Change', r'-log₁₀(P-value)', 'Volcano Plot').optimize_ax()
        p.ax.legend(loc='upper right', frameon=True)
        if save_path: p.save_all_formats(save_path)


def plot_sankey(flows, title="Sankey Diagram", size="poster_large", save_path=None):
    """
    模板：绘制桑基图。
    """
    with plotter.use_theme("nature").use_size(size).create_fig() as p:
        from collections import defaultdict
        from matplotlib.path import Path
        import matplotlib.patches as patches

        df = pd.DataFrame(flows, columns=['source', 'target', 'value'])
        all_nodes = sorted(pd.unique(df[['source', 'target']].values.ravel('K')))

        node_io = {node: {'input': 0, 'output': 0} for node in all_nodes}
        for _, row in df.iterrows():
            node_io[row['source']]['output'] += row['value']
            node_io[row['target']]['input'] += row['value']

        node_heights = {node: max(io['input'], io['output']) for node, io in node_io.items()}
        y_pos = 0
        node_positions = {}
        for node in all_nodes:
            node_positions[node] = {'y': y_pos, 'height': node_heights[node]}
            y_pos += node_heights[node] * 1.5

        source_nodes = [node for node, io in node_io.items() if io['input'] == 0]
        for node, pos_data in node_positions.items():
            x_pos = 0.0 if node in source_nodes else 0.8
            p.ax.add_patch(plt.Rectangle((x_pos, pos_data['y']), 0.1, pos_data['height'],
                                         facecolor=color_manager.get_palette("pastel")[len(node) % 5],
                                         edgecolor='black'))
            p.ax.text(x_pos + 0.05, pos_data['y'] + pos_data['height'] / 2, node, ha='center', va='center')
            node_positions[node]['x'] = x_pos

        src_y_offset = defaultdict(float)
        tgt_y_offset = defaultdict(float)
        max_flow_val = df['value'].max()

        for _, row in df.iterrows():
            src, tgt, val = row['source'], row['target'], row['value']
            y_start = node_positions[src]['y'] + src_y_offset[src]
            y_end = node_positions[tgt]['y'] + tgt_y_offset[tgt]
            x_start = node_positions[src]['x'] + 0.1
            x_end = node_positions[tgt]['x']
            verts = [(x_start, y_start), (x_start + (x_end - x_start) / 3, y_start),
                     (x_start + 2 * (x_end - x_start) / 3, y_end), (x_end, y_end)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='grey', lw=val / max_flow_val * 10, alpha=0.5)
            p.ax.add_patch(patch)
            src_y_offset[src] += val
            tgt_y_offset[tgt] += val

        p.ax.autoscale_view()
        p.ax.axis('off')
        p.set_labels("", "", title)
        if save_path: p.save_all_formats(save_path)


def plot_raincloud(
        data, x_labels=None, y_label="Value", title="",
        palette="pastel", journal="science", save_path=None
):
    """
    模板：绘制云雨图。
    """
    with plotter.use_journal(journal).create_fig() as p:
        colors = color_manager.get_palette(palette)

        if isinstance(data, pd.DataFrame):
            df = data.copy()
            x_col = x_labels if isinstance(x_labels, str) and x_labels in df.columns else df.columns[0]
            if y_label in df.columns:
                y_col = y_label
            elif len(df.columns) > 1:
                y_col = df.columns[1] if df.columns[1] != x_col else df.columns[0]
            else:
                raise ValueError("无法自动确定Y轴数据列。")
        else:
            if not hasattr(data, '__iter__'): raise TypeError("数据必须是DataFrame或可迭代对象。")
            group_names = x_labels if (x_labels and not isinstance(x_labels, str)) else [f'Group {i + 1}' for i in
                                                                                         range(len(data))]
            data_long = []
            for i, group_data in enumerate(data):
                if not hasattr(group_data, '__iter__') or isinstance(group_data, str): group_data = [group_data]
                for val in group_data:
                    data_long.append({'_internal_group': group_names[i], '_internal_val': val})
            df = pd.DataFrame(data_long)
            x_col, y_col = '_internal_group', '_internal_val'

        sns.violinplot(x=x_col, y=y_col, data=df, ax=p.ax, palette=colors, inner=None, cut=0, linewidth=1.5,
                       saturation=0.8)

        for collection in p.ax.collections:
            if isinstance(collection, plt.matplotlib.collections.PolyCollection):
                collection.set_alpha(0.7)

        sns.stripplot(x=x_col, y=y_col, data=df, ax=p.ax, palette=colors, edgecolor="white", size=4, jitter=0.2,
                      zorder=1)
        sns.boxplot(x=x_col, y=y_col, data=df, ax=p.ax, width=0.15, boxprops={'zorder': 2, 'facecolor': 'white'},
                    whiskerprops={'zorder': 2, 'color': 'black'}, capprops={'zorder': 2, 'color': 'black'},
                    medianprops={'zorder': 2, 'color': 'red'}, showfliers=False)

        p.set_labels("", y_label, title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_correlogram(df, title="Correlation Matrix", save_path=None):
    """
    模板：绘制相关性矩阵热图。
    """
    with plotter.use_journal("nature").use_size("nature_double").create_fig() as p:
        from scipy import stats
        corr = df.corr()
        p_values = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", ax=p.ax, vmin=-1, vmax=1)
        for i in range(len(corr.columns)):
            for j in range(i):
                p_val, star = p_values.iloc[i, j], ""
                if p_val < 0.001:
                    star = '***'
                elif p_val < 0.01:
                    star = '**'
                elif p_val < 0.05:
                    star = '*'
                if star: p.ax.text(j + 0.5, i + 0.3, star, ha='center', va='center', color='white', fontsize=12)
        p.set_labels("", "", title)
        if save_path: p.save_all_formats(save_path)


def plot_upset(data, title="Set Intersections", save_path=None):
    """
    模板：绘制Upset图。
    """
    try:
        from upsetplot import plot as upset_plot
        with plotter.use_theme("simple").use_size("nature_double").create_fig() as p:
            upset_plot(data, fig=p.fig, show_counts=True)
            p.fig.suptitle(title, fontsize=14, fontweight='bold')
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：Upset图需要 'upsetplot' 库。请运行 'pip install upsetplot'")


def plot_bland_altman(data1, data2, title="Bland-Altman Plot", save_path=None):
    """
    模板：绘制Bland-Altman图。
    """
    with plotter.use_journal("science").create_fig() as p:
        mean, diff = np.mean([data1, data2], axis=0), data1 - data2
        md, sd = np.mean(diff), np.std(diff, ddof=1)
        p.ax.scatter(mean, diff, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
        p.ax.axhline(md, color='gray', linestyle='--')
        p.ax.axhline(md + 1.96 * sd, color='red', linestyle='--')
        p.ax.axhline(md - 1.96 * sd, color='red', linestyle='--')
        p.ax.text(np.max(mean), md, f' Mean ({md:.2f})', ha='left', va='bottom', color='gray')
        p.ax.text(np.max(mean), md + 1.96 * sd, f' +1.96SD ({md + 1.96 * sd:.2f})', ha='left', va='bottom', color='red')
        p.ax.text(np.max(mean), md - 1.96 * sd, f' -1.96SD ({md - 1.96 * sd:.2f})', ha='left', va='top', color='red')
        p.set_labels('Average of two methods', 'Difference between two methods', title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_pca(X, groups, title="PCA Plot", save_path=None):
    """
    模板：绘制主成分分析（PCA）图。
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        with plotter.use_journal("nature").create_fig() as p:
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(X_scaled)
            pc_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
            pc_df['group'] = groups
            sns.scatterplot(x='PC1', y='PC2', hue='group', data=pc_df, ax=p.ax, s=60, alpha=0.8, edgecolor='k')
            p.set_labels(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})',
                         f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', title)
            p.optimize_ax()
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：PCA图需要 'scikit-learn' 库。请运行 'pip install scikit-learn'")


def plot_forest(data, title="Forest Plot", save_path=None):
    """
    模板：绘制森林图。
    """
    with plotter.use_journal("science").use_size("science_double").create_fig() as p:
        data = data.iloc[::-1].reset_index(drop=True)  # 倒序排列，让第一个研究在最上方
        # 计算95%置信区间误差范围
        errors = [
            data['odds_ratio'] - data['ci_low'],  # 下界误差
            data['ci_high'] - data['odds_ratio']  # 上界误差
        ]

        # 关键修复：使用固定标记大小（标量），而非数组
        p.ax.errorbar(
            x=data['odds_ratio'],  # x轴：OR值
            y=data.index,  # y轴：研究索引（倒序后）
            xerr=errors,  # x方向误差棒（95%CI）
            fmt='o',  # 标记样式为圆形
            capsize=3,  # 误差棒帽子长度
            color='black',  # 线条颜色
            markerfacecolor='black',  # 标记填充色
            markersize=8  # 固定标记大小（标量值，如8）
        )

        # 添加无效线（OR=1，代表无差异）
        p.ax.axvline(x=1, linestyle='--', color='red', zorder=0)
        # 设置y轴标签为研究名称
        p.ax.set_yticks(data.index)
        p.ax.set_yticklabels(data['study'])
        # x轴使用对数刻度（适合OR值展示）
        p.ax.set_xscale('log')
        # 设置标签和优化布局
        p.set_labels("Odds Ratio (95% CI)", "Study", title).optimize_ax()

        if save_path:
            p.save_all_formats(save_path)


def plot_stacked_bar(df, title="Composition Plot", save_path=None, as_percent=True):
    """
    模板：绘制堆叠柱状图。
    """
    with plotter.use_journal("nature").create_fig() as p:
        plot_df, y_label = (df.divide(df.sum(axis=1), axis=0) * 100, "Proportion (%)") if as_percent else (df, "Value")
        plot_df.plot(kind='bar', stacked=True, ax=p.ax, width=0.8,
                     colormap=sns.color_palette("pastel", len(df.columns)).as_hex())
        p.ax.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.setp(p.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        p.set_labels("Group", y_label, title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_clustermap(df, title="Clustermap", save_path=None, **kwargs):
    """
    模板：绘制聚类热图。
    """
    plotter.use_journal("nature")
    g = sns.clustermap(df, cmap="viridis", standard_scale=1,**kwargs)
    g.fig.suptitle(title, y=1.02, fontweight='bold')
    if save_path:
        base_name = save_path.rsplit('.', 1)[0]
        for ext in ['svg', 'png', 'pdf']:
            path = f"{base_name}.{ext}"
            g.savefig(path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至：{path}")
    else:
        return g


# --- 顶级期刊专业图表模板 (第二批) ---

def plot_manhattan(df, chr_col, pos_col, pval_col, title="Manhattan Plot", sig_level=5e-8, save_path=None):
    """
    模板：绘制曼哈顿图。
    """
    with plotter.use_journal("nature").use_size("poster_large").create_fig() as p:
        df['-log10p'] = -np.log10(df[pval_col])
        df[chr_col] = df[chr_col].astype('category')
        df = df.sort_values([chr_col, pos_col])
        df['ind'] = range(len(df))
        df_grouped = df.groupby(chr_col)

        colors = color_manager.get_palette("pastel")
        x_labels, x_labels_pos = [], []

        for i, (name, group) in enumerate(df_grouped):
            group.plot(kind='scatter', x='ind', y='-log10p', color=colors[i % len(colors)], s=10, alpha=0.8, ax=p.ax)
            x_labels.append(name)
            x_labels_pos.append(group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2)

        p.ax.axhline(y=-np.log10(sig_level), color='r', linestyle='--')
        p.ax.set_xticks(x_labels_pos)
        p.ax.set_xticklabels(x_labels)
        p.set_labels("Chromosome", r'-log₁₀(P-value)', title).optimize_ax(grid=False)
        if save_path: p.save_all_formats(save_path)


def plot_gene_dot_plot(df, title="Gene Expression Dot Plot", save_path=None):
    """
    模板：绘制基因表达气泡图。
    """
    with plotter.use_journal("nature").use_size("nature_double").create_fig() as p:
        sns.scatterplot(data=df, x="gene", y="group", size="percent_expressed", hue="avg_expression",
                        palette="viridis", ax=p.ax, sizes=(20, 400), edgecolor='k', linewidth=0.5)
        plt.setp(p.ax.get_xticklabels(), rotation=90)
        legend = p.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=10)
        legend.set_title("Avg Expression\n\n% Expressed")
        p.set_labels("Gene", "Group", title).optimize_ax(grid=True)
        if save_path: p.save_all_formats(save_path)


def plot_ma(df, logfc_col, mean_expr_col, sig_col, title="MA Plot", save_path=None):
    """
    模板：绘制MA图。
    """
    with plotter.use_journal("science").create_fig() as p:
        p.ax.scatter(x=np.log10(df[mean_expr_col]), y=df[logfc_col], c=np.where(df[sig_col], 'red', 'grey'), s=5,
                     alpha=0.6)
        p.ax.axhline(0, linestyle='--', color='blue')
        p.set_labels("Mean Expression (log10)", "Log2 Fold Change", title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_sequence_logo(pwm, title="Sequence Logo", save_path=None):
    """
    模板：绘制序列Logo图。
    """
    try:
        import logomaker as lm
        with plotter.use_theme("science").create_fig() as p:
            lm.Logo(pwm, ax=p.ax)
            p.set_labels("Position", "Bits", title).optimize_ax(grid=False)
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：序列Logo图需要 'logomaker' 库。请运行 'pip install logomaker'")


def plot_circos_like(links_df, title="Circos-like Plot", save_path=None):
    """
    模板：绘制简化版环形基因组图。
    """
    # --- FIX: Use subplot_kw to create a polar projection axis ---
    with plotter.use_theme("nature").use_size("ppt_standard").create_fig(
        subplot_kw={'projection': 'polar'}
    ) as p:
        chromosomes = sorted(pd.unique(links_df[['source_chr', 'target_chr']].values.ravel('K')))
        angles = np.linspace(0, 2 * np.pi, len(chromosomes) + 1)
        chr_map = {name: (angles[i], angles[i + 1]) for i, name in enumerate(chromosomes)}

        for chr_name, (start_angle, end_angle) in chr_map.items():
            p.ax.plot([start_angle, end_angle], [1, 1], lw=5, solid_capstyle='round')
            p.ax.text(np.mean([start_angle, end_angle]), 1.1, chr_name, ha='center', va='center')

        for _, row in links_df.iterrows():
            start = np.interp(row['source_pos'], [0, 1], [chr_map[row['source_chr']][0], chr_map[row['source_chr']][1]])
            end = np.interp(row['target_pos'], [0, 1], [chr_map[row['target_chr']][0], chr_map[row['target_chr']][1]])
            p.ax.plot([start, end], [1, 1], lw=row['value'] * 2, alpha=0.5)

        p.ax.set_ylim(0, 1.2);
        p.ax.set_yticklabels([]);
        p.ax.set_xticklabels([])
        p.ax.grid(False);
        try:
            p.ax.spines['polar'].set_visible(False)
        except KeyError:
            pass

        p.set_labels("", "", title)
        if save_path: p.save_all_formats(save_path)


# --- 2. 机器学习 & 统计学 ---

def plot_roc_curve(y_true, y_pred_prob, title="ROC Curve", save_path=None):
    """
    模板：绘制受试者工作特征（ROC）曲线。
    """
    try:
        from sklearn.metrics import roc_curve, auc
        with plotter.use_journal("nature").create_fig() as p:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            p.ax.plot(fpr, tpr, color=color_manager.get_palette("nature")[1], lw=2,
                      label=f'ROC curve (area = {roc_auc:0.2f})')
            p.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            p.set_labels('False Positive Rate', 'True Positive Rate', title).optimize_ax(grid=False)
            p.ax.legend(loc="lower right")
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：ROC曲线需要 'scikit-learn' 库。请运行 'pip install scikit-learn'")


def plot_precision_recall_curve(y_true, y_pred_prob, title="Precision-Recall Curve", save_path=None):
    """
    模板：绘制精准率-召回率（PR）曲线。
    """
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        with plotter.use_journal("science").create_fig() as p:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
            ap = average_precision_score(y_true, y_pred_prob)
            p.ax.plot(recall, precision, color=color_manager.get_palette("science")[2], lw=2, label=f'AP = {ap:0.2f}')
            p.set_labels('Recall', 'Precision', title).optimize_ax()
            p.ax.legend(loc="upper right")
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：PR曲线需要 'scikit-learn' 库。请运行 'pip install scikit-learn'")


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    """
    模板：绘制混淆矩阵热图。
    """
    with plotter.use_theme("simple").create_fig() as p:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=p.ax, xticklabels=classes, yticklabels=classes)
        p.set_labels('Predicted label', 'True label', title)
        plt.setp(p.ax.get_yticklabels(), rotation=0)
        if save_path: p.save_all_formats(save_path)


def plot_feature_importance(importances, feature_names, title="Feature Importance", save_path=None):
    """
    模板：绘制模型特征重要性排序图。
    """
    with plotter.use_journal("nature").create_fig() as p:
        df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance',
                                                                                             ascending=True)
        p.ax.barh(df['feature'], df['importance'])
        p.set_labels("Importance", "Feature", title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_dendrogram(X, title="Dendrogram", save_path=None):
    """
    模板：绘制层次聚类树状图。
    """
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        with plotter.use_journal("science").create_fig() as p:
            linked = linkage(X, 'ward')
            dendrogram(linked, ax=p.ax, orientation='top', distance_sort='descending', show_leaf_counts=True)
            p.set_labels("Sample Index", "Distance", title).optimize_ax(grid=False)
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：树状图需要 'scipy' 库。请运行 'pip install scipy'")


# --- 3. 临床与流行病学 ---

def plot_survival_curve(durations, event_observed, groups, title="Survival Analysis", save_path=None):
    """
    模板：绘制生存曲线（Kaplan-Meier估计）。
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        with plotter.use_journal("science").create_fig() as p:
            df = pd.DataFrame({'duration': durations, 'event': event_observed, 'group': groups})
            for name, group_df in df.groupby('group'):
                kmf = KaplanMeierFitter()
                kmf.fit(group_df['duration'], event_observed=group_df['event'], label=name)
                kmf.plot_survival_function(ax=p.ax)

            if df['group'].nunique() > 1:
                group_names = df['group'].unique()
                if len(group_names) == 2:
                    results = logrank_test(df[df['group'] == group_names[0]]['duration'],
                                           df[df['group'] == group_names[1]]['duration'],
                                           df[df['group'] == group_names[0]]['event'],
                                           df[df['group'] == group_names[1]]['event'])
                    p.ax.text(0.1, 0.1, f'Log-rank p={results.p_value:.3f}', transform=p.ax.transAxes)

            p.set_labels("Time", "Survival Probability", title).optimize_ax()
            p.ax.legend(title="Group")
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：生存曲线需要 'lifelines' 库。请运行 'pip install lifelines'")


def plot_funnel(df, effect_col, se_col, title="Funnel Plot", save_path=None):
    """
    模板：绘制漏斗图。
    """
    with plotter.use_journal("nature").create_fig() as p:
        mean_effect = np.average(df[effect_col],
                                 weights=1 / df[se_col] **2) if 'weight' not in df.columns else np.average(df[effect_col], weights=df['weight'])
        df['precision'] = 1 / df[se_col]

        p.ax.scatter(df[effect_col], df['precision'], alpha=0.6)
        p.ax.axvline(mean_effect, color='red', linestyle='--')

        prec_range = np.linspace(df['precision'].min(), df['precision'].max(), 100)[1:]
        # 【修复】公式使用SE，而不是1/sqrt(precision)
        upper_bound = mean_effect + 1.96 * (1 / prec_range)
        lower_bound = mean_effect - 1.96 * (1 / prec_range)
        p.ax.plot(upper_bound, prec_range, 'k--')
        p.ax.plot(lower_bound, prec_range, 'k--')

        p.set_labels("Effect Size", "Precision (1/SE)", title).optimize_ax()
        p.ax.set_ylim(0, df['precision'].max() * 1.1)
        if save_path: p.save_all_formats(save_path)


def plot_spaghetti(df, time_col, value_col, id_col, title="Spaghetti Plot", save_path=None):
    """
    模板：绘制意大利面条图。
    """
    with plotter.use_journal("science").create_fig() as p:
        sns.lineplot(data=df, x=time_col, y=value_col, hue=id_col, ax=p.ax, legend=None, alpha=0.3)
        sns.lineplot(data=df, x=time_col, y=value_col, ax=p.ax, color='black', lw=3, label='Average', errorbar=None)
        p.set_labels("Time", "Value", title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


# --- 4. 数据探索与构成 ---

def plot_paired_dumbbell(df, before_col, after_col, group_col, title="Paired Comparison", save_path=None):
    """
    模板：绘制配对哑铃图。
    """
    with plotter.use_journal("science").create_fig() as p:
        df_sorted = df.sort_values(by=before_col).reset_index()
        p.ax.hlines(y=df_sorted.index, xmin=df_sorted[before_col], xmax=df_sorted[after_col], color="grey", alpha=0.4)
        p.ax.scatter(df_sorted[before_col], df_sorted.index, color=color_manager.get_palette("pastel")[0], alpha=1,
                     s=80, label='Before')
        p.ax.scatter(df_sorted[after_col], df_sorted.index, color=color_manager.get_palette("science")[0], alpha=1,
                     s=80, label='After')
        p.ax.set_yticks(df_sorted.index)
        p.ax.set_yticklabels(df_sorted[group_col])
        p.set_labels("Value", "Group", title).optimize_ax()
        p.ax.legend()
        if save_path: p.save_all_formats(save_path)


def plot_ridge_plot(df, x_col, y_col, title="Ridge Plot", save_path=None):
    """
    模板：绘制山脊图。
    """
    with plotter.use_theme("science").use_size("nature_double").create_fig() as p:
        plotter.fig.subplots_adjust(hspace=-.5)
        groups = df[y_col].unique()
        palette = sns.color_palette("viridis", n_colors=len(groups))
        for i, group in enumerate(groups):
            p.ax.axhline(y=i, color='white', linewidth=3)
            sns.kdeplot(data=df[df[y_col] == group], x=x_col, fill=True, alpha=0.7, color=palette[i], ax=p.ax, cut=0)
            p.ax.text(df[x_col].min(), i + 0.1, group, ha="left", va="center", transform=p.ax.get_yaxis_transform())
        p.ax.set_yticks([]);
        p.ax.set_ylabel("");
        p.ax.set_xlabel(x_col);
        p.ax.set_title(title)
        if save_path: p.save_all_formats(save_path)


def plot_waffle(data, title="Waffle Chart", save_path=None):
    """
    模板：绘制华夫饼图。
    """
    try:
        from pywaffle import Waffle
        with plotter.use_theme("simple").use_size("ppt_standard").create_fig() as p:
            Waffle.make_waffle(ax=p.ax, values=data, title={'label': title, 'loc': 'center'},
                               legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)})
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：华夫饼图需要 'pywaffle' 库。请运行 'pip install pywaffle'")


def plot_treemap(sizes, labels, title="Treemap", save_path=None):
    """
    模板：绘制树状图。
    """
    try:
        import squarify
        with plotter.use_theme("nature").create_fig() as p:
            squarify.plot(sizes=sizes, label=[f"{l}\n({s})" for l, s in zip(labels, sizes)], ax=p.ax, alpha=.8,
                          text_kwargs={'fontsize': 10, 'color': 'white', 'fontweight': 'bold'})
            p.ax.set_title(title, fontsize=15, fontweight='bold')
            p.ax.axis('off')
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：树状图需要 'squarify' 库。请运行 'pip install squarify'")


def plot_parallel_coordinates(df, class_column, title="Parallel Coordinates", save_path=None):
    """
    模板：绘制平行坐标图。
    """
    with plotter.use_theme("nature").use_size("poster_large").create_fig() as p:
        from pandas.plotting import parallel_coordinates
        parallel_coordinates(df, class_column, ax=p.ax, colormap='viridis', alpha=0.5)
        p.ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        p.set_labels("", "", title).optimize_ax(grid=False)
        if save_path: p.save_all_formats(save_path)


# --- 5. 专业与综合可视化 ---

def plot_calendar_heatmap(data, title="Calendar Heatmap", save_path=None):
    """
    模板：绘制日历热图。
    """
    try:
        import calmap
        with plotter.use_theme("science").use_size("poster_large").create_fig() as p:
            calmap.yearplot(data, ax=p.ax, cmap='viridis')
            p.fig.suptitle(title, fontsize=16, fontweight='bold')
            if save_path: p.save_all_formats(save_path)
    except ImportError:
        print("错误：日历热图需要 'calmap' 库。请运行 'pip install calmap'")


def plot_gantt(tasks_df, title="Gantt Chart", save_path=None):
    """
    模板：绘制甘特图。
    """
    with plotter.use_theme("simple").create_fig() as p:
        tasks_df['Start'] = pd.to_datetime(tasks_df['Start'])
        tasks_df['Finish'] = pd.to_datetime(tasks_df['Finish'])
        tasks_df['Duration'] = (tasks_df['Finish'] - tasks_df['Start'])

        p.ax.barh(tasks_df['Task'], tasks_df['Duration'], left=tasks_df['Start'], color=tasks_df.get('Color', 'blue'))
        p.ax.xaxis_date()
        p.set_labels("Date", "Task", title).optimize_ax()
        if save_path: p.save_all_formats(save_path)


# --- 补充高级模板 (第二批) ---

def plot_survival_curve(survival_data, time_col, event_col, group_col=None, title="Survival Curve",
                        journal="nature", save_path=None):
    """
    模板：绘制生存曲线（适用于医学/生物学期刊）。
    需要 lifelines 库支持：pip install lifelines
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        raise ImportError("请安装 lifelines: pip install lifelines")

    with plotter.use_journal(journal).create_fig() as p:
        kmf = KaplanMeierFitter()
        if group_col:
            groups = survival_data[group_col].unique()
            colors = color_manager.get_palette(journal)[:len(groups)]
            for i, group in enumerate(groups):
                mask = survival_data[group_col] == group
                kmf.fit(survival_data.loc[mask, time_col],
                        event_observed=survival_data.loc[mask, event_col],
                        label=str(group))
                kmf.plot_survival_function(ax=p.ax, color=colors[i], linewidth=1.5)
        else:
            kmf.fit(survival_data[time_col], event_observed=survival_data[event_col])
            kmf.plot_survival_function(ax=p.ax, linewidth=1.5)

        p.set_labels("Time", "Survival Probability", title)
        p.ax.legend(loc="lower left")
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_roc_curve(fpr_list, tpr_list, labels=None, title="ROC Curve", journal="science", save_path=None):
    """
    模板：绘制ROC曲线（适用于机器学习/诊断学相关期刊）。
    fpr_list/tpr_list: 可传入单组数据(列表)或多组数据(列表的列表)
    """
    with plotter.use_journal(journal).create_fig() as p:
        # 处理单组数据情况
        if not isinstance(fpr_list[0], list):
            fpr_list = [fpr_list]
            tpr_list = [tpr_list]

        colors = color_manager.get_palette(journal)[:len(fpr_list)]
        labels = labels or [f"Model {i + 1}" for i in range(len(fpr_list))]

        for fpr, tpr, label, color in zip(fpr_list, tpr_list, labels, colors):
            p.ax.plot(fpr, tpr, label=label, color=color, linewidth=1.5)

        # 对角线参考线
        p.ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        p.set_labels("False Positive Rate", "True Positive Rate", title)
        p.ax.legend(loc="lower right")
        p.ax.set_xlim([0.0, 1.0])
        p.ax.set_ylim([0.0, 1.05])
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_pca_biplot(pca_result, components=[0, 1], sample_labels=None, group_col=None,
                    title="PCA Biplot", journal="nature", save_path=None):
    """
    模板：绘制PCA双标图（适用于多组学/高通量数据分析）。
    pca_result: sklearn.decomposition.PCA的fit_transform输出结果
    """
    with plotter.use_journal(journal).create_fig() as p:
        x = pca_result[:, components[0]]
        y = pca_result[:, components[1]]

        if group_col is not None and sample_labels is not None:
            groups = sample_labels[group_col].unique()
            colors = color_manager.get_palette(journal)[:len(groups)]
            for group, color in zip(groups, colors):
                mask = sample_labels[group_col] == group
                p.ax.scatter(x[mask], y[mask], label=str(group), color=color, alpha=0.7, s=30)
            p.ax.legend(loc="best")
        else:
            p.ax.scatter(x, y, color=color_manager.get_palette(journal)[0], alpha=0.7, s=30)

        p.set_labels(f"PC{components[0] + 1}", f"PC{components[1] + 1}", title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_violin_box(data, x_labels=None, y_label="Value", title="Distribution",
                    journal="science", save_path=None):
    """
    模板：小提琴图+箱线图组合（强调分布形状与统计分位数）。
    """
    with plotter.use_journal(journal).create_fig() as p:
        colors = color_manager.get_palette(journal)[:len(data)]
        sns.violinplot(data=data, ax=p.ax, palette=colors, inner=None, linewidth=1)
        sns.boxplot(data=data, ax=p.ax, width=0.15, boxprops={'facecolor': 'white', 'zorder': 2})

        if x_labels:
            p.ax.set_xticklabels(x_labels)
        p.set_labels("Group", y_label, title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_bar_line_dual(y1_data, y2_data, x_labels, y1_label, y2_label, title="Bar-Line Plot",
                       journal="nature", save_path=None):
    """
    模板：柱状图+折线图双轴组合（适用于展示相关但尺度不同的指标）。
    """
    with plotter.use_journal(journal).create_fig() as p:
        x_pos = np.arange(len(x_labels))

        # 主轴柱状图
        p.ax.bar(x_pos, y1_data, color=color_manager.get_palette(journal)[0], alpha=0.7, label=y1_label)
        p.ax.set_ylabel(y1_label)
        p.ax.set_xticks(x_pos)
        p.ax.set_xticklabels(x_labels)

        # 副轴折线图
        ax2 = p.ax.twinx()
        ax2.plot(x_pos, y2_data, color=color_manager.get_palette(journal)[1], marker='o', linewidth=2, label=y2_label)
        ax2.set_ylabel(y2_label)

        # 合并图例
        lines, labels = p.ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        p.ax.legend(lines + lines2, labels + labels2, loc='upper right')

        p.set_labels("", "", title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_bubble(data, x_col, y_col, size_col, color_col=None, title="Bubble Plot",
                journal="science", save_path=None):
    """
    模板：气泡图（通过大小和颜色展示多维度数据）。
    data: pandas DataFrame
    """
    with plotter.use_journal(journal).create_fig() as p:
        sizes = (data[size_col] - data[size_col].min()) / (data[size_col].max() - data[size_col].min()) * 500 + 50

        if color_col:
            unique_colors = data[color_col].nunique()
            colors = color_manager.get_palette(journal)[:unique_colors]
            color_map = {val: colors[i] for i, val in enumerate(data[color_col].unique())}
            p.ax.scatter(data[x_col], data[y_col], s=sizes,
                         c=data[color_col].map(color_map), alpha=0.6, edgecolors='w', linewidth=0.5)
        else:
            p.ax.scatter(data[x_col], data[y_col], s=sizes,
                         c=color_manager.get_palette(journal)[0], alpha=0.6, edgecolors='w', linewidth=0.5)

        p.set_labels(x_col, y_col, title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_stacked_bar(data, x_labels, category_labels, title="Stacked Bar Plot",
                     journal="nature", save_path=None):
    """
    模板：堆叠柱状图（展示类别占比随分组的变化）。
    data: 二维列表，shape=(n_groups, n_categories)
    """
    with plotter.use_journal(journal).create_fig() as p:
        x_pos = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))
        colors = color_manager.get_palette(journal)[:len(category_labels)]

        for i, (category_data, color) in enumerate(zip(np.array(data).T, colors)):
            p.ax.bar(x_pos, category_data, bottom=bottom, color=color, alpha=0.8,
                     edgecolor='black', label=category_labels[i])
            bottom += category_data

        p.ax.set_xticks(x_pos)
        p.ax.set_xticklabels(x_labels)
        p.ax.legend(loc='upper right')
        p.set_labels("Group", "Value", title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_multi_panel(plots_funcs, labels=None, layout=(1, 1), title="Multi-panel Plot",
                     journal="science", save_path=None):
    """
    模板：多面板组合图（期刊常用的ABC分图格式）。
    plots_funcs: 绘图函数列表，每个函数接收(ax, ...)参数
    """
    with plotter.use_journal(journal).create_fig(subplot=layout) as p:
        labels = labels or [chr(65 + i) for i in range(len(plots_funcs))]  # A, B, C...

        for i, (func, label) in enumerate(zip(plots_funcs, labels)):
            ax = p.get_ax(i)
            func(ax)  # 调用子图绘制函数
            p.add_panel_label(label, ax=ax, x=-0.1, y=1.1)  # 面板标签位置

        p.fig.suptitle(title, y=1.02)
        p.fig.tight_layout()
        if save_path: p.save_all_formats(save_path)


def plot_density_heatmap(x_data, y_data, bins=50, title="Density Heatmap",
                         cmap="viridis", journal="nature", save_path=None):
    """
    模板：密度热图（展示二维数据分布密度）。
    """
    with plotter.use_journal(journal).create_fig() as p:
        hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = p.ax.imshow(hist.T, origin='lower', extent=extent, cmap=cmap, aspect='auto')
        p.fig.colorbar(im, ax=p.ax, label='Density')
        p.set_labels("X", "Y", title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_line_with_shaded(data_dict, x_data, x_label="", y_label="", title="Line with Shaded Error",
                          journal="science", save_path=None):
    """
    模板：带阴影误差范围的折线图（适用于时间序列/剂量反应数据）。
    data_dict: {label: (mean, std)} 格式
    """
    with plotter.use_journal(journal).create_fig() as p:
        colors = color_manager.get_palette(journal)[:len(data_dict)]

        for (label, (mean, std)), color in zip(data_dict.items(), colors):
            p.ax.plot(x_data, mean, label=label, color=color, linewidth=1.5)
            p.ax.fill_between(x_data, mean - std, mean + std, color=color, alpha=0.2)

        p.ax.legend(loc='best')
        p.set_labels(x_label, y_label, title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


def plot_radar_chart(categories, data_dict, title="Radar Chart", journal="nature", save_path=None):
    """
    模板：雷达图（对比多组数据在多个维度的表现）。
    """
    with plotter.use_journal(journal).create_fig() as p:
        # 计算角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        colors = color_manager.get_palette(journal)[:len(data_dict)]

        for (label, values), color in zip(data_dict.items(), colors):
            values = list(values) + [values[0]]  # 闭合图形
            p.ax.plot(angles, values, label=label, color=color, linewidth=2, alpha=0.5)
            p.ax.fill(angles, values, color=color, alpha=0.25)

        p.ax.set_xticks(angles[:-1])
        p.ax.set_xticklabels(categories)
        p.ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        p.set_labels("", "", title)
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)


# --- 终极模板 ---

def create_multi_panel_figure(
        layout, size="nature_double", theme="nature",
        plot_functions=None, save_path=None
):
    """
    模板：创建多面板组合图 (Figure 1A, 1B, 1C...)。
    """
    if plot_functions is None: plot_functions = []
    rows, cols = layout
    if len(plot_functions) > rows * cols: raise ValueError("提供的绘图函数数量超过了布局容量。")

    with plotter.use_theme(theme).use_size(size).create_fig(subplot=layout) as p:
        panel_labels = [chr(65 + i) for i in range(len(plot_functions))]
        for i, plot_func in enumerate(plot_functions):
            ax = p.axes[i]
            plot_func(ax)
            add_panel_label(ax, panel_labels[i])
        for i in range(len(plot_functions), len(p.axes)):
            p.axes[i].axis('off')
        p.optimize_ax()
        if save_path: p.save_all_formats(save_path)
