import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 假设 templates.py 位于 svg_academic 包中
# 如果您的结构不同，请相应地调整导入路径
# Assuming templates.py is in the svg_academic package
# Adjust the import path if your structure is different
import svg_academic.templates as sat


def create_output_directory(dir_name="demo_plots"):
    """创建一个目录用于存放生成的图表，如果目录已存在则不执行任何操作。"""
    # Create a directory to store the generated plots. If it exists, do nothing.
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created.")
    return dir_name


def main():
    """主函数，为每个模板生成并保存一个示例图表。"""
    # Main function to generate and save an example plot for each template.
    OUTPUT_DIR = create_output_directory()

    print("\n--- 1. Starting generation of basic and intermediate templates ---")

    # --- 基础模板 (Basic Templates) ---
    print("Generating: Bar plot with error bars...")
    bar_data = {'Group A': (1.5, 0.2), 'Group B': (2.8, 0.4), 'Group C': (2.1, 0.3)}
    sat.plot_bar_with_error(bar_data, x_label="Experiment Group", y_label="Measurement",
                            title="Bar Plot with Error Bars Example",
                            save_path=os.path.join(OUTPUT_DIR, "bar_with_error"))

    print("Generating: Scatter plot with regression line...")
    x_scatter = np.random.rand(50) * 10
    y_scatter = 2 * x_scatter + 1 + np.random.randn(50) * 2
    sat.plot_scatter_with_regression(x_scatter, y_scatter, x_label="Independent Variable", y_label="Dependent Variable",
                                     title="Scatter Plot with Linear Regression",
                                     save_path=os.path.join(OUTPUT_DIR, "scatter_with_regression"))

    # --- 中级模板 (Intermediate Templates) ---
    print("Generating: Academic style heatmap...")
    heatmap_data = np.random.rand(8, 10)
    sat.plot_heatmap(heatmap_data, x_labels=[f'X{i + 1}' for i in range(10)], y_labels=[f'Y{i + 1}' for i in range(8)],
                     title="Heatmap Example", save_path=os.path.join(OUTPUT_DIR, "heatmap"))

    print("Generating: Combined distribution plot...")
    dist_data = [np.random.normal(loc, 1, 30) for loc in [0, 2, 1.5]]
    sat.plot_distribution(dist_data, x_labels=['Control', 'Treat A', 'Treat B'],
                          title="Violin, Box, and Strip Plot Combo",
                          save_path=os.path.join(OUTPUT_DIR, "distribution_plot"))

    print("\n--- 2. Starting generation of advanced templates ---")

    # --- 高级模板 (第一批) (Advanced Templates - Part 1) ---
    print("Generating: Volcano plot...")
    volcano_df = pd.DataFrame({
        'gene': [f'Gene{i}' for i in range(200)],
        'log2FC': np.random.randn(200) * 2,
        'p_value': np.random.power(0.5, 200)
    })
    genes_to_label = volcano_df.sort_values('p_value').head(5)['gene'].tolist()
    sat.plot_volcano(volcano_df, 'log2FC', 'p_value', 'gene', genes_to_label=genes_to_label,
                     save_path=os.path.join(OUTPUT_DIR, "volcano_plot"))

    print("Generating: Sankey diagram...")
    sankey_flows = [
        ['Input A', 'Process 1', 5], ['Input B', 'Process 1', 10],
        ['Process 1', 'Process 2', 8], ['Process 1', 'Loss', 7],
        ['Process 2', 'Output C', 4], ['Process 2', 'Output D', 4]
    ]
    sat.plot_sankey(sankey_flows, title="Sankey Diagram Example", save_path=os.path.join(OUTPUT_DIR, "sankey_diagram"))

    print("Generating: Raincloud plot...")
    rain_data_df = pd.DataFrame({
        'Group': np.repeat(['A', 'B', 'C'], 50),
        'Value': np.concatenate(
            [np.random.normal(0, 1, 50), np.random.normal(2, 1.5, 50), np.random.normal(-1, 0.8, 50)])
    })
    sat.plot_raincloud(rain_data_df, x_labels='Group', y_label='Measurement', title='Raincloud Plot Example',
                       save_path=os.path.join(OUTPUT_DIR, "raincloud_plot"))

    print("Generating: Correlogram...")
    corr_df = pd.DataFrame(np.random.rand(10, 5), columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])
    corr_df['Var3'] += corr_df['Var1'] * 2
    corr_df['Var5'] -= corr_df['Var2'] * 1.5
    sat.plot_correlogram(corr_df, save_path=os.path.join(OUTPUT_DIR, "correlogram"))

    print("Generating: Upset plot...")
    upset_data_list = [
        {'name': f'item_{i}',
         'sets': np.random.choice(['Set1', 'Set2', 'Set3', 'Set4'], np.random.randint(1, 4), replace=False).tolist()}
        for i in range(100)
    ]
    upset_df = pd.DataFrame(upset_data_list)
    upset_data_from_df = upset_df.set_index('name')['sets'].str.join('|').str.get_dummies('|')
    sat.plot_upset(upset_data_from_df.groupby(list(upset_data_from_df.columns)).size(),
                   save_path=os.path.join(OUTPUT_DIR, "upset_plot"))

    print("Generating: Bland-Altman plot...")
    method1 = np.random.rand(50) * 20 + 5
    method2 = method1 + (np.random.rand(50) - 0.5) * 4
    sat.plot_bland_altman(method1, method2, save_path=os.path.join(OUTPUT_DIR, "bland_altman_plot"))

    print("Generating: Principal Component Analysis (PCA) plot...")
    pca_X = np.random.rand(60, 10)
    pca_groups = np.repeat(['Group A', 'Group B', 'Group C'], 20)
    sat.plot_pca(pca_X, pca_groups, save_path=os.path.join(OUTPUT_DIR, "pca_plot"))

    print("Generating: Forest plot...")
    forest_data = pd.DataFrame({
        'study': [f'Study {i}' for i in range(1, 6)],
        'odds_ratio': [1.2, 0.8, 2.1, 1.5, 0.9],
        'ci_low': [0.9, 0.6, 1.5, 1.1, 0.7],
        'ci_high': [1.6, 1.1, 2.9, 2.2, 1.2],
    })
    sat.plot_forest(forest_data, save_path=os.path.join(OUTPUT_DIR, "forest_plot"))

    print("Generating: Stacked bar plot...")
    stacked_bar_df = pd.DataFrame({
        'Category A': [20, 35, 30, 35],
        'Category B': [25, 32, 34, 20],
        'Category C': [15, 20, 22, 26]
    }, index=['Group 1', 'Group 2', 'Group 3', 'Group 4'])
    percent_df = stacked_bar_df.divide(stacked_bar_df.sum(axis=1), axis=0)
    data_values = percent_df.values
    x_labels_stacked = percent_df.index.tolist()
    category_labels_stacked = percent_df.columns.tolist()
    sat.plot_stacked_bar(data_values, x_labels=x_labels_stacked, category_labels=category_labels_stacked,
                         title="Stacked Bar Plot (Proportion)", save_path=os.path.join(OUTPUT_DIR, "stacked_bar_plot"))

    print("Generating: Clustermap...")
    clustermap_df = pd.DataFrame(np.random.randn(10, 10), index=[f'Gene_{i}' for i in range(10)],
                                 columns=[f'Sample_{i}' for i in range(10)])
    sat.plot_clustermap(clustermap_df, title="Clustermap Example",
                        save_path=os.path.join(OUTPUT_DIR, "clustermap_plot.svg"))

    # --- 顶级期刊专业图表模板 (第二批) (Top-tier Journal Templates - Part 2) ---
    print("Generating: Manhattan plot...")
    manhattan_data = []
    for i in range(1, 6):  # 5 chromosomes
        n_snps = np.random.randint(500, 1000)
        manhattan_data.append(pd.DataFrame({
            'CHR': i,
            'POS': np.sort(np.random.randint(1, 1000000, n_snps)),
            'P': np.random.uniform(0, 1, n_snps) ** 2
        }))
    manhattan_df = pd.concat(manhattan_data)
    sat.plot_manhattan(manhattan_df, 'CHR', 'POS', 'P', save_path=os.path.join(OUTPUT_DIR, "manhattan_plot"))

    print("Generating: Gene expression dot plot...")
    gene_dot_df = pd.DataFrame({
        'gene': np.tile(['GeneA', 'GeneB', 'GeneC', 'GeneD'], 3),
        'group': np.repeat(['Cluster1', 'Cluster2', 'Cluster3'], 4),
        'percent_expressed': np.random.rand(12) * 100,
        'avg_expression': np.random.rand(12) * 5
    })
    sat.plot_gene_dot_plot(gene_dot_df, save_path=os.path.join(OUTPUT_DIR, "gene_dot_plot"))

    print("Generating: MA plot...")
    ma_df = pd.DataFrame({
        'mean_expression': np.random.lognormal(mean=3, sigma=1, size=1000),
        'logfc': np.random.normal(loc=0, scale=1.5, size=1000),
    })
    ma_df['significant'] = (np.abs(ma_df['logfc']) > 2) & (ma_df['mean_expression'] > 10)
    sat.plot_ma(ma_df, 'logfc', 'mean_expression', 'significant', save_path=os.path.join(OUTPUT_DIR, "ma_plot"))

    print("Generating: Sequence logo...")
    pwm_df = pd.DataFrame(np.random.rand(4, 10), index=['A', 'C', 'G', 'T'])
    pwm_df = pwm_df.div(pwm_df.sum(axis=0), axis=1)  # Normalize
    sat.plot_sequence_logo(pwm_df, save_path=os.path.join(OUTPUT_DIR, "sequence_logo"))

    print("Generating: Circos-like plot...")
    circos_df = pd.DataFrame({
        'source_chr': ['chr1', 'chr2', 'chr3', 'chr1'],
        'source_pos': [0.2, 0.5, 0.8, 0.9],
        'target_chr': ['chr2', 'chr3', 'chr1', 'chr3'],
        'target_pos': [0.3, 0.6, 0.1, 0.4],
        'value': [0.8, 0.5, 1.0, 0.3]
    })
    sat.plot_circos_like(circos_df, save_path=os.path.join(OUTPUT_DIR, "circos_like_plot"))

    print("\n--- 3. Starting generation of Machine Learning & Statistics plots ---")

    # Prepare ML data
    y_true = np.random.randint(0, 2, 100)
    y_pred_prob = np.clip(y_true * 0.4 + np.random.rand(100), 0, 1)

    print("Generating: ROC curve...")
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    sat.plot_roc_curve([fpr], [tpr], labels=['Model A'], save_path=os.path.join(OUTPUT_DIR, "roc_curve"))

    print("Generating: Precision-Recall (PR) curve...")
    sat.plot_precision_recall_curve(y_true, y_pred_prob, save_path=os.path.join(OUTPUT_DIR, "pr_curve"))

    print("Generating: Confusion matrix...")
    from sklearn.metrics import confusion_matrix
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sat.plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'],
                              save_path=os.path.join(OUTPUT_DIR, "confusion_matrix"))

    print("Generating: Feature importance plot...")
    features = [f'Feature_{i}' for i in range(10)]
    importances = np.random.rand(10) * 10
    sat.plot_feature_importance(importances, features, save_path=os.path.join(OUTPUT_DIR, "feature_importance"))

    print("Generating: Dendrogram...")
    dendrogram_X = np.random.rand(15, 5)
    sat.plot_dendrogram(dendrogram_X, save_path=os.path.join(OUTPUT_DIR, "dendrogram"))

    print("\n--- 4. Starting generation of Clinical & Epidemiology plots ---")

    print("Generating: Survival curve...")
    n_samples = 100
    survival_df = pd.DataFrame({
        'time': np.random.exponential(scale=365, size=n_samples) + 30,
        'event': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
        'group': np.random.choice(['Control', 'Treatment'], size=n_samples)
    })
    sat.plot_survival_curve(survival_df, time_col='time', event_col='event', group_col='group',
                            save_path=os.path.join(OUTPUT_DIR, "survival_curve"))

    print("Generating: Funnel plot...")
    funnel_df = pd.DataFrame({
        'effect_size': np.random.normal(loc=0.5, scale=0.5, size=20),
        'std_error': np.random.uniform(0.05, 0.4, size=20)
    })
    sat.plot_funnel(funnel_df, effect_col='effect_size', se_col='std_error',
                    save_path=os.path.join(OUTPUT_DIR, "funnel_plot"))

    print("Generating: Spaghetti plot...")
    spaghetti_df = pd.DataFrame({
        'time': np.tile(np.arange(5), 10),
        'value': np.random.randn(50).cumsum() + np.repeat(np.random.randn(10) * 5, 5),
        'id': np.repeat(np.arange(10), 5)
    })
    sat.plot_spaghetti(spaghetti_df, 'time', 'value', 'id', save_path=os.path.join(OUTPUT_DIR, "spaghetti_plot"))

    print("\n--- 5. Starting generation of Data Exploration & Composition plots ---")

    print("Generating: Paired dumbbell plot...")
    dumbbell_df = pd.DataFrame({
        'Patient': [f'P{i}' for i in range(10)],
        'Before': np.random.rand(10) * 10,
        'After': np.random.rand(10) * 10 + 1
    })
    sat.plot_paired_dumbbell(dumbbell_df, 'Before', 'After', 'Patient',
                             save_path=os.path.join(OUTPUT_DIR, "dumbbell_plot"))

    print("Generating: Ridge plot...")
    ridge_df = pd.DataFrame({
        'Value': np.concatenate([np.random.normal(loc=i, size=100) for i in range(5)]),
        'Group': np.repeat([f'Group {chr(65 + i)}' for i in range(5)], 100)
    })
    sat.plot_ridge_plot(ridge_df, 'Value', 'Group', save_path=os.path.join(OUTPUT_DIR, "ridge_plot"))

    print("Generating: Waffle chart...")
    waffle_data = {'Category A': 45, 'Category B': 30, 'Category C': 15, 'Category D': 10}
    sat.plot_waffle(waffle_data, save_path=os.path.join(OUTPUT_DIR, "waffle_chart"))

    print("Generating: Treemap...")
    treemap_sizes = [500, 300, 150, 50]
    treemap_labels = ['Area A', 'Area B', 'Area C', 'Area D']
    sat.plot_treemap(treemap_sizes, treemap_labels, save_path=os.path.join(OUTPUT_DIR, "treemap"))

    print("Generating: Parallel coordinates plot...")
    from sklearn.datasets import load_iris
    iris = load_iris()
    parallel_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    parallel_df['species'] = iris.target_names[iris.target]
    sat.plot_parallel_coordinates(parallel_df, 'species', save_path=os.path.join(OUTPUT_DIR, "parallel_coordinates"))

    print("\n--- 6. Starting generation of Professional & Composite plots ---")

    print("Generating: Calendar heatmap...")
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(366)]
    calendar_data = pd.Series(np.random.randint(0, 20, size=366), index=dates)
    sat.plot_calendar_heatmap(calendar_data, save_path=os.path.join(OUTPUT_DIR, "calendar_heatmap"))

    print("Generating: Gantt chart...")
    gantt_df = pd.DataFrame({
        'Task': ['Task A', 'Task B', 'Task C'],
        'Start': ['2024-01-01', '2024-01-15', '2024-02-01'],
        'Finish': ['2024-02-10', '2024-03-05', '2024-03-20'],
        'Color': ['#E63946', '#457B9D', '#F4A261']
    })
    sat.plot_gantt(gantt_df, save_path=os.path.join(OUTPUT_DIR, "gantt_chart"))

    print("\n--- 7. Starting generation of supplementary advanced templates ---")

    print("Generating: PCA biplot...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_X)
    sample_labels = pd.DataFrame({'group': pca_groups})
    sat.plot_pca_biplot(pca_result, sample_labels=sample_labels, group_col='group',
                        save_path=os.path.join(OUTPUT_DIR, "pca_biplot"))

    print("Generating: Violin + Box plot combo...")
    vbox_data = [np.random.normal(loc, scale, 100) for loc, scale in zip([0, 3, 1], [1, 1.5, 0.8])]
    sat.plot_violin_box(vbox_data, x_labels=['C1', 'C2', 'C3'], save_path=os.path.join(OUTPUT_DIR, "violin_box"))

    print("Generating: Bar-Line dual-axis plot...")
    y1_data = np.random.randint(100, 500, 6)
    y2_data = np.random.rand(6) * 10 + 20
    x_labels_dual = [f'Month-{i}' for i in range(1, 7)]
    sat.plot_bar_line_dual(y1_data, y2_data, x_labels_dual, 'Sales', 'Temperature (°C)',
                           save_path=os.path.join(OUTPUT_DIR, "bar_line_dual"))

    print("Generating: Bubble plot...")
    bubble_df = pd.DataFrame({
        'GDP': np.random.rand(10) * 1000,
        'Life Expectancy': np.random.rand(10) * 30 + 50,
        'Population': np.random.rand(10) * 1e6,
        'Continent': np.random.choice(['Asia', 'Europe', 'Africa'], 10)
    })
    sat.plot_bubble(bubble_df, 'GDP', 'Life Expectancy', 'Population', 'Continent',
                    save_path=os.path.join(OUTPUT_DIR, "bubble_plot"))

    print("Generating: Density heatmap...")
    x_dense = np.random.randn(10000)
    y_dense = x_dense * 0.5 + np.random.randn(10000)
    sat.plot_density_heatmap(x_dense, y_dense, save_path=os.path.join(OUTPUT_DIR, "density_heatmap"))

    print("Generating: Line plot with shaded error...")
    x_line_shaded = np.linspace(0, 10, 50)
    line_shaded_dict = {
        'Model A': (np.sin(x_line_shaded), np.random.rand(50) * 0.2 + 0.1),
        'Model B': (np.cos(x_line_shaded), np.random.rand(50) * 0.2 + 0.1)
    }
    sat.plot_line_with_shaded(line_shaded_dict, x_line_shaded, "Time", "Signal",
                              save_path=os.path.join(OUTPUT_DIR, "line_with_shaded_error"))

    print("Generating: Radar chart...")
    radar_categories = ['Speed', 'Reliability', 'Comfort', 'Safety', 'Efficiency']
    radar_data = {
        'Car A': np.random.randint(1, 10, 5),
        'Car B': np.random.randint(1, 10, 5)
    }
    sat.plot_radar_chart(radar_categories, radar_data, save_path=os.path.join(OUTPUT_DIR, "radar_chart"))

    print("\n--- 8. Starting generation of the ultimate multi-panel figure template ---")

    def panel_a(ax):
        """一个简单的散点图作为子图A"""
        # A simple scatter plot for panel A
        ax.scatter(np.random.rand(20), np.random.rand(20))
        ax.set_title("Panel A: Scatter")

    def panel_b(ax):
        """一个简单的柱状图作为子图B"""
        # A simple bar plot for panel B
        ax.bar(['X', 'Y', 'Z'], np.random.randint(3, 10, 3))
        ax.set_title("Panel B: Bar")

    print("Generating: Multi-panel figure...")
    sat.create_multi_panel_figure(
        layout=(1, 2),
        plot_functions=[panel_a, panel_b],
        save_path=os.path.join(OUTPUT_DIR, "multi_panel_figure")
    )

    print(f"\n✅ All example plots have been successfully generated and saved to the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    main()