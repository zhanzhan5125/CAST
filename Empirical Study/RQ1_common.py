import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st  # 新增：用于统计分布拟合
import warnings  # 新增：用于忽略拟合过程中的警告
from typing import List, Optional

# --- 全局绘图配置 ---
sns.set_theme(style="whitegrid")

ACADEMIC_COLORS = {
    'primary': '#7AACD2',  # 之前的浅蓝
    'accent': '#EE934F',  # 橙色
    'auxiliary': '#87C27E',  # 绿色
    'gray': '#666666',
    'dark_text': '#333333'
}

# --- 从图片中提取的专用饼图色板 ---
PIE_STYLE_COLORS = [
    '#406990', '#EF9D6E', '#9688BA', '#8FD6D6', '#E6C5D6', '#C4CACE', '#D3D3D3'
]

# 统一 Matplotlib 参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 15,
    'figure.figsize': (8, 6),
    'axes.unicode_minus': False,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42
})


class CommonEmpiricalAnalyzer:
    def __init__(self, input_dir: str = './features', output_dir: str = './RQ1'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 定义哪些特征应该使用 auxiliary (绿色)
        self.extrinsic_features = [
            'ICR', 'CodeOverlap', 'MethodOverlap',
            'ParentOverlap', 'PathOverlap', 'RepoOverlap', 'SemanticSim'
        ]

        # 新增：定义要尝试拟合的常见分布列表
        # 包含了 SE 领域常见的：正态、对数正态、指数、伽马、帕累托(长尾)、Weibull
        self.distributions_to_check = [
            st.norm, st.lognorm, st.expon, st.gamma, st.beta,
            st.pareto, st.weibull_min
        ]

    def _save_figure(self, filename_base: str):
        """同时保存 PDF (矢量) 和 PNG (位图)"""
        plt.savefig(os.path.join(self.output_dir, f'{filename_base}.pdf'), format='pdf')
        # plt.savefig(os.path.join(self.output_dir, f'{filename_base}.png'), dpi=300)
        plt.close()
        print(f"  - Saved plots for: {filename_base}")

    def save_correlation_table(self, df: pd.DataFrame):
        """计算相关性矩阵并绘制热力图"""
        cols = ['Length', 'SentCount', 'TTR', 'IDF', 'Entropy', 'Redundancy',
                'ContentRatio', 'ICR', 'CodeOverlap', 'MethodOverlap',
                'ParentOverlap', 'PathOverlap', 'RepoOverlap', 'SemanticSim']

        valid_cols = [c for c in df.columns if c in cols]
        if not valid_cols:
            return

        corr_matrix = df[valid_cols].corr(method='spearman')
        csv_path = os.path.join(self.output_dir, 'Correlation_Matrix.csv')
        corr_matrix.to_csv(csv_path, float_format='%.2f')

        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=.5, cbar_kws={"shrink": .8}
        )
        plt.title('Feature Correlation Matrix (Spearman)', pad=20)
        self._save_figure('Correlation_Heatmap')

    def _plot_histogram(self, data: pd.Series, col_name: str,
                        bins=30, x_range=None, x_label=None):
        """绘制直方图 (左轴 Count) 和 核密度估计 (右轴 Density)"""
        plot_data = data.dropna()
        filter_info = ""

        if x_range:
            mask = (plot_data >= x_range[0]) & (plot_data <= x_range[1])
            subset = plot_data[mask]
            filter_str = f"Range: {x_range[0]}-{x_range[1]}"
            filter_info = f"  [Filter {filter_str}] Count: {len(subset)} ({len(subset) / len(plot_data):.1%})"
            plot_data = subset

        if col_name in self.extrinsic_features:
            bar_color = ACADEMIC_COLORS['auxiliary']
        else:
            bar_color = ACADEMIC_COLORS['primary']

        fig, ax1 = plt.subplots()

        sns.histplot(
            plot_data, bins=bins, kde=False, stat="count", ax=ax1,
            color=bar_color, edgecolor='white', linewidth=0.5, alpha=1, label='Frequency'
        )

        ax1.set_xlabel(x_label if x_label else col_name)
        ax1.set_ylabel('Frequency (Count)')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        sns.kdeplot(
            plot_data, ax=ax2, color=ACADEMIC_COLORS['accent'],
            linewidth=2.5, label='Density'
        )

        ax2.set_ylabel('Density (KDE)')
        ax2.tick_params(axis='y')
        ax2.grid(False)

        if x_range and (x_range[1] - x_range[0] < 20):
            ax1.set_xticks(range(x_range[0], x_range[1] + 1))

        plt.title(f'Distribution of {col_name}')
        safe_name = str(col_name).replace('/', '_')
        self._save_figure(f'{safe_name}_distribution')

        return filter_info

    def _plot_pie_chart(self, data: pd.Series, col_name: str):
        """饼图绘制 - 优化版"""
        counts = data.value_counts()
        total = counts.sum()

        top_n = 6
        if len(counts) > top_n:
            main_counts = counts[:top_n]
            others_val = counts[top_n:].sum()
            plot_data = main_counts.copy()
            plot_data['Others'] = others_val
        else:
            plot_data = counts

        colors = PIE_STYLE_COLORS[:len(plot_data)]
        if len(plot_data) > len(colors):
            colors.extend(['#D3D3D3'] * (len(plot_data) - len(colors)))
        if plot_data.index[-1] == 'Others':
            colors[-1] = PIE_STYLE_COLORS[5]

        def smart_autopct(pct):
            return ('%1.1f%%' % pct) if pct > 3 else ''

        fig, ax = plt.subplots(figsize=(10, 6))

        wedges, texts, autotexts = ax.pie(
            plot_data, labels=None, autopct=smart_autopct,
            startangle=90, counterclock=False, colors=colors,
            pctdistance=0.75, explode=[0.03] * len(plot_data),
            textprops={'fontsize': 12}
        )

        for i, autotext in enumerate(autotexts):
            autotext.set_weight("bold")
            if i == 0:
                autotext.set_color("white")
            else:
                autotext.set_color("#333333")

        legend_labels = [f'{label} ({val / total:.1%})' for label, val in zip(plot_data.index, plot_data.values)]

        ax.legend(
            wedges, legend_labels, title="POS Tags",
            loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
            frameon=False, fontsize=12
        )

        plt.title(f'Top {min(len(counts), top_n)} POS Tags Distribution', fontsize=16)
        plt.tight_layout()
        self._save_figure(f'{col_name}_distribution')

        top_3 = [f"{idx}({val / total:.1%})" for idx, val in counts.head(3).items()]
        return f"  Top POS Distribution: {', '.join(top_3)}"

    # --- 新增功能：拟合最佳分布 ---
    def _fit_best_distribution(self, data: pd.Series) -> str:
        """
        尝试将数据拟合到多种分布，并使用 Kolmogorov-Smirnov (KS) 测试找出最佳拟合。
        返回格式化的报告字符串。
        """
        # 1. 数据清洗（去除 NaN 和 无穷大）
        clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # 如果数据量太少或都是同一个值，跳过
        if len(clean_data) < 10 or clean_data.nunique() <= 1:
            return "  Distribution Fit: Not enough data or constant value."

        results = []

        # 忽略拟合过程中的 RuntimeWarning (例如 divide by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for dist in self.distributions_to_check:
                dist_name = dist.name
                try:
                    # 1. 拟合分布，获取参数
                    # 某些分布拟合可能很慢或失败，try-catch 保护
                    params = dist.fit(clean_data)

                    # 2. 执行 KS 测试 (Goodness of Fit)
                    # statistic (D) 越小越好; p-value 越大越表示“不能拒绝是该分布的假设”
                    D, p_value = st.kstest(clean_data, dist_name, args=params)

                    results.append({
                        'name': dist_name,
                        'D': D,
                        'p': p_value,
                        'params': params
                    })
                except Exception:
                    continue

        # 3. 按 KS 统计量 D 从小到大排序 (最好的排前面)
        results.sort(key=lambda x: x['D'])

        # 4. 构造输出报告
        if not results:
            return "  Distribution Fit: Failed to fit any distribution."

        report_str = "  [Distribution Fitting Analysis (Top 3)]\n"
        for i, res in enumerate(results[:3]):
            # 格式化参数，保留2位小数
            param_str = ", ".join([f"{p:.2f}" for p in res['params']])
            # 标记最佳
            prefix = ">> " if i == 0 else "   "
            report_str += (f"  {prefix}{i + 1}. {res['name']:<10} | KS-Stat (D): {res['D']:.4f} "
                           f"| p-value: {res['p']:.2e} | Params: ({param_str})\n")

        # 增加一行简单的结论
        best_dist = results[0]
        if best_dist['p'] > 0.05:
            verdict = f"Likely follows {best_dist['name']} distribution (p>0.05)"
        else:
            verdict = f"Best approx is {best_dist['name']}, but strictly rejects null hypothesis (p<0.05)"
        report_str += f"  -> Conclusion: {verdict}"

        return report_str

    def run_analysis(self):
        csv_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        if not csv_files:
            print(f"Warning: No .csv files found in {self.input_dir}")
            return

        df_list = []
        print("Loading data...")
        for file in csv_files:
            try:
                df = pd.read_csv(file, index_col=0)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not df_list:
            return

        all_data = pd.concat(df_list, ignore_index=True)
        print(f"Total records loaded: {len(all_data)}")

        report_lines = [
            "====== Empirical Analysis Report ======",
            f"Total Samples: {len(all_data)}",
            "=======================================\n"
        ]

        cols_to_analyze = all_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'Start' in all_data.columns:
            cols_to_analyze.append('Start')

        for col in cols_to_analyze:
            print(f"Processing feature: {col}...")
            desc = all_data[col].describe()
            report_lines.append(f"Feature: {col}")

            if col == 'Start':
                report_lines.append(f"  Count : {desc['count']}")
                report_lines.append(f"  Unique: {desc['unique']}")
                report_lines.append(f"  Top   : {desc['top']} (Freq: {desc['freq']})")
                info = self._plot_pie_chart(all_data[col], col)
                report_lines.append(info)
            else:
                report_lines.append(f"  Mean: {desc['mean']:.4f} | Std: {desc['std']:.4f}")
                report_lines.append(f"  Min : {desc['min']:.4f} | Max: {desc['max']:.4f}")

                # --- 绘图 ---
                filter_info = ""
                if col == 'Length':
                    filter_info = self._plot_histogram(
                        all_data[col], col, bins=range(3, 52), x_range=(3, 50), x_label='Length (Tokens)'
                    )
                elif col == 'ICR':
                    filter_info = self._plot_histogram(
                        all_data[col], col, bins=range(0, 36), x_range=(0, 35), x_label='ICR Value'
                    )
                else:
                    self._plot_histogram(all_data[col], col)

                if filter_info:
                    report_lines.append(filter_info)

                # --- 新增调用：分布拟合分析 ---
                print(f"  -> Fitting distributions for {col}...")
                fit_report = self._fit_best_distribution(all_data[col])
                report_lines.append(fit_report)

            report_lines.append("")

        print("Running Multi-dimensional Analysis...")
        self.save_correlation_table(all_data)

        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"Analysis finished. Report saved to {report_path}")
        print(f"Plots saved to {self.output_dir} (PDF & PNG)")


def main():
    analyzer = CommonEmpiricalAnalyzer(input_dir='./features', output_dir='./RQ1')
    analyzer.run_analysis()


if __name__ == "__main__":
    main()