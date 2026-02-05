import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 环境与绘图配置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class IntegratedEmpiricalAnalyzer:
    def __init__(self, output_dir: str = './RQ2_AllInOne'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)

    def _find_optimal_k(self, data: np.ndarray, dim_name: str) -> int:
        """自动肘部法：寻找曲线拐点"""
        distortions = []
        K_range = range(2, 10)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            distortions.append(km.inertia_)

        p1 = np.array([K_range[0], distortions[0], 0])
        p2 = np.array([K_range[-1], distortions[-1], 0])
        distances = []
        for i in range(len(K_range)):
            p3 = np.array([K_range[i], distortions[i], 0])
            d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
            distances.append(d)

        best_k = K_range[np.argmax(distances)]

        plt.figure(figsize=(8, 4))
        plt.plot(K_range, distortions, 'bo-')
        plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal K={best_k}')
        plt.title(f"Elbow Method: {dim_name}")
        plt.savefig(f"{self.output_dir}/elbow_{dim_name.lower()}.png")
        plt.close()
        return best_k

    def _remove_redundant_features(self, df: pd.DataFrame, threshold=0.85) -> pd.DataFrame:
        """变量选择过程：移除高可预测性的变量"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if to_drop:
            print(f"移除冗余变量 (相关性 > {threshold}): {to_drop}")
        return df.drop(columns=to_drop)

    def run_PCA_analysis(self):
        """执行全变量 PCA + 聚类"""
        csv_files = glob.glob("./features/*.features.csv")
        if not csv_files:
            print("未发现特征 CSV，请先运行特征提取。")
            return

        # 读取所有数据
        full_df = pd.concat([pd.read_csv(f, index_col=0) for f in csv_files], ignore_index=True)

        # 定义所有特征
        dim1_cols = ['Length', 'SentCount', 'ContentRatio', 'Redundancy', 'IDF', 'TTR', 'Entropy']
        dim2_cols = ['ICR', 'ParentOverlap', 'MethodOverlap', 'RepoOverlap', 'PathOverlap', 'CodeOverlap',
                     'SemanticSim']
        all_features = dim1_cols + dim2_cols

        # 为了处理偏态分布，建议对 ICR 取对数
        if 'ICR' in full_df.columns:
            full_df['ICR'] = np.log1p(full_df['ICR'])

        print("\n" + "=" * 50)
        print("开始执行：全变量单次 PCA + K-Means 聚类 (All-In-One PCA)")
        print("=" * 50)

        # ==========================================================
        # 1. 特征预处理
        # ==========================================================
        print("\n>>> 特征选择与标准化...")
        df_selected = self._remove_redundant_features(full_df[all_features])
        X_scaled = MinMaxScaler().fit_transform(df_selected)

        print(f"原始特征数: {len(all_features)}, 去冗余后特征数: {df_selected.shape[1]}")

        # ==========================================================
        # 2. PCA 降维 (计算所有主成分)
        # ==========================================================
        print("\n>>> 执行 PCA 降维...")
        # 修改点：n_components=None 表示保留所有成分，直到方差解释度为 100%
        pca = PCA(n_components=None)
        X_pca = pca.fit_transform(X_scaled)

        # 将前两个主成分保存到 full_df 用于可视化
        full_df['PC1'] = X_pca[:, 0]
        full_df['PC2'] = X_pca[:, 1]

        # 打印解释方差信息
        print(f"PC1 解释方差: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"PC2 解释方差: {pca.explained_variance_ratio_[1]:.2%}")
        print(f"总解释方差 (所有PC): {sum(pca.explained_variance_ratio_):.2%}")

        # --- 保存 Loadings 表 (修改了顺序和完整性) ---
        # 1. 创建 Loadings 数据帧
        n_pcs = pca.n_components_
        pc_columns = [f'PC{i + 1}' for i in range(n_pcs)]  # PC1, PC2, ..., PCn

        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=pc_columns,
            index=df_selected.columns
        )

        # 2. 创建方差统计行 (Variance Stats)
        variance_stats = pd.DataFrame(
            [pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)],
            index=['Prop. of Variance', 'Cumulative Prop.'],
            columns=pc_columns
        )

        # 3. 将方差统计行放在最前面 (Top)
        final_output_df = pd.concat([variance_stats, loadings_df])

        # 保存 CSV
        final_output_df.to_csv(f"{self.output_dir}/pca_loadings_full.csv")
        print(f"PCA Loadings 已保存 (包含累积方差)，路径: {self.output_dir}/pca_loadings_full.csv")

        # ==========================================================
        # 3. K-Means 聚类
        # ==========================================================
        print("\n>>> 执行 K-Means 聚类...")
        # 注意：这里使用全维度的 X_pca 进行聚类，包含了所有信息
        best_k = self._find_optimal_k(X_pca, "All_Features_Space")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        full_df['Cluster'] = kmeans.fit_predict(X_pca)

        print(f"聚类完成，最佳簇数 K={best_k}")

        # ==========================================================
        # 4. 可视化 (仅展示前两个主成分)
        # ==========================================================
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=full_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.6, s=60)

        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.title(f"PCA Clustering (K={best_k}, Visualized on PC1-PC2)")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pca_clustering_scatter.png")
        plt.close()

        # ==========================================================
        # 5. 详细统计
        # ==========================================================
        print("正在计算详细簇统计特征...")
        counts = full_df['Cluster'].value_counts().sort_index()
        ratios = (counts / len(full_df)) * 100
        dist_df = pd.DataFrame({'Count': counts, 'Ratio(%)': ratios})

        grouped = full_df.groupby('Cluster')[all_features]
        means = grouped.mean()
        medians = grouped.median()
        stats_summary = grouped.agg(['mean', 'median'])
        stats_summary.to_csv(f"{self.output_dir}/cluster_statistics_summary.csv")

        print("\n[Cluster Size Distribution]:")
        print(dist_df)

        # ==========================================================
        # 6. 判定黄金簇
        # ==========================================================
        scaler = MinMaxScaler()
        norm_means = pd.DataFrame(scaler.fit_transform(means), columns=all_features, index=means.index)
        norm_means.to_csv(f"{self.output_dir}/cluster_normalized_scores.csv")

        # 启发式打分：SemanticSim 高 + Length 低
        score_series = norm_means['SemanticSim'] + (1 - norm_means['Length'])
        best_id = score_series.idxmax()

        print(f"\n>>> 判定黄金簇为: Cluster {best_id}")

        # 热力图
        plt.figure(figsize=(15, 12))
        sns.heatmap(norm_means.drop(columns=['Entropy'], errors='ignore'), annot=True, cmap='YlGnBu', fmt=".2f")
        plt.title(f"Feature Heatmap - Best: Cluster {best_id}")
        plt.savefig(f"{self.output_dir}/feature_heatmap.png")
        plt.close()

        # ==========================================================
        # 7. 导出最优簇数据
        # ==========================================================
        print("\n" + "=" * 50)
        print(f"正在导出最优簇 (Cluster {best_id}) 的数据...")
        golden_df = full_df[full_df['Cluster'] == best_id].copy()
        save_path = f"{self.output_dir}/golden_cluster_{best_id}_data.csv"
        golden_df.to_csv(save_path, index=True)
        print(f"导出成功！文件路径: {save_path}")
        print(f"包含数据行数: {len(golden_df)}")
        print("=" * 50 + "\n")


def main():
    output_dir = './RQ2_AllInOne'
    existing_csvs = glob.glob(os.path.join('./features', "*.features.csv"))
    analyzer = IntegratedEmpiricalAnalyzer(output_dir=output_dir)
    print(f"输出目录中已有 {len(existing_csvs)} 个特征 CSV 文件")
    analyzer.run_PCA_analysis()
    print("PCA聚类分析完毕")


if __name__ == "__main__":
    main()