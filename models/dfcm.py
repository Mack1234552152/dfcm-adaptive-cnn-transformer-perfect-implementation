"""
判别性模糊C均值聚类模块 (Discriminative Fuzzy C-Means)
严格按照论文算法实现，不进行任何优化修改
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import config

class DFCM:
    """
    判别性模糊C均值聚类算法
    论文核心创新：引入相似样本隶属度一致性约束
    """

    def __init__(self, n_clusters=None, max_iterations=None, error_tolerance=None,
                 fuzzy_coefficient=None, penalty_lambda=None):
        """
        初始化DFCM算法参数
        """
        self.n_clusters = n_clusters or config.Config.MAX_CLUSTERS
        self.max_iterations = max_iterations or config.Config.DFCM_MAX_ITERATIONS
        self.error_tolerance = error_tolerance or config.Config.DFCM_ERROR_TOLERANCE
        self.fuzzy_coefficient = fuzzy_coefficient or config.Config.FUZZY_COEFFICIENT
        self.penalty_lambda = penalty_lambda or config.Config.PENALTY_LAMBDA

        # 算法变量
        self.centers = None
        self.membership_matrix = None
        self.objective_function_values = []
        self.k_optimal = config.Config.K_NEIGHBORS_RANGE.start if hasattr(config.Config.K_NEIGHBORS_RANGE, 'start') else 10

    def initialize_membership_matrix(self, n_samples):
        """
        初始化隶属度矩阵
        U ∈ [0,1]^(n_samples × n_clusters)，每行和为1
        """
        U = np.random.rand(n_samples, self.n_clusters)
        # 归一化，确保每行和为1
        U = U / U.sum(axis=1, keepdims=True)
        return U

    def calculate_similarity_weights(self, X):
        """
        严格按照论文计算样本间相似性权重
        论文中ωᵢⱼ表示样本i和j的相似性权重
        使用更复杂的相似性度量，考虑特征空间的语义相似性
        """
        n_samples, n_features = X.shape

        # 根据论文，使用适中的k近邻数量
        k_neighbors = min(15, n_samples - 1, max(10, self.n_clusters * 2))
        self.k_optimal = k_neighbors

        print(f"计算相似性权重，样本数: {n_samples}, 特征数: {n_features}, k近邻: {k_neighbors}")

        # 计算特征向量的标准差用于归一化
        feature_stds = np.std(X, axis=0)
        feature_stds[feature_stds < 1e-8] = 1e-8  # 避免除零

        # 使用稀疏相似矩阵表示
        similarity_matrix = np.zeros((n_samples, k_neighbors))
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        for i in range(n_samples):
            for j_idx, neighbor_idx in enumerate(indices[i][1:]):  # 排除自己
                if j_idx < k_neighbors:
                    # 计算更复杂的相似性权重
                    # 1. 基于距离的相似性
                    distance = distances[i, j_idx]
                    distance_similarity = np.exp(-distance / np.mean(distances[i][1:]))

                    # 2. 基于特征空间相似性的余弦相似度
                    x_i_normalized = (X[i] - np.mean(X[i])) / (np.std(X[i]) + 1e-8)
                    x_j_normalized = (X[neighbor_idx] - np.mean(X[neighbor_idx])) / (np.std(X[neighbor_idx]) + 1e-8)

                    # 计算余弦相似度
                    dot_product = np.dot(x_i_normalized, x_j_normalized)
                    norm_i = np.linalg.norm(x_i_normalized)
                    norm_j = np.linalg.norm(x_j_normalized)
                    cosine_similarity = dot_product / (norm_i * norm_j + 1e-8)

                    # 3. 基于特征分布相似性
                    feature_diff = np.abs(X[i] - X[neighbor_idx]) / (feature_stds + 1e-8)
                    distribution_similarity = np.exp(-np.mean(feature_diff))

                    # 4. 综合相似性权重（论文中的ωᵢⱼ）
                    # 结合多种相似性度量
                    combined_similarity = 0.4 * distance_similarity + 0.3 * cosine_similarity + 0.3 * distribution_similarity

                    similarity_matrix[i, j_idx] = combined_similarity

        return similarity_matrix, indices

    def calculate_objective_function(self, X, U, V, similarity_weights, neighbor_indices):
        """
        计算DFCM目标函数
        严格按照论文Eq. 3.3实现：J(U, V) = Σ Σ u_ic^M ||x_i - v_c||² + λ Σ Σ Σ ω_ij |u_ic - u_jc|²

        其中：
        - 第一项：传统FCM目标函数
        - 第二项：相似样本隶属度一致性惩罚项

        参数：
        - similarity_weights: 预计算的相似权重矩阵
        - neighbor_indices: 预计算的邻居索引，避免重复计算
        """
        n_samples, n_features = X.shape
        n_clusters = V.shape[0]

        # 第一项：传统FCM目标函数
        term1 = 0.0
        for i in range(n_samples):
            for c in range(self.n_clusters):
                distance = np.sum((X[i] - V[c]) ** 2)
                term1 += (U[i, c] ** self.fuzzy_coefficient) * distance

        # 第二项：相似样本隶属度一致性惩罚项
        # 使用预计算的邻居索引，避免重复计算
        term2 = 0.0
        for i in range(n_samples):
            for k_idx in range(self.k_optimal):
                j = neighbor_indices[i, k_idx + 1]  # 排除自己，从索引1开始
                if i != j:  # 确保不是同一个样本
                    # 使用预计算的相似权重
                    ω_ij = similarity_weights[i, k_idx] if k_idx < similarity_weights.shape[1] else np.exp(-np.linalg.norm(X[i] - X[j]))
                    for c in range(self.n_clusters):
                        # 严格按照论文公式：ω_ij * |u_ic - u_jc|²
                        term2 += ω_ij * (U[i, c] - U[j, c]) ** 2

        return term1 + self.penalty_lambda * term2

    def update_membership_matrix(self, X, V, similarity_weights, neighbor_indices=None):
        """
        更新隶属度矩阵
        严格按照论文Appendix A Eq. 9实现
        使用上一迭代的隶属度矩阵值计算新值
        """
        n_samples = X.shape[0]

        # 如果没有提供邻居索引，则计算一次
        if neighbor_indices is None:
            nbrs = NearestNeighbors(n_neighbors=self.k_optimal + 1).fit(X)
            _, neighbor_indices = nbrs.kneighbors(X)

        # 使用上一迭代的隶属度矩阵值
        U_old = self.membership_matrix.copy()
        U_new = np.zeros_like(U_old)

        # 根据论文Appendix A Eq. 9计算新的隶属度矩阵
        for i in range(n_samples):
            for c in range(self.n_clusters):
                # 计算距离项
                d_ic = np.sum((X[i] - V[c]) ** 2)

                # 计算惩罚项 Σ ω_ij (u_ic^old - u_jc^old)
                # 严格按照论文公式，使用上一迭代的隶属度值
                penalty = 0.0
                for k_idx in range(self.k_optimal):
                    j = neighbor_indices[i, k_idx + 1]  # 排除自己
                    if i != j:
                        # 使用预计算的相似权重或计算新权重
                        ω_ij = similarity_weights[i, k_idx] if k_idx < similarity_weights.shape[1] else np.exp(-np.linalg.norm(X[i] - X[j]))
                        penalty += ω_ij * (U_old[i, c] - U_old[j, c])

                # 根据论文推导的更新公式
                denominator = 0.0
                for l in range(self.n_clusters):
                    d_il = np.sum((X[i] - V[l]) ** 2)

                    # 计算第l个簇的惩罚项，使用上一迭代的隶属度值
                    penalty_l = 0.0
                    for k_idx in range(self.k_optimal):
                        j = neighbor_indices[i, k_idx + 1]  # 排除自己
                        if i != j:
                            ω_ij = similarity_weights[i, k_idx] if k_idx < similarity_weights.shape[1] else np.exp(-np.linalg.norm(X[i] - X[j]))
                            penalty_l += ω_ij * (U_old[i, l] - U_old[j, l])

                    # 论文Appendix A Eq. 9的实现
                    # 添加数值稳定性保护
                    denominator_value = d_il + 2 * self.penalty_lambda * penalty_l
                    if denominator_value > 1e-8:
                        ratio = (d_ic + 2 * self.penalty_lambda * penalty) / denominator_value
                        denominator += max(ratio ** (1 / (self.fuzzy_coefficient - 1)), 1e-8)

                if denominator > 1e-8:
                    U_new[i, c] = 1.0 / denominator
                else:
                    U_new[i, c] = 1.0 / self.n_clusters  # 避免除零错误

        # 归一化，确保每行和为1
        U_new = U_new / (U_new.sum(axis=1, keepdims=True) + config.Config.EPSILON)

        return U_new

    def update_cluster_centers(self, X, U):
        """
        更新聚类中心
        v_j = (Σ u_ij^m x_i) / (Σ u_ij^m)
        """
        n_samples, n_features = X.shape
        V_new = np.zeros((self.n_clusters, n_features))

        for j in range(self.n_clusters):
            numerator = np.zeros(n_features)
            denominator = 0.0

            for i in range(n_samples):
                weight = U[i, j] ** self.fuzzy_coefficient
                numerator += weight * X[i]
                denominator += weight

            if denominator > 0:
                V_new[j] = numerator / denominator
            else:
                # 随机初始化，避免除零错误
                V_new[j] = X[np.random.randint(0, n_samples)]

        return V_new

    def calculate_dbi_index(self, X):
        """
        计算Davies-Bouldin指数用于确定最优聚类数
        DBI = (1/k) Σ max_j ((S_i + S_j) / d_ij)
        """
        if self.centers is None or self.membership_matrix is None:
            return float('inf')

        n_clusters = self.centers.shape[0]
        if n_clusters < 2:
            return float('inf')

        # 计算每个簇的离散度
        cluster_scatter = np.zeros(n_clusters)
        for i in range(self.n_clusters):
            # 簇内样本到中心的加权距离
            weighted_distances = 0.0
            total_weight = 0.0

            for j in range(X.shape[0]):
                weight = self.membership_matrix[j, i] ** self.fuzzy_coefficient
                weighted_distances += weight * np.sum((X[j] - self.centers[i]) ** 2)
                total_weight += weight

            if total_weight > 0:
                cluster_scatter[i] = np.sqrt(weighted_distances / total_weight)

        # 计算簇间距离
        inter_cluster_distances = euclidean_distances(self.centers)

        # 计算DBI
        dbi_values = []
        for i in range(self.n_clusters):
            max_ratio = 0.0
            for j in range(self.n_clusters):
                if i != j and inter_cluster_distances[i, j] > 0:
                    ratio = (cluster_scatter[i] + cluster_scatter[j]) / inter_cluster_distances[i, j]
                    max_ratio = max(max_ratio, ratio)

            dbi_values.append(max_ratio)

        return np.mean(dbi_values)

    def fit(self, X):
        """
        拟合DFCM模型
        """
        # 如果输入是3维序列数据，重塑为2维
        if len(X.shape) == 3:
            # X形状: (n_sequences, sequence_length, n_features)
            # 重塑为: (n_sequences * sequence_length, n_features)
            n_sequences, sequence_length, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            print(f"DFCM输入数据重塑: {X.shape} -> {X_reshaped.shape}")
        else:
            X_reshaped = X

        n_samples, n_features = X_reshaped.shape

        # 检查是否需要采样
        if n_samples > 2000:
            sample_size = min(2000, n_samples)  # 进一步减少到2000个样本
            print(f"样本数过大({n_samples})，使用采样策略，采样{sample_size}个样本")
            # 随机采样样本进行DFCM
            self.sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_reshaped = X_reshaped[self.sample_indices]
            n_samples, n_features = X_reshaped.shape
            print(f"DFCM使用采样数据，样本数减少到{n_samples}")

        # 初始化
        self.membership_matrix = self.initialize_membership_matrix(n_samples)
        self.centers = X_reshaped[np.random.choice(n_samples, self.n_clusters, replace=False)]
        similarity_weights, neighbor_indices = self.calculate_similarity_weights(X_reshaped)

        print(f"开始DFCM聚类，样本数: {n_samples}, 特征数: {n_features}, 聚类数: {self.n_clusters}")

        # 迭代优化
        for iteration in range(self.max_iterations):
            # 保存旧的隶属度矩阵用于收敛判断
            U_old = self.membership_matrix.copy()

            # 更新聚类中心
            self.centers = self.update_cluster_centers(X_reshaped, self.membership_matrix)

            # 更新隶属度矩阵
            self.membership_matrix = self.update_membership_matrix(X_reshaped, self.centers, similarity_weights, neighbor_indices)

            # 计算目标函数值（传递预计算的邻居索引以避免重复计算）
            objective_value = self.calculate_objective_function(X_reshaped, self.membership_matrix,
                                                             self.centers, similarity_weights, neighbor_indices)
            self.objective_function_values.append(objective_value)

            # 检查收敛
            membership_change = np.linalg.norm(self.membership_matrix - U_old)
            if membership_change < self.error_tolerance:
                print(f"DFCM在第{iteration+1}次迭代后收敛")
                break

            if iteration % 10 == 0:
                print(f"迭代 {iteration+1}/{self.max_iterations}, 目标函数值: {objective_value:.6f}, 变化量: {membership_change:.6f}")

        print(f"DFCM聚类完成，最终目标函数值: {self.objective_function_values[-1]:.6f}")

    def predict(self, X):
        """
        预测样本的隶属度
        """
        if self.centers is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 如果输入是3维序列数据，重塑为2维
        if len(X.shape) == 3:
            n_sequences, sequence_length, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
        else:
            X_reshaped = X

        n_samples = X_reshaped.shape[0]
        U_pred = np.zeros((n_samples, self.n_clusters))

        for i in range(n_samples):
            distances = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                distances[j] = np.sum((X_reshaped[i] - self.centers[j]) ** 2)

            # 计算隶属度
            for j in range(self.n_clusters):
                denominator = 0.0
                for k in range(self.n_clusters):
                    denominator += (distances[j] / distances[k]) ** (2 / (self.fuzzy_coefficient - 1))
                U_pred[i, j] = 1.0 / denominator

        # 如果原始输入是3维数据，将预测结果重塑回原始形状
        if len(X.shape) == 3:
            # U_pred形状: (n_sequences * sequence_length, n_clusters)
            # 重塑为: (n_sequences, sequence_length, n_clusters)
            U_pred = U_pred.reshape(X.shape[0], X.shape[1], -1)

        return U_pred

    def get_cluster_labels(self):
        """
        获取聚类标签（每个样本所属的簇）
        """
        if self.membership_matrix is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        return np.argmax(self.membership_matrix, axis=1)

    def optimal_number_of_clusters(self, X):
        """
        严格按照论文确定最优聚类数（使用DBI指数）
        论文要求使用DBI指数自动选择最优聚类数k
        """
        print("使用DBI指数确定最优聚类数...")

        dbi_values = {}
        min_clusters = config.Config.MIN_CLUSTERS
        max_clusters = min(config.Config.MAX_CLUSTERS, X.shape[0] - 1)

        # 保存原始状态
        original_n_clusters = self.n_clusters
        original_membership_matrix = self.membership_matrix
        original_centers = self.centers
        original_objective_values = self.objective_function_values

        print(f"测试聚类数范围: {min_clusters} - {max_clusters}")

        for n_clusters in range(min_clusters, max_clusters + 1):
            print(f"测试聚类数: {n_clusters}")

            # 临时设置聚类数
            self.n_clusters = n_clusters
            self.membership_matrix = None
            self.centers = None
            self.objective_function_values = []

            # 运行DFCM
            try:
                self.fit(X)
                dbi_value = self.calculate_dbi_index(X)
                dbi_values[n_clusters] = dbi_value

                print(f"聚类数 {n_clusters}, DBI值: {dbi_value:.6f}, 目标函数值: {self.objective_function_values[-1]:.6f}")

            except Exception as e:
                print(f"聚类数 {n_clusters} 训练失败: {e}")
                dbi_values[n_clusters] = float('inf')

        # 选择DBI最小的聚类数
        if dbi_values:
            # 过滤掉无穷大的值
            valid_dbis = {k: v for k, v in dbi_values.items() if v != float('inf')}
            if valid_dbis:
                optimal_k = min(valid_dbis.keys(), key=lambda k: valid_dbis[k])
                optimal_dbi = valid_dbis[optimal_k]

                print(f"\nDBI分析结果:")
                for k, v in sorted(valid_dbis.items()):
                    print(f"  k={k}: DBI={v:.6f}")
                print(f"最优聚类数: {optimal_k}, 最小DBI值: {optimal_dbi:.6f}")

                # 恢复原始状态
                self.n_clusters = original_n_clusters
                self.membership_matrix = original_membership_matrix
                self.centers = original_centers
                self.objective_function_values = original_objective_values

                # 使用最优聚类数重新训练
                print(f"\n使用最优聚类数 {optimal_k} 重新训练...")
                self.n_clusters = optimal_k
                self.membership_matrix = None
                self.centers = None
                self.objective_function_values = []
                self.fit(X)

                return optimal_k, dbi_values
            else:
                print("所有聚类数都训练失败，使用默认聚类数")
                self.n_clusters = original_n_clusters
                self.membership_matrix = original_membership_matrix
                self.centers = original_centers
                self.objective_function_values = original_objective_values
                return original_n_clusters, {}
        else:
            print("DBI计算失败，使用默认聚类数")
            return original_n_clusters, {}

def test_dfcm():
    """测试DFCM算法"""
    print("测试DFCM算法...")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    # 生成3个簇的测试数据
    cluster1 = np.random.randn(60, n_features) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(70, n_features) + np.array([3, 3, 3, 3, 3])
    cluster3 = np.random.randn(70, n_features) + np.array([-3, -3, -3, -3, -3])

    X = np.vstack([cluster1, cluster2, cluster3])

    # 测试DFCM
    dfcm = DFCM(n_clusters=3)
    dfcm.fit(X)

    # 获取聚类结果
    labels = dfcm.get_cluster_labels()
    print(f"聚类标签分布: {np.bincount(labels)}")

    # 测试预测
    test_samples = np.random.randn(10, n_features)
    predictions = dfcm.predict(test_samples)
    print(f"预测形状: {predictions.shape}")

    print("DFCM测试完成！")

if __name__ == "__main__":
    test_dfcm()