"""
自适应卷积神经网络模块 (Adaptive CNN)
严格按照论文Algorithm 3实现，动态确定网络层数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class AdaptiveCNN(nn.Module):
    """
    自适应CNN模型
    严格按照论文Algorithm 3实现动态层数确定机制
    停止条件：|ȳᵢ - x̄ᵢ| ≤ θ，其中ȳᵢ是CNN输出，x̄ᵢ是聚类均值
    """

    def __init__(self, input_size, output_size, stop_threshold=None, max_layers=None):
        super(AdaptiveCNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.stop_threshold = stop_threshold or config.Config.CNN_STOP_THRESHOLD
        self.max_layers = max_layers or config.Config.CNN_MAX_LAYERS

        # 论文Algorithm 3的参数
        self.L = 0  # 当前层数
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        # 全连接层
        self.fc1 = None
        self.fc2 = None

        # 存储每层的输出
        self.layer_outputs = []

        print(f"初始化自适应CNN: 输入维度={input_size}, 输出维度={output_size}")
        print(f"停止阈值θ={self.stop_threshold}, 最大层数={self.max_layers}")

    def add_conv_layer(self, in_channels, out_channels, kernel_size=3):
        """
        添加一个卷积层
        按照论文Algorithm 3步骤
        """
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        pool_layer = nn.MaxPool1d(2)

        self.conv_layers.append(conv_layer)
        self.pool_layers.append(pool_layer)
        self.L += 1
        print(f"添加第{self.L}层卷积: {in_channels}->{out_channels} 通道")

    def forward(self, x, cluster_means=None):
        """
        前向传播
        严格按照论文Algorithm 3实现动态层数确定

        参数:
        - x: 输入数据
        - cluster_means: 聚类均值x̄ᵢ，用于停止条件判断
        """
        batch_size = x.shape[0]

        # 处理输入维度
        if len(x.shape) == 3:
            # (batch_size, sequence_length, n_features) -> (batch_size, n_features, sequence_length)
            x = x.transpose(1, 2)
        elif len(x.shape) == 2:
            # (batch_size, features) -> (batch_size, 1, features)
            x = x.unsqueeze(1)

        # 如果没有提供聚类均值，使用默认行为
        if cluster_means is None:
            return self._forward_default(x)

        # 论文Algorithm 3的动态层数确定
        current_x = x.clone()

        # 重置网络结构（如果需要）
        if self.L == 0:
            self._initialize_first_layer(current_x.shape[1])

        # 逐层添加并检查停止条件
        for layer_idx in range(self.max_layers):
            # 如果当前层不存在，添加新层
            if layer_idx >= len(self.conv_layers):
                in_channels = current_x.shape[1]
                out_channels = min(32 * (layer_idx + 2), 128)  # 动态增加通道数
                self.add_conv_layer(in_channels, out_channels)

                # 将新层移动到正确的设备
                if current_x.is_cuda:
                    self.conv_layers[-1] = self.conv_layers[-1].cuda()
                    self.pool_layers[-1] = self.pool_layers[-1].cuda()

            # 通过当前层
            current_x = F.relu(self.conv_layers[layer_idx](current_x))
            
            # 确保池化不会导致输出尺寸为0
            if current_x.shape[-1] > 1:
                current_x = self.pool_layers[layer_idx](current_x)
            else:
                # 如果池化会导致尺寸为0，则跳过池化
                print(f"跳过第{layer_idx+1}层池化以避免输出尺寸为0")
                break

            # 计算当前层的输出ȳᵢ
            layer_output = self._compute_layer_output(current_x)

            # 检查停止条件：|ȳᵢ - x̄ᵢ| ≤ θ
            # 严格按照论文，需要在相同特征空间中比较
            if cluster_means is not None:
                # 确保cluster_means维度匹配
                if cluster_means.dim() == 3:
                    cluster_means_current = cluster_means[:, -1, :]  # 取最后一个时间步
                else:
                    cluster_means_current = cluster_means

                # 动态创建特征空间映射层（如果不存在）
                # 将原始特征空间的聚类均值映射到CNN当前层的特征空间
                if not hasattr(self, 'cluster_mean_projection') or self.cluster_mean_projection.out_features != layer_output.shape[1]:
                    original_dim = cluster_means_current.shape[1]
                    current_dim = layer_output.shape[1]
                    self.cluster_mean_projection = nn.Linear(original_dim, current_dim).to(layer_output.device)
                    print(f"创建/更新聚类均值投影层: {original_dim} -> {current_dim}")

                # 将聚类均值投影到当前层的特征空间
                cluster_means_projected = self.cluster_mean_projection(cluster_means_current)

                # 计算距离 |ȳᵢ - x̄ᵢ|
                # 现在layer_output和cluster_means_projected在同一特征空间
                distances = torch.norm(layer_output - cluster_means_projected, dim=1)
                avg_distance = distances.mean().item()

                print(f"第{layer_idx + 1}层特征空间距离: {avg_distance:.6f}, 阈值: {self.stop_threshold}")

                # 如果满足停止条件，停止添加层
                if avg_distance <= self.stop_threshold:
                    print(f"停止条件满足，停止在第{layer_idx + 1}层")
                    break

        # 最终输出处理
        final_output = self._compute_final_output(current_x)
        return final_output

    def _initialize_first_layer(self, input_channels):
        """初始化第一层"""
        if len(self.conv_layers) == 0:
            out_channels = 32
            self.add_conv_layer(input_channels, out_channels)

    def _compute_layer_output(self, x):
        """
        计算层的输出ȳᵢ
        按照论文定义
        """
        batch_size = x.shape[0]

        # 全局平均池化
        if x.dim() == 3:
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        else:
            x = x.view(batch_size, -1)

        return x

    def _compute_final_output(self, x):
        """
        计算最终输出
        """
        batch_size = x.shape[0]

        # 展平
        if x.dim() == 3:
            x = x.view(batch_size, -1)
        else:
            x = x.view(batch_size, -1)

        # 动态初始化全连接层
        flattened_size = x.shape[1]
        if self.fc1 is None:
            self.fc1 = nn.Linear(flattened_size, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.output_size).to(x.device)

        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def _forward_default(self, x):
        """
        默认前向传播（无聚类均值时）
        """
        # 如果没有卷积层，初始化第一层
        if len(self.conv_layers) == 0:
            self._initialize_first_layer(x.shape[1])

        # 通过现有的卷积层
        current_x = x
        for i in range(len(self.conv_layers)):
            current_x = F.relu(self.conv_layers[i](current_x))
            # 确保池化不会导致输出尺寸为0
            if current_x.shape[-1] > 1:
                current_x = self.pool_layers[i](current_x)
            else:
                break

        # 计算最终输出
        return self._compute_final_output(current_x)

    def get_current_depth(self):
        """获取当前网络深度"""
        return len(self.conv_layers)

    def reset_network(self):
        """重置网络结构"""
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.L = 0
        self.fc1 = None
        self.fc2 = None
        print("重置自适应CNN网络结构")

def test_adaptive_cnn():
    """测试自适应CNN"""
    print("测试自适应CNN...")

    # 创建测试数据
    batch_size = 32
    sequence_length = 10
    n_features = 4
    output_size = 1

    x = torch.randn(batch_size, sequence_length, n_features)
    cluster_means = torch.randn(batch_size, n_features)  # 模拟聚类均值

    # 创建自适应CNN
    model = AdaptiveCNN(n_features, output_size, stop_threshold=0.5, max_layers=5)

    # 测试前向传播
    output = model(x, cluster_means)
    print(f"输出形状: {output.shape}")
    print(f"最终网络深度: {model.get_current_depth()}")

    # 测试无聚类均值的情况
    output2 = model(x)
    print(f"无聚类均值输出形状: {output2.shape}")

    print("自适应CNN测试完成！")

if __name__ == "__main__":
    test_adaptive_cnn()