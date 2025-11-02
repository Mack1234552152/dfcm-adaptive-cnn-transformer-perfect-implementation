"""
Configuration file: Adaptive CNN-Transformer Fusion for Discriminative Fuzzy C-Means Clustering
严格按照论文参数设置，不进行任何优化修改
"""

import numpy as np

class Config:
    """论文算法参数配置类"""

    # ==================== 数据配置 ====================
    # 数据文件路径
    DATA_PATH = "AirQualityUCI.xlsx"

    # 序列长度（历史数据步长）
    SEQUENCE_LENGTH = 24

    # 预测范围（未来预测步长）
    PREDICTION_HORIZON = 1

    # 数据分割比例
    TRAIN_RATIO = 0.8
    VALIDATION_RATIO = 0.1
    TEST_RATIO = 0.1

    # ==================== DFCM参数 ====================
    # 模糊C均值聚类的模糊系数（论文标准值）
    FUZZY_COEFFICIENT = 2.0

    # 惩罚项系数λ（论文中重要参数，平衡相似性约束）
    PENALTY_LAMBDA = 0.1

    # 最大迭代次数（论文实验设置）
    DFCM_MAX_ITERATIONS = 100

    # 收敛误差容忍度（论文中用于判断算法收敛）
    DFCM_ERROR_TOLERANCE = 1e-5

    # 最小聚类数（基于论文数据集特点）
    MIN_CLUSTERS = 2

    # 最大聚类数（根据论文实验设置）
    MAX_CLUSTERS = 8

    # k近邻的k值范围（用于相似性权重计算）
    K_NEIGHBORS_RANGE = range(3, 11)

    # ==================== 自适应CNN参数 ====================
    # 停止条件阈值θ（论文中设为0.1）
    CNN_STOP_THRESHOLD = 0.1

    # 最大层数限制
    CNN_MAX_LAYERS = 10

    # 卷积核大小
    CNN_KERNEL_SIZE = 3

    # 池化窗口大小
    CNN_POOL_SIZE = 2

    # 卷积滤波器数量
    CNN_FILTERS = [32, 64, 128]

    # Dropout率
    CNN_DROPOUT_RATE = 0.2

    # 激活函数
    CNN_ACTIVATION = 'relu'

    # ==================== 自适应Transformer参数 ====================
    # 模型维度
    TRANSFORMER_D_MODEL = 512

    # 注意力头数
    TRANSFORMER_NUM_HEADS = 8

    # 前馈网络维度
    TRANSFORMER_DIM_FEEDFORWARD = 2048

    # Dropout率
    TRANSFORMER_DROPOUT_RATE = 0.1

    # 层数
    TRANSFORMER_NUM_LAYERS = 6

    # ==================== 训练参数 ====================
    # 学习率
    LEARNING_RATE = 0.001

    # 批次大小
    BATCH_SIZE = 32

    # 训练轮数
    EPOCHS = 3  # 快速测试版本

    # 优化器
    OPTIMIZER = 'adam'

    # 损失函数
    LOSS_FUNCTION = 'mse'

    # 学习率衰减
    LEARNING_RATE_DECAY = 0.95

    # 早停耐心值
    EARLY_STOPPING_PATIENCE = 10

    # ==================== 评估参数 ====================
    # 评估指标
    METRICS = ['mae', 'mse', 'rmse']

    # 小数值稳定性
    EPSILON = 1e-8

    # ==================== 随机种子 ====================
    # 确保结果可重现
    RANDOM_SEED = 42

    @classmethod
    def setup_random_seed(cls):
        """设置随机种子以确保结果可重现"""
        import random
        import torch
        random.seed(cls.RANDOM_SEED)
        np.random.seed(cls.RANDOM_SEED)
        torch.manual_seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.RANDOM_SEED)
            torch.cuda.manual_seed_all(cls.RANDOM_SEED)

    @classmethod
    def validate_config(cls):
        """验证配置参数的合理性"""
        assert cls.SEQUENCE_LENGTH > 0, "序列长度必须大于0"
        assert cls.PREDICTION_HORIZON > 0, "预测范围必须大于0"
        assert cls.MIN_CLUSTERS < cls.MAX_CLUSTERS, "最小聚类数必须小于最大聚类数"
        assert 0 < cls.CNN_STOP_THRESHOLD < 1, "CNN停止阈值必须在0到1之间"
        assert cls.TRANSFORMER_D_MODEL % cls.TRANSFORMER_NUM_HEADS == 0, "d_model必须能被num_heads整除"
        print("配置参数验证通过！")

if __name__ == "__main__":
    # 测试配置文件
    Config.validate_config()
    print("配置文件加载成功！")
    print(f"数据路径: {Config.DATA_PATH}")
    print(f"序列长度: {Config.SEQUENCE_LENGTH}")
    print(f"预测范围: {Config.PREDICTION_HORIZON}")