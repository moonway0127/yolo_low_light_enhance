# 配置模块
# 定义全局参数、路径参数和训练参数

import os

class Config:
    """
    配置类，定义系统的所有参数
    """
    
    # ==================== 全局参数 ====================
    # 输入尺寸
    INPUT_SIZE = 640
    # 缓存超时时间（秒）
    CACHE_TIMEOUT = 3600
    # 场景匹配阈值
    SCENE_MATCH_THRESHOLD = 0.6
    # Transformer 头数
    TRANSFORMER_HEADS = 2
    # 特征维度
    FEATURE_DIM = 128
    # 暗部灰度阈值
    DARK_THRESHOLD = 50
    
    # ==================== 路径参数 ====================
    # 项目根目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 高清图目录
    HIGH_RES_DIR = os.path.join(ROOT_DIR, 'sample_data', 'high_res')
    # 低光数据集目录
    LOW_LIGHT_DIR = os.path.join(ROOT_DIR, 'sample_data', 'low_light')
    # 缓存文件路径
    CACHE_PATH = os.path.join(ROOT_DIR, 'cache', 'feat.pkl')
    # 模型权重保存路径
    WEIGHTS_DIR = os.path.join(ROOT_DIR, 'runs', 'train')
    # 日志路径
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')
    
    # ==================== 训练参数 ====================
    # 学习率
    LEARNING_RATE = 1e-4
    # 批次大小
    BATCH_SIZE = 8
    # 训练轮数
    EPOCHS = 50
    # 暗部损失权重
    DARK_LOSS_WEIGHT = 1.8
    # 跨特征对齐损失权重
    ALIGNMENT_LOSS_WEIGHT = 0.5
    
    @classmethod
    def initialize(cls):
        """
        初始化配置，创建必要的目录
        """
        # 创建缓存目录
        os.makedirs(os.path.dirname(cls.CACHE_PATH), exist_ok=True)
        # 创建权重保存目录
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)
        # 创建日志目录
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        # 创建示例数据目录
        os.makedirs(cls.HIGH_RES_DIR, exist_ok=True)
        os.makedirs(cls.LOW_LIGHT_DIR, exist_ok=True)

# 初始化配置
Config.initialize()
