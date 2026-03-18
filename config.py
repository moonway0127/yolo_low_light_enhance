# 配置模块
# 定义全局参数、路径参数和训练参数
# 原理：使用类的方式组织所有配置参数，便于统一管理和修改

import os  # 导入操作系统接口库，提供路径操作功能

class Config:
    """
    配置类，定义系统的所有参数
    原理：使用类变量存储所有配置，通过类方法初始化系统环境
    """
    
    # ==================== 全局参数 ====================
    # 输入尺寸：模型处理的图像大小 (640x640 像素)
    # YOLO 模型的标准输入尺寸，保持宽高一致避免形变
    INPUT_SIZE = 640
    
    # 缓存超时时间（秒）：高清特征缓存的有效期
    # 超过此时间的缓存会被认为失效，需要重新提取
    # 3600 秒 = 1 小时，平衡内存使用和性能
    CACHE_TIMEOUT = 3600
    
    # 场景匹配阈值：SIFT 特征匹配的最低要求
    # 用于判断当前场景与缓存场景是否相同
    # 0.6 表示 60% 的特征点匹配才认为场景未变化
    SCENE_MATCH_THRESHOLD = 0.6
    
    # Transformer 头数：Multi-Head Attention 中的头数
    # 多头机制允许模型同时关注不同位置的信息
    # 2 个头是比较轻量的配置，适合实时应用
    TRANSFORMER_HEADS = 2
    
    # 特征维度：高清特征的压缩维度
    # 将高维特征压缩到 128 维，减少内存占用
    FEATURE_DIM = 128
    
    # 暗部灰度阈值：判断像素是否属于暗部的标准
    # 灰度值低于 50 的像素被认为是暗部区域
    # 暗部区域会应用更强的增强和特征融合
    DARK_THRESHOLD = 50
    
    # ==================== 路径参数 ====================
    # 项目根目录：获取当前配置文件所在的目录
    # os.path.abspath(__file__) 获取 config.py 的绝对路径
    # os.path.dirname() 获取其父目录，即项目根目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 高清图目录：存储高清参考图像的目录
    # 用于提供场景的高清特征，辅助低光检测
    HIGH_RES_DIR = os.path.join(ROOT_DIR, 'sample_data', 'high_res')
    
    # 低光数据集目录：存储低光图像的目录
    # 训练数据的主要来源，模拟真实低光环境
    LOW_LIGHT_DIR = os.path.join(ROOT_DIR, 'sample_data', 'low_light')
    
    # 缓存文件路径：存储预提取的高清特征的 pickle 文件
    # 避免重复提取特征，加速推理过程
    CACHE_PATH = os.path.join(ROOT_DIR, 'cache', 'feat.pkl')
    
    # 模型权重保存路径：训练过程中保存模型权重的目录
    # 每个训练任务会创建一个时间戳子目录
    WEIGHTS_DIR = os.path.join(ROOT_DIR, 'runs', 'train')
    
    # 日志路径：存储训练和推理日志的目录
    # 记录运行状态、错误信息和性能指标
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')
    
    # ==================== 训练参数 ====================
    # 学习率：优化器的步长大小
    # 1e-4 = 0.001，较小的学习率保证训练稳定
    # Adam 优化器的默认学习率通常是 1e-3 或 1e-4
    LEARNING_RATE = 1e-4
    
    # 批次大小：每次迭代处理的样本数
    # 8 是一个较小的批次，适合显存有限的 GPU
    # 较大的批次可以加速训练但需要更多内存
    BATCH_SIZE = 8
    
    # 训练轮数：整个数据集被遍历的次数
    # 50 轮通常足够模型收敛
    # 过多的轮数可能导致过拟合
    EPOCHS = 50
    
    # 暗部损失权重：暗部区域损失的加权系数
    # 1.8 表示暗部区域的损失比普通区域更重要
    # 目的是让模型更关注难以检测的暗部目标
    DARK_LOSS_WEIGHT = 1.8
    
    # 跨特征对齐损失权重：特征对齐损失的加权系数
    # 0.5 表示对齐损失的重要性是检测损失的一半
    # 用于约束低光特征与高清特征的一致性
    ALIGNMENT_LOSS_WEIGHT = 0.5
    
    @classmethod
    def initialize(cls):
        """
        初始化配置，创建必要的目录
        原理：在系统启动时确保所有需要的目录存在
        避免后续操作因目录不存在而失败
        """
        # 创建缓存目录
        # os.path.dirname(cls.CACHE_PATH) 获取缓存文件的父目录
        # exist_ok=True 表示目录已存在时不报错
        os.makedirs(os.path.dirname(cls.CACHE_PATH), exist_ok=True)
        
        # 创建权重保存目录
        # 用于存放训练过程中保存的模型文件
        os.makedirs(cls.WEIGHTS_DIR, exist_ok=True)
        
        # 创建日志目录
        # 用于存放运行日志文件
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        # 创建示例数据目录
        # 高清图和低光图目录，用于存储训练和测试数据
        os.makedirs(cls.HIGH_RES_DIR, exist_ok=True)
        os.makedirs(cls.LOW_LIGHT_DIR, exist_ok=True)

# 初始化配置
# 在模块导入时自动执行初始化，创建所有必要的目录
Config.initialize()
