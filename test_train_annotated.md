# test_train.py 完整注释

```python
# 测试训练模块
# 训练低光照视频目标检测增强模型 (快速版)
# 原理：使用采样的小数据集快速验证训练流程

import os  # 操作系统接口
import time  # 时间处理
import logging  # 日志模块
import torch  # PyTorch 框架
import torch.optim as optim  # 优化器
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度器
from torch.utils.data import DataLoader, Subset  # 数据加载器和子集
import random  # 随机数生成
from yolo_transformer import YOLOTransformerLowLight  # 主模型
from dataset import LowLightPairDataset  # 数据集
from high_res_cache import HighResFeatureCache  # 特征缓存
from config import Config  # 配置参数

# 自定义 collate_fn 函数
# 作用：处理 batch 中的数据，特别是不同长度的标签列表
def collate_fn(batch):
    """
    自定义 batch 整理函数
    
    Args:
        batch: 样本列表 [(high_res, low_light, labels), ...]
    
    Returns:
        (high_res_batch, low_light_batch, labels_list)
    """
    # 堆叠高清图像
    # torch.stack: 在新维度上堆叠张量
    high_res = torch.stack([item[0] for item in batch])
    
    # 堆叠低光图像
    low_light = torch.stack([item[1] for item in batch])
    
    # 标签列表保持不变 (因为每个样本的标签数量不同)
    labels_list = [item[2] for item in batch]
    
    return high_res, low_light, labels_list

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        # 文件处理器
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'test_train.log')),
        # 控制台处理器
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestTrainer:
    """
    测试训练器类
    用于快速训练和验证模型
    
    特点:
    - 使用小数据集 (sample_size=1000)
    - 少轮次训练 (epochs=5)
    - 快速验证训练流程
    """
    
    def __init__(self, high_res_dir, low_light_dir, sample_size=1000, model_path='yolo26n.pt'):
        """
        初始化测试训练器
        
        Args:
            high_res_dir: 高清图像目录
            low_light_dir: 低光图像目录
            sample_size: 采样图像数量
            model_path: 预训练模型路径
        """
        # 初始化数据集
        self.dataset = LowLightPairDataset(high_res_dir, low_light_dir)
        
        # 随机采样
        if sample_size < len(self.dataset):
            # random.sample: 不重复随机采样
            indices = random.sample(range(len(self.dataset)), sample_size)
            
            # Subset: 创建子集
            self.dataset = Subset(self.dataset, indices)
            logger.info(f"随机采样 {sample_size} 个图像进行训练")
        else:
            logger.info(f"使用全部 {len(self.dataset)} 个图像进行训练")
        
        # 初始化数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=Config.BATCH_SIZE,  # 批次大小
            shuffle=True,  # 打乱数据
            num_workers=4,  # 数据加载线程数
            collate_fn=collate_fn  # 自定义 collate_fn
        )
        
        # 初始化模型
        self.model = YOLOTransformerLowLight(model_path)
        
        # 设备选择
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Apple Silicon
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')  # NVIDIA GPU
        else:
            self.device = torch.device('cpu')  # CPU
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 初始化缓存
        self.cache = HighResFeatureCache()
        
        # 配置优化器 - 只优化特定模块
        params = []
        for name, param in self.model.named_parameters():
            # 只训练增强、融合和检测头
            if 'light_enhance' in name or 'fusion' in name or 'head' in name:
                param.requires_grad = True  # 可训练
                params.append(param)
            else:
                param.requires_grad = False  # 冻结
        
        # Adam 优化器
        self.optimizer = optim.Adam(params, lr=Config.LEARNING_RATE)
        
        # 训练轮次 (快速版)
        self.epochs = 5
        
        # 余弦退火学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6  # 最小学习率
        )
        
        # 训练参数
        self.best_loss = float('inf')  # 最佳损失
        
        # 创建保存目录
        self.save_dir = os.path.join(Config.WEIGHTS_DIR, f'test_train_{int(time.time())}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 日志信息
        logger.info(f"测试训练器初始化完成，使用设备：{self.device}")
        logger.info(f"模型参数数量：{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        logger.info(f"数据集大小：{len(self.dataset)}")
        logger.info(f"批处理大小：{Config.BATCH_SIZE}")
        logger.info(f"总批次数：{len(self.dataloader)}")
        logger.info(f"总训练轮数：{self.epochs}")
    
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        
        Args:
            epoch: 当前轮次
        
        Returns:
            平均损失
        """
        # 设置训练模式
        for param in self.model.parameters():
            param.requires_grad = True
        
        total_loss = 0  # 总损失
        batch_count = 0  # batch 计数
        batch_start_time = time.time()  # batch 计时
        
        # 遍历数据加载器
        for batch_idx, (high_res, low_light, labels_list) in enumerate(self.dataloader):
            try:
                # 打印进度
                print(f"处理图像：Batch {batch_idx}, 图像数量：{len(high_res)}")
                
                # 移动到设备
                high_res = high_res.to(self.device)
                low_light = low_light.to(self.device)
                
                # 处理标签
                targets = []
                for i, labels in enumerate(labels_list):
                    if labels:
                        # 提取类别
                        cls = torch.tensor([label[0] for label in labels], device=self.device)
                        
                        # 提取边界框
                        bboxes = torch.tensor([label[1:] for label in labels], device=self.device)
                        
                        if len(bboxes) > 0:
                            targets.append({
                                'img_idx': i,
                                'cls': cls,
                                'bboxes': bboxes
                            })
                
                # 提取高清特征
                high_res_feat = []
                for i in range(high_res.shape[0]):
                    with torch.no_grad():
                        # 1. 增强
                        enhanced_high_res = self.model.light_enhance(high_res[i:i+1])
                        
                        # 2. Backbone 提取特征
                        high_res_backbone_feat = self.model.backbone(enhanced_high_res)
                        
                        # 3. 全局池化
                        high_res_feat_global = torch.nn.functional.adaptive_avg_pool2d(
                            high_res_backbone_feat, (1, 1)
                        ).squeeze()
                        
                        # 4. 维度调整
                        if high_res_feat_global.shape[-1] != Config.FEATURE_DIM:
                            feat_adjust = torch.nn.Linear(
                                high_res_feat_global.shape[-1], 
                                Config.FEATURE_DIM
                            ).to(self.device)
                            high_res_feat_global = feat_adjust(high_res_feat_global)
                    
                    high_res_feat.append(high_res_feat_global)
                
                # 堆叠特征
                high_res_feat = torch.stack(high_res_feat)
                
                # 前向传播
                outputs = self.model(low_light, high_res_feat, is_training=True)
                
                # 获取低光特征
                with torch.no_grad():
                    enhanced = self.model.light_enhance(low_light)
                    low_light_feat = self.model.backbone(enhanced)
                
                # 计算损失
                loss_dict = self.model.loss_fn(outputs, targets, low_light_feat, high_res_feat)
                loss = loss_dict['total_loss']
                
                # 反向传播
                self.optimizer.zero_grad()  # 清零梯度
                loss.backward()  # 计算梯度
                self.optimizer.step()  # 更新参数
                
                # 累计损失
                total_loss += loss.item()
                batch_count += 1
                
                # 打印日志
                if batch_idx % 10 == 0:
                    batch_time = time.time() - batch_start_time
                    batch_start_time = time.time()
                    logger.info(
                        f"Epoch [{epoch}/{self.epochs}], "
                        f"Batch [{batch_idx}/{len(self.dataloader)}], "
                        f"Loss: {loss.item():.4f}, Time: {batch_time:.2f}s"
                    )
            except Exception as e:
                import traceback
                print(f"处理图像失败：Batch {batch_idx}")
                logger.error(f"训练过程中出错：{e}")
                logger.error(traceback.format_exc())
                continue
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def validate(self):
        """
        验证模型
        
        Returns:
            验证损失
        """
        total_loss = 0
        batch_count = 0
        
        # 禁用梯度
        with torch.no_grad():
            for batch_idx, (high_res, low_light, labels_list) in enumerate(self.dataloader):
                try:
                    # 移动到设备
                    high_res = high_res.to(self.device)
                    low_light = low_light.to(self.device)
                    
                    # 处理标签 (与 train_epoch 相同)
                    targets = []
                    for i, labels in enumerate(labels_list):
                        if labels:
                            cls = torch.tensor([label[0] for label in labels], device=self.device)
                            bboxes = torch.tensor([label[1:] for label in labels], device=self.device)
                            if len(bboxes) > 0:
                                targets.append({
                                    'img_idx': i,
                                    'cls': cls,
                                    'bboxes': bboxes
                                })
                    
                    # 提取高清特征
                    high_res_feat = []
                    for i in range(high_res.shape[0]):
                        with torch.no_grad():
                            enhanced_high_res = self.model.light_enhance(high_res[i:i+1])
                            high_res_backbone_feat = self.model.backbone(enhanced_high_res)
                            high_res_feat_global = torch.nn.functional.adaptive_avg_pool2d(
                                high_res_backbone_feat, (1, 1)
                            ).squeeze()
                            if high_res_feat_global.shape[-1] != Config.FEATURE_DIM:
                                feat_adjust = torch.nn.Linear(
                                    high_res_feat_global.shape[-1], 
                                    Config.FEATURE_DIM
                                ).to(self.device)
                                high_res_feat_global = feat_adjust(high_res_feat_global)
                        high_res_feat.append(high_res_feat_global)
                    high_res_feat = torch.stack(high_res_feat)
                    
                    # 前向传播
                    outputs = self.model(low_light, high_res_feat, is_training=True)
                    
                    # 获取低光特征
                    enhanced = self.model.light_enhance(low_light)
                    low_light_feat = self.model.backbone(enhanced)
                    
                    # 计算损失
                    loss_dict = self.model.loss_fn(outputs, targets, low_light_feat, high_res_feat)
                    loss = loss_dict['total_loss']
                    
                    total_loss += loss.item()
                    batch_count += 1
                except Exception as e:
                    logger.error(f"验证过程中出错：{e}")
                    continue
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def train(self):
        """
        训练模型
        """
        logger.info(f"开始测试训练，共 {self.epochs} 个 epoch")
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch [{epoch}/{self.epochs}] - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 保存模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_model_path = os.path.join(self.save_dir, 'best.pt')
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"保存最佳模型到：{best_model_path}")
            
            # 保存最后模型
            last_model_path = os.path.join(self.save_dir, 'last.pt')
            torch.save(self.model.state_dict(), last_model_path)
        
        # 导出 ONNX
        self.export_onnx()
        
        total_time = time.time() - start_time
        logger.info(f"测试训练完成！总耗时：{total_time:.2f}s")
    
    def export_onnx(self):
        """
        导出模型为 ONNX 格式
        """
        try:
            # 创建示例输入
            dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(self.device)
            high_res_feat = torch.randn(1, Config.FEATURE_DIM).to(self.device)
            
            # 导出路径
            onnx_path = os.path.join(self.save_dir, 'model.onnx')
            
            # 导出 ONNX
            torch.onnx.export(
                self.model,
                (dummy_input, high_res_feat),
                onnx_path,
                input_names=['low_light_frame', 'high_res_feat'],
                output_names=['output'],
                dynamic_axes={
                    'low_light_frame': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'high_res_feat': {0: 'batch_size'}
                },
                opset_version=11
            )
            logger.info(f"成功导出 ONNX 模型到：{onnx_path}")
        except Exception as e:
            logger.error(f"导出 ONNX 失败：{e}")

def test_train_model(high_res_dir, low_light_dir, sample_size=1000):
    """
    测试训练入口函数
    
    Args:
        high_res_dir: 高清图像目录
        low_light_dir: 低光图像目录
        sample_size: 采样数量
    """
    try:
        trainer = TestTrainer(high_res_dir, low_light_dir, sample_size)
        trainer.train()
    except Exception as e:
        logger.error(f"测试训练失败：{e}")
        raise

if __name__ == "__main__":
    # 示例调用
    high_res_dir = 'sample_data/high_res'
    low_light_dir = 'sample_data/low_light'
    test_train_model(high_res_dir, low_light_dir, sample_size=1000)
```
