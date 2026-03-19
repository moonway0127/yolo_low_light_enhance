# 训练模块
# 训练低光照视频目标检测增强模型

import os
import time
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from yolo_transformer import YOLOTransformerLowLight
from dataset import LowLightPairDataset
from high_res_cache import HighResFeatureCache
from config import Config

# 自定义collate_fn函数，处理不同长度的标签列表
def collate_fn(batch):
    high_res = torch.stack([item[0] for item in batch])
    low_light = torch.stack([item[1] for item in batch])
    labels_list = [item[2] for item in batch]
    return high_res, low_light, labels_list

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    """
    训练器类
    用于训练低光照视频目标检测增强模型
    """
    
    def __init__(self, high_res_dir, low_light_dir, model_path='yolo26n.pt'):
        """
        初始化训练器
        
        Args:
            high_res_dir: 高清图像目录
            low_light_dir: 低光图像目录
            model_path: 预训练模型路径
        """
        # 初始化数据集
        self.dataset = LowLightPairDataset(high_res_dir, low_light_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        # 初始化模型
        self.model = YOLOTransformerLowLight(model_path)
        # 优先使用MPS（Apple Silicon GPU），然后是CUDA，最后是CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # 初始化高清特征缓存
        self.cache = HighResFeatureCache()
        
        # 初始化优化器和学习率调度器
        # 仅优化融合模块和检测头
        params = []
        for name, param in self.model.named_parameters():
            if 'light_enhance' in name or 'fusion' in name or 'head' in name:
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
        
        self.optimizer = optim.Adam(params, lr=Config.LEARNING_RATE)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=Config.EPOCHS,
            eta_min=1e-6
        )
        
        # 训练参数
        self.epochs = Config.EPOCHS
        self.best_loss = float('inf')
        
        # 创建保存目录
        self.save_dir = os.path.join(Config.WEIGHTS_DIR, f'train_{int(time.time())}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"训练器初始化完成，使用设备: {self.device}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch
            
        Returns:
            平均损失
        """
        # 设置模型为训练模式
        for param in self.model.parameters():
            param.requires_grad = True
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (high_res, low_light, labels_list) in enumerate(self.dataloader):
            try:
                # 移动到设备
                high_res = high_res.to(self.device)
                low_light = low_light.to(self.device)
                
                # 处理标签
                targets = []
                for i, labels in enumerate(labels_list):
                    if labels:
                        # 提取类别和边界框
                        cls = torch.tensor([label[0] for label in labels], device=self.device)
                        bboxes = torch.tensor([label[1:] for label in labels], device=self.device)
                        if len(bboxes) > 0:
                            targets.append({
                                'img_idx': i,
                                'cls': cls,
                                'bboxes': bboxes
                            })
                
                # 提取高清特征（与推理流程一致）
                high_res_feat = []
                for i in range(high_res.shape[0]):
                    # 从实际高清图像中提取特征
                    # 1. 增强高清图像
                    with torch.no_grad():
                        enhanced_high_res = self.model.light_enhance(high_res[i:i+1])
                        # 2. 通过backbone提取特征
                        high_res_backbone_feat = self.model.backbone(enhanced_high_res)
                        # 3. 全局池化
                        high_res_feat_global = torch.nn.functional.adaptive_avg_pool2d(high_res_backbone_feat, (1, 1)).squeeze()
                        # 4. 确保特征维度与配置一致
                        if high_res_feat_global.shape[-1] != Config.FEATURE_DIM:
                            # 使用线性层调整维度
                            feat_adjust = torch.nn.Linear(high_res_feat_global.shape[-1], Config.FEATURE_DIM).to(self.device)
                            high_res_feat_global = feat_adjust(high_res_feat_global)
                    high_res_feat.append(high_res_feat_global)
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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # 打印进度
                logger.info(f"Epoch [{epoch}/{self.epochs}], Batch [{batch_idx}/{len(self.dataloader)}], Loss: {loss.item():.4f}")
            except Exception as e:
                logger.error(f"训练过程中出错: {e}")
                continue
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def validate(self):
        """
        验证模型
        
        Returns:
            验证损失
        """
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, (high_res, low_light, labels_list) in enumerate(self.dataloader):
                try:
                    # 移动到设备
                    high_res = high_res.to(self.device)
                    low_light = low_light.to(self.device)
                    
                    # 处理标签
                    targets = []
                    for i, labels in enumerate(labels_list):
                        if labels:
                            # 提取类别和边界框
                            cls = torch.tensor([label[0] for label in labels], device=self.device)
                            bboxes = torch.tensor([label[1:] for label in labels], device=self.device)
                            if len(bboxes) > 0:
                                targets.append({
                                    'img_idx': i,
                                    'cls': cls,
                                    'bboxes': bboxes
                                })
                    
                    # 提取高清特征（与推理流程一致）
                    high_res_feat = []
                    for i in range(high_res.shape[0]):
                        # 从实际高清图像中提取特征
                        # 1. 增强高清图像
                        with torch.no_grad():
                            enhanced_high_res = self.model.light_enhance(high_res[i:i+1])
                            # 2. 通过backbone提取特征
                            high_res_backbone_feat = self.model.backbone(enhanced_high_res)
                            # 3. 全局池化
                            high_res_feat_global = torch.nn.functional.adaptive_avg_pool2d(high_res_backbone_feat, (1, 1)).squeeze()
                            # 4. 确保特征维度与配置一致
                            if high_res_feat_global.shape[-1] != Config.FEATURE_DIM:
                                # 使用线性层调整维度
                                feat_adjust = torch.nn.Linear(high_res_feat_global.shape[-1], Config.FEATURE_DIM).to(self.device)
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
                except Exception as e:
                    logger.error(f"验证过程中出错: {e}")
                    continue
        
        return total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0
    
    def train(self):
        """
        训练模型
        """
        logger.info(f"开始训练，共 {self.epochs} 个 epoch")
        
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            logger.info(f"Epoch [{epoch}/{self.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}, Time: {time.time() - start_time:.2f}s")
            
            # 保存模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_model_path = os.path.join(self.save_dir, 'best.pt')
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"保存最佳模型到: {best_model_path}")
            
            # 保存最后模型
            last_model_path = os.path.join(self.save_dir, 'last.pt')
            torch.save(self.model.state_dict(), last_model_path)
        
        # 导出 ONNX
        self.export_onnx()
        
        logger.info("训练完成！")
    
    def export_onnx(self):
        """
        导出模型为 ONNX 格式
        """
        try:
            # 创建示例输入
            dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).to(self.device)
            high_res_feat = torch.randn(1, Config.FEATURE_DIM).to(self.device)
            
            # 导出 ONNX
            onnx_path = os.path.join(self.save_dir, 'model.onnx')
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
            logger.info(f"成功导出 ONNX 模型到: {onnx_path}")
        except Exception as e:
            logger.error(f"导出 ONNX 失败: {e}")

def train_model(high_res_dir, low_light_dir):
    """
    训练模型的入口函数
    
    Args:
        high_res_dir: 高清图像目录
        low_light_dir: 低光图像目录
    """
    try:
        trainer = Trainer(high_res_dir, low_light_dir)
        trainer.train()
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise
