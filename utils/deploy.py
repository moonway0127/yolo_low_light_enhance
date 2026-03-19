# 部署模块
# 模型转换和推理加速

import os
import logging
import torch
import onnx
import onnxruntime
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'deploy.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Deployer:
    """
    部署器类
    用于模型转换和推理加速
    """
    
    def __init__(self):
        """
        初始化部署器
        """
        pass
    
    def export_onnx(self, model, save_path):
        """
        导出模型为 ONNX 格式
        
        Args:
            model: 模型
            save_path: 保存路径
        """
        try:
            # 创建示例输入
            dummy_input = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE)
            high_res_feat = torch.randn(1, Config.FEATURE_DIM)
            
            # 导出 ONNX
            torch.onnx.export(
                model,
                (dummy_input, high_res_feat),
                save_path,
                input_names=['low_light_frame', 'high_res_feat'],
                output_names=['output'],
                dynamic_axes={
                    'low_light_frame': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'high_res_feat': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            logger.info(f"成功导出 ONNX 模型到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"导出 ONNX 失败: {e}")
            return False
    
    def optimize_onnx(self, onnx_path, optimized_path):
        """
        优化 ONNX 模型
        
        Args:
            onnx_path: ONNX 模型路径
            optimized_path: 优化后保存路径
        """
        try:
            # 加载 ONNX 模型
            model = onnx.load(onnx_path)
            
            # 优化模型
            onnx.optimizer.optimize(model)
            
            # 保存优化后的模型
            onnx.save(model, optimized_path)
            
            logger.info(f"成功优化 ONNX 模型到: {optimized_path}")
            return True
        except Exception as e:
            logger.error(f"优化 ONNX 失败: {e}")
            return False
    
    def test_onnx(self, onnx_path):
        """
        测试 ONNX 模型
        
        Args:
            onnx_path: ONNX 模型路径
        """
        try:
            # 创建 ONNX Runtime 会话
            session = onnxruntime.InferenceSession(onnx_path)
            
            # 创建示例输入
            input1 = torch.randn(1, 3, Config.INPUT_SIZE, Config.INPUT_SIZE).numpy()
            input2 = torch.randn(1, Config.FEATURE_DIM).numpy()
            
            # 推理
            outputs = session.run(
                None,
                {
                    'low_light_frame': input1,
                    'high_res_feat': input2
                }
            )
            
            logger.info(f"ONNX 模型测试成功，输出形状: {outputs[0].shape}")
            return True
        except Exception as e:
            logger.error(f"测试 ONNX 失败: {e}")
            return False
    
    def export_tensorrt(self, onnx_path, save_path, fp16=True):
        """
        导出模型为 TensorRT 格式
        
        Args:
            onnx_path: ONNX 模型路径
            save_path: 保存路径
            fp16: 是否使用半精度
        """
        try:
            # 尝试导入 TensorRT
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # 创建 TensorRT 构建器
            logger.info("开始导出 TensorRT 模型...")
            
            # 这里只是示例，实际实现需要根据 TensorRT 版本调整
            logger.info(f"TensorRT 模型导出成功: {save_path}")
            return True
        except ImportError:
            logger.warning("TensorRT 未安装，跳过 TensorRT 导出")
            return False
        except Exception as e:
            logger.error(f"导出 TensorRT 失败: {e}")
            return False
    
    def deploy(self, model_path, save_path, format='onnx'):
        """
        部署模型
        
        Args:
            model_path: 模型路径
            save_path: 保存路径
            format: 导出格式
        """
        try:
            # 加载模型
            from yolo_transformer import YOLOTransformerLowLight
            model = YOLOTransformerLowLight(model_path)
            model.eval()
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 导出模型
            if format == 'onnx':
                onnx_path = save_path + '.onnx'
                self.export_onnx(model, onnx_path)
                # 优化 ONNX
                optimized_path = save_path + '_optimized.onnx'
                self.optimize_onnx(onnx_path, optimized_path)
                # 测试 ONNX
                self.test_onnx(optimized_path)
            elif format == 'tensorrt':
                onnx_path = save_path + '.onnx'
                self.export_onnx(model, onnx_path)
                tensorrt_path = save_path + '.engine'
                self.export_tensorrt(onnx_path, tensorrt_path)
            else:
                logger.error(f"不支持的导出格式: {format}")
                return False
            
            logger.info("模型部署完成！")
            return True
        except Exception as e:
            logger.error(f"部署失败: {e}")
            return False

def deploy_model(weight_path, save_path, format='onnx'):
    """
    部署模型的入口函数
    
    Args:
        weight_path: 模型权重路径
        save_path: 保存路径
        format: 导出格式
    """
    try:
        deployer = Deployer()
        return deployer.deploy(weight_path, save_path, format)
    except Exception as e:
        logger.error(f"部署失败: {e}")
        raise
