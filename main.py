# 主程序
# 系统的入口点，支持训练、推理和部署三种模式
# 原理：通过 argparse 解析命令行参数，根据不同模式调用相应的功能模块

import argparse  # 导入命令行参数解析库，用于处理用户输入的参数
import os  # 导入操作系统接口库，提供文件和目录操作功能
from train import train_model  # 从 train 模块导入训练函数，用于模型训练
from infer import infer_video, infer_image  # 从 infer 模块导入视频和图像推理函数
from deploy import deploy_model  # 从 deploy 模块导入模型部署函数
from config import Config  # 从 config 模块导入配置类，包含所有系统参数

def parse_args():
    """
    解析命令行参数
    原理：使用 argparse 创建参数解析器，定义所有可用的命令行选项
    """
    # 创建参数解析器对象，description 参数显示在帮助信息开头
    parser = argparse.ArgumentParser(description="低光照视频目标检测增强系统")
    
    # 模式选择参数
    # required=True 表示该参数必须提供
    # choices=['train', 'infer', 'deploy'] 限制参数值只能是这三个选项之一
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer', 'deploy'],
                        help='运行模式：train (训练), infer (推理), deploy (部署)')
    
    # 训练参数 - 高清图像目录
    # default=Config.HIGH_RES_DIR 使用配置类中的默认值
    parser.add_argument('--high_res_dir', type=str, default=Config.HIGH_RES_DIR,
                        help='高清图像目录')
    # 训练参数 - 低光图像目录
    parser.add_argument('--low_light_dir', type=str, default=Config.LOW_LIGHT_DIR,
                        help='低光图像目录')
    # 训练参数 - 预训练模型路径
    parser.add_argument('--model', type=str, default='yolo26n.pt',
                        help='预训练模型路径')
    
    # 推理参数 - 视频路径或摄像头 ID
    # 可用于加载视频文件或打开摄像头 (如 0 表示默认摄像头)
    parser.add_argument('--video_path', type=str,
                        help='视频路径或摄像头 ID')
    # 推理参数 - 单张图像路径
    parser.add_argument('--image_path', type=str,
                        help='图像路径')
    # 推理参数 - 缓存文件路径
    # 用于加载预提取的高清特征缓存，加速推理
    parser.add_argument('--cache_path', type=str, default=Config.CACHE_PATH,
                        help='缓存文件路径')
    # 推理参数 - 输出路径
    # 保存检测结果的视频或图像
    parser.add_argument('--output_path', type=str,
                        help='输出路径')
    
    # 部署参数 - 模型权重路径
    # 指定要导出的训练好的模型权重文件
    parser.add_argument('--weight', type=str,
                        help='模型权重路径')
    # 部署参数 - 保存路径
    parser.add_argument('--save_path', type=str, default='./deploy_model',
                        help='部署模型保存路径')
    # 部署参数 - 导出格式
    # onnx: 开放神经网络格式，跨平台
    # tensorrt: NVIDIA TensorRT 格式，用于 GPU 加速推理
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'tensorrt'],
                        help='导出格式')
    
    # 训练配置参数 - 训练轮数
    # 控制训练的迭代次数
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                        help='训练轮数')
    # 训练配置参数 - 批次大小
    # 控制每次迭代使用的样本数量，影响内存使用和训练稳定性
    parser.add_argument('--batch', type=int, default=Config.BATCH_SIZE,
                        help='批次大小')
    
    # 解析并返回命令行参数
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 执行对应模式
    if args.mode == 'train':
        # 训练模式
        print("开始训练模型...")
        print(f"高清图像目录: {args.high_res_dir}")
        print(f"低光图像目录: {args.low_light_dir}")
        print(f"预训练模型: {args.model}")
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch}")
        
        # 执行训练
        train_model(args.high_res_dir, args.low_light_dir)
        
    elif args.mode == 'infer':
        # 推理模式
        if args.video_path:
            # 视频推理
            print("开始视频推理...")
            print(f"视频路径: {args.video_path}")
            print(f"模型路径: {args.model}")
            print(f"缓存路径: {args.cache_path}")
            if args.output_path:
                print(f"输出路径: {args.output_path}")
            
            infer_video(args.video_path, args.model, args.cache_path, args.output_path)
        
        elif args.image_path:
            # 图像推理
            print("开始图像推理...")
            print(f"图像路径: {args.image_path}")
            print(f"模型路径: {args.model}")
            print(f"缓存路径: {args.cache_path}")
            if args.output_path:
                print(f"输出路径: {args.output_path}")
            
            infer_image(args.image_path, args.model, args.cache_path, args.output_path)
        
        else:
            print("错误: 推理模式需要指定 --video_path 或 --image_path")
    
    elif args.mode == 'deploy':
        # 部署模式
        if not args.weight:
            print("错误: 部署模式需要指定 --weight 参数")
            return
        
        print("开始部署模型...")
        print(f"模型权重路径: {args.weight}")
        print(f"保存路径: {args.save_path}")
        print(f"导出格式: {args.format}")
        
        deploy_model(args.weight, args.save_path, args.format)
    
    else:
        print(f"错误: 不支持的模式: {args.mode}")

if __name__ == "__main__":
    main()
