# 项目结构说明

## 📁 目录结构

```
yolo_low_light_enhance/
├── README.md                    # 项目主文档 (v2.1+)
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖包列表
├── .gitignore                   # Git 忽略文件
│
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── yolo_transformer.py      # YOLO 主模型 (含检测头)
│   ├── fusion_module.py         # 特征融合模块 (v2.1+ 核心)
│   ├── light_enhance.py         # 低光增强模块
│   └── high_res_cache.py        # 高清特征缓存
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── config.py                # 配置参数
│   ├── dataset.py               # 数据集加载
│   ├── train.py                 # 训练脚本
│   ├── infer.py                 # 推理脚本
│   └── deploy.py                # 部署脚本
│
├── tests/                       # 测试代码
│   ├── __init__.py
│   ├── test_train.py            # 训练测试
│   ├── test_infer.py            # 推理测试
│   ├── test_model.py            # 模型测试
│   ├── test_async_head.py       # 异步测试头测试
│   └── test_detail_protection.py # 细节保护测试
│
├── scripts/                     # 脚本工具
│   ├── analyze_detail_loss.py   # 细节丢失分析
│   ├── benchmark_performance_impact.py  # 性能基准测试
│   └── find_working_image.py    # 查找可用图像
│
└── docs/                        # 文档
    ├── architecture.md          # 架构说明
    ├── detail_protection_analysis.md   # 细节保护分析
    └── speed_impact_analysis.md        # 速度影响分析
```

## 📂 文件夹说明

### models/ - 模型定义
存放所有深度学习模型相关代码:
- **yolo_transformer.py**: 主模型，整合 YOLO Backbone + 融合模块 + 检测头
- **fusion_module.py**: v2.1+ 核心，四线索人体估计 + 金字塔配准 + 细节保护
- **light_enhance.py**: 低光图像增强模块 (Zero-DCE 轻量化)
- **high_res_cache.py**: 高清特征缓存管理

### utils/ - 工具函数
存放功能性代码:
- **config.py**: 全局配置参数
- **dataset.py**: 数据集加载和预处理
- **train.py**: 训练流程和损失函数
- **infer.py**: 视频/图像推理
- **deploy.py**: 模型导出和部署

### tests/ - 测试代码
存放所有测试脚本:
- **test_train.py**: 训练流程测试
- **test_infer.py**: 推理流程测试
- **test_model.py**: 模型结构测试
- **test_async_head.py**: 异步测试头测试
- **test_detail_protection.py**: 细节保护效果测试

### scripts/ - 脚本工具
存放独立运行的分析脚本:
- **analyze_detail_loss.py**: 详细分析细节丢失情况
- **benchmark_performance_impact.py**: 性能基准测试
- **find_working_image.py**: 查找可用测试图像

### docs/ - 文档
存放详细技术文档:
- **architecture.md**: 系统架构详细说明
- **detail_protection_analysis.md**: 细节保护机制分析
- **speed_impact_analysis.md**: 推理速度影响分析

## 🔄 导入关系

```python
# 从 models 导入
from models import YOLOTransformerLowLight, MobileViTAttention

# 从 utils 导入
from utils import Config, train_model, infer_image

# 完整路径导入
from models.yolo_transformer import YOLOTransformerLowLight
from utils.config import Config
```

## 📝 使用说明

### 训练
```bash
python main.py --mode train \
    --high_res_dir sample_data/high_res \
    --low_light_dir sample_data/low_light \
    --epochs 50 \
    --batch 8
```

### 推理
```bash
# 图像推理
python main.py --mode infer \
    --image_path test.jpg \
    --cache_path cache/feat.pkl

# 视频推理
python main.py --mode infer \
    --video_path test.mp4 \
    --cache_path cache/feat.pkl
```

### 测试
```bash
# 运行所有测试
cd tests && python -m pytest

# 运行单个测试
python tests/test_model.py
```

### 性能分析
```bash
# 性能基准测试
python scripts/benchmark_performance_impact.py

# 细节保护分析
python scripts/analyze_detail_loss.py
```

## 📊 版本信息

**版本**: v2.1+  
**更新日期**: 2026-03-19  
**项目结构整理**: 已完成
