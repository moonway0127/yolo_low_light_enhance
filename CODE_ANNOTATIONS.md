# 低光照视频目标检测增强系统 - 代码详细注释与原理说明

本文档为项目中所有 Python 文件的详细注释和原理说明。

---

## 1. main.py - 主程序入口

**功能**: 系统的入口点，支持训练、推理和部署三种模式

**核心原理**:
- 使用 `argparse` 解析命令行参数
- 根据 `--mode` 参数选择不同的执行分支
- 三种模式:
  - `train`: 训练低光照检测模型
  - `infer`: 对视频/图像进行低光照目标检测
  - `deploy`: 将模型导出为 ONNX/TensorRT 格式

**关键代码说明**:
```python
# 参数解析
parser.add_argument('--mode', required=True, choices=['train', 'infer', 'deploy'])
# 训练模式调用
train_model(args.high_res_dir, args.low_light_dir)
# 推理模式调用
infer_video(args.video_path, args.model, args.cache_path, args.output_path)
# 部署模式调用
deploy_model(args.weight, args.save_path, args.format)
```

---

## 2. config.py - 配置模块 ✓ 已完整注释

**功能**: 定义所有全局参数、路径参数和训练参数

**核心原理**:
- 使用类变量统一管理所有配置
- 启动时自动创建必要的目录
- 参数分类:
  - **全局参数**: INPUT_SIZE(640), CACHE_TIMEOUT(3600s), DARK_THRESHOLD(50) 等
  - **路径参数**: ROOT_DIR, HIGH_RES_DIR, LOW_LIGHT_DIR, CACHE_PATH 等
  - **训练参数**: LEARNING_RATE(1e-4), BATCH_SIZE(8), EPOCHS(50) 等

**参数详解**:
- `INPUT_SIZE = 640`: YOLO 模型的标准输入尺寸
- `DARK_THRESHOLD = 50`: 灰度值低于 50 判定为暗部区域
- `CACHE_TIMEOUT = 3600`: 缓存 1 小时后失效
- `SCENE_MATCH_THRESHOLD = 0.6`: SIFT 特征匹配率需达 60% 才认为场景未变

---

## 3. yolo_transformer.py - 核心检测模型

**功能**: 整合 YOLO26n + 低光增强 + Transformer 融合 + 自定义检测头

**核心原理**:
```
输入低光图像 → 低光增强 → Backbone 提取特征 → 与高清特征融合 → 检测头输出结果
```

**关键组件**:

### 3.1 CustomDetectHead (自定义检测头)
```python
class CustomDetectHead(nn.Module):
    # 作用：处理融合特征并输出标准格式的检测结果
    # 原理：
    # 1. 使用卷积层输出边界框、置信度和类别
    # 2. 通过 postprocess 方法进行 NMS 后处理
    # 3. 返回标准格式的检测结果
```

**关键方法**:
- `forward(x)`: 前向传播，通过检测层输出原始预测
- `postprocess(outputs, img_shape)`: 后处理，包括:
  - 解码边界框坐标
  - 应用置信度阈值过滤
  - 使用 NMS(非极大值抑制)去除重叠框
  - 返回最终检测结果

### 3.2 YOLOTransformerLowLight (主模型类)
```python
class YOLOTransformerLowLight(nn.Module):
    # 整合所有模块的核心模型
    # 组件:
    # - self.backbone: YOLO 的特征提取器
    # - self.light_enhance: 低光增强模块
    # - self.fusion: Transformer 特征融合模块
    # - self.custom_head: 自定义检测头
```

**前向传播流程**:
```python
def forward(self, low_light_frame, high_res_feat=None, is_training=False):
    # 1. 低光增强 (约 5-10ms)
    enhanced_frame = self.light_enhance(low_light_frame)
    
    # 2. Backbone 提取特征 (约 10-20ms)
    low_light_feat = self.backbone(enhanced_frame)
    
    # 3. 生成暗部掩码 (约 1-2ms)
    dark_mask = self.get_dark_mask(low_light_frame)
    
    # 4. 特征融合 (约 5-10ms)
    if high_res_feat is not None:
        fused_feat = self.fusion(low_light_feat, high_res_feat, dark_mask)
    else:
        fused_feat = low_light_feat
    
    # 5. 检测头输出 (约 5-10ms)
    outputs = self.custom_head(fused_feat)
    
    return outputs
```

**损失函数**:
```python
def loss_fn(self, outputs, targets, low_light_feat, high_res_feat):
    # 包含以下损失项:
    # 1. box_loss: 边界框定位损失 (MSE)
    # 2. obj_loss: 目标置信度损失 (BCE)
    # 3. cls_loss: 分类损失 (BCE)
    # 4. alignment_loss: 特征对齐损失 (MSE)
    
    # 总损失 = 检测损失 + 对齐损失
    # 检测损失 = box_loss + obj_loss + cls_loss
```

---

## 4. light_enhance.py - 低光增强模块

**功能**: 使用轻量化 Zero-DCE 实现低光图像增强

**核心原理**:
- 基于 Zero-DCE(Zero-Reference Deep Curve Estimation) 算法
- 通过学习曲线映射来增强图像亮度和对比度
- 轻量化设计，只有 5 个卷积层

**网络结构**:
```python
class LightEnhance(nn.Module):
    conv1: Conv2d(3, 16)   # 输入 3 通道，输出 16 通道
    conv2: Conv2d(16, 32)  # 增加到 32 通道
    conv3: Conv2d(32, 32)  # 保持 32 通道
    conv4: Conv2d(32, 16)  # 减少到 16 通道
    conv5: Conv2d(16, 3)   # 输出 3 通道 (RGB)
```

**增强原理**:
```python
def forward(self, x):
    # 1. 输入归一化到 [0, 1]
    x = x / 255.0 if x.max() > 1.0 else x
    
    # 2. 通过卷积层提取特征
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.relu(self.conv3(out))
    out = self.relu(self.conv4(out))
    
    # 3. 生成增强映射曲线 (sigmoid 激活，输出 [0, 1])
    enhancement_map = self.sigmoid(self.conv5(out))
    
    # 4. 应用增强映射：enhanced = x * (1 + enhancement_map)
    #    相当于对每个像素应用非线性曲线变换
    enhanced = x * (1 + enhancement_map)
    
    # 5. 裁剪到 [0, 1] 范围
    enhanced = torch.clamp(enhanced, 0, 1)
    
    return enhanced
```

**优势**:
- 无需参考图像，自监督学习
- 轻量快速，适合实时应用
- 参数少，不易过拟合

---

## 5. fusion_module.py - Transformer 融合模块

**功能**: 使用轻量化 MobileViT 注意力实现特征融合

**核心原理**:
- 将低光帧特征与高清缓存特征融合
- 使用注意力机制，只在暗部区域应用融合
- 轻量化设计，使用 Conv2d 而非完整的 Transformer

**MobileViTAttention 结构**:
```python
class MobileViTAttention(nn.Module):
    # 特征投影层
    high_res_proj: Conv2d(128, 16)  # 将高清特征投影到 16 通道
    output_proj: Conv2d(32, 16)     # 融合后投影回 16 通道
```

**融合流程**:
```python
def forward(self, low_light_feat, high_res_feat, dark_mask):
    # 1. 处理高清特征
    if high_res_feat.dim() == 2:
        # 全局特征 (B, 128) → 空间特征 (B, 128, H, W)
        high_res_feat = high_res_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
    
    # 2. 维度适配
    high_res_feat_proj = self.high_res_proj(high_res_feat)
    
    # 3. 特征拼接 + 卷积融合
    combined = torch.cat([low_light_feat, high_res_feat_proj], dim=1)
    fused_feat = self.output_proj(combined)
    
    # 4. 基于暗部掩码的加权融合
    # 原理：只在暗部区域 (dark_mask=1) 应用融合增强
    # 亮部区域 (dark_mask=0) 保持原始低光特征
    fused_feat = low_light_feat * (1 - dark_mask) + \
                 (low_light_feat + fused_feat) * dark_mask
    
    return fused_feat
```

**暗部掩码生成**:
```python
def get_dark_mask(self, image):
    # 将 RGB 图像转换为灰度图
    # 使用标准灰度系数：0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    
    # 生成二值掩码：灰度 < 50 的区域为 1(暗部), 其他为 0(亮部)
    mask = (gray < Config.DARK_THRESHOLD).float().unsqueeze(1)
    
    return mask
```

---

## 6. dataset.py - 数据集模块

**功能**: 加载高清图 + 低光帧配对数据集

**核心原理**:
- 支持两种数据源:
  1. 真实配对的高清图 + 低光图
  2. 使用 Zero-DCE 从高清图生成低光图
- 自动标注：使用 YOLO 模型为高清图生成标签
- 数据增强：翻转、裁剪、亮度扰动、噪声等

**关键类**:

### 6.1 ZeroDCE (简化版)
```python
class ZeroDCE(nn.Module):
    # 用于从高清图生成低光图
    # 原理：学习从正常光到低光的逆映射
    # 输出：low_light = x * (1 - enhancement_map)
```

### 6.2 LowLightPairDataset (数据集类)
```python
class LowLightPairDataset(Dataset):
    # 初始化流程:
    # 1. 获取高清图和低光图文件列表
    # 2. 匹配成对的图像
    # 3. 如果没有配对，使用高清图生成低光图
    
    def __getitem__(self, idx):
        # 1. 加载高清图像
        high_res = cv2.imread(high_res_path)
        
        # 2. 加载或生成低光图像
        if low_light_path exists:
            low_light = cv2.imread(low_light_path)
        else:
            low_light = self.generate_low_light(high_res)
        
        # 3. 应用数据增强
        high_res, low_light = self._apply_augmentation(high_res, low_light)
        
        # 4. 自动标注 (使用 YOLO 检测高清图)
        labels = self._auto_label(high_res)
        
        # 5. 转换为张量并返回
        return high_res_tensor, low_light_tensor, labels
```

**数据增强方法**:
```python
def _apply_augmentation(self, high_res, low_light):
    # 1. 随机水平翻转 (50% 概率)
    if random() > 0.5:
        cv2.flip(high_res, 1)
        cv2.flip(low_light, 1)
    
    # 2. 随机裁剪 (50% 概率)
    # 裁剪到原图的 80% 大小
    
    # 3. 亮度扰动 (50% 概率)
    # 乘以 0.8-1.2 的随机因子
    
    # 4. 高斯噪声 (50% 概率)
    # 添加均值为 0，标准差为 10 的噪声
    
    # 5. 调整大小到 INPUT_SIZE
    cv2.resize(img, (640, 640))
```

---

## 7. high_res_cache.py - 高清特征缓存模块

**功能**: 离线提取和缓存高清图特征，供实时推理使用

**核心原理**:
- 预提取高清图像的特征并保存到 pickle 文件
- 推理时直接加载缓存特征，避免重复计算
- 使用 SIFT 特征匹配验证场景是否变化
- 定时更新缓存，保证特征时效性

**关键方法**:

### 7.1 特征提取
```python
def extract_high_res_feat(self, image_path):
    # 1. 加载高清图像
    image = cv2.imread(image_path)
    
    # 2. 使用 YOLO 提取特征
    with torch.no_grad():
        results = self.model(image, verbose=False)
        feat = results[0].orig_img
    
    # 3. 转换为张量并归一化
    feat = torch.from_numpy(feat).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # 4. 特征压缩 (512 通道 → 128 通道)
    compressed_feat = self.feature_compress(feat)
    
    # 5. 全局池化 (HxW → 1x1)
    global_feat = F.adaptive_avg_pool2d(compressed_feat, (1, 1)).squeeze()
    
    return global_feat.numpy()
```

### 7.2 缓存有效性验证
```python
def is_cache_valid(self, current_image, cache_key):
    # 1. 检查缓存是否存在
    if cache_key not in self.cache:
        return False
    
    # 2. 检查时间戳 (超过 1 小时失效)
    if time.time() - timestamp > CACHE_TIMEOUT:
        return False
    
    # 3. SIFT 特征匹配
    # 计算当前帧与缓存高清图的匹配率
    match_ratio = good_matches / max(kp1, kp2)
    
    # 4. 匹配率需达到阈值 (60%)
    return match_ratio >= SCENE_MATCH_THRESHOLD
```

### 7.3 运动检测
```python
def detect_motion(self, prev_frame, current_frame, threshold=300):
    # 1. 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 计算帧差
    frame_diff = cv2.absdiff(prev_gray, current_gray)
    
    # 3. 二值化 (阈值 30)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # 4. 计算非零像素数
    non_zero_count = cv2.countNonZero(thresh)
    
    # 5. 超过阈值认为有运动
    return non_zero_count > threshold
```

### 7.4 缓存更新策略
```python
def should_update_cache(self, current_frame, cache_key):
    # 不更新缓存的情况:
    # 1. 场景中有移动物体
    # 2. 场景中有人
    # 3. 清晰度差异超过 20%
    
    # 更新缓存的情况:
    # 1. 缓存不存在
    # 2. 缓存文件丢失
    # 3. 场景稳定且清晰度相当
```

---

## 8. train.py - 训练模块

**功能**: 训练低光照视频目标检测增强模型

**核心原理**:
- 使用配对的高清图 + 低光图进行训练
- 冻结 YOLO backbone，只训练增强和融合模块
- 多任务损失：检测损失 + 特征对齐损失

**训练流程**:
```python
class Trainer:
    def __init__(self, high_res_dir, low_light_dir):
        # 1. 初始化数据集
        self.dataset = LowLightPairDataset(high_res_dir, low_light_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=8, shuffle=True)
        
        # 2. 初始化模型
        self.model = YOLOTransformerLowLight('yolo26n.pt')
        
        # 3. 选择设备 (MPS > CUDA > CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # 4. 冻结 backbone，只训练特定模块
        for name, param in self.model.named_parameters():
            if 'light_enhance' in name or 'fusion' in name or 'head' in name:
                param.requires_grad = True  # 可训练
            else:
                param.requires_grad = False  # 冻结
        
        # 5. 初始化优化器 (Adam)
        self.optimizer = optim.Adam(params, lr=1e-4)
        
        # 6. 学习率调度器 (余弦退火)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
    
    def train_epoch(self, epoch):
        # 训练一个 epoch
        for batch_idx, (high_res, low_light, labels) in enumerate(self.dataloader):
            # 1. 移动到设备
            high_res = high_res.to(self.device)
            low_light = low_light.to(self.device)
            
            # 2. 提取高清特征
            with torch.no_grad():
                enhanced_high_res = self.model.light_enhance(high_res)
                high_res_backbone_feat = self.model.backbone(enhanced_high_res)
                high_res_feat = adaptive_avg_pool2d(high_res_backbone_feat)
            
            # 3. 前向传播
            outputs = self.model(low_light, high_res_feat, is_training=True)
            
            # 4. 计算损失
            loss_dict = self.model.loss_fn(outputs, targets, low_light_feat, high_res_feat)
            loss = loss_dict['total_loss']
            
            # 5. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self):
        # 完整训练循环
        for epoch in range(1, EPOCHS + 1):
            # 1. 训练
            train_loss = self.train_epoch(epoch)
            
            # 2. 验证
            val_loss = self.validate()
            
            # 3. 更新学习率
            self.scheduler.step()
            
            # 4. 保存模型
            if val_loss < best_loss:
                save_best_model()
            save_last_model()
        
        # 5. 导出 ONNX
        self.export_onnx()
```

---

## 9. infer.py - 实时推理模块

**功能**: 支持视频流、摄像头和图片输入的低光照目标检测

**核心原理**:
- 加载预训练模型和缓存特征
- 对每一帧进行低光增强和目标检测
- 可视化检测结果和 FPS

**推理流程**:
```python
class Inferencer:
    def __init__(self, model_path, cache_path):
        # 1. 加载模型
        self.model = YOLOTransformerLowLight(model_path)
        self.device = select_device()  # MPS > CUDA > CPU
        self.model.to(self.device)
        
        # 2. 设置评估模式 (冻结参数)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 3. 加载缓存
        self.cache = HighResFeatureCache()
        if cache_path exists:
            self.cache.load_cache()
    
    def infer_image(self, image, high_res_feat=None):
        # 1. 预处理
        image_tensor = preprocess(image)  # resize + to_tensor
        
        # 2. 推理
        with torch.no_grad():
            if high_res_feat is not None:
                # 使用融合增强
                enhanced_frame = self.model.light_enhance(image_tensor)
                low_light_feat = self.model.backbone(enhanced_frame)
                dark_mask = self.model.get_dark_mask(image_tensor)
                fused_feat = self.model.fusion(low_light_feat, high_res_feat, dark_mask)
                outputs = self.model.head(fused_feat)
                
                # 使用原生 YOLO 检测增强后的图像
                results = self.native_yolo(enhanced_image_bgr)
            else:
                # 降级为原生 YOLO
                results = self.native_yolo(image)
        
        return results
    
    def process_video(self, video_path):
        # 1. 打开视频
        cap = cv2.VideoCapture(video_path)
        
        # 2. 逐帧处理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 3. 获取高清特征
            high_res_feat = self.cache.get_feature(frame, 'default')
            
            # 4. 推理
            results = self.infer_image(frame, high_res_feat)
            
            # 5. 可视化
            vis_frame = self.visualize(frame, results, use_enhance)
            
            # 6. 显示/保存
            cv2.imshow('Low Light Detection', vis_frame)
```

---

## 10. deploy.py - 部署模块

**功能**: 模型转换和推理加速

**核心原理**:
- 将 PyTorch 模型导出为 ONNX 格式
- 优化 ONNX 模型 (算子融合、常量折叠等)
- 可选导出为 TensorRT 格式 (GPU 加速)

**部署流程**:
```python
class Deployer:
    def export_onnx(self, model, save_path):
        # 1. 创建示例输入
        dummy_input = torch.randn(1, 3, 640, 640)
        high_res_feat = torch.randn(1, 128)
        
        # 2. 导出 ONNX
        torch.onnx.export(
            model,
            (dummy_input, high_res_feat),
            save_path,
            input_names=['low_light_frame', 'high_res_feat'],
            output_names=['output'],
            dynamic_axes={...},  # 支持动态 batch 和分辨率
            opset_version=11
        )
    
    def optimize_onnx(self, onnx_path, optimized_path):
        # 1. 加载 ONNX 模型
        model = onnx.load(onnx_path)
        
        # 2. 优化模型
        onnx.optimizer.optimize(model)
        
        # 3. 保存优化后的模型
        onnx.save(model, optimized_path)
    
    def test_onnx(self, onnx_path):
        # 1. 创建 ONNX Runtime 会话
        session = onnxruntime.InferenceSession(onnx_path)
        
        # 2. 推理测试
        outputs = session.run(None, {
            'low_light_frame': input1,
            'high_res_feat': input2
        })
```

---

## 总结

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    低光照检测系统                        │
├─────────────────────────────────────────────────────────┤
│  训练阶段：                                              │
│  高清图 + 低光图 → 数据增强 → 模型训练 → 保存权重        │
├─────────────────────────────────────────────────────────┤
│  推理阶段：                                              │
│  输入帧 → 低光增强 → 特征提取 → 特征融合 → 检测头 → 输出│
│         ↑                                                │
│         └──── 高清特征缓存 (离线提取)                     │
└─────────────────────────────────────────────────────────┘
```

### 核心技术

1. **低光增强**: Zero-DCE 轻量化网络，快速提升图像质量
2. **特征融合**: MobileViT 注意力机制，只在暗部区域融合高清特征
3. **缓存机制**: SIFT 匹配 + 时间戳验证，智能更新缓存
4. **检测头**: 自定义 YOLO 风格检测头，支持多尺度预测

### 性能优化

- **冻结 Backbone**: 只训练增强和融合模块，减少参数量
- **特征压缩**: 512 维 → 128 维，降低内存占用
- **动态批处理**: 支持不同分辨率和 batch size
- **设备自适应**: MPS (Apple Silicon) > CUDA > CPU

---

**文档版本**: v1.0  
**最后更新**: 2026-03-18
