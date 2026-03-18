# 增强 YOLO 低光检测系统架构

## 本次更新 (v2.0) - 核心升级

### 🚀 升级亮点

| 特性 | v1.0 | v2.0 (本次) | 改进效果 |
|------|------|-------------|----------|
| **位置偏移处理** | ❌ 无 | ✅ 图像配准网络 | 解决风/触碰导致的镜头偏移 |
| **行人保护** | ❌ 暗部区域可能稀释 | ✅ 人体区域估计 | 保护行人特征不被背景稀释 |
| **融合策略** | 直接相加 | ✅ 三维权重 + 残差 | 智能选择融合区域 |
| **检测任务** | 通用目标 | ✅ 专注行人 | 针对行人检测优化 |
| **检测头** | 255 通道 | ✅ 18 通道 | 参数量减少 14 倍，推理更快 |

### v2.0 核心改进详解

```mermaid
graph TB
    subgraph v2.0 改进模块
        A[轻量级配准网络] --> D[解决位置偏移]
        B[人体区域估计] --> E[保护行人特征]
        C[三维权重融合] --> F[智能融合策略]
    end
    
    subgraph 场景适配
        G[高清背景图: 无目标] --> I[配准对齐]
        H[低光视频: 有行人] --> I
        I --> J[残差融合]
    end
    
    subgraph 效果
        J --> K[背景增强]
        J --> L[行人保护]
        K --> M[检测精度提升]
        L --> M
    end
    
    style A fill:#ff9800
    style B fill:#ff9800
    style C fill:#ff9800
    style M fill:#4caf50
```

---

## 1. 系统整体架构

```mermaid
graph TB
    subgraph 输入层
        A[低光视频帧] --> B[图像预处理]
        C[高清参考帧] --> D[特征提取与缓存]
    end
    
    subgraph 特征处理层
        B --> E[低光增强模块]
        E --> F[Backbone 特征提取]
        D --> G[高清特征缓存]
    end
    
    subgraph 特征融合层[v2.0 升级]
        F --> H1[暗部掩码生成]
        G --> I1[高清特征投影]
        H1 --> J1[配准网络对齐]
        I1 --> J1
        J1 --> K1[人体区域估计]
        K1 --> L1[三维权重计算]
        L1 --> M1[残差融合输出]
    end
    
    subgraph 检测头层
        M1 --> N[自定义检测头]
        N --> O[边界框预测]
        N --> P[类别预测]
        N --> Q[置信度预测]
    end
    
    subgraph 后处理层
        O --> R[NMS 非极大值抑制]
        P --> R
        Q --> R
        R --> S[最终检测结果]
    end
    
    style A fill:#e1f5ff
    style C fill:#e1f5ff
    style S fill:#e8f5e9
    style J1 fill:#ff9800
    style K1 fill:#ff9800
    style L1 fill:#ff9800
    style M1 fill:#ff9800
```

---

## 2. v2.0 特征融合模块架构（核心升级）

```mermaid
graph TB
    subgraph 输入
        A[低光特征 B-16-H-W]
        B[高清特征 B-128]
    end
    
    subgraph Step1-维度适配
        B --> C[1x1 卷积]
        C --> D[高清投影 B-16-H-W]
    end
    
    subgraph Step2-掩码生成
        A --> E[暗部掩码计算]
        E --> F[暗部掩码 B-1-H-W]
        A --> G[人体估计网络]
        G --> H[人体掩码 B-1-H-W]
    end
    
    subgraph Step3-配准-v2.0
        A --> I[拼接]
        D --> I
        I --> J[配准网络 3 层卷积]
        J --> K[光流场 B-2-H-W]
        K --> L[网格采样]
        L --> M[对齐后高清特征]
    end
    
    subgraph Step4-权重-v2.0
        H --> N[人体权重]
        F --> O[暗部权重]
        M --> P[差异权重]
        N --> Q[综合权重计算]
        O --> Q
        P --> Q
    end
    
    subgraph Step5-融合-v2.0
        M --> R[残差计算]
        A --> R
        R --> S[应用权重]
        Q --> S
        S --> T[输出投影]
        T --> U[最终特征 B-16-H-W]
    end
    
    style J fill:#ff9800
    style G fill:#ff9800
    style Q fill:#ff9800
    style R fill:#ff9800
    style U fill:#4caf50
```

### v2.0 融合策略详解

```mermaid
graph LR
    subgraph 三维权重计算
        A[人体权重 human_weight] --> D[综合权重]
        B[暗部权重 dark_weight] --> D
        C[差异权重 diff_weight] --> D
    end
    
    subgraph 融合公式
        D --> E["fuse_weight = (1-human) * dark * (1-diff)"]
    end
    
    subgraph 融合效果
        E --> F{区域判断}
        F -->|人体区域 | G[weight=0 不融合]
        F -->|暗部背景 | H[weight=1 融合增强]
        F -->|亮部区域 | I[weight=0 不融合]
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style G fill:#ffcdd2
    style H fill:#c8e6c9
    style I fill:#ffcdd2
```

---

## 3. 训练流程

```mermaid
graph LR
    subgraph 数据准备
        A1[低光图像<br/>含行人] --> B1[数据加载器]
        C1[高清图像<br/>纯背景] --> B1
    end
    
    subgraph 前向传播
        B1 --> D1[低光增强]
        D1 --> E1[Backbone]
        E1 --> F1[v2.0 配准对齐]
        F1 --> G1[v2.0 人体估计]
        G1 --> H1[v2.0 融合]
        H1 --> I1[检测头]
        I1 --> J1[预测输出]
    end
    
    subgraph 损失计算
        J1 --> K1[定位损失]
        J1 --> L1[置信度损失]
        J1 --> M1[分类损失]
        H1 --> N1[对齐损失]
    end
    
    subgraph 反向传播
        K1 --> O1[总损失]
        L1 --> O1
        M1 --> O1
        N1 --> O1
        O1 --> P1[梯度更新]
        P1 --> Q1[优化器]
        Q1 --> D1
    end
    
    style A1 fill:#ffebee
    style C1 fill:#ffebee
    style O1 fill:#ffcdd2
    style Q1 fill:#c8e6c9
```

---

## 4. 推理流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant System as 系统
    participant Cache as 特征缓存
    participant Model as 增强 YOLO 模型
    participant Output as 输出模块
    
    User->>System: 输入低光视频帧
    System->>Cache: 查询高清缓存特征
    
    alt 缓存命中
        Cache-->>System: 返回缓存特征
    else 缓存未命中
        User->>System: 输入高清参考帧
        System->>System: 提取并缓存特征
    end
    
    System->>Model: 低光帧 + 高清特征
    Model->>Model: 低光增强
    Model->>Model: Backbone 特征提取
    Model->>Model: v2.0 配准对齐（解决偏移）
    Model->>Model: v2.0 人体区域估计（保护行人）
    Model->>Model: v2.0 三维权重融合
    Model->>Model: 检测头预测
    Model->>Model: NMS 后处理
    Model-->>Output: 检测结果（行人）
    Output-->>User: 显示检测框
```

---

## 5. v2.0 核心创新点

```mermaid
graph TB
    subgraph 创新 1-图像配准
        A1[问题] --> B1[高清背景和低光帧 可能有位置偏移]
        A1 --> C1[原因：风 触碰 镜头微小移动]
        B1 --> D1[轻量级光流网络 3 层卷积估计偏移]
        D1 --> E1[对齐后融合 精度提升]
    end
    
    subgraph 创新 2-人体估计
        A2[问题] --> B2[行人通常在暗部 直接融合会稀释特征]
        A2 --> C2[不能用 YOLO 检测 低光下检测不准]
        C2 --> D2[利用特征统计特性 纹理复杂度估计]
        D2 --> E2[人体区域不融合 保护目标特征]
    end
    
    subgraph 创新 3-三维权重
        A3[问题] --> B3[不同区域需要 不同融合策略]
        B3 --> C3[人体权重 * 暗部权重 * 差异权重]
        C3 --> D3[自适应选择 融合区域]
        D3 --> E3[智能融合 效果最优]
    end
    
    subgraph 创新 4-残差融合
        A4[问题] --> B4[直接相加会 破坏原有特征]
        B4 --> C4[只融合差异部分 residual = high - low]
        C4 --> D4[背景增强 + 目标保护]
        D4 --> E4[融合更自然]
    end
    
    style D1 fill:#ff9800
    style D2 fill:#ff9800
    style D3 fill:#ff9800
    style D4 fill:#ff9800
```

---

## 6. 检测头架构（行人检测优化版）

```mermaid
graph TB
    subgraph 输入
        A[融合特征<br/>16×H×W] --> B[Conv 3×3]
    end
    
    subgraph 特征提取
        B --> C[特征图<br/>64×H×W]
        C --> D[Conv 1×1]
    end
    
    subgraph 多任务输出
        D --> E[边界框分支]
        D --> F[置信度分支]
        D --> G[分类分支]
        
        E --> H[4 个坐标<br/>x,y,w,h]
        F --> I[1 个置信度<br/>obj]
        G --> J[1 个类别<br/>行人]
    end
    
    subgraph 输出格式
        H --> K[输出张量<br/>3×H×W×18]
        I --> K
        J --> K
    end
    
    style A fill:#e3f2fd
    style K fill:#c8e6c9
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
```

---

## 7. 损失函数组成

```mermaid
mindmap
  root((总损失))
    检测损失
      定位损失<br/>Box Loss
      置信度损失<br/>Obj Loss
      分类损失<br/>Cls Loss
    对齐损失
      特征对齐<br/>Alignment Loss
    v2.0 新增
      人体区域保护损失<br/>Human Protection Loss
    权重配置
      box_gain: 0.05
      obj_gain: 1.0
      cls_gain: 0.5
      align_gain: 0.5
```

---

## 8. 数据流维度变化

```mermaid
graph LR
    A[输入图像<br/>3×640×640] --> B[低光增强<br/>3×640×640]
    B --> C[Backbone<br/>16×80×80]
    C --> D[v2.0 配准对齐<br/>16×80×80]
    D --> E[v2.0 人体估计<br/>1×80×80]
    E --> F[v2.0 融合<br/>16×80×80]
    F --> G[检测头<br/>3×80×80×18]
    G --> H[NMS 后处理<br/>N×6]
    H --> I[最终输出<br/>boxes, conf, cls]
    
    style A fill:#e3f2fd
    style I fill:#c8e6c9
    style D fill:#ff9800
    style E fill:#ff9800
    style F fill:#ff9800
```

---

## 9. 性能优化策略

```mermaid
graph TB
    subgraph 算法优化[v2.0]
        A1[NMS 优化<br/>max_det=50] --> B1[后处理加速]
        A2[v2.0 轻量级配准<br/>3层卷积] --> B2[融合加速]
        A3[v2.0 人体估计<br/>5层卷积] --> B2
    end
    
    subgraph 架构优化
        A4[预初始化层] --> B2
        A5[减少动态创建] --> B2
    end
    
    subgraph 硬件加速
        A6[TensorRT] --> B3[预期加速]
        A7[INT8 量化] --> B3
        A8[FP16 推理] --> B3
    end
    
    subgraph v2.0 性能
        B1 --> C1[总耗时<br/>约 30ms]
        B2 --> C1
        B3 --> C2[未来目标<br/>10ms]
    end
    
    style A1 fill:#ffe0b2
    style A2 fill:#ff9800
    style A3 fill:#ff9800
    style C1 fill:#4caf50
```

### v2.0 时间分解（预估）

```mermaid
pie title v2.0 时间分布（总计约 30ms）
    "预处理" : 0.5
    "低光增强" : 1.0
    "Backbone 提取" : 0.3
    "v2.0 配准对齐" : 1.5
    "v2.0 人体估计" : 0.5
    "v2.0 融合" : 1.0
    "检测头" : 5.0
    "NMS 后处理" : 20.0
    "结果转换" : 0.2
```

---

## 10. 应用场景

```mermaid
graph TB
    subgraph v2.0 适用场景
        A[夜间监控] --> D[v2.0 增强 YOLO]
        B[低光视频] --> D
        C[隧道/地下] --> D
        D --> E[专注行人检测]
    end
    
    subgraph 典型应用
        E --> F[安防监控]
        E --> G[自动驾驶]
        E --> H[视频分析]
    end
    
    subgraph 输出
        F --> I[行人跟踪]
        G --> J[障碍物检测]
        H --> K[行为分析]
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style E fill:#ff9800
    style I fill:#c8e6c9
    style J fill:#c8e6c9
    style K fill:#c8e6c9
```

---

## 11. 模块依赖关系

```mermaid
graph BT
    A[增强 YOLO v2.0] --> B[低光增强模块]
    A --> C[Backbone 冻结]
    A --> D[v2.0 融合模块]
    A --> E[自定义检测头]
    
    D --> F[v2.0 暗部掩码]
    D --> G[v2.0 配准网络]
    D --> H[v2.0 人体估计]
    D --> I[v2.0 权重计算]
    D --> J[残差融合]
    
    E --> K[边界框回归]
    E --> L[目标置信度]
    E --> M[多类别分类]
    
    style A fill:#ff9800
    style D fill:#ff9800
    style G fill:#ff9800
    style H fill:#ff9800
```

---

## 12. v1.0 vs v2.0 对比

| 特性 | v1.0 | v2.0 | 改进说明 |
|------|------|------|----------|
| **位置偏移** | ❌ 不处理 | ✅ 配准网络 | 解决风/触碰偏移 |
| **行人保护** | ❌ 暗部融合 | ✅ 人体估计 | 不依赖检测器 |
| **融合方式** | 直接相加 | ✅ 残差融合 | 保护原特征 |
| **权重策略** | 一维暗部掩码 | ✅ 三维权重 | 更智能选择 |
| **目标类型** | 80 类通用 | ✅ 专注行人 | 针对优化 |
| **检测头** | 255 通道 | ✅ 18 通道 | 参数量减少 14 倍 |
| **计算开销** | ~25ms | ~30ms | 增加 5ms |
| **检测效果** | 基线 | ✅ 显著提升 | 行人特征保护 |

---

## 13. 使用说明

### 快速开始

```bash
# 训练
python main.py --mode train --high_res_dir sample_data/high_res --low_light_dir sample_data/low_light

# 推理（单图）
python main.py --mode infer --image_path test.jpg --cache_path cache/feat.pkl

# 推理（视频）
python main.py --mode infer --video_path test.mp4 --cache_path cache/feat.pkl
```

### v2.0 特色使用

```python
# v2.0 优势场景
# 1. 摄像头有轻微晃动
# 2. 背景相对固定，有行人经过
# 3. 夜间/低光环境
# 4. 专注检测行人
```

---

## 14. 技术规格

| 项目 | 规格 |
|------|------|
| **输入尺寸** | 640×640 |
| **特征维度** | 128 |
| **暗部阈值** | 50 (灰度值) |
| **配准网络** | 3 层卷积 |
| **人体估计网络** | 5 层卷积 |
| **检测类别** | 1 (行人) |
| **检测头输出** | 18 通道 (3×6) |
| **推理设备** | GPU/CPU/MPS |
| **目标帧率** | 30+ FPS |

---

## 15. 未来规划

- [ ] 端到端配准训练
- [ ] 多目标跟踪集成
- [ ] TensorRT 加速部署
- [ ] 移动端优化
