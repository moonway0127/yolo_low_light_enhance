# 增强 YOLO 低光检测系统架构

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
    
    subgraph 特征融合层
        F --> H[暗部掩码生成]
        G --> I[特征投影]
        H --> J[MobileViT 注意力融合]
        I --> J
    end
    
    subgraph 检测头层
        J --> K[自定义检测头]
        K --> L[边界框预测]
        K --> M[类别预测]
        K --> N[置信度预测]
    end
    
    subgraph 后处理层
        L --> O[NMS 非极大值抑制]
        M --> O
        N --> O
        O --> P[最终检测结果]
    end
    
    style A fill:#e1f5ff
    style C fill:#e1f5ff
    style P fill:#e8f5e9
    style J fill:#fff3e0
    style K fill:#fff3e0
```

## 2. 训练流程

```mermaid
graph LR
    subgraph 数据准备
        A1[低光图像] --> B1[数据加载器]
        C1[高清图像] --> B1
    end
    
    subgraph 前向传播
        B1 --> D1[低光增强]
        D1 --> E1[Backbone]
        E1 --> F1[特征融合]
        F1 --> G1[检测头]
        G1 --> H1[预测输出]
    end
    
    subgraph 损失计算
        H1 --> I1[定位损失]
        H1 --> J2[置信度损失]
        H1 --> K1[分类损失]
        F1 --> L1[对齐损失]
    end
    
    subgraph 反向传播
        I1 --> M1[总损失]
        J2 --> M1
        K1 --> M1
        L1 --> M1
        M1 --> N1[梯度更新]
        N1 --> O1[优化器]
        O1 --> D1
    end
    
    style A1 fill:#ffebee
    style C1 fill:#ffebee
    style M1 fill:#ffcdd2
    style O1 fill:#c8e6c9
```

## 3. 推理流程

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
    Model->>Model: 暗部掩码生成
    Model->>Model: 特征融合
    Model->>Model: 检测头预测
    Model->>Model: NMS 后处理
    Model-->>Output: 检测结果
    Output-->>User: 显示检测框
```

## 4. 特征融合模块详细架构

```mermaid
graph TB
    subgraph 输入特征
        A[低光特征<br/>16×H×W] --> C[拼接]
        B[高清特征<br/>128×H×W] --> D[1×1 卷积]
    end
    
    subgraph 特征处理
        D --> E[高清特征投影<br/>16×H×W]
        E --> C
        C --> F[拼接特征<br/>32×H×W]
    end
    
    subgraph 注意力融合
        F --> G[1×1 卷积]
        G --> H[融合特征<br/>16×H×W]
        H --> I[暗部掩码<br/>1×H×W]
        I --> J[加权融合]
    end
    
    subgraph 输出
        J --> K[最终特征<br/>16×H×W]
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style K fill:#c8e6c9
    style G fill:#fff3e0
```

## 5. 检测头架构

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
        G --> J[80 个类别<br/>COCO]
    end
    
    subgraph 输出格式
        H --> K[输出张量<br/>3×H×W×85]
        I --> K
        J --> K
    end
    
    style A fill:#e3f2fd
    style K fill:#c8e6c9
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
```

## 6. 损失函数组成

```mermaid
mindmap
  root((总损失))
    检测损失
      定位损失<br/>Box Loss
      置信度损失<br/>Obj Loss
      分类损失<br/>Cls Loss
    对齐损失
      特征对齐<br/>Alignment Loss
    权重配置
      box_gain: 0.05
      obj_gain: 1.0
      cls_gain: 0.5
      align_gain: 0.1
```

## 7. 数据流维度变化

```mermaid
graph LR
    A[输入图像<br/>3×640×640] --> B[低光增强<br/>3×640×640]
    B --> C[Backbone<br/>16×80×80]
    C --> D[特征融合<br/>16×80×80]
    D --> E[检测头<br/>3×80×80×85]
    E --> F[NMS 后处理<br/>N×6]
    F --> G[最终输出<br/>boxes, conf, cls]
    
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

## 8. 性能优化策略

```mermaid
graph TB
    subgraph 算法优化
        A1[NMS 优化<br/>max_det=50] --> B1[后处理加速<br/>67ms→56%]
        A2[阈值调整<br/>conf=0.35] --> B1
        A3[向量化运算] --> B2[融合加速<br/>40%]
    end
    
    subgraph 架构优化
        A4[预初始化层] --> B2
        A5[减少动态创建] --> B2
    end
    
    subgraph 硬件加速
        A6[TensorRT] --> B3[预期加速<br/>4 倍]
        A7[INT8 量化] --> B3
        A8[FP16 推理] --> B3
    end
    
    subgraph 最终性能
        B1 --> C1[总耗时<br/>195ms→123ms]
        B2 --> C1
        B3 --> C2[未来目标<br/>30ms]
    end
    
    style A1 fill:#ffe0b2
    style A6 fill:#ffe0b2
    style C1 fill:#c8e6c9
    style C2 fill:#c8e6c9
```

## 9. 模块依赖关系

```mermaid
graph BT
    A[增强 YOLO] --> B[低光增强模块]
    A --> C[Backbone 冻结]
    A --> D[融合模块]
    A --> E[自定义检测头]
    
    D --> F[暗部掩码]
    D --> G[特征投影]
    D --> H[注意力机制]
    
    E --> I[边界框回归]
    E --> J[目标置信度]
    E --> K[多类别分类]
    
    style A fill:#ffcdd2
    style D fill:#fff3e0
    style E fill:#fff3e0
```

## 10. 应用场景

```mermaid
graph TB
    subgraph 输入场景
        A[夜间监控] --> D[增强 YOLO 系统]
        B[低光视频] --> D
        C[隧道/地下] --> D
    end
    
    subgraph 处理流程
        D --> E[实时检测]
        D --> F[特征缓存]
        D --> G[多帧融合]
    end
    
    subgraph 输出应用
        E --> H[目标跟踪]
        E --> I[行为分析]
        E --> J[异常检测]
        F --> K[场景理解]
        G --> L[视频增强]
    end
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style H fill:#c8e6c9
    style I fill:#c8e6c9
    style J fill:#c8e6c9
```

## 11. 时间分解（性能分析）

### 甘特图

```mermaid
gantt
    title 增强 YOLO 推理时间分解（总耗时：123ms）
    dateFormat  X
    axisFormat %Lms
    section 预处理
    图像转张量       :0, 1
    section 前向传播
    低光增强         :1, 1
    Backbone 提取    :2, 1
    暗部掩码生成     :3, 8
    特征融合         :11, 11
    检测头           :22, 8
    section 后处理
    NMS 等           :30, 67
    section 结果转换
    格式转换         :97, 3
```

### 时间分布饼图

```mermaid
pie title 时间分布（总计 123ms）
    "预处理" : 0.5
    "低光增强" : 1.2
    "Backbone 提取" : 0.3
    "暗部掩码生成" : 7.9
    "特征融合" : 11.4
    "检测头" : 8.0
    "NMS 后处理" : 67.4
    "结果转换" : 2.9
```

### 横向柱状图

```mermaid
graph LR
    subgraph 时间分解
        A[预处理 0.5ms<br/>0.4%] --> B[前向传播 28.9ms<br/>23.5%]
        B --> C[后处理 67.4ms<br/>54.8%]
        C --> D[结果转换 2.9ms<br/>2.4%]
    end
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#ffcdd2
    style D fill:#e8f5e9
```

## 12. 关键创新点

```mermaid
graph TB
    subgraph 核心创新
        A[场景缓存机制] --> D[学术价值]
        B[暗部掩码引导] --> D
        C[端到端训练] --> D
    end
    
    subgraph 技术优势
        A --> E[利用时间信息]
        B --> F[自适应融合]
        C --> G[联合优化]
    end
    
    subgraph 应用价值
        E --> H[提升检测精度]
        F --> I[减少伪影]
        G --> J[实时性能]
    end
    
    style A fill:#ffcdd2
    style B fill:#ffcdd2
    style C fill:#ffcdd2
    style D fill:#c8e6c9
```
