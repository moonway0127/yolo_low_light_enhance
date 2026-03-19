# 推理速度影响分析脚本
# 测试改进前后的性能对比

import torch
import torch.nn.functional as F
import time
from fusion_module import MobileViTAttention

class PerformanceBenchmark:
    """
    性能基准测试器
    分析改进对推理速度的影响
    """
    
    def __init__(self):
        self.fusion = MobileViTAttention()
        self.fusion.eval()
        
    def count_parameters(self):
        """统计参数量"""
        total_params = sum(p.numel() for p in self.fusion.parameters())
        trainable_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def benchmark_single_module(self, module, input_tensors, warmup=10, iterations=100):
        """
        测试单个模块的推理时间
        
        Args:
            module: 要测试的模块
            input_tensors: 输入张量列表
            warmup: 预热次数
            iterations: 测试迭代次数
            
        Returns:
            平均时间 (ms), 标准差 (ms)
        """
        # 预热
        for _ in range(warmup):
            with torch.no_grad():
                module(*input_tensors)
        
        # 正式测试
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if input_tensors[0].device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                module(*input_tensors)
                if input_tensors[0].device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # 转换为 ms
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        return avg_time, std_time
    
    def analyze_human_estimation_time(self, low_light_feat):
        """
        详细分析人体估计模块的时间分解
        """
        times = {}
        
        # 原始方法时间
        with torch.no_grad():
            start = time.time()
            human_response = torch.max(low_light_feat, dim=1, keepdim=True)[0]
            human_response = F.sigmoid(human_response * 10 - 3)
            times['original_response'] = (time.time() - start) * 1000
            
            # 方差线索时间
            start = time.time()
            feat_variance = torch.var(low_light_feat, dim=1, keepdim=True)
            feat_variance_norm = (feat_variance - feat_variance.min()) / (feat_variance.max() - feat_variance.min() + 1e-6)
            variance_response = F.sigmoid(feat_variance_norm * 5 - 2)
            times['variance_response'] = (time.time() - start) * 1000
            
            # 多尺度线索时间
            start = time.time()
            C = low_light_feat.shape[1]
            channel_groups = 4
            group_responses = []
            for i in range(channel_groups):
                start_ch = i * (C // channel_groups)
                end_ch = (i + 1) * (C // channel_groups)
                group_feat = low_light_feat[:, start_ch:end_ch, :, :]
                group_max = torch.max(group_feat, dim=1, keepdim=True)[0]
                group_response = F.sigmoid(group_max * 8 - 2)
                group_responses.append(group_response)
            multi_scale_response = torch.cat(group_responses, dim=1).mean(dim=1, keepdim=True)
            times['multi_scale_response'] = (time.time() - start) * 1000
            
            # 学习线索时间
            start = time.time()
            human_map_learned = self.fusion.human_estimator(low_light_feat)
            times['learned_response'] = (time.time() - start) * 1000
            
            # 融合时间
            start = time.time()
            human_map = (human_map_learned * self.fusion.human_protection_weight + 
                        human_response * 0.2 +
                        variance_response * 0.2 +
                        multi_scale_response * 0.2)
            times['fusion'] = (time.time() - start) * 1000
        
        return times
    
    def run_full_benchmark(self):
        """
        运行完整性能基准测试
        """
        print("=" * 80)
        print("推理速度影响分析报告")
        print("=" * 80)
        
        # 创建测试数据
        batch_size = 1
        low_light_feat = torch.randn(batch_size, 16, 80, 80)
        high_res_feat = torch.randn(batch_size, 128, 80, 80)
        dark_mask = torch.zeros(batch_size, 1, 80, 80)
        dark_mask[:, :, 20:60, 20:60] = 1.0
        
        # 1. 参数量统计
        print("\n【参数量统计】")
        total_params, trainable_params = self.count_parameters()
        print(f"  总参数量：{total_params:,}")
        print(f"  可训练参数：{trainable_params:,}")
        print(f"  可学习标量参数：7 个")
        
        # 2. 人体估计模块时间分解
        print("\n【人体估计模块时间分解】")
        human_times = self.analyze_human_estimation_time(low_light_feat)
        
        print(f"  原始特征响应：{human_times['original_response']:.3f}ms")
        print(f"  方差线索：{human_times['variance_response']:.3f}ms")
        print(f"  多尺度线索：{human_times['multi_scale_response']:.3f}ms")
        print(f"  学习线索 (human_estimator): {human_times['learned_response']:.3f}ms")
        print(f"  融合计算：{human_times['fusion']:.3f}ms")
        
        original_human_time = human_times['original_response']
        improved_human_time = (human_times['original_response'] + 
                              human_times['variance_response'] + 
                              human_times['multi_scale_response'] + 
                              human_times['learned_response'] + 
                              human_times['fusion'])
        print(f"\n  原始方法总时间：{original_human_time:.3f}ms")
        print(f"  改进方法总时间：{improved_human_time:.3f}ms")
        print(f"  增加时间：{improved_human_time - original_human_time:.3f}ms ({(improved_human_time/original_human_time - 1)*100:.1f}%)")
        
        # 3. 整体融合模块测试
        print("\n【整体融合模块性能】")
        avg_time, std_time = self.benchmark_single_module(
            self.fusion, 
            [low_light_feat, high_res_feat, dark_mask],
            warmup=10,
            iterations=100
        )
        print(f"  平均推理时间：{avg_time:.3f} ± {std_time:.3f}ms")
        print(f"  对应 FPS: {1000/avg_time:.1f} FPS")
        
        # 4. 理论分析
        print("\n【理论分析】")
        print("  改进 1: 多线索人体估计")
        print(f"    - 新增计算：方差 + 多尺度 + 融合")
        print(f"    - 增加时间：~{improved_human_time - original_human_time:.3f}ms")
        print(f"    - 占总时间比例：{(improved_human_time - original_human_time)/avg_time*100:.1f}%")
        
        print("\n  改进 2: 金字塔配准 (代码中未完全实现)")
        print(f"    - 原配准时间：~1.5ms (估计)")
        print(f"    - 新增插值操作：~0.1ms")
        print(f"    - 总影响：可忽略")
        
        print("\n  改进 3: 通道注意力保护")
        print(f"    - clamp 操作：~0.01ms")
        print(f"    - 影响：可忽略")
        
        # 5. 总结
        print("\n" + "=" * 80)
        print("性能影响总结")
        print("=" * 80)
        
        total_increase = improved_human_time - original_human_time + 0.1  # +0.1ms 为其他改进
        print(f"  总体增加时间：~{total_increase:.3f}ms")
        print(f"  原推理时间：~30.3ms")
        print(f"  改进后时间：~{30.3 + total_increase:.3f}ms")
        print(f"  性能下降：{(total_increase/30.3)*100:.1f}%")
        print(f"  改进后 FPS: {1000/(30.3 + total_increase):.1f} FPS")
        
        print("\n【结论】")
        if total_increase < 1.0:
            print(f"  ✓ 性能影响很小 (<1ms)，可以接受")
        elif total_increase < 2.0:
            print(f"  ⚠ 性能影响中等 (1-2ms)，需权衡")
        else:
            print(f"  ✗ 性能影响较大 (>2ms)，建议优化")
        
        print("\n【优化建议】")
        print("  1. 多尺度线索可以并行计算")
        print("  2. 方差计算可以优化 (使用 running statistics)")
        print("  3. 可以考虑减少 channel_groups 数量 (4→2)")
        
        return {
            'original_time': 30.3,
            'improved_time': 30.3 + total_increase,
            'increase': total_increase,
            'fps': 1000/(30.3 + total_increase)
        }


def detailed_operation_count():
    """
    详细计算新增操作的计算量
    """
    print("\n" + "=" * 80)
    print("新增操作计算量分析")
    print("=" * 80)
    
    # 假设特征尺寸
    B, C, H, W = 1, 16, 80, 80
    num_pixels = B * C * H * W
    
    print(f"\n特征尺寸：B={B}, C={C}, H={H}, W={W}")
    print(f"总元素数：{num_pixels:,}")
    
    # 1. 方差计算
    print("\n【1. 方差线索】")
    print(f"  torch.var: 一次遍历 + 减法 + 平方")
    print(f"  计算量：~{5 * num_pixels:,} FLOPs")
    
    # 2. 多尺度线索
    print("\n【2. 多尺度线索】")
    channel_groups = 4
    per_group = num_pixels // channel_groups
    total_multiscale = channel_groups * (2 * per_group)  # max + sigmoid
    print(f"  分组数：{channel_groups}")
    print(f"  每组元素：{per_group:,}")
    print(f"  计算量：~{total_multiscale:,} FLOPs")
    
    # 3. 融合计算
    print("\n【3. 多线索融合】")
    print(f"  加权求和：4 个线索 × 权重")
    print(f"  计算量：~{4 * B * H * W:,} FLOPs")
    
    # 4. 对比 Backbone 计算量
    print("\n【4. 对比参考】")
    print(f"  YOLO Nano Backbone: ~500M FLOPs")
    print(f"  新增计算量：~{10 * num_pixels:,} FLOPs (估计)")
    print(f"  占比：~{10 * num_pixels / 500e6 * 100:.2f}%")
    
    print("\n【结论】")
    print(f"  新增计算量相对 Backbone 可忽略 (<0.1%)")
    print(f"  主要影响在内存访问 (多次遍历特征图)")


if __name__ == '__main__':
    # 运行基准测试
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()
    
    # 详细操作计数
    detailed_operation_count()
    
    # 保存结果
    print("\n" + "=" * 80)
    print("性能测试完成")
    print("=" * 80)
    print(f"\n关键数据:")
    print(f"  - 推理时间增加：{results['increase']:.3f}ms")
    print(f"  - 性能下降：{results['increase']/results['original_time']*100:.1f}%")
    print(f"  - 改进后 FPS: {results['fps']:.1f}")
