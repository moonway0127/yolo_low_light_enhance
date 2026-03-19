# 细节保护效果验证脚本
# 对比改进前后的细节丢失情况

import torch
import torch.nn.functional as F
import numpy as np
from fusion_module import MobileViTAttention

class DetailProtectionTester:
    """
    细节保护效果测试器
    """
    
    def __init__(self):
        self.fusion = MobileViTAttention()
        self.fusion.eval()
        
    def create_test_scenarios(self):
        """
        创建测试场景
        模拟可能丢失细节的极端情况
        """
        scenarios = {}
        
        # 场景 1: 低光人体 (特征响应弱)
        low_light_weak = torch.randn(1, 16, 80, 80) * 0.3  # 弱特征
        dark_mask_weak = torch.zeros(1, 1, 80, 80)
        dark_mask_weak[:, :, 20:60, 20:60] = 1.0
        high_res_bg = torch.randn(1, 128, 80, 80)
        scenarios['weak_human'] = {
            'low_light_feat': low_light_weak,
            'high_res_feat': high_res_bg,
            'dark_mask': dark_mask_weak,
            'description': '低光人体 - 特征响应弱'
        }
        
        # 场景 2: 暗部人体 (完全在暗部)
        low_light_dark = torch.randn(1, 16, 80, 80) * 0.5
        dark_mask_full = torch.ones(1, 1, 80, 80)  # 全暗
        scenarios['dark_human'] = {
            'low_light_feat': low_light_dark,
            'high_res_feat': high_res_bg,
            'dark_mask': dark_mask_full,
            'description': '暗部人体 - 完全暗部区域'
        }
        
        # 场景 3: 小目标人体 (特征少)
        low_light_small = torch.randn(1, 16, 80, 80)
        dark_mask_small = torch.zeros(1, 1, 80, 80)
        dark_mask_small[:, :, 35:45, 35:45] = 1.0  # 小暗部区域
        scenarios['small_target'] = {
            'low_light_feat': low_light_small,
            'high_res_feat': high_res_bg,
            'dark_mask': dark_mask_small,
            'description': '小目标 - 特征区域小'
        }
        
        # 场景 4: 正常情况 (基线)
        low_light_normal = torch.randn(1, 16, 80, 80)
        dark_mask_normal = torch.zeros(1, 1, 80, 80)
        dark_mask_normal[:, :, 20:60, 20:60] = 1.0
        scenarios['normal'] = {
            'low_light_feat': low_light_normal,
            'high_res_feat': high_res_bg,
            'dark_mask': dark_mask_normal,
            'description': '正常情况 - 基线'
        }
        
        return scenarios
    
    def test_human_estimation(self, low_light_feat):
        """
        测试人体估计的准确性
        """
        with torch.no_grad():
            # 原始方法
            human_response_old = torch.max(low_light_feat, dim=1, keepdim=True)[0]
            human_response_old = F.sigmoid(human_response_old * 10 - 3)
            
            # 改进方法 (多线索)
            human_response_new = torch.max(low_light_feat, dim=1, keepdim=True)[0]
            human_response_new = F.sigmoid(human_response_new * 10 - 3)
            
            # 方差线索
            feat_variance = torch.var(low_light_feat, dim=1, keepdim=True)
            feat_variance_norm = (feat_variance - feat_variance.min()) / (feat_variance.max() - feat_variance.min() + 1e-6)
            variance_response = F.sigmoid(feat_variance_norm * 5 - 2)
            
            # 多尺度线索
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
            
            # 学习线索
            human_map_learned = self.fusion.human_estimator(low_light_feat)
            
            # 融合
            human_map_new = (human_map_learned * self.fusion.human_protection_weight + 
                           human_response_new * 0.2 +
                           variance_response * 0.2 +
                           multi_scale_response * 0.2)
            
        return {
            'old_response': human_response_old,
            'new_response': human_map_new,
            'variance_response': variance_response,
            'multi_scale_response': multi_scale_response,
            'learned_response': human_map_learned
        }
    
    def test_fusion_weight(self, low_light_feat, high_res_feat, dark_mask, human_map):
        """
        测试融合权重分布
        """
        with torch.no_grad():
            # 计算差异权重
            feat_diff = torch.abs(low_light_feat - high_res_feat).mean(dim=1, keepdim=True)
            diff_weight = F.sigmoid(feat_diff * self.fusion.diff_sensitivity - 1)
            
            # 暗部权重
            dark_weight = dark_mask.float() * self.fusion.dark_fusion_weight
            
            # 人体权重
            human_weight = human_map * self.fusion.human_protection_weight
            
            # 综合权重
            fuse_weight = (1 - human_weight) * dark_weight * (1 - diff_weight)
            fuse_weight = fuse_weight.clamp(0, 1)
            
        return {
            'fuse_weight': fuse_weight,
            'human_weight': human_weight,
            'dark_weight': dark_weight,
            'diff_weight': diff_weight,
            'fuse_weight_mean': fuse_weight.mean().item(),
            'human_area_weight': (fuse_weight * human_map).sum() / (human_map.sum() + 1e-6)
        }
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        scenarios = self.create_test_scenarios()
        
        print("=" * 80)
        print("细节保护效果测试报告")
        print("=" * 80)
        
        for name, scenario in scenarios.items():
            print(f"\n{'='*60}")
            print(f"场景：{scenario['description']}")
            print(f"{'='*60}")
            
            low_light = scenario['low_light_feat']
            high_res = scenario['high_res_feat']
            dark_mask = scenario['dark_mask']
            
            # 1. 测试人体估计
            human_test = self.test_human_estimation(low_light)
            
            print(f"\n【人体估计对比】")
            print(f"  原始方法响应均值：{human_test['old_response'].mean().item():.4f}")
            print(f"  改进方法响应均值：{human_test['new_response'].mean().item():.4f}")
            print(f"  方差线索响应均值：{human_test['variance_response'].mean().item():.4f}")
            print(f"  多尺度线索响应均值：{human_test['multi_scale_response'].mean().item():.4f}")
            print(f"  学习线索响应均值：{human_test['learned_response'].mean().item():.4f}")
            
            # 2. 测试融合权重
            weight_test = self.test_fusion_weight(low_light, high_res, dark_mask, human_test['new_response'])
            
            print(f"\n【融合权重分布】")
            print(f"  平均融合权重：{weight_test['fuse_weight_mean']:.4f}")
            print(f"  人体区域权重：{weight_test['human_area_weight'].item():.4f}")
            print(f"  暗部权重均值：{weight_test['dark_weight'].mean().item():.4f}")
            print(f"  差异权重均值：{weight_test['diff_weight'].mean().item():.4f}")
            
            # 3. 完整融合测试
            with torch.no_grad():
                fused_feat = self.fusion(low_light, high_res, dark_mask)
                feat_change = torch.abs(fused_feat - low_light_feat).mean()
            
            print(f"\n【特征变化】")
            print(f"  平均特征变化：{feat_change.item():.4f}")
            
            # 4. 评估细节保护效果
            print(f"\n【细节保护评估】")
            human_protection = 1.0 - weight_test['human_area_weight'].item()
            if human_protection > 0.7:
                print(f"  ✓ 人体保护良好 (保护率：{human_protection:.2%})")
            elif human_protection > 0.5:
                print(f"  ⚠ 人体保护中等 (保护率：{human_protection:.2%})")
            else:
                print(f"  ✗ 人体保护不足 (保护率：{human_protection:.2%})")
        
        print(f"\n{'='*80}")
        print("测试完成")
        print(f"{'='*80}")
        
        return scenarios


# 可视化对比
def visualize_improvement():
    """
    可视化改进效果
    """
    import matplotlib.pyplot as plt
    
    fusion = MobileViTAttention()
    fusion.eval()
    
    # 创建测试数据
    low_light = torch.randn(1, 16, 80, 80) * 0.3
    high_res = torch.randn(1, 128, 80, 80)
    dark_mask = torch.zeros(1, 1, 80, 80)
    dark_mask[:, :, 20:60, 20:60] = 1.0
    
    with torch.no_grad():
        # 人体估计
        human_old = torch.max(low_light, dim=1, keepdim=True)[0]
        human_old = F.sigmoid(human_old * 10 - 3)
        
        human_new = fusion.human_estimator(low_light)
        
        # 融合
        fused = fusion(low_light, high_res, dark_mask)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始低光特征
    im0 = axes[0, 0].imshow(low_light.mean(dim=1).squeeze().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Low-Light Feature')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 2. 原始人体估计
    im1 = axes[0, 1].imshow(human_old.squeeze().numpy(), cmap='hot')
    axes[0, 1].set_title('Old Human Response')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3. 改进人体估计
    im2 = axes[0, 2].imshow(human_new.squeeze().numpy(), cmap='hot')
    axes[0, 2].set_title('Improved Human Map (Learned)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 4. 暗部掩码
    im3 = axes[1, 0].imshow(dark_mask.squeeze().numpy(), cmap='gray')
    axes[1, 0].set_title('Dark Mask')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 5. 融合特征
    im4 = axes[1, 1].imshow(fused.mean(dim=1).squeeze().numpy(), cmap='gray')
    axes[1, 1].set_title('Fused Feature')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 6. 特征差异
    diff = torch.abs(fused - low_light).mean(dim=1).squeeze().numpy()
    im5 = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Feature Change (Lower is Better)')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('detail_protection_improvement.png', dpi=150)
    print(f"可视化已保存到：detail_protection_improvement.png")


if __name__ == '__main__':
    # 运行测试
    tester = DetailProtectionTester()
    tester.run_all_tests()
    
    # 可视化
    visualize_improvement()
