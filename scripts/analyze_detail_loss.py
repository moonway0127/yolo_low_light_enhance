# 目标细节丢失分析工具
# 用于诊断融合过程中是否丢失目标细节

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from fusion_module import MobileViTAttention

class DetailLossAnalyzer:
    """
    细节丢失分析器
    分析融合过程中目标区域的特征变化
    """
    
    def __init__(self, fusion_module: MobileViTAttention):
        self.fusion = fusion_module
        self.hooks = []
        self.activations = {}
        
    def register_hooks(self):
        """注册钩子函数捕获中间特征"""
        
        def make_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        # 注册钩子
        self.hooks.append(self.fusion.human_estimator.register_forward_hook(make_hook('human_map')))
        self.hooks.append(self.fusion.channel_attention.register_forward_hook(make_hook('channel_weights')))
        
    def analyze_fusion(self, low_light_feat, high_res_feat, dark_mask, target_mask=None):
        """
        分析融合过程中的细节损失
        
        Args:
            low_light_feat: 低光特征 (B, C, H, W)
            high_res_feat: 高清特征
            dark_mask: 暗部掩码 (B, 1, H, W)
            target_mask: 目标掩码 (如果有 GT)(B, 1, H, W)
            
        Returns:
            分析报告字典
        """
        self.activations = {}
        
        with torch.no_grad():
            # 执行融合
            fused_feat = self.fusion(low_light_feat, high_res_feat, dark_mask)
            
            # 获取中间结果
            human_map = self.activations.get('human_map', None)
            channel_weights = self.activations.get('channel_weights', None)
            
            # 1. 计算人体区域保护程度
            if human_map is not None:
                human_protection_rate = self._analyze_human_protection(
                    human_map, target_mask
                )
            else:
                human_protection_rate = {'error': 'human_map not captured'}
            
            # 2. 计算融合权重分布
            fuse_weight_stats = self._analyze_fuse_weight(
                low_light_feat, high_res_feat, dark_mask, human_map
            )
            
            # 3. 计算特征相似度变化
            feature_sim = self._analyze_feature_similarity(
                low_light_feat, fused_feat, target_mask
            )
            
            # 4. 计算通道注意力影响
            if channel_weights is not None:
                channel_impact = self._analyze_channel_attention(
                    channel_weights, target_mask
                )
            else:
                channel_impact = {'error': 'channel_weights not captured'}
            
            # 5. 检测潜在细节丢失区域
            detail_loss_map = self._detect_detail_loss(
                low_light_feat, fused_feat, human_map, dark_mask
            )
            
        return {
            'human_protection': human_protection_rate,
            'fuse_weight_stats': fuse_weight_stats,
            'feature_similarity': feature_sim,
            'channel_attention_impact': channel_impact,
            'detail_loss_map': detail_loss_map,
            'fused_feat': fused_feat,
            'low_light_feat': low_light_feat
        }
    
    def _analyze_human_protection(self, human_map, target_mask):
        """分析人体区域保护程度"""
        if target_mask is None:
            # 无 GT 时，统计人体掩码的分布
            return {
                'human_map_mean': human_map.mean().item(),
                'human_map_std': human_map.std().item(),
                'high_human_ratio': (human_map > 0.5).float().mean().item()
            }
        
        # 有 GT 时，计算保护率
        with torch.no_grad():
            target_area = (target_mask > 0.5).float()
            human_response = (human_map > 0.5).float()
            
            # 真正率：目标区域被正确识别为人体
            tp = (target_area * human_response).sum()
            fn = (target_area * (1 - human_response)).sum()
            
            protection_rate = tp / (tp + fn + 1e-6)
            
            return {
                'protection_rate': protection_rate.item(),
                'human_map_mean': human_map.mean().item(),
                'target_coverage': (human_response * target_area).sum() / (target_area.sum() + 1e-6)
            }
    
    def _analyze_fuse_weight(self, low_light_feat, high_res_feat, dark_mask, human_map):
        """分析融合权重分布"""
        with torch.no_grad():
            # 重建融合权重计算过程
            if human_map is None:
                human_map = self.fusion.human_estimator(low_light_feat)
            
            feat_diff = torch.abs(low_light_feat - high_res_feat).mean(dim=1, keepdim=True)
            diff_weight = F.sigmoid(feat_diff * self.fusion.diff_sensitivity - 1)
            dark_weight = dark_mask.float() * self.fusion.dark_fusion_weight
            
            human_weight = human_map * self.fusion.human_protection_weight
            
            fuse_weight = (1 - human_weight) * dark_weight * (1 - diff_weight)
            fuse_weight = fuse_weight.clamp(0, 1)
            
            return {
                'fuse_weight_mean': fuse_weight.mean().item(),
                'fuse_weight_std': fuse_weight.std().item(),
                'high_fuse_ratio': (fuse_weight > 0.5).float().mean().item(),
                'dark_area_fuse_weight': (fuse_weight * dark_mask).sum() / (dark_mask.sum() + 1e-6),
                'human_area_fuse_weight': (fuse_weight * human_map).sum() / (human_map.sum() + 1e-6)
            }
    
    def _analyze_feature_similarity(self, low_light_feat, fused_feat, target_mask):
        """分析特征相似度变化"""
        with torch.no_grad():
            # 计算融合前后特征差异
            feat_diff = torch.abs(fused_feat - low_light_feat).mean(dim=1, keepdim=True)
            
            if target_mask is not None:
                # 目标区域和非目标区域的特征变化
                target_area = (target_mask > 0.5).float()
                bg_area = 1 - target_area
                
                target_feat_change = (feat_diff * target_area).sum() / (target_area.sum() + 1e-6)
                bg_feat_change = (feat_diff * bg_area).sum() / (bg_area.sum() + 1e-6)
                
                return {
                    'overall_feat_change': feat_diff.mean().item(),
                    'target_feat_change': target_feat_change.item(),
                    'bg_feat_change': bg_feat_change.item(),
                    'target_bg_ratio': (target_feat_change / (bg_feat_change + 1e-6)).item()
                }
            
            return {
                'overall_feat_change': feat_diff.mean().item(),
                'feat_change_std': feat_diff.std().item()
            }
    
    def _analyze_channel_attention(self, channel_weights, target_mask):
        """分析通道注意力的影响"""
        with torch.no_grad():
            # 统计通道权重分布
            channel_stats = {
                'channel_weights_mean': channel_weights.mean().item(),
                'channel_weights_std': channel_weights.std().item(),
                'strong_channels': (channel_weights > 0.7).float().sum().item(),
                'weak_channels': (channel_weights < 0.3).float().sum().item()
            }
            
            if target_mask is not None:
                # 分析目标区域和非目标区域的通道注意力差异
                target_area = (target_mask > 0.5).float()
                bg_area = 1 - target_area
                
                # 空间平均通道权重
                spatial_avg_weights = channel_weights.mean(dim=[2, 3], keepdim=True)
                
                target_attention = (spatial_avg_weights * target_area).sum() / (target_area.sum() + 1e-6)
                bg_attention = (spatial_avg_weights * bg_area).sum() / (bg_area.sum() + 1e-6)
                
                channel_stats['target_attention'] = target_attention.item()
                channel_stats['bg_attention'] = bg_attention.item()
                channel_stats['attention_bias'] = (target_attention - bg_attention).item()
            
            return channel_stats
    
    def _detect_detail_loss(self, low_light_feat, fused_feat, human_map, dark_mask):
        """检测潜在的细节丢失区域"""
        with torch.no_grad():
            # 特征变化大的区域
            feat_change = torch.abs(fused_feat - low_light_feat).mean(dim=1, keepdim=True)
            
            # 归一化
            feat_change_norm = (feat_change - feat_change.min()) / (feat_change.max() - feat_change.min() + 1e-6)
            
            # 潜在细节丢失区域：特征变化大 + 人体区域 + 暗部
            potential_loss = feat_change_norm * human_map * dark_mask
            
            # 风险等级
            high_risk = (potential_loss > 0.7).float()
            medium_risk = ((potential_loss > 0.4) & (potential_loss <= 0.7)).float()
            low_risk = ((potential_loss > 0.2) & (potential_loss <= 0.4)).float()
            
            return {
                'risk_map': potential_loss,
                'high_risk_pixels': high_risk.sum().item(),
                'medium_risk_pixels': medium_risk.sum().item(),
                'low_risk_pixels': low_risk.sum().item(),
                'total_pixels': potential_loss.numel(),
                'risk_ratio': high_risk.sum() / (potential_loss.numel() + 1e-6)
            }
    
    def visualize_analysis(self, analysis_result, save_path='analysis_vis.png'):
        """可视化分析结果"""
        import matplotlib.pyplot as plt
        
        detail_loss_map = analysis_result['detail_loss_map']['risk_map']
        human_map = analysis_result.get('human_protection', {})
        
        # 如果有 human_map
        if 'human_map_mean' in human_map:
            # 从 fusion module 重新获取
            pass
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 风险热力图
        im0 = axes[0, 0].imshow(detail_loss_map.squeeze().cpu().numpy(), cmap='hot')
        axes[0, 0].set_title('Detail Loss Risk Map')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # 2. 融合特征
        fused = analysis_result['fused_feat'].mean(dim=1).squeeze().cpu().numpy()
        im1 = axes[0, 1].imshow(fused, cmap='gray')
        axes[0, 1].set_title('Fused Feature')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. 原始特征
        original = analysis_result['low_light_feat'].mean(dim=1).squeeze().cpu().numpy()
        im2 = axes[1, 0].imshow(original, cmap='gray')
        axes[1, 0].set_title('Original Low-Light Feature')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 4. 特征差异
        diff = torch.abs(analysis_result['fused_feat'] - analysis_result['low_light_feat'])
        diff = diff.mean(dim=1).squeeze().cpu().numpy()
        im3 = axes[1, 1].imshow(diff, cmap='hot')
        axes[1, 1].set_title('Feature Change')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"分析可视化已保存到：{save_path}")
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()


# 使用示例
if __name__ == '__main__':
    # 示例：分析单张图像
    fusion = MobileViTAttention()
    analyzer = DetailLossAnalyzer(fusion)
    analyzer.register_hooks()
    
    # 创建测试数据
    low_light_feat = torch.randn(1, 16, 80, 80)
    high_res_feat = torch.randn(1, 128, 80, 80)
    dark_mask = torch.zeros(1, 1, 80, 80)
    dark_mask[:, :, 20:60, 20:60] = 1.0  # 模拟暗部区域
    
    # 分析
    result = analyzer.analyze_fusion(low_light_feat, high_res_feat, dark_mask)
    
    # 打印报告
    print("\n=== 融合分析报告 ===")
    print(f"人体保护率：{result['human_protection']}")
    print(f"\n融合权重统计：{result['fuse_weight_stats']}")
    print(f"\n特征相似度：{result['feature_similarity']}")
    print(f"\n通道注意力影响：{result['channel_attention_impact']}")
    print(f"\n细节丢失风险：{result['detail_loss_map']}")
    
    # 可视化
    analyzer.visualize_analysis(result, 'detail_analysis.png')
    
    analyzer.remove_hooks()
