import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graphics_utils import batch_quaternion_multiply

def poc_fre(input_data, poc_buf):
    """Positional encoding function"""
    if isinstance(poc_buf, int):
        # Create buffer on the fly if integer is provided
        poc_buf = torch.FloatTensor([(2**i) for i in range(poc_buf)]).to(input_data.device)
    
    output = []
    for freq in poc_buf:
        for func in [torch.sin, torch.cos]:
            output.append(func(input_data * freq))
    return torch.cat([input_data] + output, -1)

class SpatialRelationModule(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # 减小特征维度以节省显存
        mid_dim = feature_dim // 2
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Linear(feature_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, feature_dim)
        )
        
        # 位置编码
        self.position_encoding = nn.Sequential(
            nn.Linear(3, mid_dim),  # 3D坐标
            nn.ReLU(),
            nn.Linear(mid_dim, feature_dim)
        )
        
    def forward(self, features, points):
        # 位置编码
        pos_encoding = self.position_encoding(points)
        
        # 空间注意力
        spatial_weights = self.spatial_attention(features)
        
        # 特征增强
        enhanced_features = features + spatial_weights * pos_encoding
        return enhanced_features

class DynamicFeatureAdapter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.mid_dim = feature_dim // 4
        
        # 特征重要性估计器
        self.importance_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), # Changed from mid_dim to feature_dim
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim), # Ensure output matches input dimension
            nn.Sigmoid()
        )
        
        # 条件特征转换
        self.condition_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), # Changed from mid_dim to feature_dim
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 特征门控单元
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features, condition_features=None):
        # 估计特征重要性
        importance = self.importance_estimator(features)
        
        # 如果有条件特征，进行条件特征转换
        if condition_features is not None:
            condition = self.condition_transform(condition_features)
            # 计算门控值
            gate = self.gate(torch.cat([features, condition], dim=-1))
            # 应用门控和重要性
            adapted_features = features * importance * gate + condition * (1 - gate)
        else:
            # 只应用重要性
            adapted_features = features * importance
            
        return adapted_features

class MultiScaleIntegration(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scales = [1, 2, 4]  # 多尺度系数
        self.mid_dim = feature_dim // 2
        
        # 多尺度特征转换
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, self.mid_dim),
                nn.ReLU(),
                nn.Linear(self.mid_dim, feature_dim)
            ) for _ in self.scales
        ])
        
        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * len(self.scales), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, len(self.scales)),
            nn.Softmax(dim=-1)
        )
        
        # 特征增强
        self.enhancement = nn.Sequential(
            nn.Linear(feature_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, feature_dim)
        )
        
    def forward(self, features, points=None):
        batch_size = features.shape[0]
        
        # 多尺度特征提取
        multi_scale_features = []
        for i, transform in enumerate(self.scale_transforms):
            # 应用不同尺度的特征转换
            scale_feature = transform(features)
            if points is not None:
                # 如果提供了点云数据，考虑空间信息
                scale = self.scales[i]
                scaled_points = points / scale
                scale_feature = scale_feature * (1 + torch.tanh(scaled_points.sum(-1, keepdim=True)))
            multi_scale_features.append(scale_feature)
            
        # 计算尺度注意力权重
        concat_features = torch.cat(multi_scale_features, dim=-1)
        attention_weights = self.scale_attention(concat_features)
        
        # 加权融合多尺度特征
        weighted_features = torch.zeros_like(features)
        for i, scale_feature in enumerate(multi_scale_features):
            weight = attention_weights[..., i:i+1]
            weighted_features += scale_feature * weight
            
        # 特征增强
        enhanced_features = self.enhancement(weighted_features)
        return enhanced_features + features  # 残差连接

class HybridFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256, num_heads=4):
        super().__init__()
        
        # 减小中间特征维度
        self.mid_dim = max(32, feature_dim // 4)
        
        # 简化CNN结构
        self.local_cnn = nn.Sequential(
            nn.Conv1d(3, self.mid_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.mid_dim, feature_dim, kernel_size=1)
        )
        
        # 简化时间编码
        self.time_embedding = nn.Linear(1, feature_dim)
        
        # 轻量级时间注意力
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 空间关系处理
        self.spatial_module = SpatialRelationModule(feature_dim)
        
        # 动态特征适配器
        self.dynamic_adapter = DynamicFeatureAdapter(feature_dim)
        
        # 多尺度集成
        self.multi_scale = MultiScaleIntegration(feature_dim)
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, points, time):
        # 输入处理
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        if time.dim() == 2:
            time = time.unsqueeze(0)
            
        batch_size, num_points, _ = points.shape
        
        # CNN特征提取
        points_features = points.transpose(1, 2)
        local_features = self.local_cnn(points_features)
        local_features = local_features.transpose(1, 2)
        
        # 时间特征
        temporal_features = self.temporal_attention(local_features)
        
        # 空间特征
        spatial_features = self.spatial_module(local_features, points[:,:,:3])
        
        # 动态特征适配
        temporal_features = self.dynamic_adapter(temporal_features, spatial_features)
        
        # 多尺度特征集成
        multi_scale_features = self.multi_scale(temporal_features, points)
        
        # 特征融合
        combined_features = torch.cat([multi_scale_features, spatial_features], dim=-1)
        final_features = self.fusion_layer(combined_features)
        
        if batch_size == 1:
            final_features = final_features.squeeze(0)
            
        return final_features

class UpgradedDeformation(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 设置特征维度与网络宽度一致
        self.feature_dim = args.net_width
        
        self.feature_extractor = HybridFeatureExtractor(
            feature_dim=self.feature_dim,
            num_heads=4
        )

        
        # 保持原有变形预测部分
        from scene.deformation import Deformation
        self.original_deform = Deformation(
            W=args.net_width,
            D=args.defor_depth,
            args=args
        )
        
        self.deformation_net = self.original_deform
        self.args = args
        
        # 修改特征适配器以处理grid特征维度
        self.grid_adapter = nn.Sequential(
            nn.Linear(48, self.feature_dim),  # 将grid特征升维到网络宽度
            nn.ReLU()
        )
        
        # 动态特征适配器使用正确的维度
        self.feature_adapter = DynamicFeatureAdapter(self.feature_dim)
        
        # 多尺度特征集成使用正确的维度
        self.multi_scale_integration = MultiScaleIntegration(self.feature_dim)
        
        # 特征转换层保持维度不变
        self.feature_transform = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),  # 处理拼接后的特征
            nn.ReLU()
        )
        self.initialized_device = False
        
    def initialize_device(self, device):
        """Ensure all components are on the same device"""
        if not self.initialized_device:
            self.feature_extractor = self.feature_extractor.to(device)
            self.original_deform = self.original_deform.to(device)
            self.grid_adapter = self.grid_adapter.to(device)
            self.feature_adapter = self.feature_adapter.to(device)
            self.multi_scale_integration = self.multi_scale_integration.to(device)
            self.feature_transform = self.feature_transform.to(device)
            self.initialized_device = True

    def forward(self, points, scales=None, rotations=None, opacity=None, shs=None, times=None):
        device = points.device
        self.initialize_device(device)
        # 确保时间输入格式正确
        if times is None:
            times = torch.zeros_like(points[:, 0:1])
        elif times.dim() == 1:
            times = times.unsqueeze(-1).to(device)
            
        # 使用新的特征提取器
        features = self.feature_extractor(points, times)

        if self.args.no_grid:
            grid_feature = self.original_deform.grid(points[:,:3], times)
            if self.args.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.args.grid_pe)
            
            # Ensure grid_feature is on the correct device
            grid_feature = grid_feature.to(device)
            grid_feature = self.grid_adapter(grid_feature)
        else:
            grid_feature = self.original_deform.grid(points[:,:3], times)
            if self.args.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.args.grid_pe)
                
            # 先将grid特征升维
            grid_feature = self.grid_adapter(grid_feature)
            
            # 然后应用特征适配和多尺度集成
            adapted_grid = self.feature_adapter(grid_feature)
            multi_scale_grid = self.multi_scale_integration(adapted_grid, points)
            
            # 特征融合
            combined_features = torch.cat([features, multi_scale_grid], dim=-1)
            hidden = self.feature_transform(combined_features)
            
        # 使用原有的变形预测头
        if self.args.no_dx:
            pts = points[:,:3]
        else:
            dx = self.original_deform.pos_deform(hidden)
            pts = points[:,:3] + dx
            
        if self.args.no_ds:
            scales_out = scales[:,:3]
        else:
            ds = self.original_deform.scales_deform(hidden)
            scales_out = scales[:,:3] + ds
            
        if self.args.no_dr:
            rotations_out = rotations[:,:4]
        else:
            dr = self.original_deform.rotations_deform(hidden)
            if self.args.apply_rotation:
                rotations_out = batch_quaternion_multiply(rotations, dr)
            else:
                rotations_out = rotations[:,:4] + dr
            
        if self.args.no_do:
            opacity_out = opacity[:,:1]
        else:
            do = self.original_deform.opacity_deform(hidden)
            opacity_out = opacity[:,:1] + do
            
        if self.args.no_dshs:
            shs_out = shs
        else:
            dshs = self.original_deform.shs_deform(hidden).reshape([shs.shape[0],16,3])
            shs_out = shs + dshs
            
        return pts, scales_out, rotations_out, opacity_out, shs_out
        
    def get_mlp_parameters(self):
        return (list(self.feature_extractor.parameters()) + 
                list(self.feature_transform.parameters()) + 
                list(self.feature_adapter.parameters()) +
                list(self.multi_scale_integration.parameters()) +
                list(self.original_deform.get_mlp_parameters()))
        
    def get_grid_parameters(self):
        return self.original_deform.get_grid_parameters()
        
    def to(self, device):
        self.initialize_device(device)
        return self



