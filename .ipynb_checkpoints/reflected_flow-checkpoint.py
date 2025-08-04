import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(x + self.layers(x))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        B, D = x.shape
        
        
        x_reshaped = x.view(B, D, 1)  # (B, D, 1)
        
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, self.num_heads, self.head_dim).transpose(1, 2)  
        v = self.v_proj(x).view(B, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scale = (self.head_dim ** -0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1)
        
        output = self.out_proj(attn_output)
        return self.norm(x + output)


class FlowNetwork(nn.Module):
    """速度场网络"""
    def __init__(self, dim, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1, use_attention=True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # 时间嵌入 - 使用正弦位置编码
        time_embed_dim = hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 空间位置嵌入
        self.spatial_embed = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 主干网络：交替使用残差块和注意力块
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ResidualBlock(hidden_dim, dropout))
            if use_attention and i % 2 == 1:  # 每两层添加一个注意力块
                self.layers.append(AttentionBlock(hidden_dim, num_heads))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, dim)
        )
        
        # 条件缩放网络 - 根据时间调整输出幅度
        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, dim),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        # 时间嵌入
        t_embed = self.time_embed(t.float())
        
        # 空间嵌入
        x_embed = self.spatial_embed(x)
        
        # 融合时空特征
        h = self.input_proj(torch.cat([x_embed, t_embed], dim=-1))
        
        # 通过主干网络
        for layer in self.layers:
            h = layer(h)
        
        # 输出速度场
        velocity = self.output_proj(h)
        
        # 时间条件缩放
        scale = self.scale_net(t_embed)
        velocity = velocity * scale
        
        return velocity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class SphericalFlowModel(nn.Module):
    """Spherical Reflected Flow (Reflected Flow adapted to unit hypersphere)"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.velocity_net = FlowNetwork(latent_dim, hidden_dim=512, num_layers=8, use_attention=True)

    def slerp(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # Ensure unit vectors
        x0 = F.normalize(x0, dim=-1)
        x1 = F.normalize(x1, dim=-1)

        dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)

        t = t.unsqueeze(-1)
        coeff0 = torch.sin((1 - t) * omega) / sin_omega
        coeff1 = torch.sin(t * omega) / sin_omega

        return coeff0 * x0 + coeff1 * x1

    def tangent_velocity(self, x0: Tensor, x1: Tensor) -> Tensor:
        """在超球面上计算x0到x1路径的单位切向量"""
        dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        tangent = (x1 - dot * x0)
        return tangent / (sin_omega + 1e-6)

    def velocity_field(self, x: Tensor, t: Tensor) -> Tensor:
        return self.velocity_net(x, t)

    def forward_flow(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return self.slerp(x0, x1, t)

    def backward_flow(self, x1: Tensor, num_steps: int = 100) -> Tensor:
        """反向积分：从噪声走向数据分布"""
        dt = 1.0 / num_steps
        x = x1.clone()

        for i in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device)
            v = self.velocity_field(x, t)
            x = x - v * dt
            x = F.normalize(x, dim=-1)  # 保持在球面上

        return x

    def sample_noise(self, batch_size: int, device) -> Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=device)
        z = F.normalize(z, dim=-1)
        return z

    def flow_matching_loss(self, x0: Tensor) -> Tensor:
        batch_size = x0.size(0)
        device = x0.device
        x1 = self.sample_noise(batch_size, device)
        t = torch.rand(batch_size, device=device)

        x_t = self.forward_flow(x0, x1, t)
        target_velocity = self.tangent_velocity(x_t, x1)
        predicted_velocity = self.velocity_field(x_t, t)

        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss

    def train_step(self, x0_batch: Tensor) -> Tensor:
        return self.flow_matching_loss(x0_batch)

    @torch.no_grad()
    def sample(self, num_samples: int, device, num_steps: int = 200) -> Tensor:
        x = self.sample_noise(num_samples, device)
        x = self.backward_flow(x, num_steps)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.train_step(x)


class ReflectedFlowModel(nn.Module):
    """Reflected Flow"""
    def __init__(self, latent_dim, bounds=(-1.0,1.0)):
        super().__init__()
        self.latent_dim = latent_dim
        self.bounds = bounds
        self.velocity_net = FlowNetwork(latent_dim, hidden_dim=512, num_layers=8, use_attention=True)
        
    def reflect_point(self, x):
        x_reflected = x.clone()
        lower, upper = self.bounds
        mask_lower = x < lower
        x_reflected[mask_lower] = 2 * lower - x[mask_lower]
         
        mask_upper = x > upper
        x_reflected[mask_upper] = 2 * upper - x[mask_upper]
        
        return x_reflected
    
    def get_reflection_multiplier(self, x):
        multiplier = torch.ones_like(x)
        lower, upper = self.bounds
        
        mask_lower = x < lower
        mask_upper = x > upper
        
        multiplier[mask_lower] = -1.0
        multiplier[mask_upper] = -1.0
        
        return multiplier
    
    # def velocity_field(self, x, t):
    #     x_reflected = self.reflect_point(x)
        
    #     v = self.velocity_net(x_reflected, t)
        
    #     multiplier = self.get_reflection_multiplier(x)
        
    #     v_reflected = v * multiplier
        
    #     return v_reflected
    
    def forward_flow(self, x0, x1, t):
        # 直接使用线性插值：x_t = (1-t) * x0 + t * x1
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        return x_t
    
    def backward_flow(self, x1, num_steps=200):
        """反向流：从噪声分布到数据分布"""
        dt = 1.0 / num_steps
        x = x1.clone()
        
        for i in range(num_steps):
            
            t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device)
            v = v = self.velocity_net(x, t)
            x = x - v * dt  # 反向积分
            x = F.normalize(x, dim=-1)

        return x
    
    def sample_noise(self, batch_size, device):
        """从边界内的均匀分布采样噪声"""
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        noise = F.normalize(noise, dim=-1)  # 投影到单位超球面
        return noise

    
    def flow_matching_loss(self, x0):
        """Flow matching损失"""
        batch_size = x0.shape[0]
        device = x0.device
        
        # 采样目标噪声
        x1 = self.sample_noise(batch_size, device)
        
        # 随机采样时间
        t = torch.rand(batch_size, device=device)
        
        # 使用线性插值获得中间状态
        x_t = self.forward_flow(x0, x1, t)
        
        # 目标速度场（沿直线的切向量）
        target_velocity = x1 - x0
        
        # 预测速度场
        predicted_velocity = self.velocity_net(x_t, t)
        
        # MSE损失
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def train_step(self, x0_batch):
        """训练步骤"""
        # 直接计算flow matching损失
        loss = self.flow_matching_loss(x0_batch)
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, device, num_steps=200):
        """生成样本"""
        # 从噪声开始
        x = self.sample_noise(num_samples, device)
        
        # 反向流采样
        x = self.backward_flow(x, num_steps)
        
        return x
    
    def forward(self, x):
        """训练时调用"""
        return self.train_step(x)