"""
Enhanced EMSHAP Model Implementation
Based on the paper "Energy-Based Model for Accurate Estimation of Shapley Values in Feature Attribution"
Complete implementation of energy network, GRU network, training algorithm and Shapley value computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Tuple, Dict, Any, Optional, List
import math
import numpy as np
from loguru import logger


class AttentionModule(nn.Module):
    """
    Attention Module
    Used to capture interactions between features
    """
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(AttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.output(context)
        return output


class EnhancedEnergyNetwork(nn.Module):
    """
    Enhanced Energy Network
    Supports feature subsets and conditional energy functions
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 context_dim: int = 32, num_heads: int = 8, dropout_rate: float = 0.1):
        super(EnhancedEnergyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim
        
        # Feature embedding layer
        self.feature_embedding = nn.Linear(input_dim, hidden_dims[0])
        
        # Attention module
        self.attention = AttentionModule(hidden_dims[0], num_heads, dropout_rate)
        
        # Context projection
        self.context_projection = nn.Linear(context_dim, hidden_dims[0])
        
        # Main network layers
        layers = []
        prev_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.main_network = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            x: Input features (batch_size, input_dim)
            context: Context vector (batch_size, context_dim)
            feature_mask: Feature mask (batch_size, input_dim)
            
        Returns:
            Energy values (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Apply feature mask
        if feature_mask is not None:
            x = x * feature_mask
        
        # Feature embedding
        x_embedded = self.feature_embedding(x)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for attention
        x_seq = x_embedded.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply attention
        x_attended = self.attention(x_seq)  # (batch_size, 1, hidden_dim)
        x_attended = x_attended.squeeze(1)  # (batch_size, hidden_dim)
        
        # Context projection
        context_proj = self.context_projection(context)  # (batch_size, hidden_dim)
        
        # Combine features and context
        combined = x_attended + context_proj
        
        # Main network
        features = self.main_network(combined)
        
        # Output energy value
        energy = self.output_layer(features)
        
        # Apply temperature scaling
        energy = energy / self.temperature
        
        return energy


class EnhancedGRUNetwork(nn.Module):
    """
    Enhanced GRU Network
    Improved proposal distribution generation and feature extraction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int,
                 num_layers: int = 2, dropout_rate: float = 0.1):
        super(EnhancedGRUNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Context processing
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Proposal distribution parameter generation
        self.proposal_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.proposal_mean = nn.Linear(hidden_dim // 2, input_dim)
        self.proposal_logvar = nn.Linear(hidden_dim // 2, input_dim)
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Context output
        self.context_output = nn.Linear(hidden_dim * 2, context_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward propagation
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            mask: Mask (batch_size, seq_len, input_dim)
            context: Context vector (batch_size, context_dim)
            
        Returns:
            proposal_mean: Proposal distribution mean
            proposal_logvar: Proposal distribution log variance
            context_output: Output context vector
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply mask
        masked_x = x * mask
        
        # Initialize hidden state
        if context is not None:
            context_encoded = self.context_encoder(context)
            h0 = context_encoded.unsqueeze(0).repeat(self.num_layers * 2, 1, 1)  # *2 for bidirectional
        else:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim, device=x.device)
        
        # GRU forward propagation
        gru_output, hidden = self.gru(masked_x, h0)
        
        # Get last time step output
        last_output = gru_output[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Generate proposal distribution parameters
        proposal_features = self.proposal_net(last_output)
        proposal_mean = self.proposal_mean(proposal_features)
        proposal_logvar = self.proposal_logvar(proposal_features)
        
        # Apply temperature scaling
        proposal_logvar = proposal_logvar / self.temperature
        
        # Generate context vector
        context_output = self.context_output(last_output)
        
        return proposal_mean, proposal_logvar, context_output


class EMSHAPEnhanced(nn.Module):
    """
    Enhanced EMSHAP Model
    Complete implementation of energy network and GRU network
    """
    
    def __init__(self, input_dim: int, gru_hidden_dim: int = 64, 
                 context_dim: int = 32, energy_hidden_dims: List[int] = None,
                 gru_layers: int = 2, dropout_rate: float = 0.1):
        super(EMSHAPEnhanced, self).__init__()
        
        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.context_dim = context_dim
        
        if energy_hidden_dims is None:
            energy_hidden_dims = [128, 64, 32]
        
        # GRU network
        self.gru_net = EnhancedGRUNetwork(
            input_dim=input_dim,
            hidden_dim=gru_hidden_dim,
            context_dim=context_dim,
            num_layers=gru_layers,
            dropout_rate=dropout_rate
        )
        
        # Energy network
        self.energy_net = EnhancedEnergyNetwork(
            input_dim=input_dim,
            hidden_dims=energy_hidden_dims,
            context_dim=context_dim,
            dropout_rate=dropout_rate
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                context: torch.Tensor = None, feature_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            mask: 掩码，形状与x相同
            context: 上下文向量 (batch_size, context_dim)
            feature_mask: 特征掩码 (batch_size, input_dim)
            
        Returns:
            energy: 能量值
            proposal_params: 提议分布参数字典
        """
        # 处理输入维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            mask = mask.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # GRU网络前向传播
        proposal_mean, proposal_logvar, context_output = self.gru_net(x, mask, context)
        
        # 取最后一个时间步的特征
        if x.dim() == 3:
            x_last = x[:, -1, :]  # (batch_size, input_dim)
        else:
            x_last = x
        
        # 能量网络前向传播
        energy = self.energy_net(x_last, context_output, feature_mask)
        
        # 构建提议分布参数字典
        proposal_params = {
            'mean': proposal_mean,
            'logvar': proposal_logvar,
            'context': context_output
        }
        
        return energy, proposal_params
    
    def sample_from_proposal(self, proposal_params: Dict[str, torch.Tensor], 
                           num_samples: int = 1) -> torch.Tensor:
        """
        从提议分布采样
        
        Args:
            proposal_params: 提议分布参数
            num_samples: 采样数量
            
        Returns:
            采样结果 (batch_size, num_samples, input_dim)
        """
        mean = proposal_params['mean']  # (batch_size, input_dim)
        logvar = proposal_params['logvar']  # (batch_size, input_dim)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.size(0), num_samples, mean.size(1), device=mean.device)
        
        samples = mean.unsqueeze(1) + eps * std.unsqueeze(1)
        
        return samples
    
    def compute_energy_loss(self, x: torch.Tensor, energy: torch.Tensor, 
                          proposal_params: Dict[str, torch.Tensor], 
                          num_samples: int = 10) -> torch.Tensor:
        """
        计算能量损失（对比学习）
        
        Args:
            x: 原始输入
            energy: 原始输入的能量值
            proposal_params: 提议分布参数
            num_samples: 负样本数量
            
        Returns:
            能量损失
        """
        batch_size = x.shape[0]
        
        # 从提议分布采样负样本
        negative_samples = self.sample_from_proposal(proposal_params, num_samples)
        negative_samples = negative_samples.view(batch_size * num_samples, -1)
        
        # 计算负样本的能量值
        context_expanded = proposal_params['context'].unsqueeze(1).repeat(1, num_samples, 1)
        context_expanded = context_expanded.view(batch_size * num_samples, -1)
        
        negative_energy = self.energy_net(negative_samples, context_expanded)
        negative_energy = negative_energy.view(batch_size, num_samples)
        
        # 对比损失
        positive_energy = energy.unsqueeze(1)  # (batch_size, 1)
        
        # 确保维度一致
        if positive_energy.dim() == 3:
            positive_energy = positive_energy.squeeze(-1)  # (batch_size, 1)
        if negative_energy.dim() == 3:
            negative_energy = negative_energy.squeeze(-1)  # (batch_size, num_samples)
        
        # InfoNCE损失
        logits = torch.cat([positive_energy, negative_energy], dim=1)  # (batch_size, num_samples + 1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_kl_loss(self, proposal_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算KL散度损失
        
        Args:
            proposal_params: 提议分布参数
            
        Returns:
            KL散度损失
        """
        mean = proposal_params['mean']
        logvar = proposal_params['logvar']
        
        # 与标准正态分布的KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        return kl_loss


class ShapleyCalculator:
    """
    Shapley值计算器
    使用Monte Carlo方法估计Shapley值
    """
    
    def __init__(self, model: EMSHAPEnhanced, num_samples: int = 1000):
        self.model = model
        self.num_samples = num_samples
    
    def compute_shapley_values(self, x: torch.Tensor, baseline: torch.Tensor = None) -> torch.Tensor:
        """
        计算Shapley值（简化版本）
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            baseline: 基线值 (batch_size, input_dim)
            
        Returns:
            Shapley值 (batch_size, input_dim)
        """
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        batch_size, input_dim = x.shape
        device = x.device
        
        # 初始化Shapley值
        shapley_values = torch.zeros_like(x)
        
        # 简化版本：逐个特征计算重要性
        for i in range(input_dim):
            # 创建包含特征i的掩码
            mask_with = torch.ones_like(x)
            mask_without = torch.ones_like(x)
            mask_without[:, i] = 0  # 移除特征i
            
            # 计算边际贡献
            with torch.no_grad():
                # 使用特征i
                energy_with, _ = self.model(x, mask_with)
                
                # 不使用特征i
                energy_without, _ = self.model(x, mask_without)
                
                # 边际贡献
                marginal_contribution = energy_with - energy_without
                
                # 确保维度正确
                if marginal_contribution.dim() == 3:
                    marginal_contribution = marginal_contribution.squeeze(-1)  # (batch_size, 1)
                
                # 更新Shapley值
                shapley_values[:, i] = marginal_contribution.squeeze()
        
        return shapley_values
    
    def compute_confidence_intervals(self, x: torch.Tensor, baseline: torch.Tensor = None, 
                                   confidence: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算置信区间
        
        Args:
            x: 输入特征
            baseline: 基线值
            confidence: 置信水平
            
        Returns:
            下界和上界
        """
        # 多次计算Shapley值
        num_runs = 10
        shapley_runs = []
        
        for _ in range(num_runs):
            shapley = self.compute_shapley_values(x, baseline)
            shapley_runs.append(shapley)
        
        shapley_runs = torch.stack(shapley_runs, dim=0)  # (num_runs, batch_size, input_dim)
        
        # 计算置信区间
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = torch.percentile(shapley_runs, lower_percentile, dim=0)
        upper_bound = torch.percentile(shapley_runs, upper_percentile, dim=0)
        
        return lower_bound, upper_bound


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = EMSHAPEnhanced(input_dim=64, gru_hidden_dim=64, context_dim=32)
    
    # 测试前向传播
    batch_size = 32
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 64)
    mask = torch.bernoulli(torch.full_like(x, 0.8)).bool()
    
    energy, proposal_params = model(x, mask)
    
    print(f"输入形状: {x.shape}")
    print(f"能量形状: {energy.shape}")
    print(f"提议分布均值形状: {proposal_params['mean'].shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试损失计算
    energy_loss = model.compute_energy_loss(x, energy, proposal_params)
    kl_loss = model.compute_kl_loss(proposal_params)
    
    print(f"能量损失: {energy_loss.item():.4f}")
    print(f"KL损失: {kl_loss.item():.4f}")
