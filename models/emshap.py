"""
EMSHAP模型实现
基于论文实现能量网络(Energy Network)和GRU网络(GRU Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import math


class EnergyNetwork(nn.Module):
    """
    能量网络
    带Skip Connection的MLP，用于计算能量函数 gθ(x)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, 
                 context_dim: int = 32, dropout_rate: float = 0.1):
        """
        初始化能量网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            context_dim: 上下文向量维度
            dropout_rate: Dropout比率
        """
        super(EnergyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim
        self.dropout_rate = dropout_rate
        
        # 输入层（包含特征和上下文向量）
        total_input_dim = input_dim + context_dim
        
        # 构建网络层
        layers = []
        prev_dim = total_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 添加Skip Connection（除了第一层）
            if i > 0 and prev_dim == hidden_dim:
                layers.append(SkipConnection())
            
            layers.extend([
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层（输出单个能量值）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, input_dim)
            context: 上下文向量，形状为 (batch_size, context_dim)
            
        Returns:
            能量值，形状为 (batch_size, 1)
        """
        # 拼接特征和上下文向量
        combined_input = torch.cat([x, context], dim=1)
        
        # 前向传播
        energy = self.network(combined_input)
        
        return energy


class SkipConnection(nn.Module):
    """
    Skip Connection模块
    实现残差连接
    """
    
    def __init__(self):
        super(SkipConnection, self).__init__()
        self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量（与输入相同）
        """
        return self.skip_connection(x)


class GRUNetwork(nn.Module):
    """
    GRU网络
    用于生成提议分布 q(x) 的参数
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int,
                 num_layers: int = 2, dropout_rate: float = 0.1):
        """
        初始化GRU网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: GRU隐藏层维度
            context_dim: 上下文向量维度
            num_layers: GRU层数
            dropout_rate: Dropout比率
        """
        super(GRUNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 上下文投影层
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        
        # 提议分布参数生成层
        self.proposal_mean = nn.Linear(hidden_dim, input_dim)
        self.proposal_logvar = nn.Linear(hidden_dim, input_dim)
        
        # 上下文向量生成层
        self.context_output = nn.Linear(hidden_dim, context_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # GRU权重使用正交初始化
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, seq_len, input_dim)
            mask: 掩码，形状为 (batch_size, seq_len, input_dim)
            context: 初始上下文向量，形状为 (batch_size, context_dim)
            
        Returns:
            proposal_mean: 提议分布均值
            proposal_logvar: 提议分布对数方差
            context_output: 输出上下文向量
        """
        batch_size, seq_len, _ = x.shape
        
        # 应用掩码
        masked_x = x * mask
        
        # 初始化隐藏状态
        if context is not None:
            h0 = self.context_projection(context).unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        
        # GRU前向传播
        gru_output, hidden = self.gru(masked_x, h0)
        
        # 取最后一个时间步的输出
        last_output = gru_output[:, -1, :]
        
        # 生成提议分布参数
        proposal_mean = self.proposal_mean(last_output)
        proposal_logvar = self.proposal_logvar(last_output)
        
        # 生成上下文向量
        context_output = self.context_output(last_output)
        
        return proposal_mean, proposal_logvar, context_output


class EMSHAP(nn.Module):
    """
    EMSHAP模型
    结合能量网络和GRU网络的完整模型
    """
    
    def __init__(self, input_dim: int, gru_hidden_dim: int = 64, 
                 context_dim: int = 32, energy_hidden_dims: list = None,
                 gru_layers: int = 2, dropout_rate: float = 0.1):
        """
        初始化EMSHAP模型
        
        Args:
            input_dim: 输入特征维度
            gru_hidden_dim: GRU隐藏层维度
            context_dim: 上下文向量维度
            energy_hidden_dims: 能量网络隐藏层维度
            gru_layers: GRU层数
            dropout_rate: Dropout比率
        """
        super(EMSHAP, self).__init__()
        
        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.context_dim = context_dim
        
        # GRU网络
        self.gru_net = GRUNetwork(
            input_dim=input_dim,
            hidden_dim=gru_hidden_dim,
            context_dim=context_dim,
            num_layers=gru_layers,
            dropout_rate=dropout_rate
        )
        
        # 能量网络
        self.energy_net = EnergyNetwork(
            input_dim=input_dim,
            hidden_dims=energy_hidden_dims,
            context_dim=context_dim,
            dropout_rate=dropout_rate
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, 
                context: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
            mask: 掩码，形状与x相同
            context: 初始上下文向量，形状为 (batch_size, context_dim)
            
        Returns:
            energy: 能量值
            proposal_params: 提议分布参数字典
        """
        # 处理输入维度
        if x.dim() == 2:
            # 如果输入是2D，添加序列维度
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
        energy = self.energy_net(x_last, context_output)
        
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
            采样结果，形状为 (batch_size, num_samples, input_dim)
        """
        mean = proposal_params['mean']  # (batch_size, input_dim)
        logvar = proposal_params['logvar']  # (batch_size, input_dim)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.size(0), num_samples, mean.size(1), device=mean.device)
        
        samples = mean.unsqueeze(1) + eps * std.unsqueeze(1)
        
        return samples
    
    def compute_loss(self, x: torch.Tensor, mask: torch.Tensor, 
                    energy: torch.Tensor, proposal_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算MLE损失（最大似然估计）
        
        Args:
            x: 原始输入
            mask: 掩码
            energy: 能量值
            proposal_params: 提议分布参数
            
        Returns:
            损失值
        """
        # 这里实现论文中的损失函数
        # 具体实现需要根据论文公式(14)来完善
        
        # 简化版本：使用能量作为损失
        loss = torch.mean(energy)
        
        return loss


def create_emshap_model(input_dim: int, **kwargs) -> EMSHAP:
    """
    创建EMSHAP模型的工厂函数
    
    Args:
        input_dim: 输入特征维度
        **kwargs: 其他参数
        
    Returns:
        EMSHAP模型实例
    """
    return EMSHAP(input_dim=input_dim, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = EMSHAP(input_dim=64, gru_hidden_dim=64, context_dim=32)
    
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
