"""
功耗预测模型
实现一个简单的多层感知机(MLP)用于预测功耗
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PowerPredictor(nn.Module):
    """
    功耗预测模型
    使用多层感知机(MLP)架构预测功耗
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.1, activation: str = 'relu'):
        """
        初始化功耗预测模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，默认为[128, 64, 32]
            dropout_rate: Dropout比率
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid')
        """
        super(PowerPredictor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            预测的功耗值，形状为 (batch_size, 1)
        """
        # 应用激活函数到隐藏层
        for i, layer in enumerate(self.network[:-1]):  # 除了最后一层
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                x = self.activation(x)
        
        # 最后一层（输出层）不使用激活函数
        x = self.network[-1](x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测方法（用于推理）
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class PowerPredictorWithResidual(PowerPredictor):
    """
    带残差连接的功耗预测模型
    在原始MLP基础上添加残差连接以提高性能
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.1, activation: str = 'relu'):
        super().__init__(input_dim, hidden_dims, dropout_rate, activation)
        
        # 为残差连接创建投影层
        self.residual_projection = nn.Linear(input_dim, hidden_dims[-1]) if hidden_dims else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        带残差连接的前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果
        """
        residual = x
        
        # 前向传播
        for i, layer in enumerate(self.network[:-1]):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            else:
                x = self.activation(x)
        
        # 添加残差连接（如果维度匹配）
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
            if x.shape == residual.shape:
                x = x + residual
        
        # 输出层
        x = self.network[-1](x)
        
        return x


def create_power_predictor(input_dim: int, model_type: str = 'mlp', 
                          **kwargs) -> PowerPredictor:
    """
    创建功耗预测模型的工厂函数
    
    Args:
        input_dim: 输入特征维度
        model_type: 模型类型 ('mlp', 'residual')
        **kwargs: 其他参数
        
    Returns:
        功耗预测模型实例
    """
    if model_type == 'mlp':
        return PowerPredictor(input_dim, **kwargs)
    elif model_type == 'residual':
        return PowerPredictorWithResidual(input_dim, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = PowerPredictor(input_dim=64, hidden_dims=[128, 64, 32])
    
    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, 64)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
