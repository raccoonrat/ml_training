"""
基于强化学习的智能调度器
集成EmSHAP归因信息，实现Shapley值驱动的动态调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger


class PrecisionType(Enum):
    """数值精度类型"""
    FP32 = "fp32"
    FP16 = "fp16" 
    INT8 = "int8"
    BFloat16 = "bfloat16"


class TaskType(Enum):
    """任务类型"""
    VISION = "vision"
    LANGUAGE = "language"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"


@dataclass
class SystemState:
    """系统状态表示"""
    # 系统级宏观信息
    gpu_utilization: List[float]
    memory_utilization: List[float]
    cpu_utilization: float
    memory_bandwidth: List[float]
    pcie_bandwidth: List[float]
    total_power: float
    queue_length: int
    power_budget: float
    
    # 任务级微观信息
    task_type: TaskType
    model_size: int
    input_size: Tuple[int, ...]
    batch_size: int
    
    # EmSHAP归因向量 (关键创新点)
    attribution_vector: Dict[str, float]
    
    def to_tensor(self) -> torch.Tensor:
        """转换为张量表示"""
        # 系统状态特征
        system_features = []
        system_features.extend(self.gpu_utilization)
        system_features.extend(self.memory_utilization)
        system_features.append(self.cpu_utilization)
        system_features.extend(self.memory_bandwidth)
        system_features.extend(self.pcie_bandwidth)
        system_features.extend([self.total_power, self.queue_length, self.power_budget])
        
        # 任务特征
        task_features = []
        task_features.append(self.task_type.value if isinstance(self.task_type.value, (int, float)) else 0)
        task_features.append(self.model_size)
        task_features.extend(self.input_size)
        task_features.append(self.batch_size)
        
        # 归因特征 (EmSHAP输出)
        attribution_features = list(self.attribution_vector.values())
        
        # 合并所有特征
        all_features = system_features + task_features + attribution_features
        return torch.tensor(all_features, dtype=torch.float32)


@dataclass
class SchedulingAction:
    """调度动作"""
    target_gpu: int
    precision: PrecisionType
    batch_size: int
    memory_limit: int
    priority: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_gpu': self.target_gpu,
            'precision': self.precision.value,
            'batch_size': self.batch_size,
            'memory_limit': self.memory_limit,
            'priority': self.priority
        }


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(PolicyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        self.value_head = nn.Linear(prev_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value


class ActionSpace:
    """动作空间定义"""
    
    def __init__(self, num_gpus: int = 4, max_batch_size: int = 128):
        self.num_gpus = num_gpus
        self.max_batch_size = max_batch_size
        self.precision_types = list(PrecisionType)
        
        self.action_dim = (
            num_gpus * 
            len(self.precision_types) * 
            4
        )
        
    def decode_action(self, action_idx: int) -> SchedulingAction:
        gpu_idx = action_idx % self.num_gpus
        remaining = action_idx // self.num_gpus
        
        precision_idx = remaining % len(self.precision_types)
        batch_level = (remaining // len(self.precision_types)) % 4
        
        batch_sizes = [16, 32, 64, self.max_batch_size]
        batch_size = batch_sizes[batch_level]
        
        memory_multiplier = {
            PrecisionType.FP32: 4,
            PrecisionType.FP16: 2,
            PrecisionType.BFloat16: 2,
            PrecisionType.INT8: 1
        }
        
        memory_limit = batch_size * memory_multiplier[self.precision_types[precision_idx]]
        
        return SchedulingAction(
            target_gpu=gpu_idx,
            precision=self.precision_types[precision_idx],
            batch_size=batch_size,
            memory_limit=memory_limit,
            priority=1.0
        )


class RLScheduler:
    """基于强化学习的智能调度器"""
    
    def __init__(self, 
                 state_dim: int,
                 action_space: ActionSpace,
                 algorithm: str = 'ppo',
                 device: str = 'auto'):
        
        self.action_space = action_space
        self.algorithm = algorithm
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # 创建网络
        if algorithm == 'ppo':
            self.policy_net = PolicyNetwork(state_dim, action_space.action_dim).to(self.device)
            self.value_net = self.policy_net
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # 决策历史 (用于可解释性分析)
        self.decision_history = []
        
        logger.info(f"Initialized RL Scheduler with {algorithm} algorithm")
        logger.info(f"State dimension: {state_dim}, Action dimension: {action_space.action_dim}")
    
    def get_action(self, state: SystemState, training: bool = True) -> SchedulingAction:
        state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
        
        if self.algorithm == 'ppo':
            action_logits, value = self.policy_net(state_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if training:
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()
                action_prob = action_probs[0, action_idx].item()
            else:
                action_idx = torch.argmax(action_probs, dim=1)
                action_prob = action_probs[0, action_idx].item()
            
            action_idx = action_idx.item()
        
        # 解码动作
        action = self.action_space.decode_action(action_idx)
        
        # 记录决策 (用于可解释性分析)
        decision_record = {
            'timestamp': np.datetime64('now'),
            'state': {
                'system_features': state_tensor.cpu().numpy().tolist(),
                'attribution_vector': state.attribution_vector
            },
            'action': action.to_dict(),
            'action_idx': action_idx,
            'action_prob': action_prob,
            'algorithm': self.algorithm
        }
        self.decision_history.append(decision_record)
        
        return action
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.policy_net.state_dict(), path)
        
        # 保存决策历史
        history_path = path.replace('.pth', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.decision_history, f, default=str)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
    
    def get_decision_history(self) -> List[Dict]:
        """获取决策历史"""
        return self.decision_history.copy()
