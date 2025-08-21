"""
调度器可解释性分析模块
使用Shapley值解释RL调度器的决策过程
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import json
from dataclasses import dataclass

from models.rl_scheduler import RLScheduler, SystemState, SchedulingAction


@dataclass
class DecisionExplanation:
    """决策解释结果"""
    decision_id: str
    timestamp: str
    action: Dict[str, Any]
    feature_contributions: Dict[str, float]
    attribution_contributions: Dict[str, float]
    system_contributions: Dict[str, float]
    key_factors: List[str]
    reasoning: str
    confidence: float


class SchedulerExplainer:
    """调度器可解释性分析器"""
    
    def __init__(self, scheduler: RLScheduler, device: str = 'auto'):
        self.scheduler = scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.feature_names = self._build_feature_names()
        logger.info("Initialized Scheduler Explainer")
    
    def _build_feature_names(self) -> List[str]:
        """构建特征名称列表"""
        names = []
        names.extend([f"gpu_util_{i}" for i in range(4)])
        names.extend([f"mem_util_{i}" for i in range(4)])
        names.append("cpu_util")
        names.extend([f"mem_bw_{i}" for i in range(4)])
        names.extend([f"pcie_bw_{i}" for i in range(4)])
        names.extend(["total_power", "queue_length", "power_budget"])
        names.extend(["task_type", "model_size", "input_h", "input_w", "batch_size"])
        names.extend([
            "attribution_cpu_cycles",
            "attribution_l3_misses", 
            "attribution_memory_bandwidth",
            "attribution_pcie_bandwidth",
            "attribution_operator_conv",
            "attribution_operator_fc",
            "attribution_operator_attention"
        ])
        return names
    
    def explain_decision(self, decision_record: Dict[str, Any]) -> DecisionExplanation:
        """解释单个调度决策"""
        state_features = np.array(decision_record['state']['system_features'][0])
        attribution_vector = decision_record['state']['attribution_vector']
        action = decision_record['action']
        action_idx = decision_record['action_idx']
        
        # 计算Shapley值
        shapley_values = self._compute_shapley_values(state_features, action_idx)
        
        # 分类特征贡献
        feature_contributions = {}
        attribution_contributions = {}
        system_contributions = {}
        
        for i, (name, value) in enumerate(zip(self.feature_names, shapley_values)):
            if name.startswith('attribution_'):
                attribution_contributions[name] = value
            elif name in ['gpu_util_', 'mem_util_', 'cpu_util', 'mem_bw_', 'pcie_bw_', 
                         'total_power', 'queue_length', 'power_budget']:
                system_contributions[name] = value
            else:
                feature_contributions[name] = value
        
        # 生成解释性洞察
        key_factors = self._identify_key_factors(shapley_values)
        reasoning = self._generate_reasoning(action, attribution_contributions, system_contributions)
        confidence = self._calculate_confidence(shapley_values)
        
        return DecisionExplanation(
            decision_id=decision_record.get('timestamp', 'unknown'),
            timestamp=str(decision_record.get('timestamp', '')),
            action=action,
            feature_contributions=feature_contributions,
            attribution_contributions=attribution_contributions,
            system_contributions=system_contributions,
            key_factors=key_factors,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def _compute_shapley_values(self, state_features: np.ndarray, action_idx: int, num_samples: int = 1000) -> np.ndarray:
        """计算Shapley值 (Monte Carlo采样)"""
        n_features = len(state_features)
        shapley_values = np.zeros(n_features)
        
        baseline_pred = self._predict_action_prob(state_features * 0, action_idx)
        
        for _ in range(num_samples):
            feature_mask = np.random.choice([0, 1], size=n_features, p=[0.5, 0.5])
            
            for i in range(n_features):
                mask_with_i = feature_mask.copy()
                mask_without_i = feature_mask.copy()
                
                mask_with_i[i] = 1
                mask_without_i[i] = 0
                
                pred_with_i = self._predict_action_prob(state_features * mask_with_i, action_idx)
                pred_without_i = self._predict_action_prob(state_features * mask_without_i, action_idx)
                
                marginal_contribution = pred_with_i - pred_without_i
                shapley_values[i] += marginal_contribution / num_samples
        
        return shapley_values
    
    def _predict_action_prob(self, state_features: np.ndarray, target_action: int) -> float:
        """预测特定动作的概率"""
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, _ = self.scheduler.policy_net(state_tensor)
            action_probs = torch.softmax(action_logits, dim=1)
            return action_probs[0, target_action].item()
    
    def _identify_key_factors(self, shapley_values: np.ndarray, top_k: int = 5) -> List[str]:
        """识别关键影响因素"""
        abs_values = np.abs(shapley_values)
        top_indices = np.argsort(abs_values)[-top_k:][::-1]
        
        key_factors = []
        for idx in top_indices:
            factor_name = self.feature_names[idx]
            contribution = shapley_values[idx]
            key_factors.append(f"{factor_name}: {contribution:.4f}")
        
        return key_factors
    
    def _generate_reasoning(self, action: Dict[str, Any], attribution_contributions: Dict[str, float], system_contributions: Dict[str, float]) -> str:
        """生成决策推理"""
        top_attribution = max(attribution_contributions.items(), key=lambda x: abs(x[1]))
        top_system = max(system_contributions.items(), key=lambda x: abs(x[1]))
        
        reasoning_parts = []
        
        if top_attribution[1] > 0.1:
            if 'memory_bandwidth' in top_attribution[0]:
                reasoning_parts.append("任务显示内存密集型特征")
            elif 'cpu_cycles' in top_attribution[0]:
                reasoning_parts.append("任务显示计算密集型特征")
            elif 'pcie_bandwidth' in top_attribution[0]:
                reasoning_parts.append("任务显示I/O密集型特征")
        
        if 'gpu_util' in top_system[0]:
            gpu_id = top_system[0].split('_')[-1]
            if top_system[1] < -0.1:
                reasoning_parts.append(f"GPU {gpu_id} 利用率较低，适合分配任务")
            else:
                reasoning_parts.append(f"GPU {gpu_id} 负载较高，避免分配")
        
        if action['precision'] == 'int8':
            reasoning_parts.append("选择INT8精度以降低能耗")
        elif action['precision'] == 'fp16':
            reasoning_parts.append("选择FP16精度平衡性能和能效")
        
        return "；".join(reasoning_parts) if reasoning_parts else "基于综合因素做出决策"
    
    def _calculate_confidence(self, shapley_values: np.ndarray) -> float:
        """计算决策置信度"""
        total_contribution = np.sum(np.abs(shapley_values))
        max_contribution = np.max(np.abs(shapley_values))
        confidence = min(1.0, max_contribution / max(total_contribution, 1e-6))
        return confidence
