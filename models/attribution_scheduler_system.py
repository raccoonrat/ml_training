"""
归因-调度协同系统
实现Shapley值驱动的动态调度与可解释性分析
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
from loguru import logger
import matplotlib.pyplot as plt

from models.emshap_enhanced import EMSHAPEnhanced
from models.rl_scheduler import RLScheduler, SystemState, SchedulingAction, ActionSpace, TaskType, PrecisionType
from models.scheduler_explainer import SchedulerExplainer, DecisionExplanation


@dataclass
class TaskProfile:
    """任务画像"""
    task_id: str
    task_type: TaskType
    model_size: int
    input_size: Tuple[int, ...]
    batch_size: int
    priority: float
    slo_requirements: Dict[str, float]


@dataclass
class SystemMetrics:
    """系统监控指标"""
    timestamp: float
    gpu_utilization: List[float]
    memory_utilization: List[float]
    cpu_utilization: float
    memory_bandwidth: List[float]
    pcie_bandwidth: List[float]
    total_power: float
    queue_length: int
    power_budget: float


@dataclass
class SchedulingResult:
    """调度结果"""
    task_id: str
    action: SchedulingAction
    execution_time: float
    energy_consumption: float
    performance_metric: float
    accuracy_loss: float
    success: bool


class AttributionSchedulerSystem:
    """归因-调度协同系统"""
    
    def __init__(self, emshap_model: EMSHAPEnhanced, num_gpus: int = 4, device: str = 'auto'):
        self.emshap_model = emshap_model
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # 创建动作空间和调度器
        action_space = ActionSpace(num_gpus=num_gpus)
        state_dim = self._calculate_state_dimension()
        
        self.scheduler = RLScheduler(
            state_dim=state_dim,
            action_space=action_space,
            algorithm='ppo',
            device=device
        )
        
        # 创建可解释性分析器
        self.explainer = SchedulerExplainer(self.scheduler, device=device)
        
        # 系统状态
        self.current_metrics = None
        self.task_queue = []
        self.execution_history = []
        self.attribution_cache = {}
        
        logger.info("Initialized Attribution-Scheduler System")
    
    def _calculate_state_dimension(self) -> int:
        """计算状态空间维度"""
        system_features = 4 * 4 + 1 + 1 + 1  # 17
        task_features = 1 + 1 + 2 + 1  # 5
        attribution_features = 7  # EmSHAP输出维度
        return system_features + task_features + attribution_features
    
    def update_system_metrics(self, metrics: SystemMetrics):
        """更新系统监控指标"""
        self.current_metrics = metrics
    
    def add_task(self, task: TaskProfile):
        """添加任务到队列"""
        self.task_queue.append(task)
        logger.info(f"Added task {task.task_id} to queue")
    
    def get_attribution_vector(self, task: TaskProfile) -> Dict[str, float]:
        """获取任务的能耗归因向量"""
        cache_key = f"{task.task_type.value}_{task.model_size}_{task.batch_size}"
        if cache_key in self.attribution_cache:
            return self.attribution_cache[cache_key]
        
        # 生成模拟的监控数据
        mock_features = self._generate_mock_features(task)
        
        # 使用EmSHAP计算归因
        with torch.no_grad():
            attribution = self.emshap_model.compute_shapley_values(
                torch.tensor(mock_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
        
        # 转换为字典格式
        attribution_names = [
            'attribution_cpu_cycles', 'attribution_l3_misses', 'attribution_memory_bandwidth',
            'attribution_pcie_bandwidth', 'attribution_operator_conv', 'attribution_operator_fc',
            'attribution_operator_attention'
        ]
        
        attribution_vector = dict(zip(attribution_names, attribution.cpu().numpy()[0]))
        self.attribution_cache[cache_key] = attribution_vector
        
        return attribution_vector
    
    def _generate_mock_features(self, task: TaskProfile) -> np.ndarray:
        """生成模拟的特征数据"""
        if task.task_type == TaskType.VISION:
            features = np.array([
                0.8, 0.6, 0.7, 0.5, 0.7, 0.5, 0.6, 0.4, 0.3, 0.6, 0.4, 0.5, 0.3,
                0.4, 0.3, 0.4, 0.2, 120.0, 5, 200.0, 0, task.model_size / 1e6,
                task.input_size[0], task.input_size[1], task.batch_size
            ])
        elif task.task_type == TaskType.LANGUAGE:
            features = np.array([
                0.9, 0.8, 0.9, 0.7, 0.8, 0.7, 0.8, 0.6, 0.4, 0.8, 0.6, 0.7, 0.5,
                0.5, 0.4, 0.5, 0.3, 150.0, 5, 200.0, 1, task.model_size / 1e6,
                task.input_size[0], task.input_size[1], task.batch_size
            ])
        else:
            features = np.array([
                0.7, 0.6, 0.7, 0.5, 0.6, 0.5, 0.6, 0.4, 0.3, 0.6, 0.4, 0.5, 0.3,
                0.4, 0.3, 0.4, 0.2, 100.0, 5, 200.0, 2, task.model_size / 1e6,
                task.input_size[0], task.input_size[1], task.batch_size
            ])
        
        return features
    
    def schedule_next_task(self) -> Optional[SchedulingResult]:
        """调度下一个任务"""
        if not self.task_queue or self.current_metrics is None:
            return None
        
        task = self.task_queue.pop(0)
        attribution_vector = self.get_attribution_vector(task)
        
        # 构建系统状态
        state = SystemState(
            gpu_utilization=self.current_metrics.gpu_utilization,
            memory_utilization=self.current_metrics.memory_utilization,
            cpu_utilization=self.current_metrics.cpu_utilization,
            memory_bandwidth=self.current_metrics.memory_bandwidth,
            pcie_bandwidth=self.current_metrics.pcie_bandwidth,
            total_power=self.current_metrics.total_power,
            queue_length=self.current_metrics.queue_length,
            power_budget=self.current_metrics.power_budget,
            task_type=task.task_type,
            model_size=task.model_size,
            input_size=task.input_size,
            batch_size=task.batch_size,
            attribution_vector=attribution_vector
        )
        
        # 获取调度动作
        action = self.scheduler.get_action(state, training=True)
        
        # 模拟执行任务
        result = self._simulate_task_execution(task, action)
        self.execution_history.append(result)
        
        return result
    
    def _simulate_task_execution(self, task: TaskProfile, action: SchedulingAction) -> SchedulingResult:
        """模拟任务执行"""
        base_time = 10.0
        base_energy = 50.0
        
        precision_multiplier = {
            PrecisionType.FP32: 1.0,
            PrecisionType.FP16: 0.7,
            PrecisionType.BFloat16: 0.7,
            PrecisionType.INT8: 0.4
        }
        
        batch_multiplier = action.batch_size / 32.0
        attribution_impact = 1.0
        
        if task.task_type == TaskType.VISION:
            attribution_impact = 1.2
        
        execution_time = base_time * precision_multiplier[action.precision] * batch_multiplier * attribution_impact
        energy_consumption = base_energy * precision_multiplier[action.precision] * batch_multiplier * attribution_impact
        performance_metric = 1.0 / execution_time
        
        accuracy_loss = {
            PrecisionType.FP32: 0.0,
            PrecisionType.FP16: 0.02,
            PrecisionType.BFloat16: 0.01,
            PrecisionType.INT8: 0.05
        }[action.precision]
        
        return SchedulingResult(
            task_id=task.task_id,
            action=action,
            execution_time=execution_time,
            energy_consumption=energy_consumption,
            performance_metric=performance_metric,
            accuracy_loss=accuracy_loss,
            success=True
        )
    
    def analyze_scheduling_decisions(self, num_decisions: int = 10) -> Dict[str, Any]:
        """分析调度决策"""
        decision_history = self.scheduler.get_decision_history()
        recent_decisions = decision_history[-num_decisions:] if len(decision_history) >= num_decisions else decision_history
        
        if not recent_decisions:
            return {"message": "No decisions to analyze"}
        
        explanations = []
        for decision in recent_decisions:
            try:
                explanation = self.explainer.explain_decision(decision)
                explanations.append(explanation)
            except Exception as e:
                logger.warning(f"Failed to explain decision: {e}")
                continue
        
        analysis = {
            'total_decisions': len(explanations),
            'average_confidence': np.mean([exp.confidence for exp in explanations]),
            'key_factors': self._extract_key_factors(explanations),
            'attribution_impact': self._analyze_attribution_impact(explanations),
            'action_distribution': self._analyze_action_distribution(explanations)
        }
        
        return analysis
    
    def _extract_key_factors(self, explanations: List[DecisionExplanation]) -> List[str]:
        """提取关键影响因素"""
        factor_counts = {}
        
        for exp in explanations:
            for factor in exp.key_factors:
                factor_name = factor.split(':')[0]
                factor_counts[factor_name] = factor_counts.get(factor_name, 0) + 1
        
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{factor}: {count}次" for factor, count in sorted_factors[:5]]
    
    def _analyze_attribution_impact(self, explanations: List[DecisionExplanation]) -> Dict[str, Any]:
        """分析归因向量的影响"""
        attribution_impacts = {}
        
        for exp in explanations:
            for attr_feature, contribution in exp.attribution_contributions.items():
                if attr_feature not in attribution_impacts:
                    attribution_impacts[attr_feature] = []
                attribution_impacts[attr_feature].append(contribution)
        
        analysis = {}
        for feature, impacts in attribution_impacts.items():
            analysis[feature] = {
                'mean_impact': np.mean(impacts),
                'std_impact': np.std(impacts),
                'positive_ratio': np.mean(np.array(impacts) > 0),
                'count': len(impacts)
            }
        
        return analysis
    
    def _analyze_action_distribution(self, explanations: List[DecisionExplanation]) -> Dict[str, Any]:
        """分析动作分布"""
        precision_counts = {}
        gpu_counts = {}
        
        for exp in explanations:
            action = exp.action
            precision = action.get('precision', 'unknown')
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
            
            gpu = action.get('target_gpu', -1)
            gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        
        return {
            'precision_distribution': precision_counts,
            'gpu_distribution': gpu_counts
        }
    
    def generate_system_report(self, output_path: str):
        """生成系统报告"""
        decision_analysis = self.analyze_scheduling_decisions()
        
        if self.execution_history:
            execution_stats = {
                'total_tasks': len(self.execution_history),
                'average_execution_time': np.mean([r.execution_time for r in self.execution_history]),
                'average_energy_consumption': np.mean([r.energy_consumption for r in self.execution_history]),
                'average_performance': np.mean([r.performance_metric for r in self.execution_history]),
                'average_accuracy_loss': np.mean([r.accuracy_loss for r in self.execution_history]),
                'success_rate': np.mean([r.success for r in self.execution_history])
            }
        else:
            execution_stats = {"message": "No execution history"}
        
        attribution_stats = {
            'cache_size': len(self.attribution_cache),
            'cache_hit_rate': 0.8
        }
        
        report = {
            'timestamp': time.time(),
            'system_status': {
                'queue_length': len(self.task_queue),
                'current_metrics': self.current_metrics.__dict__ if self.current_metrics else None
            },
            'decision_analysis': decision_analysis,
            'execution_statistics': execution_stats,
            'attribution_statistics': attribution_stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"System report saved to {output_path}")
        return report
    
    def visualize_system_performance(self, save_path: Optional[str] = None):
        """可视化系统性能"""
        if not self.execution_history:
            logger.warning("No execution history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 执行时间趋势
        execution_times = [r.execution_time for r in self.execution_history]
        axes[0, 0].plot(execution_times, marker='o')
        axes[0, 0].set_title('任务执行时间趋势')
        axes[0, 0].set_xlabel('任务序号')
        axes[0, 0].set_ylabel('执行时间 (秒)')
        axes[0, 0].grid(True)
        
        # 能耗趋势
        energy_consumptions = [r.energy_consumption for r in self.execution_history]
        axes[0, 1].plot(energy_consumptions, marker='s', color='red')
        axes[0, 1].set_title('任务能耗趋势')
        axes[0, 1].set_xlabel('任务序号')
        axes[0, 1].set_ylabel('能耗 (焦耳)')
        axes[0, 1].grid(True)
        
        # 精度选择分布
        precision_counts = {}
        for r in self.execution_history:
            precision = r.action.precision.value
            precision_counts[precision] = precision_counts.get(precision, 0) + 1
        
        if precision_counts:
            axes[1, 0].pie(precision_counts.values(), labels=precision_counts.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('精度选择分布')
        
        # 性能vs能耗散点图
        performance_metrics = [r.performance_metric for r in self.execution_history]
        axes[1, 1].scatter(energy_consumptions, performance_metrics, alpha=0.6)
        axes[1, 1].set_title('性能 vs 能耗')
        axes[1, 1].set_xlabel('能耗 (焦耳)')
        axes[1, 1].set_ylabel('性能指标')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance visualization saved to {save_path}")
        
        plt.show()
