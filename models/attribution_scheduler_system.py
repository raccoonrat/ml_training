"""
归因-调度协同系统
整合EmSHAP、RL调度器和可解释性分析
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import json

from models.emshap_enhanced import EMSHAPEnhanced
from models.rl_scheduler import RLScheduler, ActionSpace, SystemState, TaskType, PrecisionType
from models.scheduler_explainer import SchedulerExplainer, DecisionExplanation


@dataclass
class TaskProfile:
    """任务配置文件"""
    task_id: str
    task_type: TaskType
    model_size: int
    input_size: tuple
    batch_size: int
    priority: float
    slo_requirements: Dict[str, float]


@dataclass
class SystemMetrics:
    """系统指标"""
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
    action: Any
    execution_time: float
    energy_consumption: float
    performance_metric: float
    accuracy_loss: float
    success: bool


class AttributionSchedulerSystem:
    """归因-调度协同系统"""
    
    def __init__(self, 
                 emshap_model: EMSHAPEnhanced,
                 num_gpus: int = 4,
                 device: str = 'auto'):
        
        self.emshap_model = emshap_model
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # 创建RL调度器
        action_space = ActionSpace(num_gpus=num_gpus)
        state_dim = 32  # 系统特征(20) + 任务特征(5) + 归因特征(7)
        self.scheduler = RLScheduler(
            state_dim=state_dim,
            action_space=action_space,
            algorithm='ppo',
            device=device
        )
        
        # 创建可解释性分析器
        self.explainer = SchedulerExplainer(self.scheduler, device=device)
        
        # 系统状态
        self.task_queue = []
        self.current_metrics = None
        self.scheduling_history = []
        
        # 归因结果缓存
        self.attribution_cache = {}
        
        logger.info("Attribution-Driven Scheduling System initialized")
    
    def add_task(self, task: TaskProfile):
        """添加任务到队列"""
        self.task_queue.append(task)
        logger.info(f"Task {task.task_id} added to queue")
    
    def update_system_metrics(self, metrics: SystemMetrics):
        """更新系统指标"""
        self.current_metrics = metrics
        logger.debug(f"System metrics updated: power={metrics.total_power:.1f}W, queue={metrics.queue_length}")
    
    def _get_attribution_vector(self, task: TaskProfile) -> Dict[str, float]:
        """获取任务的归因向量"""
        # 检查缓存
        cache_key = f"{task.task_id}_{task.task_type.value}"
        if cache_key in self.attribution_cache:
            return self.attribution_cache[cache_key]
        
        # 模拟EmSHAP归因计算
        # 在实际应用中，这里会调用EmSHAP模型
        if task.task_type == TaskType.VISION:
            attribution = {
                'attribution_cpu_cycles': 0.3,
                'attribution_l3_misses': 0.2,
                'attribution_memory_bandwidth': 0.4,
                'attribution_pcie_bandwidth': 0.1,
                'attribution_operator_conv': 0.6,
                'attribution_operator_fc': 0.2,
                'attribution_operator_attention': 0.1
            }
        elif task.task_type == TaskType.LANGUAGE:
            attribution = {
                'attribution_cpu_cycles': 0.2,
                'attribution_l3_misses': 0.3,
                'attribution_memory_bandwidth': 0.5,
                'attribution_pcie_bandwidth': 0.1,
                'attribution_operator_conv': 0.1,
                'attribution_operator_fc': 0.3,
                'attribution_operator_attention': 0.7
            }
        else:
            attribution = {
                'attribution_cpu_cycles': 0.4,
                'attribution_l3_misses': 0.1,
                'attribution_memory_bandwidth': 0.3,
                'attribution_pcie_bandwidth': 0.1,
                'attribution_operator_conv': 0.4,
                'attribution_operator_fc': 0.3,
                'attribution_operator_attention': 0.1
            }
        
        # 缓存结果
        self.attribution_cache[cache_key] = attribution
        return attribution
    
    def schedule_next_task(self) -> Optional[SchedulingResult]:
        """调度下一个任务"""
        if not self.task_queue or self.current_metrics is None:
            return None
        
        # 获取任务
        task = self.task_queue.pop(0)
        
        # 获取归因向量
        attribution_vector = self._get_attribution_vector(task)
        
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
        
        # 获取调度决策
        action = self.scheduler.get_action(state, training=False)
        
        # 模拟任务执行
        execution_result = self._simulate_execution(task, action)
        
        # 创建结果
        result = SchedulingResult(
            task_id=task.task_id,
            action=action,
            execution_time=execution_result['execution_time'],
            energy_consumption=execution_result['energy_consumption'],
            performance_metric=execution_result['performance_metric'],
            accuracy_loss=execution_result['accuracy_loss'],
            success=execution_result['success']
        )
        
        # 记录历史
        self.scheduling_history.append(result)
        
        logger.info(f"Task {task.task_id} scheduled: GPU={action.target_gpu}, "
                   f"precision={action.precision.value}, batch_size={action.batch_size}")
        
        return result
    
    def _simulate_execution(self, task: TaskProfile, action: Any) -> Dict[str, float]:
        """模拟任务执行"""
        # 基础参数
        base_time = 10.0
        base_energy = 50.0
        
        # 精度影响因子
        precision_multiplier = {
            PrecisionType.FP32: 1.0,
            PrecisionType.FP16: 0.7,
            PrecisionType.BFloat16: 0.6,
            PrecisionType.INT8: 0.4
        }
        
        # 任务类型影响因子
        task_multiplier = {
            TaskType.VISION: 1.0,
            TaskType.LANGUAGE: 1.5,
            TaskType.RECOMMENDATION: 0.8
        }
        
        # 批次大小影响因子
        batch_multiplier = action.batch_size / 32.0
        
        # 计算执行时间和能耗
        execution_time = base_time * precision_multiplier[action.precision] * task_multiplier[task.task_type] * batch_multiplier
        energy_consumption = base_energy * precision_multiplier[action.precision] * task_multiplier[task.task_type] * batch_multiplier
        
        # 计算性能指标
        performance_metric = 1.0 / execution_time
        
        # 计算精度损失
        accuracy_loss = 0.0 if action.precision == PrecisionType.FP32 else 0.02
        
        return {
            'execution_time': execution_time,
            'energy_consumption': energy_consumption,
            'performance_metric': performance_metric,
            'accuracy_loss': accuracy_loss,
            'success': True
        }
    
    def analyze_scheduling_decisions(self) -> Dict[str, Any]:
        """分析调度决策"""
        if not self.scheduling_history:
            return {}
        
        # 获取决策历史
        decision_history = self.scheduler.get_decision_history()
        
        # 分析最近的决策
        if decision_history:
            recent_decision = decision_history[-1]
            explanation = self.explainer.explain_decision(recent_decision)
            
            analysis = {
                'total_decisions': len(decision_history),
                'average_confidence': explanation.confidence,
                'key_factors': explanation.key_factors,
                'reasoning': explanation.reasoning,
                'attribution_impact': {
                    name: {'mean_impact': abs(value)} 
                    for name, value in explanation.attribution_contributions.items()
                }
            }
        else:
            analysis = {
                'total_decisions': 0,
                'average_confidence': 0.0,
                'key_factors': [],
                'reasoning': "No decisions to analyze",
                'attribution_impact': {}
            }
        
        return analysis
    
    def generate_system_report(self, output_path: str):
        """生成系统报告"""
        report = {
            'system_info': {
                'total_tasks_processed': len(self.scheduling_history),
                'current_queue_length': len(self.task_queue),
                'attribution_cache_size': len(self.attribution_cache)
            },
            'performance_summary': {
                'total_execution_time': sum(r.execution_time for r in self.scheduling_history),
                'total_energy_consumption': sum(r.energy_consumption for r in self.scheduling_history),
                'average_performance': np.mean([r.performance_metric for r in self.scheduling_history]) if self.scheduling_history else 0.0
            },
            'scheduling_patterns': {
                'precision_distribution': {},
                'gpu_distribution': {},
                'batch_size_distribution': {}
            }
        }
        
        # 统计分布
        if self.scheduling_history:
            precision_counts = {}
            gpu_counts = {}
            batch_counts = {}
            
            for result in self.scheduling_history:
                precision = result.action.precision.value
                gpu = result.action.target_gpu
                batch_size = result.action.batch_size
                
                precision_counts[precision] = precision_counts.get(precision, 0) + 1
                gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
                batch_counts[batch_size] = batch_counts.get(batch_size, 0) + 1
            
            report['scheduling_patterns']['precision_distribution'] = precision_counts
            report['scheduling_patterns']['gpu_distribution'] = gpu_counts
            report['scheduling_patterns']['batch_size_distribution'] = batch_counts
        
        # 保存报告
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"System report saved to {output_path}")
    
    def visualize_system_performance(self, output_path: str):
        """可视化系统性能"""
        if not self.scheduling_history:
            logger.warning("No scheduling history to visualize")
            return
        
        # 创建数据框
        data = []
        for result in self.scheduling_history:
            data.append({
                'task_id': result.task_id,
                'execution_time': result.execution_time,
                'energy_consumption': result.energy_consumption,
                'performance_metric': result.performance_metric,
                'precision': result.action.precision.value,
                'gpu': result.action.target_gpu,
                'batch_size': result.action.batch_size
            })
        
        df = pd.DataFrame(data)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attribution-Driven Scheduling System Performance', fontsize=16, fontweight='bold')
        
        # 1. 执行时间趋势
        axes[0, 0].plot(range(len(df)), df['execution_time'], 'o-', color='blue')
        axes[0, 0].set_title('Execution Time Trend')
        axes[0, 0].set_xlabel('Task Index')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 能耗趋势
        axes[0, 1].plot(range(len(df)), df['energy_consumption'], 's-', color='red')
        axes[0, 1].set_title('Energy Consumption Trend')
        axes[0, 1].set_xlabel('Task Index')
        axes[0, 1].set_ylabel('Energy Consumption (joules)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 精度分布
        precision_counts = df['precision'].value_counts()
        axes[1, 0].pie(precision_counts.values, labels=precision_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Precision Selection Distribution')
        
        # 4. 性能vs能耗散点图
        scatter = axes[1, 1].scatter(df['energy_consumption'], df['performance_metric'], 
                                   c=df['gpu'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Performance vs Energy Consumption')
        axes[1, 1].set_xlabel('Energy Consumption (joules)')
        axes[1, 1].set_ylabel('Performance Metric (1/time)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('GPU ID')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance visualization saved to {output_path}")
