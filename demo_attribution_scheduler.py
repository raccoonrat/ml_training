"""
归因-调度协同系统演示脚本
演示Shapley值驱动的动态调度方法
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.emshap_enhanced import EMSHAPEnhanced
from models.rl_scheduler import RLScheduler, ActionSpace, SystemState, TaskType, PrecisionType
from models.scheduler_explainer import SchedulerExplainer


def create_demo_tasks() -> List[Dict[str, Any]]:
    """创建演示任务"""
    tasks = [
        {
            'task_id': 'vision_resnet50',
            'task_type': TaskType.VISION,
            'model_size': 50 * 1e6,
            'input_size': (224, 224),
            'batch_size': 32,
            'priority': 1.0
        },
        {
            'task_id': 'language_bert',
            'task_type': TaskType.LANGUAGE,
            'model_size': 110 * 1e6,
            'input_size': (512, 768),
            'batch_size': 16,
            'priority': 1.0
        },
        {
            'task_id': 'vision_efficientnet',
            'task_type': TaskType.VISION,
            'model_size': 7 * 1e6,
            'input_size': (224, 224),
            'batch_size': 64,
            'priority': 1.0
        }
    ]
    return tasks


def create_attribution_vector(task_type: TaskType) -> Dict[str, float]:
    """创建归因向量（模拟EmSHAP输出）"""
    if task_type == TaskType.VISION:
        return {
            'attribution_cpu_cycles': 0.3,
            'attribution_l3_misses': 0.2,
            'attribution_memory_bandwidth': 0.4,
            'attribution_pcie_bandwidth': 0.1,
            'attribution_operator_conv': 0.6,
            'attribution_operator_fc': 0.2,
            'attribution_operator_attention': 0.1
        }
    elif task_type == TaskType.LANGUAGE:
        return {
            'attribution_cpu_cycles': 0.2,
            'attribution_l3_misses': 0.3,
            'attribution_memory_bandwidth': 0.5,
            'attribution_pcie_bandwidth': 0.1,
            'attribution_operator_conv': 0.1,
            'attribution_operator_fc': 0.3,
            'attribution_operator_attention': 0.7
        }
    else:
        return {
            'attribution_cpu_cycles': 0.4,
            'attribution_l3_misses': 0.1,
            'attribution_memory_bandwidth': 0.3,
            'attribution_pcie_bandwidth': 0.1,
            'attribution_operator_conv': 0.4,
            'attribution_operator_fc': 0.3,
            'attribution_operator_attention': 0.1
        }


def simulate_task_execution(task: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, float]:
    """模拟任务执行"""
    # 基础执行时间和能耗
    base_time = 10.0
    base_energy = 50.0
    
    # 精度影响因子
    precision_multiplier = {
        'fp32': 1.0,
        'fp16': 0.7,
        'bfloat16': 0.6,
        'int8': 0.4
    }
    
    # 任务类型影响因子
    task_multiplier = {
        TaskType.VISION: 1.0,
        TaskType.LANGUAGE: 1.5,
        TaskType.RECOMMENDATION: 0.8
    }
    
    # 批次大小影响因子
    batch_multiplier = action['batch_size'] / 32.0
    
    # 计算执行时间和能耗
    execution_time = base_time * precision_multiplier[action['precision']] * task_multiplier[task['task_type']] * batch_multiplier
    energy_consumption = base_energy * precision_multiplier[action['precision']] * task_multiplier[task['task_type']] * batch_multiplier
    
    # 计算性能指标（吞吐量）
    performance_metric = 1.0 / execution_time
    
    return {
        'execution_time': execution_time,
        'energy_consumption': energy_consumption,
        'performance_metric': performance_metric,
        'accuracy_loss': 0.0 if action['precision'] == 'fp32' else 0.02,
        'success': True
    }


def demonstrate_attribution_scheduling():
    """演示归因-调度协同系统"""
    logger.info("Starting Attribution-Driven Scheduling Demonstration")
    
    # 创建EmSHAP模型
    emshap_model = EMSHAPEnhanced(
        input_dim=20,
        gru_hidden_dim=64,
        context_dim=32,
        energy_hidden_dims=[128, 64, 32],
        gru_layers=2,
        dropout_rate=0.1
    )
    
    # 创建RL调度器
    action_space = ActionSpace(num_gpus=4)
    state_dim = 32  # 系统特征(20) + 任务特征(5) + 归因特征(7)
    scheduler = RLScheduler(
        state_dim=state_dim,
        action_space=action_space,
        algorithm='ppo',
        device='cpu'
    )
    
    # 创建可解释性分析器
    explainer = SchedulerExplainer(scheduler, device='cpu')
    
    # 创建演示任务
    tasks = create_demo_tasks()
    logger.info(f"Created {len(tasks)} demo tasks")
    
    # 模拟调度过程
    results = []
    
    for i, task in enumerate(tasks):
        logger.info(f"Processing task {i+1}: {task['task_id']}")
        
        # 创建归因向量
        attribution_vector = create_attribution_vector(task['task_type'])
        
        # 创建系统状态
        system_state = SystemState(
            gpu_utilization=[0.7, 0.6, 0.8, 0.5],
            memory_utilization=[0.6, 0.5, 0.7, 0.4],
            cpu_utilization=0.3,
            memory_bandwidth=[0.6, 0.4, 0.5, 0.3],
            pcie_bandwidth=[0.4, 0.3, 0.4, 0.2],
            total_power=120.0,
            queue_length=len(tasks) - i,
            power_budget=200.0,
            task_type=task['task_type'],
            model_size=task['model_size'],
            input_size=task['input_size'],
            batch_size=task['batch_size'],
            attribution_vector=attribution_vector
        )
        
        # 获取调度决策
        action = scheduler.get_action(system_state, training=False)
        
        # 模拟任务执行
        execution_result = simulate_task_execution(task, action.to_dict())
        
        # 记录结果
        result = {
            'step': i + 1,
            'task_id': task['task_id'],
            'task_type': task['task_type'].value,
            'gpu': action.target_gpu,
            'precision': action.precision.value,
            'batch_size': action.batch_size,
            'execution_time': execution_result['execution_time'],
            'energy_consumption': execution_result['energy_consumption'],
            'performance_metric': execution_result['performance_metric']
        }
        results.append(result)
        
        logger.info(f"Task {task['task_id']} scheduled to GPU {action.target_gpu}, "
                   f"precision {action.precision.value}, batch size {action.batch_size}")
        logger.info(f"Execution time: {execution_result['execution_time']:.2f}s, "
                   f"Energy: {execution_result['energy_consumption']:.2f}J")
    
    # 分析结果
    df = pd.DataFrame(results)
    
    # 计算统计信息
    total_time = df['execution_time'].sum()
    total_energy = df['energy_consumption'].sum()
    avg_performance = df['performance_metric'].mean()
    
    # 统计分布
    precision_dist = df['precision'].value_counts()
    gpu_dist = df['gpu'].value_counts()
    
    logger.info("=== Scheduling Results Summary ===")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Total energy consumption: {total_energy:.2f} joules")
    logger.info(f"Average performance metric: {avg_performance:.4f}")
    logger.info(f"Precision distribution: {dict(precision_dist)}")
    logger.info(f"GPU distribution: {dict(gpu_dist)}")
    
    # 保存结果到EmSHAP输出目录
    output_dir = 'data/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV结果
    csv_path = os.path.join(output_dir, 'attribution_scheduler_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    # 创建可视化
    create_visualizations(df, output_dir)
    
    return df


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """创建可视化图表"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attribution-Driven Scheduling Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. 执行时间和能耗对比
    ax1 = axes[0, 0]
    x = range(len(df))
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar([i-0.2 for i in x], df['execution_time'], width=0.4, 
                    label='Execution Time (s)', color='skyblue', alpha=0.7)
    bars2 = ax1_twin.bar([i+0.2 for i in x], df['energy_consumption'], width=0.4,
                         label='Energy Consumption (J)', color='lightcoral', alpha=0.7)
    
    ax1.set_xlabel('Task Index')
    ax1.set_ylabel('Execution Time (seconds)', color='skyblue')
    ax1_twin.set_ylabel('Energy Consumption (joules)', color='lightcoral')
    ax1.set_title('Execution Time vs Energy Consumption')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['task_id'], rotation=45)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. 精度选择分布
    ax2 = axes[0, 1]
    precision_counts = df['precision'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax2.pie(precision_counts.values, labels=precision_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Precision Selection Distribution')
    
    # 3. GPU分配分布
    ax3 = axes[1, 0]
    gpu_counts = df['gpu'].value_counts().sort_index()
    bars = ax3.bar(gpu_counts.index, gpu_counts.values, color='lightgreen', alpha=0.7)
    ax3.set_xlabel('GPU ID')
    ax3.set_ylabel('Number of Tasks')
    ax3.set_title('GPU Allocation Distribution')
    ax3.set_xticks(gpu_counts.index)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. 性能指标对比
    ax4 = axes[1, 1]
    x_pos = range(len(df))
    bars = ax4.bar(x_pos, df['performance_metric'], color='gold', alpha=0.7)
    ax4.set_xlabel('Task Index')
    ax4.set_ylabel('Performance Metric (1/time)')
    ax4.set_title('Task Performance Metrics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(df['task_id'], rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    viz_path = os.path.join(output_dir, 'attribution_scheduler_analysis.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {viz_path}")
    plt.close()


def main():
    """主函数"""
    logger.info("Starting Attribution-Driven Scheduling System Demo")
    
    try:
        # 运行演示
        results_df = demonstrate_attribution_scheduling()
        
        logger.info("Demo completed successfully!")
        logger.info("Check the results in data/visualizations/ directory")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
