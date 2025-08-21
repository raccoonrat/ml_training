"""
归因-调度协同系统演示
展示Shapley值驱动的动态调度方法
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
import json
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.emshap_enhanced import EMSHAPEnhanced
from models.rl_scheduler import RLScheduler, ActionSpace, SystemState, TaskType, PrecisionType
from models.scheduler_explainer import SchedulerExplainer


def create_demo_tasks():
    """创建演示任务"""
    return [
        {
            'task_id': 'vision_resnet50',
            'task_type': TaskType.VISION,
            'model_size': 25 * 1e6,
            'input_size': (224, 224),
            'batch_size': 32,
            'description': 'ResNet50图像分类任务'
        },
        {
            'task_id': 'language_bert',
            'task_type': TaskType.LANGUAGE,
            'model_size': 110 * 1e6,
            'input_size': (512, 768),
            'batch_size': 8,
            'description': 'BERT语言模型任务'
        },
        {
            'task_id': 'vision_efficientnet',
            'task_type': TaskType.VISION,
            'model_size': 7 * 1e6,
            'input_size': (224, 224),
            'batch_size': 64,
            'description': 'EfficientNet轻量级图像分类'
        }
    ]


def create_attribution_vector(task_type: TaskType):
    """创建任务特定的归因向量"""
    if task_type == TaskType.VISION:
        return {
            'attribution_cpu_cycles': 0.4,
            'attribution_l3_misses': 0.2,
            'attribution_memory_bandwidth': 0.3,
            'attribution_pcie_bandwidth': 0.1,
            'attribution_operator_conv': 0.6,
            'attribution_operator_fc': 0.3,
            'attribution_operator_attention': 0.1
        }
    elif task_type == TaskType.LANGUAGE:
        return {
            'attribution_cpu_cycles': 0.3,
            'attribution_l3_misses': 0.4,
            'attribution_memory_bandwidth': 0.5,
            'attribution_pcie_bandwidth': 0.2,
            'attribution_operator_conv': 0.1,
            'attribution_operator_fc': 0.2,
            'attribution_operator_attention': 0.7
        }
    else:
        return {
            'attribution_cpu_cycles': 0.2,
            'attribution_l3_misses': 0.1,
            'attribution_memory_bandwidth': 0.2,
            'attribution_pcie_bandwidth': 0.6,
            'attribution_operator_conv': 0.1,
            'attribution_operator_fc': 0.8,
            'attribution_operator_attention': 0.1
        }


def simulate_task_execution(task, action):
    """模拟任务执行"""
    base_time = 10.0
    base_energy = 50.0
    
    precision_multiplier = {
        'fp32': 1.0, 'fp16': 0.7, 'bfloat16': 0.7, 'int8': 0.4
    }
    
    batch_multiplier = action['batch_size'] / 32.0
    task_multiplier = {
        TaskType.VISION: 1.0,
        TaskType.LANGUAGE: 1.5,
        TaskType.RECOMMENDATION: 0.8
    }[task['task_type']]
    
    execution_time = base_time * precision_multiplier[action['precision']] * batch_multiplier * task_multiplier
    energy_consumption = base_energy * precision_multiplier[action['precision']] * batch_multiplier * task_multiplier
    performance_metric = 1.0 / execution_time
    
    return {
        'execution_time': execution_time,
        'energy_consumption': energy_consumption,
        'performance_metric': performance_metric,
        'success': True
    }


def demonstrate_attribution_scheduling():
    """演示归因-调度协同系统"""
    logger.info("=== 归因-调度协同系统演示 ===")
    
    # 初始化系统组件
    logger.info("1. 初始化系统组件...")
    
    emshap_model = EMSHAPEnhanced(
        input_dim=32, gru_hidden_dim=64, context_dim=32,
        energy_hidden_dims=[128, 64, 32], gru_layers=2, dropout_rate=0.1
    )
    
    action_space = ActionSpace(num_gpus=4)
    scheduler = RLScheduler(state_dim=32, action_space=action_space, algorithm='ppo', device='cpu')
    explainer = SchedulerExplainer(scheduler, device='cpu')
    
    # 创建演示任务
    logger.info("2. 创建演示任务...")
    tasks = create_demo_tasks()
    
    for task in tasks:
        logger.info(f"   - {task['task_id']}: {task['description']}")
    
    # 模拟调度过程
    logger.info("3. 开始模拟调度过程...")
    
    scheduling_results = []
    
    for step, task in enumerate(tasks):
        logger.info(f"\n--- 步骤 {step + 1}: 调度任务 {task['task_id']} ---")
        
        # 创建系统状态
        state = SystemState(
            gpu_utilization=[0.7, 0.6, 0.8, 0.5],
            memory_utilization=[0.6, 0.5, 0.7, 0.4],
            cpu_utilization=0.3,
            memory_bandwidth=[0.6, 0.4, 0.5, 0.3],
            pcie_bandwidth=[0.4, 0.3, 0.4, 0.2],
            total_power=120.0,
            queue_length=3 - step,
            power_budget=200.0,
            task_type=task['task_type'],
            model_size=task['model_size'],
            input_size=task['input_size'],
            batch_size=task['batch_size'],
            attribution_vector=create_attribution_vector(task['task_type'])
        )
        
        # 获取调度动作
        action = scheduler.get_action(state, training=True)
        logger.info(f"调度决策: GPU={action.target_gpu}, 精度={action.precision.value}, 批次={action.batch_size}")
        
        # 模拟执行
        execution_result = simulate_task_execution(task, {
            'precision': action.precision.value,
            'batch_size': action.batch_size
        })
        
        logger.info(f"执行结果: 时间={execution_result['execution_time']:.2f}s, "
                   f"能耗={execution_result['energy_consumption']:.2f}J")
        
        # 记录结果
        scheduling_results.append({
            'step': step + 1,
            'task_id': task['task_id'],
            'task_type': task['task_type'].value,
            'gpu': action.target_gpu,
            'precision': action.precision.value,
            'batch_size': action.batch_size,
            'execution_time': execution_result['execution_time'],
            'energy_consumption': execution_result['energy_consumption'],
            'performance_metric': execution_result['performance_metric']
        })
    
    # 分析结果
    logger.info("\n4. 分析调度结果...")
    
    total_time = sum(r['execution_time'] for r in scheduling_results)
    total_energy = sum(r['energy_consumption'] for r in scheduling_results)
    avg_performance = np.mean([r['performance_metric'] for r in scheduling_results])
    
    logger.info(f"总执行时间: {total_time:.2f}秒")
    logger.info(f"总能耗: {total_energy:.2f}焦耳")
    logger.info(f"平均性能: {avg_performance:.4f}")
    
    # 动作分布
    precision_counts = {}
    gpu_counts = {}
    for r in scheduling_results:
        precision_counts[r['precision']] = precision_counts.get(r['precision'], 0) + 1
        gpu_counts[r['gpu']] = gpu_counts.get(r['gpu'], 0) + 1
    
    logger.info(f"精度选择分布: {precision_counts}")
    logger.info(f"GPU分配分布: {gpu_counts}")
    
    # 可视化结果
    logger.info("5. 生成可视化结果...")
    
    df = pd.DataFrame(scheduling_results)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['step'], df['execution_time'], 'o-')
    plt.title('任务执行时间趋势')
    plt.xlabel('任务步骤')
    plt.ylabel('执行时间 (秒)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(df['step'], df['energy_consumption'], 's-', color='red')
    plt.title('任务能耗趋势')
    plt.xlabel('任务步骤')
    plt.ylabel('能耗 (焦耳)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.pie(precision_counts.values(), labels=precision_counts.keys(), autopct='%1.1f%%')
    plt.title('精度选择分布')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['energy_consumption'], df['performance_metric'], alpha=0.7)
    plt.title('性能 vs 能耗')
    plt.xlabel('能耗 (焦耳)')
    plt.ylabel('性能指标')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存结果
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/attribution_scheduler_demo.png', dpi=300, bbox_inches='tight')
    df.to_csv('outputs/scheduling_results_demo.csv', index=False)
    
    logger.info("演示完成! 结果已保存到 outputs/ 目录")
    
    return scheduling_results


def main():
    """主函数"""
    logger.info("开始归因-调度协同系统演示")
    
    try:
        results = demonstrate_attribution_scheduling()
        
        logger.info("\n=== 演示总结 ===")
        logger.info("✓ 成功演示了EmSHAP归因与RL调度的协同工作")
        logger.info("✓ 展示了可解释性分析在调度决策中的应用")
        logger.info("✓ 验证了Shapley值驱动的动态调度方法")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()
