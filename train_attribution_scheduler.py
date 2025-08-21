"""
归因-调度协同系统训练和演示脚本
演示Shapley值驱动的动态调度方法
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
import json
import time
from typing import List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.emshap_enhanced import EMSHAPEnhanced
from models.attribution_scheduler_system import (
    AttributionSchedulerSystem, 
    TaskProfile, 
    SystemMetrics, 
    TaskType
)
from utils import setup_logging, create_directories


def create_sample_tasks(num_tasks: int = 20) -> List[TaskProfile]:
    """创建示例任务"""
    tasks = []
    
    task_configs = [
        # 视觉任务
        (TaskType.VISION, 50, (224, 224), 32),
        (TaskType.VISION, 100, (512, 512), 16),
        (TaskType.VISION, 25, (128, 128), 64),
        
        # 语言任务
        (TaskType.LANGUAGE, 200, (512, 768), 8),
        (TaskType.LANGUAGE, 350, (1024, 1024), 4),
        (TaskType.LANGUAGE, 100, (256, 512), 16),
        
        # 推荐任务
        (TaskType.RECOMMENDATION, 80, (1000, 256), 32),
        (TaskType.RECOMMENDATION, 150, (2000, 512), 16),
    ]
    
    for i in range(num_tasks):
        task_type, model_size, input_size, batch_size = task_configs[i % len(task_configs)]
        
        task = TaskProfile(
            task_id=f"task_{i:03d}",
            task_type=task_type,
            model_size=model_size * 1e6,  # 转换为参数数量
            input_size=input_size,
            batch_size=batch_size,
            priority=1.0,
            slo_requirements={'latency': 10.0, 'accuracy': 0.95}
        )
        tasks.append(task)
    
    return tasks


def create_sample_metrics() -> SystemMetrics:
    """创建示例系统指标"""
    return SystemMetrics(
        timestamp=time.time(),
        gpu_utilization=[0.7, 0.6, 0.8, 0.5],
        memory_utilization=[0.6, 0.5, 0.7, 0.4],
        cpu_utilization=0.3,
        memory_bandwidth=[0.6, 0.4, 0.5, 0.3],
        pcie_bandwidth=[0.4, 0.3, 0.4, 0.2],
        total_power=120.0,
        queue_length=5,
        power_budget=200.0
    )


def load_emshap_model(model_path: str, device: str = 'auto') -> EMSHAPEnhanced:
    """加载EmSHAP模型"""
    logger.info(f"Loading EmSHAP model from {model_path}")
    
    # 创建模型实例
    model = EMSHAPEnhanced(
        input_dim=20,  # 假设输入维度
        gru_hidden_dim=64,
        context_dim=32,
        energy_hidden_dims=[128, 64, 32],
        gru_layers=2,
        dropout_rate=0.1
    )
    
    # 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("EmSHAP model loaded successfully")
    else:
        logger.warning(f"Model file not found: {model_path}")
        logger.info("Using untrained EmSHAP model for demonstration")
    
    return model


def demonstrate_attribution_scheduling(config: dict):
    """演示归因-调度协同系统"""
    
    # 设置日志
    setup_logging(config.get('log_level', 'INFO'))
    
    # 创建目录
    output_dir = config.get('output_dir', 'outputs/attribution_scheduler')
    create_directories([output_dir])
    
    logger.info("开始演示归因-调度协同系统")
    
    # 加载EmSHAP模型
    emshap_model = load_emshap_model(
        config.get('emshap_model_path', 'checkpoints/emshap_enhanced.pth'),
        config.get('device', 'auto')
    )
    
    # 创建协同系统
    system = AttributionSchedulerSystem(
        emshap_model=emshap_model,
        num_gpus=config.get('num_gpus', 4),
        device=config.get('device', 'auto')
    )
    
    # 创建示例任务
    num_tasks = config.get('num_tasks', 20)
    tasks = create_sample_tasks(num_tasks)
    logger.info(f"创建了 {len(tasks)} 个示例任务")
    
    # 添加任务到队列
    for task in tasks:
        system.add_task(task)
    
    # 模拟调度过程
    logger.info("开始模拟调度过程...")
    results = []
    
    for i in range(min(num_tasks, config.get('max_scheduling_steps', 10))):
        # 更新系统指标
        metrics = create_sample_metrics()
        system.update_system_metrics(metrics)
        
        # 调度下一个任务
        result = system.schedule_next_task()
        if result:
            results.append(result)
            logger.info(f"步骤 {i+1}: 调度任务 {result.task_id}, "
                       f"GPU={result.action.target_gpu}, "
                       f"精度={result.action.precision.value}, "
                       f"执行时间={result.execution_time:.2f}s, "
                       f"能耗={result.energy_consumption:.2f}J")
        
        # 短暂延迟模拟真实环境
        time.sleep(0.1)
    
    # 分析调度决策
    logger.info("分析调度决策...")
    analysis = system.analyze_scheduling_decisions()
    
    # 打印分析结果
    logger.info("=== 调度决策分析结果 ===")
    logger.info(f"总决策数: {analysis.get('total_decisions', 0)}")
    logger.info(f"平均置信度: {analysis.get('average_confidence', 0):.3f}")
    
    if 'key_factors' in analysis:
        logger.info("关键影响因素:")
        for factor in analysis['key_factors']:
            logger.info(f"  - {factor}")
    
    if 'attribution_impact' in analysis:
        logger.info("归因向量影响:")
        for feature, impact in analysis['attribution_impact'].items():
            logger.info(f"  - {feature}: 平均影响={impact['mean_impact']:.4f}")
    
    # 生成系统报告
    report_path = os.path.join(output_dir, 'system_report.json')
    system.generate_system_report(report_path)
    
    # 可视化性能
    viz_path = os.path.join(output_dir, 'performance_visualization.png')
    system.visualize_system_performance(viz_path)
    
    # 保存详细结果
    results_data = []
    for result in results:
        results_data.append({
            'task_id': result.task_id,
            'target_gpu': result.action.target_gpu,
            'precision': result.action.precision.value,
            'batch_size': result.action.batch_size,
            'execution_time': result.execution_time,
            'energy_consumption': result.energy_consumption,
            'performance_metric': result.performance_metric,
            'accuracy_loss': result.accuracy_loss,
            'success': result.success
        })
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(output_dir, 'scheduling_results.csv')
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"结果已保存到: {output_dir}")
    logger.info(f"系统报告: {report_path}")
    logger.info(f"性能可视化: {viz_path}")
    logger.info(f"调度结果: {results_path}")
    
    return system, results, analysis


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='归因-调度协同系统演示')
    parser.add_argument('--config', type=str, default='configs/attribution_scheduler.json',
                       help='配置文件路径')
    parser.add_argument('--num_tasks', type=int, default=20,
                       help='任务数量')
    parser.add_argument('--num_gpus', type=int, default=4,
                       help='GPU数量')
    parser.add_argument('--output_dir', type=str, default='outputs/attribution_scheduler',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'num_tasks': args.num_tasks,
        'num_gpus': args.num_gpus,
        'output_dir': args.output_dir,
        'device': args.device,
        'log_level': args.log_level,
        'max_scheduling_steps': 10,
        'emshap_model_path': 'checkpoints/emshap_enhanced.pth'
    }
    
    # 运行演示
    try:
        system, results, analysis = demonstrate_attribution_scheduling(config)
        logger.info("演示完成!")
        
        # 打印总结
        if results:
            avg_time = np.mean([r.execution_time for r in results])
            avg_energy = np.mean([r.energy_consumption for r in results])
            avg_performance = np.mean([r.performance_metric for r in results])
            
            logger.info("=== 性能总结 ===")
            logger.info(f"平均执行时间: {avg_time:.2f}秒")
            logger.info(f"平均能耗: {avg_energy:.2f}焦耳")
            logger.info(f"平均性能指标: {avg_performance:.4f}")
            logger.info(f"任务成功率: {np.mean([r.success for r in results]):.1%}")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()
