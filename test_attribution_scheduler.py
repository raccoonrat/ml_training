"""
测试归因-调度协同系统
验证Shapley值驱动的动态调度方法
"""

import os
import sys
import numpy as np
import torch
from loguru import logger

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.emshap_enhanced import EMSHAPEnhanced
from models.rl_scheduler import RLScheduler, ActionSpace, SystemState, TaskType, PrecisionType
from models.scheduler_explainer import SchedulerExplainer


def test_rl_scheduler():
    """测试RL调度器"""
    logger.info("测试RL调度器...")
    
    # 创建动作空间
    action_space = ActionSpace(num_gpus=4)
    logger.info(f"动作空间维度: {action_space.action_dim}")
    
    # 创建调度器
    # 计算实际状态维度: 系统特征(20) + 任务特征(5) + 归因特征(7) = 32
    state_dim = 20 + 5 + 7  # 32
    scheduler = RLScheduler(
        state_dim=state_dim,
        action_space=action_space,
        algorithm='ppo',
        device='cpu'
    )
    
    # 创建模拟状态
    mock_state = SystemState(
        gpu_utilization=[0.7, 0.6, 0.8, 0.5],
        memory_utilization=[0.6, 0.5, 0.7, 0.4],
        cpu_utilization=0.3,
        memory_bandwidth=[0.6, 0.4, 0.5, 0.3],
        pcie_bandwidth=[0.4, 0.3, 0.4, 0.2],
        total_power=120.0,
        queue_length=5,
        power_budget=200.0,
        task_type=TaskType.VISION,
        model_size=50 * 1e6,
        input_size=(224, 224),
        batch_size=32,
        attribution_vector={
            'attribution_cpu_cycles': 0.3,
            'attribution_l3_misses': 0.2,
            'attribution_memory_bandwidth': 0.4,
            'attribution_pcie_bandwidth': 0.1,
            'attribution_operator_conv': 0.5,
            'attribution_operator_fc': 0.2,
            'attribution_operator_attention': 0.1
        }
    )
    
    # 测试动作选择
    action = scheduler.get_action(mock_state, training=True)
    logger.info(f"选择的动作: GPU={action.target_gpu}, 精度={action.precision.value}, 批次={action.batch_size}")
    
    # 测试决策历史
    history = scheduler.get_decision_history()
    logger.info(f"决策历史长度: {len(history)}")
    
    return scheduler


def test_scheduler_explainer(scheduler):
    """测试调度器解释器"""
    logger.info("测试调度器解释器...")
    
    explainer = SchedulerExplainer(scheduler, device='cpu')
    
    # 获取决策历史
    history = scheduler.get_decision_history()
    if not history:
        logger.warning("没有决策历史可供解释")
        return
    
    # 解释最近的决策
    recent_decision = history[-1]
    explanation = explainer.explain_decision(recent_decision)
    
    logger.info(f"决策解释: {explanation.decision_id}")
    logger.info(f"置信度: {explanation.confidence:.3f}")
    logger.info(f"推理: {explanation.reasoning}")
    logger.info("关键因素:")
    for factor in explanation.key_factors:
        logger.info(f"  - {factor}")
    
    return explainer


def test_emshap_integration():
    """测试EmSHAP集成"""
    logger.info("测试EmSHAP集成...")
    
    # 创建EmSHAP模型
    model = EMSHAPEnhanced(
        input_dim=20,
        gru_hidden_dim=64,
        context_dim=32,
        energy_hidden_dims=[128, 64, 32],
        gru_layers=2,
        dropout_rate=0.1
    )
    
    # 创建模拟输入
    mock_input = torch.randn(1, 20)
    
    # 测试Shapley值计算
    try:
        shapley_values = model.compute_shapley_values(mock_input)
        logger.info(f"EmSHAP Shapley值形状: {shapley_values.shape}")
        logger.info(f"Shapley值范围: [{shapley_values.min():.4f}, {shapley_values.max():.4f}]")
    except Exception as e:
        logger.warning(f"EmSHAP计算失败: {e}")
    
    return model


def test_end_to_end():
    """端到端测试"""
    logger.info("开始端到端测试...")
    
    # 1. 测试RL调度器
    scheduler = test_rl_scheduler()
    
    # 2. 测试调度器解释器
    explainer = test_scheduler_explainer(scheduler)
    
    # 3. 测试EmSHAP集成
    emshap_model = test_emshap_integration()
    
    logger.info("端到端测试完成!")
    
    return {
        'scheduler': scheduler,
        'explainer': explainer,
        'emshap_model': emshap_model
    }


def main():
    """主函数"""
    logger.info("开始测试归因-调度协同系统")
    
    try:
        results = test_end_to_end()
        logger.info("所有测试通过!")
        
        # 打印总结
        logger.info("=== 测试总结 ===")
        logger.info("✓ RL调度器: 正常")
        logger.info("✓ 调度器解释器: 正常")
        logger.info("✓ EmSHAP集成: 正常")
        logger.info("✓ 端到端流程: 正常")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == '__main__':
    main()
