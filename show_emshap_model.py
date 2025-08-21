"""
展示EMSHAP增强模型的结构和输出
"""

import torch
import numpy as np
import pandas as pd
from models.emshap_enhanced import EMSHAPEnhanced
from data_pipeline.feature_vector import FEATURE_COLUMNS, TARGET_COLUMN

def show_model_structure():
    """展示模型结构"""
    print("=" * 60)
    print("EMSHAP增强模型结构")
    print("=" * 60)
    
    # 创建模型
    model = EMSHAPEnhanced(
        input_dim=20,
        gru_hidden_dim=64,
        context_dim=32,
        energy_hidden_dims=[128, 64, 32],
        gru_layers=2,
        dropout_rate=0.1
    )
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入维度: {model.input_dim}")
    print(f"GRU隐藏维度: {model.gru_hidden_dim}")
    print(f"上下文维度: {model.context_dim}")
    print(f"能量网络隐藏层: {model.energy_net.hidden_dims}")
    
    print("\n模型组件:")
    print("1. 增强GRU网络 (EnhancedGRUNetwork)")
    print("   - 双向GRU层")
    print("   - 上下文编码器")
    print("   - 提议分布参数生成器")
    
    print("\n2. 增强能量网络 (EnhancedEnergyNetwork)")
    print("   - 特征嵌入层")
    print("   - 注意力模块")
    print("   - 上下文投影")
    print("   - 主网络层")
    print("   - 温度参数")
    
    print("\n3. Shapley值计算器 (ShapleyCalculator)")
    print("   - Monte Carlo方法")
    print("   - 置信区间计算")
    
    return model

def show_model_outputs(model):
    """展示模型输出"""
    print("\n" + "=" * 60)
    print("模型输出示例")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 1
    input_dim = 20
    
    # 模拟特征数据
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones_like(x).bool()
    
    print(f"输入形状: {x.shape}")
    print(f"掩码形状: {mask.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        energy, proposal_params = model(x, mask)
    
    print(f"\n能量输出形状: {energy.shape}")
    print(f"能量值范围: [{energy.min().item():.4f}, {energy.max().item():.4f}]")
    
    print(f"\n提议分布参数:")
    print(f"  均值形状: {proposal_params['mean'].shape}")
    print(f"  对数方差形状: {proposal_params['logvar'].shape}")
    print(f"  上下文形状: {proposal_params['context'].shape}")
    
    # 计算损失
    energy_loss = model.compute_energy_loss(x, energy, proposal_params)
    kl_loss = model.compute_kl_loss(proposal_params)
    
    print(f"\n损失值:")
    print(f"  能量损失: {energy_loss.item():.4f}")
    print(f"  KL损失: {kl_loss.item():.4f}")

def show_feature_importance():
    """展示特征重要性"""
    print("\n" + "=" * 60)
    print("特征重要性排名")
    print("=" * 60)
    
    # 读取特征重要性文件
    try:
        importance_df = pd.read_csv('evaluation_results/feature_importance.csv')
        
        print("前10个最重要的特征:")
        for i, row in importance_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:15s}: {row['importance']:.6f}")
        
        print(f"\n特征总数: {len(importance_df)}")
        print(f"重要性范围: [{importance_df['importance'].min():.6f}, {importance_df['importance'].max():.6f}]")
        
    except FileNotFoundError:
        print("未找到特征重要性文件，请先运行训练脚本")

def show_training_results():
    """展示训练结果"""
    print("\n" + "=" * 60)
    print("训练结果摘要")
    print("=" * 60)
    
    try:
        with open('evaluation_results/evaluation_summary.json', 'r') as f:
            import json
            results = json.load(f)
        
        model_info = results['model_info']
        print(f"模型信息:")
        print(f"  输入维度: {model_info['input_dim']}")
        print(f"  参数数量: {model_info['num_parameters']:,}")
        print(f"  最佳验证损失: {model_info['best_val_loss']:.6f}")
        
        training_history = results['training_history']
        print(f"\n训练历史:")
        print(f"  训练轮数: {len(training_history['total_loss'])}")
        print(f"  最终训练损失: {training_history['total_loss'][-1]:.6f}")
        print(f"  最终验证损失: {training_history['val_loss'][-1]:.6f}")
        print(f"  学习率: {training_history['learning_rate'][-1]:.6f}")
        
    except FileNotFoundError:
        print("未找到评估摘要文件，请先运行训练脚本")

def show_model_architecture():
    """展示模型架构详情"""
    print("\n" + "=" * 60)
    print("模型架构详情")
    print("=" * 60)
    
    model = EMSHAPEnhanced(input_dim=20)
    
    print("EMSHAP增强模型包含以下核心组件:")
    
    print("\n1. 增强GRU网络:")
    print("   - 双向GRU层，捕获序列依赖关系")
    print("   - 上下文编码器，处理条件信息")
    print("   - 提议分布生成器，学习特征分布")
    print("   - 温度参数，控制分布锐度")
    
    print("\n2. 增强能量网络:")
    print("   - 特征嵌入层，将原始特征映射到高维空间")
    print("   - 多头注意力机制，捕获特征间交互")
    print("   - 上下文投影，整合条件信息")
    print("   - 多层感知机，学习能量函数")
    print("   - 温度缩放，控制能量分布")
    
    print("\n3. 训练算法:")
    print("   - 对比学习损失，区分正负样本")
    print("   - KL散度损失，正则化提议分布")
    print("   - 交替优化，平衡能量和提议网络")
    
    print("\n4. Shapley值计算:")
    print("   - Monte Carlo采样，估计特征贡献")
    print("   - 边际贡献计算，量化特征重要性")
    print("   - 置信区间估计，评估计算可靠性")

def main():
    """主函数"""
    print("EMSHAP增强模型展示")
    print("=" * 60)
    
    # 展示模型结构
    model = show_model_structure()
    
    # 展示模型输出
    show_model_outputs(model)
    
    # 展示特征重要性
    show_feature_importance()
    
    # 展示训练结果
    show_training_results()
    
    # 展示模型架构
    show_model_architecture()
    
    print("\n" + "=" * 60)
    print("模型总结")
    print("=" * 60)
    print("EMSHAP增强模型是一个基于能量的可解释AI模型，具有以下特点:")
    print("1. 使用能量网络学习特征与目标的关系")
    print("2. 使用GRU网络学习特征间的时序依赖")
    print("3. 通过对比学习训练，提高模型泛化能力")
    print("4. 计算Shapley值，提供特征归因解释")
    print("5. 支持条件生成和不确定性量化")
    print("6. 适用于高维特征空间的可解释性分析")

if __name__ == "__main__":
    main()
