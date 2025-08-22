"""
使用Google数据集的EMSHAP增强模型训练脚本
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.emshap_enhanced import EMSHAPEnhanced
from models.emshap_trainer import EMSHAPTrainer
from google_dataset_integration import GoogleDatasetLoader, EMSHAPGoogleDataProcessor
from utils import setup_logging, create_directories, calculate_metrics


def load_google_data(data_type: str = 'analytics', limit: int = 10000) -> pd.DataFrame:
    """
    加载Google数据集
    
    Args:
        data_type: 数据类型 ('analytics', 'billing')
        limit: 数据条数限制
        
    Returns:
        Google数据框
    """
    loader = GoogleDatasetLoader()
    
    if data_type == 'analytics':
        return loader.load_google_analytics_data(limit=limit)
    elif data_type == 'billing':
        return loader.load_google_cloud_billing_data(limit=limit)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def process_google_data_for_emshap(df: pd.DataFrame, input_dim: int = 64, 
                                 context_dim: int = 32, target_column: str = None) -> tuple:
    """
    处理Google数据用于EMSHAP模型
    
    Args:
        df: Google数据框
        input_dim: 输入维度
        context_dim: 上下文维度
        target_column: 目标列名
        
    Returns:
        处理后的特征、目标和处理器
    """
    processor = EMSHAPGoogleDataProcessor(input_dim=input_dim, context_dim=context_dim)
    
    features, targets = processor.process_google_data_for_emshap(df, target_column)
    
    return features, targets, processor


def create_emshap_model_for_google_data(input_dim: int, config: dict) -> EMSHAPEnhanced:
    """
    为Google数据创建EMSHAP模型
    
    Args:
        input_dim: 输入维度
        config: 配置参数
        
    Returns:
        EMSHAP模型
    """
    model = EMSHAPEnhanced(
        input_dim=input_dim,
        gru_hidden_dim=config.get('gru_hidden_dim', 128),  # 增加隐藏层维度
        context_dim=config.get('context_dim', 64),  # 增加上下文维度
        energy_hidden_dims=config.get('energy_hidden_dims', [256, 128, 64]),  # 增加能量网络维度
        gru_layers=config.get('gru_layers', 3),  # 增加GRU层数
        dropout_rate=config.get('dropout_rate', 0.2)  # 增加dropout
    )
    
    logger.info(f"创建Google数据EMSHAP模型: 输入维度 {input_dim}")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_emshap_with_google_data(model: EMSHAPEnhanced, features: np.ndarray, 
                                targets: np.ndarray, config: dict) -> EMSHAPTrainer:
    """
    使用Google数据训练EMSHAP模型
    
    Args:
        model: EMSHAP模型
        features: 特征数据
        targets: 目标数据
        config: 训练配置
        
    Returns:
        训练器实例
    """
    # 创建训练器
    trainer = EMSHAPTrainer(
        model=model,
        device=config.get('device', 'auto'),
        learning_rate=config.get('learning_rate', 5e-4),  # 降低学习率
        weight_decay=config.get('weight_decay', 1e-3)  # 增加正则化
    )
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data(
        features, targets,
        test_size=config.get('test_size', 0.2),
        batch_size=config.get('batch_size', 64)  # 增加批次大小
    )
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 200),  # 增加训练轮数
        patience=config.get('patience', 30),  # 增加早停耐心
        save_dir=config.get('save_dir', 'checkpoints')
    )
    
    return trainer


def evaluate_google_emshap_model(trainer: EMSHAPTrainer, test_features: np.ndarray, 
                               test_labels: np.ndarray, processor: EMSHAPGoogleDataProcessor,
                               config: dict) -> dict:
    """
    评估Google数据EMSHAP模型
    
    Args:
        trainer: 训练器
        test_features: 测试特征
        test_labels: 测试标签
        processor: 数据处理器
        config: 配置参数
        
    Returns:
        评估结果
    """
    logger.info("开始Google数据EMSHAP模型评估...")
    
    # 计算Shapley值
    logger.info("计算Shapley值...")
    shapley_values = trainer.compute_shapley_values(
        test_features, 
        num_samples=config.get('shapley_samples', 500)  # 减少样本数以加快速度
    )
    
    # 计算特征重要性
    feature_importance = np.mean(np.abs(shapley_values), axis=0)
    
    # 获取特征名称
    feature_names = processor.feature_mapping.get('feature_columns', 
                                                 [f'feature_{i}' for i in range(len(feature_importance))])
    
    # 创建评估结果
    evaluation_results = {
        'shapley_values': shapley_values,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'test_features': test_features,
        'test_labels': test_labels,
        'processor_mapping': processor.feature_mapping
    }
    
    # 打印特征重要性排名
    logger.info("Google数据特征重要性排名:")
    importance_ranking = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feature, importance) in enumerate(importance_ranking[:15]):  # 显示前15个特征
        logger.info(f"{i+1:2d}. {feature:25s}: {importance:.4f}")
    
    return evaluation_results


def visualize_google_results(trainer: EMSHAPTrainer, evaluation_results: dict, 
                           save_dir: str = "evaluation_results_google"):
    """
    可视化Google数据结果
    
    Args:
        trainer: 训练器
        evaluation_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制训练历史
    logger.info("绘制训练历史...")
    trainer.plot_training_history(os.path.join(save_dir, 'google_training_history.png'))
    
    # 绘制特征重要性
    logger.info("绘制特征重要性...")
    feature_importance = evaluation_results['feature_importance']
    feature_names = evaluation_results['feature_names']
    
    plt.figure(figsize=(14, 10))
    top_features = feature_importance.argsort()[-20:][::-1]  # 显示前20个特征
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel('特征重要性 (Shapley值)')
    plt.title('Google数据 - 前20个重要特征 (EMSHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'google_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制Shapley值分布
    logger.info("绘制Shapley值分布...")
    shapley_values = evaluation_results['shapley_values']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 前6个重要特征的Shapley值分布
    top_6_features = feature_importance.argsort()[-6:][::-1]
    
    for i, feature_idx in enumerate(top_6_features):
        axes[i].hist(shapley_values[:, feature_idx], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Shapley值: {feature_names[feature_idx]}')
        axes[i].set_xlabel('Shapley值')
        axes[i].set_ylabel('频次')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'google_shapley_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制特征相关性热图
    logger.info("绘制特征相关性热图...")
    test_features = evaluation_results['test_features']
    feature_df = pd.DataFrame(test_features, columns=feature_names)
    correlation_matrix = feature_df.corr()
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Google数据 - 特征相关性矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'google_feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Google数据可视化结果已保存到: {save_dir}")


def save_google_results(trainer: EMSHAPTrainer, evaluation_results: dict, 
                       save_dir: str = "evaluation_results_google"):
    """
    保存Google数据结果
    
    Args:
        trainer: 训练器
        evaluation_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'google_emshap_enhanced_model.pth')
    trainer.save_model(model_path)
    
    # 保存Shapley值
    shapley_df = pd.DataFrame(
        evaluation_results['shapley_values'],
        columns=evaluation_results['feature_names']
    )
    shapley_df.to_csv(os.path.join(save_dir, 'google_shapley_values.csv'), index=False)
    
    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': evaluation_results['feature_names'],
        'importance': evaluation_results['feature_importance']
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(save_dir, 'google_feature_importance.csv'), index=False)
    
    # 保存评估结果
    results_summary = {
        'model_info': {
            'input_dim': len(evaluation_results['feature_names']),
            'num_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'best_val_loss': trainer.best_val_loss
        },
        'feature_importance': importance_df.to_dict('records'),
        'training_history': trainer.train_history,
        'processor_mapping': evaluation_results['processor_mapping']
    }
    
    with open(os.path.join(save_dir, 'google_evaluation_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Google数据结果已保存到: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用Google数据训练EMSHAP增强模型')
    parser.add_argument('--data-type', type=str, default='analytics',
                       choices=['analytics', 'billing'],
                       help='Google数据类型')
    parser.add_argument('--data-limit', type=int, default=10000,
                       help='数据条数限制')
    parser.add_argument('--input-dim', type=int, default=64,
                       help='输入特征维度')
    parser.add_argument('--context-dim', type=int, default=32,
                       help='上下文向量维度')
    parser.add_argument('--target-column', type=str, default=None,
                       help='目标列名')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_google',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger.info("开始使用Google数据训练EMSHAP增强模型")
    
    # 创建目录
    create_directories(['checkpoints', 'logs', args.output_dir, 'data'])
    
    # 加载配置
    config = {
        'device': args.device,
        'learning_rate': 5e-4,
        'weight_decay': 1e-3,
        'batch_size': 64,
        'num_epochs': 50,  # 减少轮数以快速测试
        'patience': 15,
        'test_size': 0.2,
        'gru_hidden_dim': 128,
        'context_dim': args.context_dim,
        'energy_hidden_dims': [256, 128, 64],
        'gru_layers': 3,
        'dropout_rate': 0.2,
        'shapley_samples': 200,  # 减少样本数以加快速度
        'save_dir': 'checkpoints'
    }
    
    # 1. 加载Google数据
    logger.info(f"加载Google {args.data_type}数据...")
    google_df = load_google_data(args.data_type, args.data_limit)
    
    # 2. 处理数据
    logger.info("处理Google数据...")
    features, targets, processor = process_google_data_for_emshap(
        google_df, 
        input_dim=args.input_dim,
        context_dim=args.context_dim,
        target_column=args.target_column
    )
    
    # 3. 创建模型
    logger.info("创建EMSHAP模型...")
    model = create_emshap_model_for_google_data(len(processor.feature_mapping['feature_columns']), config)
    
    # 4. 训练模型
    logger.info("训练EMSHAP模型...")
    trainer = train_emshap_with_google_data(model, features, targets, config)
    
    # 5. 评估模型
    logger.info("评估EMSHAP模型...")
    evaluation_results = evaluate_google_emshap_model(trainer, features, targets, processor, config)
    
    # 6. 可视化结果
    visualize_google_results(trainer, evaluation_results, args.output_dir)
    
    # 7. 保存结果
    save_google_results(trainer, evaluation_results, args.output_dir)
    
    logger.info("Google数据EMSHAP增强模型训练完成！")


if __name__ == "__main__":
    main()
