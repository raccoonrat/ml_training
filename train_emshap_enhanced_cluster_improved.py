"""
改进版本：使用Google Cluster Data训练EMSHAP增强模型
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
from google_cluster_data_loader_improved import GoogleClusterDataLoaderImproved
from utils import setup_logging, create_directories, calculate_metrics


def load_google_cluster_data(data_types: list = None, force_download: bool = False) -> tuple:
    """
    加载Google Cluster Data（改进版本）
    
    Args:
        data_types: 数据类型列表
        force_download: 是否强制重新下载
        
    Returns:
        特征数组、目标数组和数据加载器
    """
    loader = GoogleClusterDataLoaderImproved()
    
    # 加载集群数据
    cluster_data = loader.load_cluster_data(data_types or ['task_events', 'task_usage', 'machine_events'])
    
    # 合并数据
    merged_data = loader.merge_cluster_data(cluster_data)
    
    # 使用更大的输入维度以捕获更多特征
    features, targets = loader.preprocess_cluster_data_for_emshap(
        merged_data, 
        input_dim=128,  # 增加输入维度
        target_column='cpu_rate'
    )
    
    return features, targets, loader


def create_emshap_model_for_cluster_data(input_dim: int, config: dict) -> EMSHAPEnhanced:
    """
    为Google Cluster Data创建EMSHAP模型
    
    Args:
        input_dim: 输入维度
        config: 配置参数
        
    Returns:
        EMSHAP模型
    """
    model = EMSHAPEnhanced(
        input_dim=input_dim,
        gru_hidden_dim=config.get('gru_hidden_dim', 128),
        context_dim=config.get('context_dim', 64),
        energy_hidden_dims=config.get('energy_hidden_dims', [256, 128, 64]),
        gru_layers=config.get('gru_layers', 3),
        dropout_rate=config.get('dropout_rate', 0.2)
    )
    
    logger.info(f"创建Google Cluster Data EMSHAP模型: 输入维度 {input_dim}")
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_emshap_with_cluster_data(model: EMSHAPEnhanced, features: np.ndarray, 
                                 targets: np.ndarray, config: dict) -> EMSHAPTrainer:
    """
    使用Google Cluster Data训练EMSHAP模型
    
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
        learning_rate=config.get('learning_rate', 5e-4),
        weight_decay=config.get('weight_decay', 1e-3)
    )
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data(
        features, targets,
        test_size=config.get('test_size', 0.2),
        batch_size=config.get('batch_size', 64)
    )
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 30),  # 减少轮数以快速测试
        patience=config.get('patience', 10),
        save_dir=config.get('save_dir', 'checkpoints')
    )
    
    return trainer


def evaluate_cluster_emshap_model(trainer: EMSHAPTrainer, test_features: np.ndarray, 
                                test_labels: np.ndarray, loader: GoogleClusterDataLoaderImproved,
                                config: dict) -> dict:
    """
    评估Google Cluster Data EMSHAP模型
    
    Args:
        trainer: 训练器
        test_features: 测试特征
        test_labels: 测试标签
        loader: 数据加载器
        config: 配置参数
        
    Returns:
        评估结果
    """
    logger.info("Starting Google Cluster Data EMSHAP model evaluation...")
    
    # Calculate Shapley values
    logger.info("Calculating Shapley values...")
    shapley_values = trainer.compute_shapley_values(
        test_features, 
        num_samples=config.get('shapley_samples', 100)  # 减少样本数以加快速度
    )
    
    # Calculate feature importance
    feature_importance = np.mean(np.abs(shapley_values), axis=0)
    
    # Get feature names
    feature_names = loader.feature_mapping.get('feature_columns', 
                                             [f'feature_{i}' for i in range(len(feature_importance))])
    
    # Create evaluation results
    evaluation_results = {
        'shapley_values': shapley_values,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'test_features': test_features,
        'test_labels': test_labels,
        'loader_mapping': loader.feature_mapping
    }
    
    # Print feature importance ranking
    logger.info("Google Cluster Data Feature Importance Ranking:")
    importance_ranking = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feature, importance) in enumerate(importance_ranking[:15]):
        logger.info(f"{i+1:2d}. {feature:30s}: {importance:.4f}")
    
    return evaluation_results


def visualize_cluster_results(trainer: EMSHAPTrainer, evaluation_results: dict, 
                            save_dir: str = "evaluation_results_cluster"):
    """
    可视化Google Cluster Data结果
    
    Args:
        trainer: 训练器
        evaluation_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training history
    logger.info("Plotting training history...")
    trainer.plot_training_history(os.path.join(save_dir, 'cluster_training_history.png'))
    
    # Plot feature importance
    logger.info("Plotting feature importance...")
    feature_importance = evaluation_results['feature_importance']
    feature_names = evaluation_results['feature_names']
    
    plt.figure(figsize=(16, 12))
    top_features = feature_importance.argsort()[-20:][::-1]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel('Feature Importance (Shapley Values)')
    plt.title('Google Cluster Data - Top 20 Important Features (EMSHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Shapley value distribution
    logger.info("Plotting Shapley value distribution...")
    shapley_values = evaluation_results['shapley_values']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 前6个重要特征的Shapley值分布
    top_6_features = feature_importance.argsort()[-6:][::-1]
    
    for i, feature_idx in enumerate(top_6_features):
        axes[i].hist(shapley_values[:, feature_idx], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Shapley Values: {feature_names[feature_idx]}')
        axes[i].set_xlabel('Shapley Values')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_shapley_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature correlation heatmap
    logger.info("Plotting feature correlation heatmap...")
    test_features = evaluation_results['test_features']
    feature_df = pd.DataFrame(test_features, columns=feature_names)
    correlation_matrix = feature_df.corr()
    
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Google Cluster Data - Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot CPU usage prediction results
    logger.info("Plotting CPU usage prediction results...")
    test_labels = evaluation_results['test_labels']
    
    # 使用模型进行预测
    model = trainer.model
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_features).to(trainer.device)
        predictions = model(test_tensor, torch.ones_like(test_tensor).to(trainer.device))[0].cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(test_labels.flatten(), predictions.flatten(), alpha=0.5)
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
    plt.xlabel('Actual CPU Usage')
    plt.ylabel('Predicted CPU Usage')
    plt.title('Google Cluster Data - CPU Usage Prediction Results')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_cpu_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Google Cluster Data visualization results saved to: {save_dir}")


def save_cluster_results(trainer: EMSHAPTrainer, evaluation_results: dict, 
                        save_dir: str = "evaluation_results_cluster"):
    """
    保存Google Cluster Data结果
    
    Args:
        trainer: 训练器
        evaluation_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'cluster_emshap_enhanced_model.pth')
    trainer.save_model(model_path)
    
    # 保存Shapley值
    shapley_df = pd.DataFrame(
        evaluation_results['shapley_values'],
        columns=evaluation_results['feature_names']
    )
    shapley_df.to_csv(os.path.join(save_dir, 'cluster_shapley_values.csv'), index=False)
    
    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': evaluation_results['feature_names'],
        'importance': evaluation_results['feature_importance']
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(save_dir, 'cluster_feature_importance.csv'), index=False)
    
    # 保存评估结果
    results_summary = {
        'model_info': {
            'input_dim': len(evaluation_results['feature_names']),
            'num_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'best_val_loss': trainer.best_val_loss
        },
        'feature_importance': importance_df.to_dict('records'),
        'training_history': trainer.train_history,
        'loader_mapping': evaluation_results['loader_mapping']
    }
    
    with open(os.path.join(save_dir, 'cluster_evaluation_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Google Cluster Data results saved to: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用Google Cluster Data训练EMSHAP增强模型（改进版本）')
    parser.add_argument('--data-types', nargs='+', default=['task_events', 'task_usage', 'machine_events'],
                       choices=['task_events', 'task_usage', 'machine_events'],
                       help='要加载的数据类型')
    parser.add_argument('--input-dim', type=int, default=64,
                       help='输入特征维度')
    parser.add_argument('--context-dim', type=int, default=64,
                       help='上下文向量维度')
    parser.add_argument('--target-column', type=str, default='cpu_rate',
                       help='Target column name')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_cluster',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger.info("Starting Google Cluster Data EMSHAP enhanced model training (improved version)")
    
    # 创建目录
    create_directories(['checkpoints', 'logs', args.output_dir, 'data'])
    
    # 加载配置
    config = {
        'device': args.device,
        'learning_rate': 1e-4,  # 降低学习率以获得更稳定的训练
        'weight_decay': 1e-4,   # 减少正则化
        'batch_size': 128,      # 增大批次大小
        'num_epochs': args.num_epochs,  # 使用命令行参数
        'patience': 15,         # 增加早停耐心
        'test_size': 0.2,
        'gru_hidden_dim': 256,  # 增大隐藏层维度
        'context_dim': args.context_dim,
        'energy_hidden_dims': [512, 256, 128],  # 增大能量网络
        'gru_layers': 4,        # 增加GRU层数
        'dropout_rate': 0.3,    # 增加dropout
        'shapley_samples': 100, # 增加Shapley值采样数
        'save_dir': 'checkpoints'
    }
    
    # 1. Load Google Cluster Data
    logger.info(f"Loading Google Cluster Data: {args.data_types}")
    features, targets, loader = load_google_cluster_data(args.data_types)
    
    # 2. Create model
    logger.info("Creating EMSHAP model...")
    model = create_emshap_model_for_cluster_data(len(loader.feature_mapping['feature_columns']), config)
    
    # 3. Train model
    logger.info("Training EMSHAP model...")
    trainer = train_emshap_with_cluster_data(model, features, targets, config)
    
    # 4. Evaluate model
    logger.info("Evaluating EMSHAP model...")
    evaluation_results = evaluate_cluster_emshap_model(trainer, features, targets, loader, config)
    
    # 5. Visualize results
    visualize_cluster_results(trainer, evaluation_results, args.output_dir)
    
    # 6. Save results
    save_cluster_results(trainer, evaluation_results, args.output_dir)
    
    logger.info("Google Cluster Data EMSHAP enhanced model training completed!")


if __name__ == "__main__":
    main()
