"""
使用Google Cluster Data训练EMSHAP增强模型
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
from google_cluster_data_loader import GoogleClusterDataLoader
from utils import setup_logging, create_directories, calculate_metrics


def load_google_cluster_data(data_types: list = None, force_download: bool = False) -> tuple:
    """
    加载Google Cluster Data
    
    Args:
        data_types: 数据类型列表
        force_download: 是否强制重新下载
        
    Returns:
        特征数组、目标数组和数据加载器
    """
    loader = GoogleClusterDataLoader()
    
    # 加载集群数据
    cluster_data = loader.load_cluster_data(data_types or ['task_events', 'task_usage', 'machine_events'])
    
    # 合并数据
    merged_data = loader.merge_cluster_data(cluster_data)
    
    # 预处理数据
    features, targets = loader.preprocess_cluster_data_for_emshap(
        merged_data, 
        input_dim=64,
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
        num_epochs=config.get('num_epochs', 100),
        patience=config.get('patience', 20),
        save_dir=config.get('save_dir', 'checkpoints')
    )
    
    return trainer


def evaluate_cluster_emshap_model(trainer: EMSHAPTrainer, test_features: np.ndarray, 
                                test_labels: np.ndarray, loader: GoogleClusterDataLoader,
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
    logger.info("开始Google Cluster Data EMSHAP模型评估...")
    
    # 计算Shapley值
    logger.info("计算Shapley值...")
    shapley_values = trainer.compute_shapley_values(
        test_features, 
        num_samples=config.get('shapley_samples', 500)
    )
    
    # 计算特征重要性
    feature_importance = np.mean(np.abs(shapley_values), axis=0)
    
    # 获取特征名称
    feature_names = loader.feature_mapping.get('feature_columns', 
                                              [f'feature_{i}' for i in range(len(feature_importance))])
    
    # 创建评估结果
    evaluation_results = {
        'shapley_values': shapley_values,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'test_features': test_features,
        'test_labels': test_labels,
        'loader_mapping': loader.feature_mapping
    }
    
    # 打印特征重要性排名
    logger.info("Google Cluster Data特征重要性排名:")
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
    
    # 绘制训练历史
    logger.info("绘制训练历史...")
    trainer.plot_training_history(os.path.join(save_dir, 'cluster_training_history.png'))
    
    # 绘制特征重要性
    logger.info("绘制特征重要性...")
    feature_importance = evaluation_results['feature_importance']
    feature_names = evaluation_results['feature_names']
    
    plt.figure(figsize=(16, 12))
    top_features = feature_importance.argsort()[-20:][::-1]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel('特征重要性 (Shapley值)')
    plt.title('Google Cluster Data - 前20个重要特征 (EMSHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_feature_importance.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(save_dir, 'cluster_shapley_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制特征相关性热图
    logger.info("绘制特征相关性热图...")
    test_features = evaluation_results['test_features']
    feature_df = pd.DataFrame(test_features, columns=feature_names)
    correlation_matrix = feature_df.corr()
    
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Google Cluster Data - 特征相关性矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制CPU使用率预测结果
    logger.info("绘制CPU使用率预测结果...")
    test_labels = evaluation_results['test_labels']
    
    # 使用模型进行预测
    model = trainer.model
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_features)
        predictions = model(test_tensor, torch.ones_like(test_tensor))[0].cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.scatter(test_labels.flatten(), predictions.flatten(), alpha=0.5)
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=2)
    plt.xlabel('实际CPU使用率')
    plt.ylabel('预测CPU使用率')
    plt.title('Google Cluster Data - CPU使用率预测结果')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_cpu_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Google Cluster Data可视化结果已保存到: {save_dir}")


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
    
    logger.info(f"Google Cluster Data结果已保存到: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用Google Cluster Data训练EMSHAP增强模型')
    parser.add_argument('--data-types', nargs='+', default=['task_events', 'task_usage', 'machine_events'],
                       choices=['task_events', 'task_usage', 'machine_events'],
                       help='要加载的数据类型')
    parser.add_argument('--input-dim', type=int, default=64,
                       help='输入特征维度')
    parser.add_argument('--context-dim', type=int, default=64,
                       help='上下文向量维度')
    parser.add_argument('--target-column', type=str, default='cpu_rate',
                       help='目标列名')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_cluster',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    parser.add_argument('--force-download', action='store_true',
                       help='强制重新下载数据')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger.info("开始使用Google Cluster Data训练EMSHAP增强模型")
    
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
    
    # 1. 加载Google Cluster Data
    logger.info(f"加载Google Cluster Data: {args.data_types}")
    features, targets, loader = load_google_cluster_data(args.data_types, args.force_download)
    
    # 2. 创建模型
    logger.info("创建EMSHAP模型...")
    model = create_emshap_model_for_cluster_data(len(loader.feature_mapping['feature_columns']), config)
    
    # 3. 训练模型
    logger.info("训练EMSHAP模型...")
    trainer = train_emshap_with_cluster_data(model, features, targets, config)
    
    # 4. 评估模型
    logger.info("评估EMSHAP模型...")
    evaluation_results = evaluate_cluster_emshap_model(trainer, features, targets, loader, config)
    
    # 5. 可视化结果
    visualize_cluster_results(trainer, evaluation_results, args.output_dir)
    
    # 6. 保存结果
    save_cluster_results(trainer, evaluation_results, args.output_dir)
    
    logger.info("Google Cluster Data EMSHAP增强模型训练完成！")


if __name__ == "__main__":
    main()
