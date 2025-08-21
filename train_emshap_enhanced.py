"""
Enhanced EMSHAP Model Training Script
Complete training pipeline based on the paper implementation
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
from data_pipeline.feature_vector import FEATURE_COLUMNS, TARGET_COLUMN, METADATA_COLUMNS
from utils import setup_logging, create_directories, calculate_metrics


def load_and_preprocess_data(data_path: str) -> tuple:
    """
    Load and preprocess data
    
    Args:
        data_path: Data file path
        
    Returns:
        Feature data and label data
    """
    logger.info(f"Loading data: {data_path}")
    
    # Read data
    df = pd.read_parquet(data_path)
    
    # Extract features and labels
    features = df[FEATURE_COLUMNS].values
    labels = df[TARGET_COLUMN].values.reshape(-1, 1)
    
    logger.info(f"Data shape: features {features.shape}, labels {labels.shape}")
    logger.info(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
    logger.info(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]")
    
    return features, labels


def create_emshap_model(input_dim: int, config: dict) -> EMSHAPEnhanced:
    """
    Create EMSHAP model
    
    Args:
        input_dim: Input dimension
        config: Configuration parameters
        
    Returns:
        EMSHAP model
    """
    model = EMSHAPEnhanced(
        input_dim=input_dim,
        gru_hidden_dim=config.get('gru_hidden_dim', 64),
        context_dim=config.get('context_dim', 32),
        energy_hidden_dims=config.get('energy_hidden_dims', [128, 64, 32]),
        gru_layers=config.get('gru_layers', 2),
        dropout_rate=config.get('dropout_rate', 0.1)
    )
    
    logger.info(f"Created EMSHAP model: input dimension {input_dim}")
    logger.info(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_emshap_model(model: EMSHAPEnhanced, features: np.ndarray, labels: np.ndarray,
                      config: dict) -> EMSHAPTrainer:
    """
    训练EMSHAP模型
    
    Args:
        model: EMSHAP模型
        features: 特征数据
        labels: 标签数据
        config: 训练配置
        
    Returns:
        训练器实例
    """
    # 创建训练器
    trainer = EMSHAPTrainer(
        model=model,
        device=config.get('device', 'auto'),
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data(
        features, labels,
        test_size=config.get('test_size', 0.2),
        batch_size=config.get('batch_size', 32)
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


def evaluate_emshap_model(trainer: EMSHAPTrainer, test_features: np.ndarray, 
                         test_labels: np.ndarray, config: dict) -> dict:
    """
    评估EMSHAP模型
    
    Args:
        trainer: 训练器
        test_features: 测试特征
        test_labels: 测试标签
        config: 配置参数
        
    Returns:
        评估结果
    """
    logger.info("Starting model evaluation...")
    
    # Compute Shapley values
    logger.info("Computing Shapley values...")
    shapley_values = trainer.compute_shapley_values(
        test_features, 
        num_samples=config.get('shapley_samples', 1000)
    )
    
    # Compute feature importance
    feature_importance = np.mean(np.abs(shapley_values), axis=0)
    
    # Create evaluation results
    evaluation_results = {
        'shapley_values': shapley_values,
        'feature_importance': feature_importance,
        'feature_names': FEATURE_COLUMNS,
        'test_features': test_features,
        'test_labels': test_labels
    }
    
    # Print feature importance ranking
    logger.info("Feature importance ranking:")
    importance_ranking = sorted(
        zip(FEATURE_COLUMNS, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feature, importance) in enumerate(importance_ranking[:10]):
        logger.info(f"{i+1:2d}. {feature:20s}: {importance:.4f}")
    
    return evaluation_results


def visualize_results(trainer: EMSHAPTrainer, evaluation_results: dict, 
                     save_dir: str = "evaluation_results"):
    """
    可视化结果
    
    Args:
        trainer: 训练器
        evaluation_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training history
    logger.info("Plotting training history...")
    trainer.plot_training_history(os.path.join(save_dir, 'training_history.png'))
    
    # Plot feature importance
    logger.info("Plotting feature importance...")
    feature_importance = evaluation_results['feature_importance']
    feature_names = evaluation_results['feature_names']
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.argsort()[-15:][::-1]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance (EMSHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Shapley value distribution
    logger.info("Plotting Shapley value distribution...")
    shapley_values = evaluation_results['shapley_values']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top 4 important features Shapley value distribution
    top_4_features = feature_importance.argsort()[-4:][::-1]
    
    for i, feature_idx in enumerate(top_4_features):
        row = i // 2
        col = i % 2
        
        axes[row, col].hist(shapley_values[:, feature_idx], bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Shapley Values: {feature_names[feature_idx]}')
        axes[row, col].set_xlabel('Shapley Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shapley_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature correlation heatmap
    logger.info("Plotting feature correlation heatmap...")
    test_features = evaluation_results['test_features']
    feature_df = pd.DataFrame(test_features, columns=feature_names)
    correlation_matrix = feature_df.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization results saved to: {save_dir}")


def save_results(trainer: EMSHAPTrainer, evaluation_results: dict, 
                save_dir: str = "evaluation_results"):
    """
    保存结果
    
    Args:
        trainer: 训练器
        evaluation_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, 'emshap_enhanced_model.pth')
    trainer.save_model(model_path)
    
    # 保存Shapley值
    shapley_df = pd.DataFrame(
        evaluation_results['shapley_values'],
        columns=evaluation_results['feature_names']
    )
    shapley_df.to_csv(os.path.join(save_dir, 'shapley_values.csv'), index=False)
    
    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': evaluation_results['feature_names'],
        'importance': evaluation_results['feature_importance']
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(save_dir, 'feature_importance.csv'), index=False)
    
    # 保存评估结果
    results_summary = {
        'model_info': {
            'input_dim': len(FEATURE_COLUMNS),
            'num_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'best_val_loss': trainer.best_val_loss
        },
        'feature_importance': importance_df.to_dict('records'),
        'training_history': trainer.train_history
    }
    
    with open(os.path.join(save_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {save_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Enhanced EMSHAP Model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Data file path')
    parser.add_argument('--config', type=str, default='config/emshap_config.json',
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("Starting Enhanced EMSHAP model training")
    
    # Create directories
    create_directories(['checkpoints', 'logs', args.output_dir])
    
    # Load configuration
    config = {
        'device': args.device,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'num_epochs': 10,  # Reduced epochs for quick testing
        'patience': 5,
        'test_size': 0.2,
        'gru_hidden_dim': 64,
        'context_dim': 32,
        'energy_hidden_dims': [128, 64, 32],
        'gru_layers': 2,
        'dropout_rate': 0.1,
        'shapley_samples': 100,  # Reduced samples for quick testing
        'save_dir': 'checkpoints'
    }
    
    # Load data
    features, labels = load_and_preprocess_data(args.data_path)
    
    # Create model
    model = create_emshap_model(len(FEATURE_COLUMNS), config)
    
    # Train model
    trainer = train_emshap_model(model, features, labels, config)
    
    # Evaluate model
    evaluation_results = evaluate_emshap_model(trainer, features, labels, config)
    
    # Visualize results
    visualize_results(trainer, evaluation_results, args.output_dir)
    
    # Save results
    save_results(trainer, evaluation_results, args.output_dir)
    
    logger.info("Enhanced EMSHAP model training completed!")


if __name__ == "__main__":
    main()
