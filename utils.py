"""
工具函数模块
包含数据处理、模型评估、日志记录等通用函数
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_file: str = "logs/training.log", level: str = "INFO"):
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置loguru
    logger.remove()  # 移除默认处理器
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def create_directories(directories: List[str]):
    """
    创建必要的目录
    
    Args:
        directories: 目录路径列表
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")


def save_scaler(scaler: StandardScaler, filepath: str):
    """
    保存StandardScaler到文件
    
    Args:
        scaler: 要保存的StandardScaler对象
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"保存StandardScaler到: {filepath}")


def load_scaler(filepath: str) -> StandardScaler:
    """
    从文件加载StandardScaler
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的StandardScaler对象
    """
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"从文件加载StandardScaler: {filepath}")
    return scaler


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含各种评估指标的字典
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典，包含loss和metrics
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练历史', fontsize=16)
    
    # 损失曲线
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='训练损失')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线（如果有）
    if 'train_acc' in history:
        axes[0, 1].plot(history['train_acc'], label='训练准确率')
    if 'val_acc' in history:
        axes[0, 1].plot(history['val_acc'], label='验证准确率')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 学习率曲线（如果有）
    if 'lr' in history:
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
    
    # 梯度范数（如果有）
    if 'grad_norm' in history:
        axes[1, 1].plot(history['grad_norm'])
        axes[1, 1].set_title('梯度范数')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"保存训练历史图到: {save_path}")
    
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('预测结果分析', fontsize=16)
    
    # 散点图
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('预测值 vs 真实值')
    axes[0, 0].grid(True)
    
    # 残差图
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].grid(True)
    
    # 残差分布
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('残差')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('残差分布')
    axes[1, 0].grid(True)
    
    # 时间序列图（如果数据有序）
    axes[1, 1].plot(y_true[:100], label='真实值', alpha=0.7)
    axes[1, 1].plot(y_pred[:100], label='预测值', alpha=0.7)
    axes[1, 1].set_xlabel('样本索引')
    axes[1, 1].set_ylabel('值')
    axes[1, 1].set_title('时间序列对比（前100个样本）')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"保存预测结果图到: {save_path}")
    
    plt.show()


def early_stopping(val_losses: List[float], patience: int = 10) -> bool:
    """
    早停检查
    
    Args:
        val_losses: 验证损失历史
        patience: 耐心值
        
    Returns:
        是否应该停止训练
    """
    if len(val_losses) < patience:
        return False
    
    # 检查最近patience个epoch是否有改善
    recent_losses = val_losses[-patience:]
    min_loss = min(recent_losses)
    current_loss = recent_losses[-1]
    
    return current_loss > min_loss


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"保存检查点到: {filepath}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   filepath: str) -> Tuple[int, float]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点文件路径
        
    Returns:
        (epoch, loss) 元组
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"从检查点加载模型: {filepath}, epoch: {epoch}, loss: {loss}")
    return epoch, loss


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    获取可用的设备
    
    Returns:
        torch.device对象
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")
    
    return device


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重现性
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"设置随机种子: {seed}")


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
