"""
EMSHAP Model Trainer
Complete training algorithm implementation, including alternating optimization, loss computation and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import os
import json
from datetime import datetime

from .emshap_enhanced import EMSHAPEnhanced, ShapleyCalculator


class EMSHAPTrainer:
    """
    EMSHAP Model Trainer
    Implementation of training algorithm from the paper
    """
    
    def __init__(self, model: EMSHAPEnhanced, device: str = 'auto', 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """
        Initialize trainer
        
        Args:
            model: EMSHAP model
            device: Device
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_history = {
            'energy_loss': [],
            'kl_loss': [],
            'total_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Best model
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        logger.info(f"EMSHAP trainer initialized successfully, device: {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def prepare_data(self, data: np.ndarray, labels: np.ndarray = None,
                    test_size: float = 0.2, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        准备训练数据
        
        Args:
            data: 输入数据
            labels: 标签（可选）
            test_size: 测试集比例
            batch_size: 批次大小
            
        Returns:
            训练和验证数据加载器
        """
        # 数据标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 分割数据
        if labels is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                data_scaled, labels, test_size=test_size, random_state=42
            )
        else:
            X_train, X_val = train_test_split(data_scaled, test_size=test_size, random_state=42)
            y_train, y_val = None, None
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        if y_train is not None:
            y_train_tensor = torch.FloatTensor(y_train)
            y_val_tensor = torch.FloatTensor(y_val)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        else:
            train_dataset = TensorDataset(X_train_tensor)
            val_dataset = TensorDataset(X_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Data preparation completed: training set {len(X_train)} samples, validation set {len(X_val)} samples")
        
        return train_loader, val_loader
    
    def create_dynamic_mask(self, batch_size: int, seq_len: int, input_dim: int,
                           mask_rate: float = 0.3) -> torch.Tensor:
        """
        创建动态掩码
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            input_dim: 输入维度
            mask_rate: 掩码率
            
        Returns:
            掩码张量
        """
        if seq_len == 1:
            mask = torch.bernoulli(torch.full((batch_size, input_dim), 1 - mask_rate)).bool()
        else:
            mask = torch.bernoulli(torch.full((batch_size, seq_len, input_dim), 1 - mask_rate)).bool()
        
        return mask.to(self.device)
    
    def compute_total_loss(self, x: torch.Tensor, energy: torch.Tensor,
                          proposal_params: Dict[str, torch.Tensor],
                          energy_weight: float = 1.0, kl_weight: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            x: 输入数据
            energy: 能量值
            proposal_params: 提议分布参数
            energy_weight: 能量损失权重
            kl_weight: KL损失权重
            
        Returns:
            总损失和损失字典
        """
        # 能量损失
        energy_loss = self.model.compute_energy_loss(x, energy, proposal_params)
        
        # KL损失
        kl_loss = self.model.compute_kl_loss(proposal_params)
        
        # 总损失
        total_loss = energy_weight * energy_loss + kl_weight * kl_loss
        
        loss_dict = {
            'energy_loss': energy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练损失字典
        """
        self.model.train()
        total_losses = {'energy_loss': 0, 'kl_loss': 0, 'total_loss': 0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 获取数据
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = None
            
            x = x.to(self.device)
            batch_size, input_dim = x.shape
            
            # 创建掩码
            mask = self.create_dynamic_mask(batch_size, 1, input_dim)
            
            # 前向传播
            self.optimizer.zero_grad()
            energy, proposal_params = self.model(x, mask)
            
            # 计算损失
            total_loss, loss_dict = self.compute_total_loss(x, energy, proposal_params)
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累积损失
            for key in total_losses:
                total_losses[key] += loss_dict[key]
            num_batches += 1
            
            # 打印进度
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {total_loss.item():.4f}")
        
        # 计算平均损失
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                if len(batch) == 2:
                    x, y = batch
                else:
                    x = batch[0]
                    y = None
                
                x = x.to(self.device)
                batch_size, input_dim = x.shape
                
                # 创建掩码
                mask = self.create_dynamic_mask(batch_size, 1, input_dim)
                
                # 前向传播
                energy, proposal_params = self.model(x, mask)
                
                # 计算损失
                loss, _ = self.compute_total_loss(x, energy, proposal_params)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, patience: int = 20, save_dir: str = 'checkpoints') -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            patience: 早停耐心值
            save_dir: 保存目录
            
        Returns:
            训练历史
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_epoch = 0
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # 训练
            train_losses = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            for key, value in train_losses.items():
                self.train_history[key].append(value)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印进度
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_losses['total_loss']:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_history': self.train_history
                }, os.path.join(save_dir, 'emshap_best.pth'))
                
                logger.info(f"Saved best model, validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered, best epoch: {best_epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info(f"Training completed, best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_history
    
    def plot_training_history(self, save_path: str = None):
        """
        绘制训练历史
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_history['total_loss'], label='Training Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Energy loss
        axes[0, 1].plot(self.train_history['energy_loss'], label='Energy Loss')
        axes[0, 1].set_title('Energy Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Energy Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(self.train_history['kl_loss'], label='KL Loss')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.train_history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Changes')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def compute_shapley_values(self, data: np.ndarray, num_samples: int = 1000) -> np.ndarray:
        """
        Compute Shapley values
        
        Args:
            data: Input data
            num_samples: Number of samples
            
        Returns:
            Shapley values
        """
        self.model.eval()
        
        # Create Shapley calculator
        calculator = ShapleyCalculator(self.model, num_samples)
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Compute Shapley values
        with torch.no_grad():
            shapley_values = calculator.compute_shapley_values(data_tensor)
        
        return shapley_values.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Model loaded from {filepath}")


# 测试代码
if __name__ == "__main__":
    # Create model
    model = EMSHAPEnhanced(input_dim=64, gru_hidden_dim=64, context_dim=32)
    
    # Create trainer
    trainer = EMSHAPTrainer(model, learning_rate=1e-3)
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(1000, 64)
    labels = np.random.randn(1000, 1)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(data, labels, batch_size=32)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=10)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Compute Shapley values
    test_data = np.random.randn(10, 64)
    shapley_values = trainer.compute_shapley_values(test_data)
    
    print(f"Shapley values shape: {shapley_values.shape}")
    print(f"Shapley values range: [{shapley_values.min():.4f}, {shapley_values.max():.4f}]")
