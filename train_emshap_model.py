"""
EMSHAP模型训练脚本
训练EMSHAP模型，实现动态掩码机制
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from loguru import logger

from config import config
from utils import (
    setup_logging, create_directories, save_checkpoint, load_checkpoint,
    calculate_metrics, plot_training_history, early_stopping, 
    count_parameters, get_device, set_seed, format_time
)
from models.emshap import EMSHAP, create_emshap_model
from data_pipeline.feature_vector import FEATURE_COLUMNS, TARGET_COLUMN, METADATA_COLUMNS


class EMSHAPTrainer:
    """
    EMSHAP模型训练器
    """
    
    def __init__(self, model_config: dict = None):
        """
        初始化训练器
        
        Args:
            model_config: 模型配置
        """
        self.model_config = model_config or config.model.__dict__
        self.training_config = config.training.__dict__
        
        # 设置设备
        self.device = get_device()
        
        # 设置随机种子
        set_seed(config.data.random_state)
        
        # 创建必要的目录
        create_directories([
            config.model.model_dir,
            config.training.checkpoint_dir,
            os.path.dirname(config.training.log_file)
        ])
        
        # 设置日志
        setup_logging(config.training.log_file, config.training.log_level)
        
        # 初始化模型和优化器
        self.model = None
        self.optimizer = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        logger.info("初始化EMSHAP模型训练器")
    
    def load_data(self, data_path: str = None) -> tuple:
        """
        加载训练数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) 元组
        """
        # 如果没有指定数据路径，使用最新的处理数据
        if data_path is None:
            processed_dir = config.data.processed_data_dir
            data_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
            if not data_files:
                raise FileNotFoundError(f"在 {processed_dir} 中没有找到数据文件")
            
            # 使用最新的数据文件
            data_files.sort()
            data_path = os.path.join(processed_dir, data_files[-1])
        
        logger.info(f"加载数据: {data_path}")
        
        # 读取数据
        df = pd.read_parquet(data_path)
        
        # 获取特征列（排除元数据列）
        feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS + ['timestamp']]
        
        # 分离特征和目标
        X = df[feature_cols].values
        y = df[TARGET_COLUMN].values
        
        logger.info(f"数据形状: X={X.shape}, y={y.shape}")
        logger.info(f"特征数量: {len(feature_cols)}")
        
        # 划分数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config.data.test_size, random_state=config.data.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config.data.validation_size, 
            random_state=config.data.random_state
        )
        
        logger.info(f"训练集: {X_train.shape[0]} 样本")
        logger.info(f"验证集: {X_val.shape[0]} 样本")
        logger.info(f"测试集: {X_test.shape[0]} 样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_model(self, input_dim: int) -> EMSHAP:
        """
        创建EMSHAP模型
        
        Args:
            input_dim: 输入特征维度
            
        Returns:
            模型实例
        """
        model = create_emshap_model(
            input_dim=input_dim,
            gru_hidden_dim=self.model_config['emshap_gru_hidden_dim'],
            context_dim=self.model_config['emshap_context_dim'],
            energy_hidden_dims=[128, 64, 32],
            gru_layers=2,
            dropout_rate=0.1
        )
        
        model = model.to(self.device)
        
        logger.info(f"创建EMSHAP模型，参数数量: {count_parameters(model)}")
        return model
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        创建数据加载器
        
        Args:
            X_train, X_val, X_test: 特征数据
            y_train, y_val, y_test: 目标数据
            
        Returns:
            (train_loader, val_loader, test_loader) 元组
        """
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"创建数据加载器: 训练={len(train_loader)}批次, 验证={len(val_loader)}批次, 测试={len(test_loader)}批次")
        
        return train_loader, val_loader, test_loader
    
    def create_dynamic_mask(self, batch_size: int, seq_len: int, input_dim: int, 
                           mask_rate: float) -> torch.Tensor:
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
        # 创建伯努利分布的掩码
        if seq_len == 1:
            # 对于单时间点数据，创建2D掩码
            mask = torch.bernoulli(torch.full((batch_size, input_dim), 1 - mask_rate)).bool()
        else:
            # 对于序列数据，创建3D掩码
            mask = torch.bernoulli(torch.full((batch_size, seq_len, input_dim), 1 - mask_rate)).bool()
        return mask.to(self.device)
    
    def compute_mle_loss(self, energy: torch.Tensor, proposal_params: dict, 
                        x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算MLE损失（最大似然估计）
        
        Args:
            energy: 能量值
            proposal_params: 提议分布参数
            x: 原始输入
            mask: 掩码
            
        Returns:
            损失值
        """
        # 这里实现论文中的损失函数
        # 简化版本：使用能量作为损失，加上KL散度正则化
        
        # 能量损失
        energy_loss = torch.mean(energy)
        
        # KL散度正则化（简化版本）
        mean = proposal_params['mean']
        logvar = proposal_params['logvar']
        
        # 计算KL散度：KL(q||p) = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = energy_loss + 0.1 * kl_loss
        
        return total_loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, 
                   total_epochs: int) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            total_epochs: 总epoch数
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0
        
        # 计算当前epoch的掩码率
        mask_rate = (self.model_config['min_mask_rate'] + 
                    (self.model_config['max_mask_rate'] - self.model_config['min_mask_rate']) * 
                    (epoch / total_epochs))
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_size, input_dim = data.shape
            
            # 创建动态掩码
            mask = self.create_dynamic_mask(batch_size, 1, input_dim, mask_rate)
            
            # 前向传播
            self.optimizer.zero_grad()
            energy, proposal_params = self.model(data, mask)
            
            # 计算损失
            loss = self.compute_mle_loss(energy, proposal_params, data, mask)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size, input_dim = data.shape
                
                # 使用固定的掩码率进行验证
                mask_rate = 0.5
                mask = self.create_dynamic_mask(batch_size, 1, input_dim, mask_rate)
                
                # 前向传播
                energy, proposal_params = self.model(data, mask)
                
                # 计算损失
                loss = self.compute_mle_loss(energy, proposal_params, data, mask)
                
                # 记录损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = None) -> dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            
        Returns:
            训练历史字典
        """
        num_epochs = num_epochs or self.model_config['num_epochs']
        
        logger.info(f"开始训练，总轮数: {num_epochs}")
        start_time = time.time()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch, num_epochs)
            
            # 验证
            val_loss = self.validate_epoch(val_loader)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 计算时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印进度
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Time: {format_time(epoch_time)}"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存检查点
                checkpoint_path = os.path.join(
                    config.training.checkpoint_dir, 
                    'emshap_best.pth'
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % config.training.save_checkpoint_every == 0:
                checkpoint_path = os.path.join(
                    config.training.checkpoint_dir, 
                    f'emshap_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
            
            # 早停检查
            if patience_counter >= self.model_config['early_stopping_patience']:
                logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break
        
        total_time = time.time() - start_time
        logger.info(f"训练完成，总时间: {format_time(total_time)}")
        
        # 构建训练历史
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        logger.info("开始模型评估")
        
        self.model.eval()
        total_loss = 0
        all_energies = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size, input_dim = data.shape
                
                # 使用固定的掩码率进行评估
                mask_rate = 0.5
                mask = self.create_dynamic_mask(batch_size, 1, input_dim, mask_rate)
                
                # 前向传播
                energy, proposal_params = self.model(data, mask)
                
                # 计算损失
                loss = self.compute_mle_loss(energy, proposal_params, data, mask)
                
                # 记录损失和能量
                total_loss += loss.item()
                all_energies.extend(energy.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        all_energies = np.array(all_energies).flatten()
        
        # 计算能量统计
        energy_stats = {
            'mean_energy': np.mean(all_energies),
            'std_energy': np.std(all_energies),
            'min_energy': np.min(all_energies),
            'max_energy': np.max(all_energies)
        }
        
        logger.info("测试集评估结果:")
        logger.info(f"  平均损失: {avg_loss:.4f}")
        for stat, value in energy_stats.items():
            logger.info(f"  {stat}: {value:.4f}")
        
        return {'loss': avg_loss, **energy_stats}
    
    def export_model(self, output_path: str = None):
        """
        导出模型为ONNX格式
        
        Args:
            output_path: 输出路径
        """
        output_path = output_path or config.model.emshap_model_path
        
        # 创建示例输入
        self.model.eval()
        batch_size = 1
        input_dim = self.model.input_dim
        dummy_input = torch.randn(batch_size, input_dim).to(self.device)
        dummy_mask = torch.ones(batch_size, 1, input_dim).bool().to(self.device)
        
        # 导出ONNX（注意：EMSHAP模型有多个输出，需要特殊处理）
        torch.onnx.export(
            self.model,
            (dummy_input, dummy_mask),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input', 'mask'],
            output_names=['energy', 'proposal_mean', 'proposal_logvar', 'context'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'mask': {0: 'batch_size'},
                'energy': {0: 'batch_size'},
                'proposal_mean': {0: 'batch_size'},
                'proposal_logvar': {0: 'batch_size'},
                'context': {0: 'batch_size'}
            }
        )
        
        logger.info(f"模型已导出到: {output_path}")
    
    def run_training(self, data_path: str = None):
        """
        运行完整的训练流程
        
        Args:
            data_path: 数据文件路径
        """
        try:
            # 加载数据
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data(data_path)
            
            # 创建模型
            self.model = self.create_model(X_train.shape[1])
            
            # 创建优化器
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.model_config['learning_rate']
            )
            
            # 创建数据加载器
            train_loader, val_loader, test_loader = self.create_data_loaders(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            
            # 训练模型
            history = self.train(train_loader, val_loader)
            
            # 加载最佳模型
            best_checkpoint_path = os.path.join(
                config.training.checkpoint_dir, 
                'emshap_best.pth'
            )
            if os.path.exists(best_checkpoint_path):
                load_checkpoint(self.model, self.optimizer, best_checkpoint_path)
                logger.info("加载最佳模型进行最终评估")
            
            # 评估模型
            test_metrics = self.evaluate(test_loader)
            
            # 绘制训练历史
            plot_training_history(
                history, 
                save_path=os.path.join(config.training.checkpoint_dir, 'emshap_training_history.png')
            )
            
            # 导出模型
            self.export_model()
            
            logger.info("EMSHAP训练流程完成")
            
        except Exception as e:
            logger.error(f"EMSHAP训练流程失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练EMSHAP模型')
    parser.add_argument('--data-path', type=str, help='数据文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.epochs:
        config.model.num_epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    
    # 创建训练器并运行训练
    trainer = EMSHAPTrainer()
    trainer.run_training(args.data_path)


if __name__ == "__main__":
    main()
