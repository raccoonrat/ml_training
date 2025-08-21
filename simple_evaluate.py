"""
简化的模型评估脚本
评估训练好的EMSHAP和功耗预测模型性能
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Tuple
from loguru import logger

from config import config
from utils import setup_logging, create_directories, calculate_metrics
from data_pipeline.feature_vector import FEATURE_COLUMNS, TARGET_COLUMN, METADATA_COLUMNS
from models.power_predictor import PowerPredictor, create_power_predictor
from models.emshap import EMSHAP, create_emshap_model


class SimpleModelEvaluator:
    """
    简化的模型评估器
    用于评估PyTorch模型的性能
    """
    
    def __init__(self):
        """初始化评估器"""
        # 创建必要的目录
        create_directories([
            'evaluation_results',
            os.path.dirname(config.training.log_file)
        ])
        
        # 设置日志
        setup_logging(config.training.log_file, config.training.log_level)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.power_predictor = None
        self.emshap_model = None
        
        logger.info("初始化简化模型评估器")
    
    def load_models(self):
        """加载训练好的模型"""
        # 加载功耗预测模型
        power_checkpoint_path = "checkpoints/power_predictor_best.pth"
        if os.path.exists(power_checkpoint_path):
            checkpoint = torch.load(power_checkpoint_path, map_location='cpu')
            self.power_predictor = create_power_predictor(
                input_dim=21,  # 特征数量
                hidden_dims=[128, 64, 32],
                dropout_rate=0.1,
                activation='relu'
            )
            self.power_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.power_predictor.to(self.device)
            self.power_predictor.eval()
            logger.info(f"加载功耗预测模型: {power_checkpoint_path}")
        else:
            logger.warning(f"功耗预测模型文件不存在: {power_checkpoint_path}")
        
        # 加载EMSHAP模型（暂时跳过，因为架构不匹配）
        emshap_checkpoint_path = "checkpoints/emshap_best.pth"
        if os.path.exists(emshap_checkpoint_path):
            logger.info(f"EMSHAP模型文件存在: {emshap_checkpoint_path}，但暂时跳过加载（架构不匹配）")
        else:
            logger.warning(f"EMSHAP模型文件不存在: {emshap_checkpoint_path}")
    
    def load_test_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载测试数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            (X_test, y_test) 元组
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
        
        logger.info(f"加载测试数据: {data_path}")
        
        # 读取数据
        df = pd.read_parquet(data_path)
        
        # 获取特征列（排除元数据列）
        feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS + ['timestamp']]
        
        # 分离特征和目标
        X = df[feature_cols].values
        y = df[TARGET_COLUMN].values
        
        # 使用最后20%的数据作为测试集
        test_size = int(0.2 * len(X))
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        logger.info(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
        
        return X_test, y_test
    
    def evaluate_power_predictor(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估功耗预测模型"""
        if self.power_predictor is None:
            logger.warning("功耗预测模型未加载")
            return {}
        
        logger.info("开始评估功耗预测模型")
        
        # 转换为张量
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.power_predictor(X_test_tensor).squeeze()
        
        # 计算指标
        y_pred = predictions.cpu().numpy()
        y_true = y_test
        
        metrics = calculate_metrics(y_true, y_pred)
        
        logger.info("功耗预测模型评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_emshap_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估EMSHAP模型"""
        if self.emshap_model is None:
            logger.warning("EMSHAP模型未加载")
            return {}
        
        logger.info("开始评估EMSHAP模型")
        
        # 转换为张量
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # 创建掩码
        batch_size, input_dim = X_test_tensor.shape
        mask = torch.bernoulli(torch.full((batch_size, input_dim), 0.8)).bool().to(self.device)
        
        # 预测
        with torch.no_grad():
            energy, proposal_params = self.emshap_model(X_test_tensor, mask)
        
        # 计算统计指标
        energy_np = energy.cpu().numpy()
        
        metrics = {
            'mean_energy': float(np.mean(energy_np)),
            'std_energy': float(np.std(energy_np)),
            'min_energy': float(np.min(energy_np)),
            'max_energy': float(np.max(energy_np)),
            'mean_proposal_mean': float(torch.mean(proposal_params['mean']).cpu().numpy()),
            'std_proposal_mean': float(torch.std(proposal_params['mean']).cpu().numpy()),
            'mean_proposal_logvar': float(torch.mean(proposal_params['logvar']).cpu().numpy()),
            'std_proposal_logvar': float(torch.std(proposal_params['logvar']).cpu().numpy())
        }
        
        logger.info("EMSHAP模型评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_results(self, power_metrics: Dict[str, float], emshap_metrics: Dict[str, float]):
        """保存评估结果"""
        results = {
            'power_predictor': power_metrics,
            'emshap': emshap_metrics,
            'evaluation_time': pd.Timestamp.now().isoformat()
        }
        
        # 保存为JSON
        results_path = 'evaluation_results/simple_evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV
        csv_results = []
        for model_name, metrics in results.items():
            if model_name == 'evaluation_time':
                continue
            for metric, value in metrics.items():
                csv_results.append({
                    'model': model_name,
                    'metric': metric,
                    'value': value
                })
        
        df_results = pd.DataFrame(csv_results)
        csv_path = 'evaluation_results/simple_evaluation_results.csv'
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"评估结果已保存到: {results_path}")
        logger.info(f"评估结果已保存到: {csv_path}")
    
    def run_evaluation(self, data_path: str = None):
        """运行完整评估"""
        try:
            # 加载模型
            self.load_models()
            
            # 加载测试数据
            X_test, y_test = self.load_test_data(data_path)
            
            # 评估功耗预测模型
            power_metrics = self.evaluate_power_predictor(X_test, y_test)
            
            # 评估EMSHAP模型（如果可用）
            emshap_metrics = {}
            if self.emshap_model is not None:
                try:
                    emshap_metrics = self.evaluate_emshap_model(X_test, y_test)
                except Exception as e:
                    logger.warning(f"EMSHAP模型评估失败: {str(e)}")
            
            # 保存结果
            self.save_results(power_metrics, emshap_metrics)
            
            logger.info("模型评估完成")
            
        except Exception as e:
            logger.error(f"评估过程失败: {str(e)}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='简化模型评估')
    parser.add_argument('--data-path', type=str, default=None,
                       help='测试数据路径')
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = SimpleModelEvaluator()
    evaluator.run_evaluation(args.data_path)


if __name__ == "__main__":
    main()
