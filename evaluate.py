"""
模型评估脚本
用于评估训练好的EMSHAP和功耗预测模型性能
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import onnxruntime as ort
from typing import Dict, Any, List, Tuple
from loguru import logger

from config import config
from utils import setup_logging, create_directories, calculate_metrics, plot_predictions
from data_pipeline.feature_vector_pb2 import FEATURE_COLUMNS, TARGET_COLUMN, METADATA_COLUMNS


class ModelEvaluator:
    """
    模型评估器
    用于评估ONNX模型的性能
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
        
        # 初始化ONNX运行时会话
        self.emshap_session = None
        self.power_predictor_session = None
        
        logger.info("初始化模型评估器")
    
    def load_models(self, emshap_path: str = None, power_predictor_path: str = None):
        """
        加载ONNX模型
        
        Args:
            emshap_path: EMSHAP模型路径
            power_predictor_path: 功耗预测模型路径
        """
        emshap_path = emshap_path or config.model.emshap_model_path
        power_predictor_path = power_predictor_path or config.model.power_predictor_path
        
        # 加载EMSHAP模型
        if os.path.exists(emshap_path):
            self.emshap_session = ort.InferenceSession(emshap_path)
            logger.info(f"加载EMSHAP模型: {emshap_path}")
        else:
            logger.warning(f"EMSHAP模型文件不存在: {emshap_path}")
        
        # 加载功耗预测模型
        if os.path.exists(power_predictor_path):
            self.power_predictor_session = ort.InferenceSession(power_predictor_path)
            logger.info(f"加载功耗预测模型: {power_predictor_path}")
        else:
            logger.warning(f"功耗预测模型文件不存在: {power_predictor_path}")
    
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
        
        logger.info(f"测试数据形状: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def evaluate_power_predictor(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        评估功耗预测模型
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估结果字典
        """
        if self.power_predictor_session is None:
            logger.error("功耗预测模型未加载")
            return {}
        
        logger.info("开始评估功耗预测模型")
        
        # 准备输入
        input_name = self.power_predictor_session.get_inputs()[0].name
        output_name = self.power_predictor_session.get_outputs()[0].name
        
        # 进行预测
        predictions = []
        batch_size = 32
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            
            # 运行推理
            outputs = self.power_predictor_session.run(
                [output_name], 
                {input_name: batch_X.astype(np.float32)}
            )
            
            predictions.extend(outputs[0].flatten())
        
        predictions = np.array(predictions)
        
        # 计算评估指标
        metrics = calculate_metrics(y_test, predictions)
        
        # 保存预测结果
        results = {
            'model_type': 'power_predictor',
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': y_test.tolist()
        }
        
        logger.info("功耗预测模型评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return results
    
    def evaluate_emshap(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        评估EMSHAP模型
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估结果字典
        """
        if self.emshap_session is None:
            logger.error("EMSHAP模型未加载")
            return {}
        
        logger.info("开始评估EMSHAP模型")
        
        # 准备输入
        input_names = [input.name for input in self.emshap_session.get_inputs()]
        output_names = [output.name for output in self.emshap_session.get_outputs()]
        
        # 进行预测
        energies = []
        proposal_means = []
        proposal_logvars = []
        contexts = []
        
        batch_size = 32
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_size_actual = len(batch_X)
            
            # 创建掩码（50%掩码率）
            mask = np.random.binomial(1, 0.5, (batch_size_actual, 1, batch_X.shape[1])).astype(np.bool_)
            
            # 准备输入
            inputs = {
                input_names[0]: batch_X.astype(np.float32),
                input_names[1]: mask.astype(np.bool_)
            }
            
            # 运行推理
            outputs = self.emshap_session.run(output_names, inputs)
            
            # 收集输出
            energies.extend(outputs[0].flatten())
            proposal_means.extend(outputs[1])
            proposal_logvars.extend(outputs[2])
            contexts.extend(outputs[3])
        
        energies = np.array(energies)
        proposal_means = np.array(proposal_means)
        proposal_logvars = np.array(proposal_logvars)
        contexts = np.array(contexts)
        
        # 计算能量统计
        energy_stats = {
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies)
        }
        
        # 计算提议分布统计
        proposal_stats = {
            'mean_proposal_mean': np.mean(proposal_means),
            'std_proposal_mean': np.std(proposal_means),
            'mean_proposal_logvar': np.mean(proposal_logvars),
            'std_proposal_logvar': np.std(proposal_logvars)
        }
        
        # 计算上下文统计
        context_stats = {
            'mean_context': np.mean(contexts),
            'std_context': np.std(contexts),
            'context_dim': contexts.shape[1]
        }
        
        # 保存评估结果
        results = {
            'model_type': 'emshap',
            'energy_stats': energy_stats,
            'proposal_stats': proposal_stats,
            'context_stats': context_stats,
            'energies': energies.tolist(),
            'proposal_means': proposal_means.tolist(),
            'proposal_logvars': proposal_logvars.tolist(),
            'contexts': contexts.tolist()
        }
        
        logger.info("EMSHAP模型评估结果:")
        logger.info("能量统计:")
        for stat, value in energy_stats.items():
            logger.info(f"  {stat}: {value:.4f}")
        
        logger.info("提议分布统计:")
        for stat, value in proposal_stats.items():
            logger.info(f"  {stat}: {value:.4f}")
        
        logger.info("上下文统计:")
        for stat, value in context_stats.items():
            logger.info(f"  {stat}: {value:.4f}")
        
        return results
    
    def compare_models(self, power_results: Dict[str, Any], emshap_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个模型的性能
        
        Args:
            power_results: 功耗预测模型结果
            emshap_results: EMSHAP模型结果
            
        Returns:
            比较结果字典
        """
        logger.info("开始模型性能比较")
        
        comparison = {
            'power_predictor_metrics': power_results.get('metrics', {}),
            'emshap_energy_stats': emshap_results.get('energy_stats', {}),
            'comparison_summary': {}
        }
        
        # 添加比较摘要
        if 'metrics' in power_results:
            power_metrics = power_results['metrics']
            comparison['comparison_summary']['power_predictor_r2'] = power_metrics.get('r2', 0)
            comparison['comparison_summary']['power_predictor_mse'] = power_metrics.get('mse', 0)
            comparison['comparison_summary']['power_predictor_mae'] = power_metrics.get('mae', 0)
        
        if 'energy_stats' in emshap_results:
            emshap_stats = emshap_results['energy_stats']
            comparison['comparison_summary']['emshap_mean_energy'] = emshap_stats.get('mean_energy', 0)
            comparison['comparison_summary']['emshap_energy_std'] = emshap_stats.get('std_energy', 0)
        
        logger.info("模型性能比较:")
        for key, value in comparison['comparison_summary'].items():
            logger.info(f"  {key}: {value:.4f}")
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], output_dir: str = 'evaluation_results'):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON结果
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {json_path}")
        
        # 保存预测图表（如果有功耗预测结果）
        if 'power_predictor' in results and 'predictions' in results['power_predictor']:
            predictions = np.array(results['power_predictor']['predictions'])
            targets = np.array(results['power_predictor']['targets'])
            
            plot_path = os.path.join(output_dir, f'power_predictor_predictions_{timestamp}.png')
            plot_predictions(targets, predictions, save_path=plot_path)
    
    def run_evaluation(self, data_path: str = None, emshap_path: str = None, 
                      power_predictor_path: str = None):
        """
        运行完整的评估流程
        
        Args:
            data_path: 数据文件路径
            emshap_path: EMSHAP模型路径
            power_predictor_path: 功耗预测模型路径
        """
        try:
            # 加载模型
            self.load_models(emshap_path, power_predictor_path)
            
            # 加载测试数据
            X_test, y_test = self.load_test_data(data_path)
            
            # 评估结果
            results = {}
            
            # 评估功耗预测模型
            if self.power_predictor_session is not None:
                power_results = self.evaluate_power_predictor(X_test, y_test)
                results['power_predictor'] = power_results
            
            # 评估EMSHAP模型
            if self.emshap_session is not None:
                emshap_results = self.evaluate_emshap(X_test, y_test)
                results['emshap'] = emshap_results
            
            # 比较模型性能
            if 'power_predictor' in results and 'emshap' in results:
                comparison = self.compare_models(
                    results['power_predictor'], 
                    results['emshap']
                )
                results['comparison'] = comparison
            
            # 保存结果
            self.save_results(results)
            
            logger.info("评估流程完成")
            
        except Exception as e:
            logger.error(f"评估流程失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--data-path', type=str, help='测试数据文件路径')
    parser.add_argument('--emshap-path', type=str, help='EMSHAP模型路径')
    parser.add_argument('--power-predictor-path', type=str, help='功耗预测模型路径')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator()
    evaluator.run_evaluation(
        data_path=args.data_path,
        emshap_path=args.emshap_path,
        power_predictor_path=args.power_predictor_path
    )


if __name__ == "__main__":
    main()
