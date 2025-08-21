"""
Kafka消费者模块
用于从Kafka消费原始指标数据，进行预处理和特征工程
"""

import os
import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from confluent_kafka import Consumer, KafkaError, KafkaException
from sklearn.preprocessing import StandardScaler
from loguru import logger
import pickle

from .feature_vector import FeatureVector, feature_vector_to_dict, FEATURE_COLUMNS, TARGET_COLUMN, METADATA_COLUMNS
from config import config
from utils import setup_logging, create_directories, save_scaler


class KafkaConsumer:
    """
    Kafka消费者类
    负责从Kafka消费数据，进行预处理和特征工程
    """
    
    def __init__(self, kafka_config: Dict[str, Any] = None):
        """
        初始化Kafka消费者
        
        Args:
            kafka_config: Kafka配置字典
        """
        self.kafka_config = kafka_config or config.kafka.__dict__
        self.consumer = None
        self.scaler = None
        self.data_buffer = []
        self.is_running = False
        
        # 创建必要的目录
        create_directories([
            config.data.data_dir,
            config.data.processed_data_dir,
            os.path.dirname(config.data.scaler_path)
        ])
        
        # 设置日志
        setup_logging(config.training.log_file, config.training.log_level)
        
        logger.info("初始化Kafka消费者")
    
    def _create_consumer(self) -> Consumer:
        """
        创建Kafka消费者实例
        
        Returns:
            Consumer实例
        """
        consumer_config = {
            'bootstrap.servers': self.kafka_config['bootstrap_servers'],
            'group.id': self.kafka_config['group_id'],
            'auto.offset.reset': self.kafka_config['auto_offset_reset'],
            'enable.auto.commit': self.kafka_config['enable_auto_commit'],
            'auto.commit.interval.ms': self.kafka_config['auto_commit_interval_ms'],
            'session.timeout.ms': self.kafka_config['session_timeout_ms'],
            'max.poll.records': self.kafka_config['max_poll_records']
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([self.kafka_config['topic']])
        
        logger.info(f"创建Kafka消费者，订阅主题: {self.kafka_config['topic']}")
        return consumer
    
    def _parse_message(self, message: bytes) -> Optional[Dict[str, Any]]:
        """
        解析Kafka消息
        
        Args:
            message: 原始消息字节
            
        Returns:
            解析后的特征向量字典，如果解析失败返回None
        """
        try:
            # 解析Protobuf消息
            feature_vector = FeatureVector()
            feature_vector.ParseFromString(message)
            
            # 转换为字典
            data_dict = feature_vector_to_dict(feature_vector)
            
            # 数据验证
            if self._validate_data(data_dict):
                return data_dict
            else:
                logger.warning("数据验证失败，跳过该消息")
                return None
                
        except Exception as e:
            logger.error(f"解析消息失败: {e}")
            return None
    
    def _validate_data(self, data_dict: Dict[str, Any]) -> bool:
        """
        验证数据有效性
        
        Args:
            data_dict: 数据字典
            
        Returns:
            数据是否有效
        """
        try:
            # 检查必需字段
            required_fields = FEATURE_COLUMNS + [TARGET_COLUMN] + METADATA_COLUMNS
            for field in required_fields:
                if field not in data_dict:
                    logger.warning(f"缺少必需字段: {field}")
                    return False
            
            # 检查数值有效性
            for field in FEATURE_COLUMNS + [TARGET_COLUMN]:
                value = data_dict[field]
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    logger.warning(f"字段 {field} 包含无效数值: {value}")
                    return False
            
            # 检查数值范围
            if data_dict['power_consumption'] < 0:
                logger.warning(f"功耗值不能为负: {data_dict['power_consumption']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证异常: {e}")
            return False
    
    def _preprocess_data(self, data_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            data_list: 原始数据列表
            
        Returns:
            预处理后的DataFrame
        """
        if not data_list:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(data_list)
        
        # 按时间排序
        df = df.sort_values('time').reset_index(drop=True)
        
        # 处理缺失值
        df = self._handle_missing_values(df)
        
        # 特征工程
        df = self._feature_engineering(df)
        
        # 异常值处理
        df = self._handle_outliers(df)
        
        logger.info(f"预处理完成，数据形状: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 检查缺失值
        missing_counts = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_counts[missing_counts > 0]}")
        
        # 使用前向填充处理缺失值
        df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(method='ffill')
        
        # 如果仍有缺失值，使用后向填充
        df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(method='bfill')
        
        # 如果仍有缺失值，使用均值填充
        df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(
            df[FEATURE_COLUMNS + [TARGET_COLUMN]].mean()
        )
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 添加时间特征
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 添加滞后特征
        for col in FEATURE_COLUMNS[:5]:  # 只对前5个特征添加滞后特征
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)
        
        # 添加滚动统计特征
        for col in FEATURE_COLUMNS[:3]:  # 只对前3个特征添加滚动特征
            df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std()
        
        # 添加交互特征
        df['cpu_mem_interaction'] = df['cpu_util'] * df['mem_util']
        df['gpu_util_mem_interaction'] = df['gpu_util'] * df['gpu_mem']
        
        # 处理新特征的缺失值
        new_features = [col for col in df.columns if col not in FEATURE_COLUMNS + [TARGET_COLUMN] + METADATA_COLUMNS + ['timestamp']]
        df[new_features] = df[new_features].fillna(0)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 使用IQR方法检测和处理异常值
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 统计异常值数量
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                logger.info(f"特征 {col} 发现 {outliers_count} 个异常值")
            
            # 将异常值限制在边界内
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _fit_scaler(self, df: pd.DataFrame) -> StandardScaler:
        """
        拟合StandardScaler
        
        Args:
            df: 训练数据
            
        Returns:
            拟合好的StandardScaler
        """
        # 获取所有数值特征列
        feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS + ['timestamp']]
        
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])
        
        logger.info(f"拟合StandardScaler，特征数量: {len(feature_cols)}")
        return scaler
    
    def _transform_data(self, df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """
        使用StandardScaler转换数据
        
        Args:
            df: 输入DataFrame
            scaler: 已拟合的StandardScaler
            
        Returns:
            转换后的DataFrame
        """
        # 获取所有数值特征列
        feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS + ['timestamp']]
        
        # 转换特征
        scaled_features = scaler.transform(df[feature_cols])
        
        # 创建新的DataFrame
        result_df = df.copy()
        result_df[feature_cols] = scaled_features
        
        logger.info(f"数据标准化完成，特征数量: {len(feature_cols)}")
        return result_df
    
    def _save_data(self, df: pd.DataFrame, filename: str):
        """
        保存数据到文件
        
        Args:
            df: 要保存的DataFrame
            filename: 文件名
        """
        filepath = os.path.join(config.data.processed_data_dir, filename)
        
        # 保存为Parquet格式
        df.to_parquet(filepath, index=False)
        logger.info(f"数据已保存到: {filepath}")
    
    def start_consuming(self, max_samples: int = None, timeout: int = None):
        """
        开始消费数据
        
        Args:
            max_samples: 最大收集样本数
            timeout: 超时时间（秒）
        """
        max_samples = max_samples or config.data.max_samples
        timeout = timeout or config.data.collection_timeout
        
        self.consumer = self._create_consumer()
        self.is_running = True
        
        start_time = time.time()
        sample_count = 0
        
        logger.info(f"开始消费数据，最大样本数: {max_samples}, 超时时间: {timeout}秒")
        
        try:
            while self.is_running:
                # 检查超时
                if time.time() - start_time > timeout:
                    logger.info("达到超时时间，停止消费")
                    break
                
                # 检查样本数量
                if sample_count >= max_samples:
                    logger.info(f"达到最大样本数 {max_samples}，停止消费")
                    break
                
                # 轮询消息
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("到达分区末尾")
                    else:
                        logger.error(f"Kafka错误: {msg.error()}")
                    continue
                
                # 解析消息
                data_dict = self._parse_message(msg.value())
                if data_dict:
                    self.data_buffer.append(data_dict)
                    sample_count += 1
                    
                    if sample_count % 100 == 0:
                        logger.info(f"已收集 {sample_count} 个样本")
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止消费")
        except Exception as e:
            logger.error(f"消费过程中发生错误: {e}")
        finally:
            self.stop_consuming()
    
    def stop_consuming(self):
        """停止消费"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka消费者已关闭")
    
    def process_and_save(self, save_scaler_flag: bool = True) -> pd.DataFrame:
        """
        处理收集的数据并保存
        
        Args:
            save_scaler_flag: 是否保存scaler
            
        Returns:
            处理后的DataFrame
        """
        if not self.data_buffer:
            logger.warning("没有数据需要处理")
            return pd.DataFrame()
        
        logger.info(f"开始处理 {len(self.data_buffer)} 个样本")
        
        # 预处理数据
        df = self._preprocess_data(self.data_buffer)
        
        if df.empty:
            logger.warning("预处理后数据为空")
            return df
        
        # 拟合scaler（如果是第一次运行）
        if self.scaler is None:
            self.scaler = self._fit_scaler(df)
            
            # 保存scaler
            if save_scaler_flag:
                save_scaler(self.scaler, config.data.scaler_path)
        
        # 转换数据
        df_scaled = self._transform_data(df, self.scaler)
        
        # 保存数据
        timestamp = int(time.time())
        self._save_data(df_scaled, f"processed_data_{timestamp}.parquet")
        
        # 清空缓冲区
        self.data_buffer.clear()
        
        logger.info("数据处理和保存完成")
        return df_scaled
    
    def run_pipeline(self, max_samples: int = None, timeout: int = None):
        """
        运行完整的数据管道
        
        Args:
            max_samples: 最大收集样本数
            timeout: 超时时间（秒）
        """
        try:
            # 开始消费数据
            self.start_consuming(max_samples, timeout)
            
            # 处理并保存数据
            df = self.process_and_save()
            
            if not df.empty:
                logger.info(f"数据管道运行完成，处理了 {len(df)} 个样本")
            else:
                logger.warning("数据管道运行完成，但没有处理到有效数据")
                
        except Exception as e:
            logger.error(f"数据管道运行失败: {e}")
            raise


def main():
    """主函数"""
    # 创建消费者
    consumer = KafkaConsumer()
    
    # 运行数据管道
    consumer.run_pipeline()


if __name__ == "__main__":
    main()
