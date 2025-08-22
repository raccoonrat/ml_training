"""
Google数据集集成模块
为EMSHAP增强模型提供Google数据集支持
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from google.cloud import bigquery
from google.oauth2 import service_account
import tensorflow_datasets as tfds
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from loguru import logger


class GoogleDatasetLoader:
    """
    Google数据集加载器
    支持多种Google数据源的统一接口
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        初始化数据集加载器
        
        Args:
            credentials_path: Google Cloud服务账号密钥文件路径
        """
        self.credentials_path = credentials_path
        self.client = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        if credentials_path and os.path.exists(credentials_path):
            self._init_bigquery_client()
    
    def _init_bigquery_client(self):
        """初始化BigQuery客户端"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self.client = bigquery.Client(credentials=credentials)
            logger.info("BigQuery客户端初始化成功")
        except Exception as e:
            logger.error(f"BigQuery客户端初始化失败: {e}")
    
    def load_google_analytics_data(self, limit: int = 10000) -> pd.DataFrame:
        """
        加载Google Analytics样本数据
        
        Args:
            limit: 数据条数限制
            
        Returns:
            处理后的数据框
        """
        if not self.client:
            logger.warning("BigQuery客户端未初始化，使用模拟数据")
            return self._generate_mock_analytics_data(limit)
        
        query = f"""
        SELECT 
            visitId,
            visitNumber,
            visitStartTime,
            date,
            totals.visits,
            totals.hits,
            totals.pageviews,
            totals.timeOnSite,
            totals.bounces,
            totals.transactions,
            totals.transactionRevenue,
            device.deviceCategory,
            device.operatingSystem,
            device.browser,
            geoNetwork.country,
            geoNetwork.region,
            trafficSource.source,
            trafficSource.medium,
            trafficSource.campaign
        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        LIMIT {limit}
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"成功加载Google Analytics数据: {df.shape}")
            return self._preprocess_analytics_data(df)
        except Exception as e:
            logger.error(f"加载Google Analytics数据失败: {e}")
            return self._generate_mock_analytics_data(limit)
    
    def _preprocess_analytics_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理Google Analytics数据
        
        Args:
            df: 原始数据框
            
        Returns:
            预处理后的数据框
        """
        # 处理缺失值
        df = df.fillna(0)
        
        # 编码分类变量
        categorical_columns = ['deviceCategory', 'operatingSystem', 'browser', 
                             'country', 'region', 'source', 'medium', 'campaign']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # 创建数值特征
        df['timeOnSite_seconds'] = df['timeOnSite'].fillna(0)
        df['transactionRevenue_usd'] = df['transactionRevenue'].fillna(0) / 1000000  # 转换为美元
        
        # 选择用于EMSHAP的特征
        feature_columns = [
            'visitNumber', 'visits', 'hits', 'pageviews', 'timeOnSite_seconds',
            'bounces', 'transactions', 'transactionRevenue_usd',
            'deviceCategory_encoded', 'operatingSystem_encoded', 'browser_encoded',
            'country_encoded', 'region_encoded', 'source_encoded', 'medium_encoded'
        ]
        
        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[available_features]
    
    def _generate_mock_analytics_data(self, limit: int) -> pd.DataFrame:
        """
        生成模拟的Google Analytics数据
        
        Args:
            limit: 数据条数
            
        Returns:
            模拟数据框
        """
        np.random.seed(42)
        
        data = {
            'visitNumber': np.random.randint(1, 10, limit),
            'visits': np.random.randint(1, 5, limit),
            'hits': np.random.randint(1, 50, limit),
            'pageviews': np.random.randint(1, 20, limit),
            'timeOnSite_seconds': np.random.randint(0, 3600, limit),
            'bounces': np.random.randint(0, 2, limit),
            'transactions': np.random.randint(0, 3, limit),
            'transactionRevenue_usd': np.random.exponential(50, limit),
            'deviceCategory_encoded': np.random.randint(0, 3, limit),
            'operatingSystem_encoded': np.random.randint(0, 5, limit),
            'browser_encoded': np.random.randint(0, 8, limit),
            'country_encoded': np.random.randint(0, 10, limit),
            'region_encoded': np.random.randint(0, 15, limit),
            'source_encoded': np.random.randint(0, 6, limit),
            'medium_encoded': np.random.randint(0, 4, limit)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"生成模拟Google Analytics数据: {df.shape}")
        return df
    
    def load_google_cloud_billing_data(self, limit: int = 10000) -> pd.DataFrame:
        """
        加载Google Cloud计费数据（需要自己的项目）
        
        Args:
            limit: 数据条数限制
            
        Returns:
            处理后的数据框
        """
        if not self.client:
            logger.warning("BigQuery客户端未初始化，使用模拟数据")
            return self._generate_mock_billing_data(limit)
        
        # 注意：这需要你自己的Google Cloud项目
        query = f"""
        SELECT 
            billing_account_id,
            service,
            sku,
            usage_start_time,
            usage_end_time,
            usage_amount,
            cost,
            currency
        FROM `your-project.your-dataset.billing_export`
        LIMIT {limit}
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"成功加载Google Cloud计费数据: {df.shape}")
            return self._preprocess_billing_data(df)
        except Exception as e:
            logger.error(f"加载Google Cloud计费数据失败: {e}")
            return self._generate_mock_billing_data(limit)
    
    def _generate_mock_billing_data(self, limit: int) -> pd.DataFrame:
        """生成模拟的Google Cloud计费数据"""
        np.random.seed(42)
        
        services = ['Compute Engine', 'Cloud Storage', 'BigQuery', 'Cloud Functions']
        skus = ['CPU', 'RAM', 'Storage', 'Network', 'API Calls']
        
        data = {
            'service_encoded': np.random.randint(0, len(services), limit),
            'sku_encoded': np.random.randint(0, len(skus), limit),
            'usage_amount': np.random.exponential(100, limit),
            'cost_usd': np.random.exponential(10, limit),
            'usage_hours': np.random.randint(1, 24, limit),
            'region_encoded': np.random.randint(0, 5, limit),
            'project_encoded': np.random.randint(0, 3, limit)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"生成模拟Google Cloud计费数据: {df.shape}")
        return df
    
    def _preprocess_billing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理计费数据"""
        # 实现计费数据预处理逻辑
        return df


class EMSHAPGoogleDataProcessor:
    """
    EMSHAP模型的Google数据处理器
    将Google数据集转换为EMSHAP模型所需的格式
    """
    
    def __init__(self, input_dim: int = 64, context_dim: int = 32):
        """
        初始化数据处理器
        
        Args:
            input_dim: 输入特征维度
            context_dim: 上下文向量维度
        """
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.scaler = StandardScaler()
        self.feature_mapping = {}
        
    def process_google_data_for_emshap(self, df: pd.DataFrame, 
                                     target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将Google数据转换为EMSHAP模型格式
        
        Args:
            df: Google数据框
            target_column: 目标列名（如果为None，将使用最后一个数值列）
            
        Returns:
            特征数组和目标数组
        """
        # 选择数值列作为特征
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column is None:
            # 使用最后一个数值列作为目标
            target_column = numeric_columns[-1]
            feature_columns = numeric_columns[:-1]
        else:
            feature_columns = [col for col in numeric_columns if col != target_column]
        
        # 确保特征数量匹配
        if len(feature_columns) > self.input_dim:
            # 选择最重要的特征
            feature_importance = self._calculate_feature_importance(df[feature_columns], df[target_column])
            top_features = feature_importance.argsort()[-self.input_dim:][::-1]
            feature_columns = [feature_columns[i] for i in top_features]
        elif len(feature_columns) < self.input_dim:
            # 填充到指定维度
            padding_features = [f'padding_{i}' for i in range(self.input_dim - len(feature_columns))]
            for col in padding_features:
                df[col] = 0
            feature_columns.extend(padding_features)
        
        # 提取特征和目标
        features = df[feature_columns].values
        targets = df[target_column].values.reshape(-1, 1)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 保存特征映射
        self.feature_mapping = {
            'feature_columns': feature_columns,
            'target_column': target_column,
            'input_dim': self.input_dim,
            'context_dim': self.context_dim
        }
        
        logger.info(f"数据处理完成: 特征形状 {features_scaled.shape}, 目标形状 {targets.shape}")
        
        return features_scaled, targets
    
    def _calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> np.ndarray:
        """
        计算特征重要性（使用相关性）
        
        Args:
            features: 特征数据框
            target: 目标序列
            
        Returns:
            特征重要性数组
        """
        correlations = []
        for col in features.columns:
            corr = abs(features[col].corr(target))
            correlations.append(corr if not np.isnan(corr) else 0)
        
        return np.array(correlations)
    
    def create_emshap_dataset(self, features: np.ndarray, targets: np.ndarray,
                             sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建EMSHAP模型的数据集
        
        Args:
            features: 特征数组
            targets: 目标数组
            sequence_length: 序列长度
            
        Returns:
            特征张量、掩码张量和目标张量
        """
        num_samples = len(features) - sequence_length + 1
        
        # 创建序列数据
        feature_sequences = []
        target_sequences = []
        
        for i in range(num_samples):
            feature_seq = features[i:i+sequence_length]
            target_seq = targets[i:i+sequence_length]
            feature_sequences.append(feature_seq)
            target_sequences.append(target_seq)
        
        # 转换为张量
        feature_tensor = torch.FloatTensor(np.array(feature_sequences))
        target_tensor = torch.FloatTensor(np.array(target_sequences))
        
        # 创建掩码（这里使用全1掩码，表示所有特征都可用）
        mask_tensor = torch.ones_like(feature_tensor)
        
        logger.info(f"创建EMSHAP数据集: 特征形状 {feature_tensor.shape}, 目标形状 {target_tensor.shape}")
        
        return feature_tensor, mask_tensor, target_tensor
    
    def save_processed_data(self, features: np.ndarray, targets: np.ndarray, 
                          save_path: str = "data/google_processed_data.parquet"):
        """
        保存处理后的数据
        
        Args:
            features: 特征数组
            targets: 目标数组
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建数据框
        feature_df = pd.DataFrame(features, columns=self.feature_mapping['feature_columns'])
        target_df = pd.DataFrame(targets, columns=[self.feature_mapping['target_column']])
        
        # 合并并保存
        combined_df = pd.concat([feature_df, target_df], axis=1)
        combined_df.to_parquet(save_path, index=False)
        
        # 保存特征映射
        mapping_path = save_path.replace('.parquet', '_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.feature_mapping, f, indent=2)
        
        logger.info(f"数据已保存到: {save_path}")
        logger.info(f"特征映射已保存到: {mapping_path}")


def main():
    """主函数：演示如何使用Google数据集"""
    
    # 1. 初始化数据加载器
    loader = GoogleDatasetLoader()
    
    # 2. 加载Google Analytics数据
    logger.info("加载Google Analytics数据...")
    analytics_df = loader.load_google_analytics_data(limit=5000)
    
    # 3. 初始化EMSHAP数据处理器
    processor = EMSHAPGoogleDataProcessor(input_dim=64, context_dim=32)
    
    # 4. 处理数据
    logger.info("处理数据...")
    features, targets = processor.process_google_data_for_emshap(
        analytics_df, 
        target_column='transactionRevenue_usd'
    )
    
    # 5. 创建EMSHAP数据集
    logger.info("创建EMSHAP数据集...")
    feature_tensor, mask_tensor, target_tensor = processor.create_emshap_dataset(
        features, targets, sequence_length=10
    )
    
    # 6. 保存处理后的数据
    processor.save_processed_data(features, targets, "data/google_analytics_processed.parquet")
    
    logger.info("Google数据集处理完成！")
    
    # 7. 显示数据统计信息
    print(f"\n数据统计信息:")
    print(f"特征形状: {features.shape}")
    print(f"目标形状: {targets.shape}")
    print(f"特征范围: [{features.min():.2f}, {features.max():.2f}]")
    print(f"目标范围: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"EMSHAP张量形状: {feature_tensor.shape}")


if __name__ == "__main__":
    main()
