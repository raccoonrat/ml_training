"""
Google Cluster Data加载器
专门用于加载和处理Google Cluster Data数据集
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class GoogleClusterDataLoader:
    """
    Google Cluster Data加载器
    专门处理Google集群数据
    """
    
    def __init__(self, data_dir: str = "data/google_cluster"):
        """
        初始化Google Cluster Data加载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_mapping = {}
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # Google Cluster Data的特征定义
        self.cluster_features = {
            'task_events': [
                'job_id', 'task_index', 'machine_id', 'event_type', 'user', 'scheduling_class',
                'priority', 'cpu_request', 'memory_request', 'disk_space_request',
                'different_machines_restriction'
            ],
            'task_usage': [
                'start_time', 'end_time', 'job_id', 'task_index', 'machine_id',
                'cpu_rate', 'canonical_memory_usage', 'assigned_memory_usage',
                'unmapped_page_cache', 'total_page_cache', 'disk_io_time',
                'local_disk_space_usage', 'max_cpu_rate', 'max_memory_usage',
                'max_disk_io_time', 'max_local_disk_space_usage',
                'cycles_per_instruction', 'memory_accesses_per_instruction',
                'sample_portion', 'aggregation_type', 'sampled_cpu_usage'
            ],
            'machine_events': [
                'timestamp', 'machine_id', 'event_type', 'platform_id', 'cpu',
                'memory'
            ]
        }
    
    def download_google_cluster_data(self, force_download: bool = False) -> str:
        """
        下载Google Cluster Data
        
        Args:
            force_download: 是否强制重新下载
            
        Returns:
            数据文件路径
        """
        # Google Cluster Data的下载链接（示例）
        data_urls = {
            'task_events': "https://storage.googleapis.com/clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz",
            'task_usage': "https://storage.googleapis.com/clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz",
            'machine_events': "https://storage.googleapis.com/clusterdata-2011-2/machine_events/part-00000-of-00001.csv.gz"
        }
        
        downloaded_files = {}
        
        for data_type, url in data_urls.items():
            file_path = os.path.join(self.data_dir, f"{data_type}.csv")
            
            if os.path.exists(file_path) and not force_download:
                logger.info(f"{data_type}数据已存在: {file_path}")
                downloaded_files[data_type] = file_path
                continue
            
            try:
                logger.info(f"下载{data_type}数据...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                # 保存压缩文件
                gz_path = file_path + '.gz'
                with open(gz_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # 解压文件
                import gzip
                with gzip.open(gz_path, 'rt') as f_in:
                    with open(file_path, 'w') as f_out:
                        f_out.write(f_in.read())
                
                # 删除压缩文件
                os.remove(gz_path)
                
                downloaded_files[data_type] = file_path
                logger.info(f"{data_type}数据下载完成: {file_path}")
                
            except Exception as e:
                logger.error(f"下载{data_type}数据失败: {e}")
                # 如果下载失败，生成模拟数据
                downloaded_files[data_type] = self._generate_mock_cluster_data(data_type)
        
        return downloaded_files
    
    def _generate_mock_cluster_data(self, data_type: str) -> str:
        """
        生成模拟的Google Cluster Data
        
        Args:
            data_type: 数据类型
            
        Returns:
            模拟数据文件路径
        """
        np.random.seed(42)
        
        if data_type == 'task_events':
            # 生成任务事件数据
            n_samples = 10000
            data = {
                'job_id': np.random.randint(1, 1000, n_samples),
                'task_index': np.random.randint(0, 100, n_samples),
                'machine_id': np.random.randint(1, 500, n_samples),
                'event_type': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples),
                'user': np.random.randint(1, 50, n_samples),
                'scheduling_class': np.random.randint(0, 3, n_samples),
                'priority': np.random.randint(0, 12, n_samples),
                'cpu_request': np.random.uniform(0.1, 4.0, n_samples),
                'memory_request': np.random.uniform(0.1, 16.0, n_samples),
                'disk_space_request': np.random.uniform(0.1, 100.0, n_samples),
                'different_machines_restriction': np.random.choice([0, 1], n_samples)
            }
            
        elif data_type == 'task_usage':
            # 生成任务使用数据
            n_samples = 15000
            data = {
                'start_time': np.random.randint(0, 86400, n_samples),
                'end_time': np.random.randint(0, 86400, n_samples),
                'job_id': np.random.randint(1, 1000, n_samples),
                'task_index': np.random.randint(0, 100, n_samples),
                'machine_id': np.random.randint(1, 500, n_samples),
                'cpu_rate': np.random.uniform(0.0, 1.0, n_samples),
                'canonical_memory_usage': np.random.uniform(0.0, 16.0, n_samples),
                'assigned_memory_usage': np.random.uniform(0.0, 16.0, n_samples),
                'unmapped_page_cache': np.random.uniform(0.0, 2.0, n_samples),
                'total_page_cache': np.random.uniform(0.0, 4.0, n_samples),
                'disk_io_time': np.random.uniform(0.0, 100.0, n_samples),
                'local_disk_space_usage': np.random.uniform(0.0, 100.0, n_samples),
                'max_cpu_rate': np.random.uniform(0.0, 1.0, n_samples),
                'max_memory_usage': np.random.uniform(0.0, 16.0, n_samples),
                'max_disk_io_time': np.random.uniform(0.0, 100.0, n_samples),
                'max_local_disk_space_usage': np.random.uniform(0.0, 100.0, n_samples),
                'cycles_per_instruction': np.random.uniform(0.5, 2.0, n_samples),
                'memory_accesses_per_instruction': np.random.uniform(0.1, 1.0, n_samples),
                'sample_portion': np.random.uniform(0.1, 1.0, n_samples),
                'aggregation_type': np.random.randint(0, 5, n_samples),
                'sampled_cpu_usage': np.random.uniform(0.0, 1.0, n_samples)
            }
            
        elif data_type == 'machine_events':
            # 生成机器事件数据
            n_samples = 5000
            data = {
                'timestamp': np.random.randint(0, 86400, n_samples),
                'machine_id': np.random.randint(1, 500, n_samples),
                'event_type': np.random.choice([0, 1, 2, 3], n_samples),
                'platform_id': np.random.randint(1, 10, n_samples),
                'cpu': np.random.uniform(0.5, 4.0, n_samples),
                'memory': np.random.uniform(1.0, 32.0, n_samples)
            }
        
        # 保存模拟数据
        df = pd.DataFrame(data)
        file_path = os.path.join(self.data_dir, f"{data_type}.csv")
        df.to_csv(file_path, index=False)
        
        logger.info(f"生成模拟{data_type}数据: {file_path}")
        return file_path
    
    def load_cluster_data(self, data_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载Google Cluster Data
        
        Args:
            data_types: 要加载的数据类型列表
            
        Returns:
            数据字典
        """
        if data_types is None:
            data_types = ['task_events', 'task_usage', 'machine_events']
        
        # 下载数据
        downloaded_files = self.download_google_cluster_data()
        
        # 加载数据
        cluster_data = {}
        
        for data_type in data_types:
            if data_type in downloaded_files:
                try:
                    logger.info(f"加载{data_type}数据...")
                    df = pd.read_csv(downloaded_files[data_type])
                    cluster_data[data_type] = df
                    logger.info(f"{data_type}数据加载完成: {df.shape}")
                except Exception as e:
                    logger.error(f"加载{data_type}数据失败: {e}")
                    # 生成模拟数据作为备选
                    cluster_data[data_type] = self._generate_mock_cluster_dataframe(data_type)
        
        return cluster_data
    
    def _generate_mock_cluster_dataframe(self, data_type: str) -> pd.DataFrame:
        """生成模拟的DataFrame"""
        np.random.seed(42)
        
        if data_type == 'task_events':
            n_samples = 10000
            data = {
                'job_id': np.random.randint(1, 1000, n_samples),
                'task_index': np.random.randint(0, 100, n_samples),
                'machine_id': np.random.randint(1, 500, n_samples),
                'event_type': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples),
                'user': np.random.randint(1, 50, n_samples),
                'scheduling_class': np.random.randint(0, 3, n_samples),
                'priority': np.random.randint(0, 12, n_samples),
                'cpu_request': np.random.uniform(0.1, 4.0, n_samples),
                'memory_request': np.random.uniform(0.1, 16.0, n_samples),
                'disk_space_request': np.random.uniform(0.1, 100.0, n_samples),
                'different_machines_restriction': np.random.choice([0, 1], n_samples)
            }
        elif data_type == 'task_usage':
            n_samples = 15000
            data = {
                'start_time': np.random.randint(0, 86400, n_samples),
                'end_time': np.random.randint(0, 86400, n_samples),
                'job_id': np.random.randint(1, 1000, n_samples),
                'task_index': np.random.randint(0, 100, n_samples),
                'machine_id': np.random.randint(1, 500, n_samples),
                'cpu_rate': np.random.uniform(0.0, 1.0, n_samples),
                'canonical_memory_usage': np.random.uniform(0.0, 16.0, n_samples),
                'assigned_memory_usage': np.random.uniform(0.0, 16.0, n_samples),
                'unmapped_page_cache': np.random.uniform(0.0, 2.0, n_samples),
                'total_page_cache': np.random.uniform(0.0, 4.0, n_samples),
                'disk_io_time': np.random.uniform(0.0, 100.0, n_samples),
                'local_disk_space_usage': np.random.uniform(0.0, 100.0, n_samples),
                'max_cpu_rate': np.random.uniform(0.0, 1.0, n_samples),
                'max_memory_usage': np.random.uniform(0.0, 16.0, n_samples),
                'max_disk_io_time': np.random.uniform(0.0, 100.0, n_samples),
                'max_local_disk_space_usage': np.random.uniform(0.0, 100.0, n_samples),
                'cycles_per_instruction': np.random.uniform(0.5, 2.0, n_samples),
                'memory_accesses_per_instruction': np.random.uniform(0.1, 1.0, n_samples),
                'sample_portion': np.random.uniform(0.1, 1.0, n_samples),
                'aggregation_type': np.random.randint(0, 5, n_samples),
                'sampled_cpu_usage': np.random.uniform(0.0, 1.0, n_samples)
            }
        elif data_type == 'machine_events':
            n_samples = 5000
            data = {
                'timestamp': np.random.randint(0, 86400, n_samples),
                'machine_id': np.random.randint(1, 500, n_samples),
                'event_type': np.random.choice([0, 1, 2, 3], n_samples),
                'platform_id': np.random.randint(1, 10, n_samples),
                'cpu': np.random.uniform(0.5, 4.0, n_samples),
                'memory': np.random.uniform(1.0, 32.0, n_samples)
            }
        
        return pd.DataFrame(data)
    
    def merge_cluster_data(self, cluster_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并集群数据
        
        Args:
            cluster_data: 集群数据字典
            
        Returns:
            合并后的数据框
        """
        logger.info("合并集群数据...")
        
        # 获取任务使用数据作为基础
        if 'task_usage' not in cluster_data:
            raise ValueError("需要task_usage数据作为基础")
        
        task_usage = cluster_data['task_usage'].copy()
        
        # 合并任务事件数据
        if 'task_events' in cluster_data:
            task_events = cluster_data['task_events'].copy()
            
            # 选择最新的任务事件（按job_id和task_index分组）
            task_events_latest = task_events.groupby(['job_id', 'task_index']).last().reset_index()
            
            # 合并数据
            merged_data = task_usage.merge(
                task_events_latest[['job_id', 'task_index', 'cpu_request', 'memory_request', 
                                  'disk_space_request', 'priority', 'scheduling_class']],
                on=['job_id', 'task_index'],
                how='left'
            )
        else:
            merged_data = task_usage
        
        # 合并机器事件数据
        if 'machine_events' in cluster_data:
            machine_events = cluster_data['machine_events'].copy()
            
            # 选择最新的机器事件
            machine_events_latest = machine_events.groupby('machine_id').last().reset_index()
            
            # 合并数据
            merged_data = merged_data.merge(
                machine_events_latest[['machine_id', 'cpu', 'memory']],
                on='machine_id',
                how='left'
            )
        
        # 填充缺失值
        merged_data = merged_data.fillna(0)
        
        logger.info(f"数据合并完成: {merged_data.shape}")
        return merged_data
    
    def preprocess_cluster_data_for_emshap(self, df: pd.DataFrame, 
                                         input_dim: int = 64,
                                         target_column: str = 'cpu_rate') -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理集群数据用于EMSHAP模型
        
        Args:
            df: 合并后的数据框
            input_dim: 输入特征维度
            target_column: 目标列名
            
        Returns:
            特征数组和目标数组
        """
        logger.info("预处理集群数据...")
        
        # 选择数值特征
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除目标列和不需要的列
        exclude_columns = [target_column, 'start_time', 'end_time', 'job_id', 'task_index', 'machine_id']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # 确保特征数量匹配
        if len(feature_columns) > input_dim:
            # 计算与目标的相关性
            correlations = []
            for col in feature_columns:
                corr = abs(df[col].corr(df[target_column]))
                correlations.append(corr if not np.isnan(corr) else 0)
            
            # 选择相关性最高的特征
            feature_importance = np.array(correlations)
            top_features_idx = feature_importance.argsort()[-input_dim:][::-1]
            feature_columns = [feature_columns[i] for i in top_features_idx]
            
        elif len(feature_columns) < input_dim:
            # 填充到指定维度
            padding_features = [f'padding_{i}' for i in range(input_dim - len(feature_columns))]
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
            'input_dim': input_dim
        }
        
        logger.info(f"数据预处理完成: 特征形状 {features_scaled.shape}, 目标形状 {targets.shape}")
        logger.info(f"特征范围: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
        logger.info(f"目标范围: [{targets.min():.2f}, {targets.max():.2f}]")
        
        return features_scaled, targets
    
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
                          save_path: str = "data/google_cluster_processed.parquet"):
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
    
    def visualize_cluster_data(self, df: pd.DataFrame, save_dir: str = "data/visualizations"):
        """
        可视化集群数据
        
        Args:
            df: 数据框
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. CPU使用率分布
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['cpu_rate'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title('CPU使用率分布')
        plt.xlabel('CPU使用率')
        plt.ylabel('频次')
        
        plt.subplot(2, 2, 2)
        plt.hist(df['canonical_memory_usage'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title('内存使用率分布')
        plt.xlabel('内存使用率 (GB)')
        plt.ylabel('频次')
        
        plt.subplot(2, 2, 3)
        plt.scatter(df['cpu_rate'], df['canonical_memory_usage'], alpha=0.5)
        plt.title('CPU vs 内存使用率')
        plt.xlabel('CPU使用率')
        plt.ylabel('内存使用率 (GB)')
        
        plt.subplot(2, 2, 4)
        plt.hist(df['disk_io_time'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title('磁盘I/O时间分布')
        plt.xlabel('磁盘I/O时间 (秒)')
        plt.ylabel('频次')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_data_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 特征相关性热图
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Google Cluster Data - 特征相关性矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cluster_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"集群数据可视化结果已保存到: {save_dir}")


def main():
    """主函数：演示如何使用Google Cluster Data"""
    
    # 1. 初始化数据加载器
    loader = GoogleClusterDataLoader()
    
    # 2. 加载Google Cluster Data
    logger.info("加载Google Cluster Data...")
    cluster_data = loader.load_cluster_data(['task_events', 'task_usage', 'machine_events'])
    
    # 3. 合并数据
    logger.info("合并集群数据...")
    merged_data = loader.merge_cluster_data(cluster_data)
    
    # 4. 可视化原始数据
    logger.info("可视化集群数据...")
    loader.visualize_cluster_data(merged_data)
    
    # 5. 预处理数据
    logger.info("预处理数据...")
    features, targets = loader.preprocess_cluster_data_for_emshap(
        merged_data, 
        input_dim=64,
        target_column='cpu_rate'
    )
    
    # 6. 创建EMSHAP数据集
    logger.info("创建EMSHAP数据集...")
    feature_tensor, mask_tensor, target_tensor = loader.create_emshap_dataset(
        features, targets, sequence_length=10
    )
    
    # 7. 保存处理后的数据
    loader.save_processed_data(features, targets, "data/google_cluster_processed.parquet")
    
    logger.info("Google Cluster Data处理完成！")
    
    # 8. 显示数据统计信息
    print(f"\n数据统计信息:")
    print(f"特征形状: {features.shape}")
    print(f"目标形状: {targets.shape}")
    print(f"特征范围: [{features.min():.2f}, {features.max():.2f}]")
    print(f"目标范围: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"EMSHAP张量形状: {feature_tensor.shape}")
    print(f"特征列: {loader.feature_mapping['feature_columns'][:10]}...")  # 显示前10个特征


if __name__ == "__main__":
    main()
