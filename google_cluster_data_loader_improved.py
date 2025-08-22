"""
改进版本的Google Cluster Data加载器
能够正确处理原始数据格式并映射到标准列名
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


class GoogleClusterDataLoaderImproved:
    """
    改进版本的Google Cluster Data加载器
    能够正确处理原始数据格式
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
        
        # Google Cluster Data的标准列名映射
        self.column_mappings = {
            'task_events': [
                'timestamp', 'missing_info', 'job_id', 'task_index', 'machine_id', 
                'event_type', 'user', 'scheduling_class', 'priority', 'cpu_request', 
                'memory_request', 'disk_space_request', 'different_machines_restriction'
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
        
        # 加载数据
        cluster_data = {}
        
        for data_type in data_types:
            file_path = os.path.join(self.data_dir, f"{data_type}.csv")
            
            if os.path.exists(file_path):
                try:
                    logger.info(f"Loading {data_type} data...")
                    df = pd.read_csv(file_path, header=None)
                    
                    # 应用列名映射
                    if data_type in self.column_mappings:
                        expected_columns = self.column_mappings[data_type]
                        if len(df.columns) >= len(expected_columns):
                            df.columns = expected_columns + [f'extra_{i}' for i in range(len(df.columns) - len(expected_columns))]
                        else:
                            # 如果列数不够，用默认列名
                            df.columns = expected_columns[:len(df.columns)] + [f'extra_{i}' for i in range(len(expected_columns) - len(df.columns))]
                    
                    cluster_data[data_type] = df
                    logger.info(f"{data_type} data loading completed: {df.shape}")
                    logger.info(f"{data_type} column names: {list(df.columns)}")
                except Exception as e:
                    logger.error(f"Failed to load {data_type} data: {e}")
                    # Generate mock data as fallback
                    cluster_data[data_type] = self._generate_mock_cluster_dataframe(data_type)
            else:
                logger.warning(f"{data_type} data file does not exist, generating mock data")
                cluster_data[data_type] = self._generate_mock_cluster_dataframe(data_type)
        
        return cluster_data
    
    def _generate_mock_cluster_dataframe(self, data_type: str) -> pd.DataFrame:
        """生成模拟的DataFrame"""
        np.random.seed(42)
        
        if data_type == 'task_events':
            n_samples = 10000
            data = {
                'timestamp': np.random.randint(0, 86400, n_samples),
                'missing_info': np.random.randint(0, 2, n_samples),
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
        合并集群数据（改进版本）
        
        Args:
            cluster_data: 集群数据字典
            
        Returns:
            合并后的数据框
        """
        logger.info("Merging cluster data...")
        
        # Get task usage data as base
        if 'task_usage' not in cluster_data:
            raise ValueError("task_usage data is required as base")
        
        task_usage = cluster_data['task_usage'].copy()
        logger.info(f"task_usage column names: {list(task_usage.columns)}")
        
        # Merge task events data
        if 'task_events' in cluster_data:
            task_events = cluster_data['task_events'].copy()
            logger.info(f"task_events column names: {list(task_events.columns)}")
            
            # Check if required columns exist
            required_columns = ['job_id', 'task_index']
            if all(col in task_events.columns for col in required_columns):
                # Select latest task events (grouped by job_id and task_index)
                task_events_latest = task_events.groupby(['job_id', 'task_index']).last().reset_index()
                
                # Select columns to merge
                merge_columns = ['job_id', 'task_index']
                additional_columns = ['cpu_request', 'memory_request', 'disk_space_request', 'priority', 'scheduling_class']
                
                # Only merge existing columns
                available_columns = [col for col in additional_columns if col in task_events_latest.columns]
                merge_columns.extend(available_columns)
                
                logger.info(f"Merging task_events columns: {merge_columns}")
                
                # Merge data
                merged_data = task_usage.merge(
                    task_events_latest[merge_columns],
                    on=['job_id', 'task_index'],
                    how='left'
                )
            else:
                logger.warning("task_events missing required columns, skipping merge")
                merged_data = task_usage
        else:
            merged_data = task_usage
        
        # Merge machine events data
        if 'machine_events' in cluster_data:
            machine_events = cluster_data['machine_events'].copy()
            logger.info(f"machine_events column names: {list(machine_events.columns)}")
            
            # Check if machine_id column exists
            if 'machine_id' in machine_events.columns:
                # Select latest machine events
                machine_events_latest = machine_events.groupby('machine_id').last().reset_index()
                
                # Select columns to merge
                merge_columns = ['machine_id']
                additional_columns = ['cpu', 'memory']
                
                # Only merge existing columns
                available_columns = [col for col in additional_columns if col in machine_events_latest.columns]
                merge_columns.extend(available_columns)
                
                logger.info(f"Merging machine_events columns: {merge_columns}")
                
                # Merge data
                merged_data = merged_data.merge(
                    machine_events_latest[merge_columns],
                    on='machine_id',
                    how='left'
                )
            else:
                logger.warning("machine_events missing machine_id column, skipping merge")
        
        # Fill missing values
        merged_data = merged_data.fillna(0)
        
        logger.info(f"Data merging completed: {merged_data.shape}")
        logger.info(f"Final column names: {list(merged_data.columns)}")
        
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
        logger.info("Preprocessing cluster data...")
        logger.info(f"Available column names: {list(df.columns)}")
        
        # Check if target column exists
        if target_column not in df.columns:
            logger.warning(f"Target column {target_column} does not exist, trying to find alternative")
            # Try to find alternative target column
            possible_targets = ['cpu_rate', 'sampled_cpu_usage', 'max_cpu_rate', 'canonical_memory_usage']
            for col in possible_targets:
                if col in df.columns:
                    target_column = col
                    logger.info(f"Using alternative target column: {target_column}")
                    break
            else:
                # If no suitable target column found, use first numeric column
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    target_column = numeric_columns[0]
                    logger.info(f"Using first numeric column as target: {target_column}")
                else:
                    raise ValueError("No available numeric columns found as target")
        
        # Select numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Numeric columns: {numeric_columns}")
        
        # Remove target column and unwanted columns
        exclude_columns = [target_column, 'start_time', 'end_time', 'job_id', 'task_index', 'machine_id', 'timestamp']
        # Only exclude existing columns
        exclude_columns = [col for col in exclude_columns if col in df.columns]
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        logger.info(f"Feature columns: {feature_columns}")
        
        # Ensure feature count matches
        if len(feature_columns) > input_dim:
            # Calculate correlation with target
            correlations = []
            for col in feature_columns:
                try:
                    corr = abs(df[col].corr(df[target_column]))
                    correlations.append(corr if not np.isnan(corr) else 0)
                except:
                    correlations.append(0)
            
            # Select features with highest correlation
            feature_importance = np.array(correlations)
            top_features_idx = feature_importance.argsort()[-input_dim:][::-1]
            feature_columns = [feature_columns[i] for i in top_features_idx]
            
        elif len(feature_columns) < input_dim:
            # Pad to specified dimension
            padding_features = [f'padding_{i}' for i in range(input_dim - len(feature_columns))]
            for col in padding_features:
                df[col] = 0
            feature_columns.extend(padding_features)
        
        # Extract features and targets
        features = df[feature_columns].values
        targets = df[target_column].values.reshape(-1, 1)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Save feature mapping
        self.feature_mapping = {
            'feature_columns': feature_columns,
            'target_column': target_column,
            'input_dim': input_dim
        }
        
        logger.info(f"Data preprocessing completed: feature shape {features_scaled.shape}, target shape {targets.shape}")
        logger.info(f"Feature range: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
        logger.info(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
        
        return features_scaled, targets
    
    def create_emshap_dataset(self, features: np.ndarray, targets: np.ndarray,
                             sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create EMSHAP model dataset
        
        Args:
            features: Feature array
            targets: Target array
            sequence_length: Sequence length
            
        Returns:
            Feature tensor, mask tensor and target tensor
        """
        num_samples = len(features) - sequence_length + 1
        
        # Create sequence data
        feature_sequences = []
        target_sequences = []
        
        for i in range(num_samples):
            feature_seq = features[i:i+sequence_length]
            target_seq = targets[i:i+sequence_length]
            feature_sequences.append(feature_seq)
            target_sequences.append(target_seq)
        
        # Convert to tensors
        feature_tensor = torch.FloatTensor(np.array(feature_sequences))
        target_tensor = torch.FloatTensor(np.array(target_sequences))
        
        # Create mask (using all 1s mask, indicating all features are available)
        mask_tensor = torch.ones_like(feature_tensor)
        
        logger.info(f"Creating EMSHAP dataset: feature shape {feature_tensor.shape}, target shape {target_tensor.shape}")
        
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
        
        logger.info(f"Data saved to: {save_path}")
        logger.info(f"Feature mapping saved to: {mapping_path}")
    
    def visualize_cluster_data(self, df: pd.DataFrame, save_dir: str = "data/visualizations"):
        """
        Visualize cluster data
        
        Args:
            df: Dataframe
            save_dir: Save directory
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if required columns exist
        required_columns = ['cpu_rate', 'canonical_memory_usage', 'disk_io_time']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if len(available_columns) >= 2:
            # 1. CPU usage distribution
            plt.figure(figsize=(12, 8))
            
            if 'cpu_rate' in df.columns:
                plt.subplot(2, 2, 1)
                plt.hist(df['cpu_rate'].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('CPU Usage Distribution')
                plt.xlabel('CPU Usage')
                plt.ylabel('Frequency')
            
            if 'canonical_memory_usage' in df.columns:
                plt.subplot(2, 2, 2)
                plt.hist(df['canonical_memory_usage'].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('Memory Usage Distribution')
                plt.xlabel('Memory Usage (GB)')
                plt.ylabel('Frequency')
            
            if 'cpu_rate' in df.columns and 'canonical_memory_usage' in df.columns:
                plt.subplot(2, 2, 3)
                plt.scatter(df['cpu_rate'], df['canonical_memory_usage'], alpha=0.5)
                plt.title('CPU vs Memory Usage')
                plt.xlabel('CPU Usage')
                plt.ylabel('Memory Usage (GB)')
            
            if 'disk_io_time' in df.columns:
                plt.subplot(2, 2, 4)
                plt.hist(df['disk_io_time'].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('Disk I/O Time Distribution')
                plt.xlabel('Disk I/O Time (seconds)')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cluster_data_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
                    # 2. Feature correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            plt.figure(figsize=(16, 14))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
            plt.title('Google Cluster Data - Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cluster_correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Cluster data visualization results saved to: {save_dir}")


def main():
    """Main function: Demonstrate how to use the improved Google Cluster Data loader"""
    
    # 1. Initialize data loader
    loader = GoogleClusterDataLoaderImproved()
    
    # 2. Load Google Cluster Data
    logger.info("Loading Google Cluster Data...")
    cluster_data = loader.load_cluster_data(['task_events', 'task_usage', 'machine_events'])
    
    # 3. Merge data
    logger.info("Merging cluster data...")
    merged_data = loader.merge_cluster_data(cluster_data)
    
    # 4. Visualize raw data
    logger.info("Visualizing cluster data...")
    loader.visualize_cluster_data(merged_data)
    
    # 5. Preprocess data
    logger.info("Preprocessing data...")
    features, targets = loader.preprocess_cluster_data_for_emshap(
        merged_data, 
        input_dim=64,
        target_column='cpu_rate'
    )
    
    # 6. Create EMSHAP dataset
    logger.info("Creating EMSHAP dataset...")
    feature_tensor, mask_tensor, target_tensor = loader.create_emshap_dataset(
        features, targets, sequence_length=10
    )
    
    # 7. Save processed data
    loader.save_processed_data(features, targets, "data/google_cluster_processed.parquet")
    
    logger.info("Google Cluster Data processing completed!")
    
    # 8. Display data statistics
    print(f"\nData Statistics:")
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"EMSHAP tensor shape: {feature_tensor.shape}")
    print(f"Feature columns: {loader.feature_mapping['feature_columns'][:10]}...")  # Show first 10 features


if __name__ == "__main__":
    main()
