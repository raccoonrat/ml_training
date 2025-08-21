"""
项目配置文件
包含Kafka连接、模型训练、数据预处理等配置参数
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class KafkaConfig:
    """Kafka配置"""
    bootstrap_servers: str = "localhost:9092"
    topic: str = "raw_metrics"
    group_id: str = "ml_training_consumer"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 1000
    session_timeout_ms: int = 30000
    max_poll_records: int = 100


@dataclass
class DataConfig:
    """数据处理配置"""
    # 数据存储路径
    data_dir: str = "data"
    processed_data_dir: str = "data/processed"
    scaler_path: str = "models/scaler.pkl"
    
    # 数据预处理参数
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # 特征工程参数
    feature_columns: list = None  # 将在运行时根据protobuf定义确定
    target_column: str = "power_consumption"
    
    # 数据收集参数
    max_samples: int = 10000  # 最大收集样本数
    collection_timeout: int = 3600  # 收集超时时间（秒）


@dataclass
class ModelConfig:
    """模型配置"""
    # EMSHAP模型参数
    emshap_input_dim: int = 64  # 输入特征维度
    emshap_hidden_dim: int = 128
    emshap_gru_hidden_dim: int = 64
    emshap_context_dim: int = 32
    
    # 功耗预测模型参数
    power_predictor_input_dim: int = 64
    power_predictor_hidden_dims: list = None
    
    # 训练参数
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # 动态掩码参数
    min_mask_rate: float = 0.2
    max_mask_rate: float = 0.8
    
    # 蒙特卡洛采样参数
    num_monte_carlo_samples: int = 1000
    
    # 模型保存路径
    model_dir: str = "models"
    emshap_model_path: str = "models/emshap_model.onnx"
    power_predictor_path: str = "models/power_predictor.onnx"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 设备配置
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs/training.log"
    
    # 检查点配置
    checkpoint_dir: str = "checkpoints"
    save_checkpoint_every: int = 10
    
    # 评估配置
    evaluation_metrics: list = None
    
    # 模型版本控制
    model_version: str = "v1.0.0"


@dataclass
class Config:
    """主配置类"""
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """初始化默认值"""
        if self.model.power_predictor_hidden_dims is None:
            self.model.power_predictor_hidden_dims = [128, 64, 32]
        
        if self.training.evaluation_metrics is None:
            self.training.evaluation_metrics = ["mse", "mae", "r2"]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "kafka": self.kafka.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        return cls(
            kafka=KafkaConfig(**config_dict.get("kafka", {})),
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {}))
        )


# 全局配置实例
config = Config()
