# EMSHAP 模型训练项目

本项目实现了基于EmSHAP论文的模型训练系统，包含数据消费、预处理、模型训练和评估的完整流程。

## 项目结构

```
ml_training/
├── config.py                    # 项目配置文件
├── utils.py                     # 工具函数
├── requirements.txt             # Python依赖
├── Dockerfile                   # Docker配置
├── .gitignore                   # Git忽略文件
├── README_ML_TRAINING.md        # 项目说明文档
│
├── data_pipeline/               # 数据管道
│   ├── __init__.py
│   ├── consumer.py              # Kafka消费者
│   └── feature_vector_pb2.py    # Protobuf定义
│
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── emshap.py               # EMSHAP模型
│   └── power_predictor.py      # 功耗预测模型
│
├── train_power_model.py         # 功耗预测模型训练脚本
├── train_emshap_model.py        # EMSHAP模型训练脚本
├── evaluate.py                  # 模型评估脚本
│
├── data/                        # 数据目录
│   └── processed/              # 处理后的数据
├── models/                      # 模型输出目录
├── logs/                        # 日志目录
├── checkpoints/                 # 检查点目录
└── evaluation_results/          # 评估结果目录
```

## 核心组件

### 1. 数据管道 (data_pipeline/)

- **consumer.py**: Kafka消费者，负责从Kafka消费原始指标数据
- **feature_vector_pb2.py**: Protobuf消息定义，包含FeatureVector结构

### 2. 模型定义 (models/)

- **emshap.py**: EMSHAP模型实现，包含能量网络和GRU网络
- **power_predictor.py**: 功耗预测模型，使用MLP架构

### 3. 训练脚本

- **train_power_model.py**: 训练功耗预测模型
- **train_emshap_model.py**: 训练EMSHAP模型，实现动态掩码机制

### 4. 评估脚本

- **evaluate.py**: 评估训练好的模型性能

## 安装和配置

### 1. 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

编辑 `config.py` 文件，配置以下参数：

- Kafka连接配置
- 模型训练参数
- 数据路径配置
- 日志配置

## 使用方法

### 1. 数据消费和预处理

```bash
# 运行数据管道
python data_pipeline/consumer.py

# 或者使用Docker
docker build -t ml-training .
docker run ml-training python data_pipeline/consumer.py
```

### 2. 训练功耗预测模型

```bash
# 使用默认配置训练
python train_power_model.py

# 指定参数训练
python train_power_model.py --epochs 100 --batch-size 64 --learning-rate 0.001
```

### 3. 训练EMSHAP模型

```bash
# 使用默认配置训练
python train_emshap_model.py

# 指定参数训练
python train_emshap_model.py --epochs 200 --batch-size 32 --learning-rate 0.0005
```

### 4. 评估模型

```bash
# 评估所有模型
python evaluate.py

# 指定模型路径评估
python evaluate.py --emshap-path models/emshap_model.onnx --power-predictor-path models/power_predictor.onnx
```

## 模型架构

### EMSHAP模型

EMSHAP模型包含两个主要组件：

1. **能量网络 (Energy Network)**: 带Skip Connection的MLP，计算能量函数 gθ(x)
2. **GRU网络 (GRU Network)**: 生成提议分布 q(x) 的参数

### 功耗预测模型

使用多层感知机(MLP)架构，包含：
- 多个隐藏层
- Batch Normalization
- Dropout正则化
- 可选的残差连接

## 训练特性

### 动态掩码机制

EMSHAP模型训练中实现了动态掩码机制：
- 掩码率从 `min_mask_rate` 线性增加到 `max_mask_rate`
- 帮助模型学习不同特征组合下的条件依赖关系

### 早停机制

两个模型都实现了早停机制：
- 监控验证损失
- 当验证损失不再改善时自动停止训练
- 保存最佳模型检查点

### 模型导出

训练完成后自动导出ONNX格式模型：
- 便于在不同环境中部署
- 支持Go服务集成

## 评估指标

### 功耗预测模型

- MSE (均方误差)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)
- MAPE (平均绝对百分比误差)

### EMSHAP模型

- 能量统计 (均值、标准差、最小值、最大值)
- 提议分布统计
- 上下文向量统计

## 日志和监控

- 使用loguru进行日志记录
- 自动创建训练历史图表
- 保存预测结果可视化

## Docker支持

项目包含完整的Docker支持：

```bash
# 构建镜像
docker build -t ml-training .

# 运行容器
docker run -v $(pwd)/data:/app/data ml-training

# 运行特定脚本
docker run ml-training python train_power_model.py
```

## 配置说明

### Kafka配置

```python
kafka_config = {
    'bootstrap_servers': 'localhost:9092',
    'topic': 'raw_metrics',
    'group_id': 'ml_training_consumer'
}
```

### 模型配置

```python
model_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'early_stopping_patience': 10
}
```

### 数据配置

```python
data_config = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}
```

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减小batch_size或使用CPU训练
2. **Kafka连接失败**: 检查Kafka服务状态和配置
3. **数据文件不存在**: 确保先运行数据管道收集数据

### 调试模式

启用详细日志：

```python
# 在config.py中设置
training_config.log_level = "DEBUG"
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。
