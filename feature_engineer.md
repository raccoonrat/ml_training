

## 🔄 数据转换流程分析

### 1. **原始数据结构定义**

项目使用 `FeatureVector` 数据类来定义原始性能指标：

```python
@dataclass
class FeatureVector:
    """特征向量数据类"""
    time: int = 0                    # 时间戳
    node_id: str = ""                # 节点ID
    power_consumption: float = 0.0   # 功耗（目标变量）

    # 系统性能指标（20个特征）
    cpu_util: float = 0.0            # CPU使用率
    mem_util: float = 0.0            # 内存使用率
    disk_io: float = 0.0             # 磁盘I/O
    net_io: float = 0.0              # 网络I/O
    gpu_util: float = 0.0            # GPU使用率
    gpu_mem: float = 0.0             # GPU内存使用率
    temp_cpu: float = 0.0            # CPU温度
    temp_gpu: float = 0.0            # GPU温度
    fan_speed: float = 0.0           # 风扇转速
    voltage: float = 0.0             # 电压
    current: float = 0.0             # 电流
    frequency: float = 0.0           # 频率
    cache_miss: float = 0.0          # 缓存未命中
    cache_hit: float = 0.0           # 缓存命中
    context_switch: float = 0.0      # 上下文切换
    page_fault: float = 0.0          # 页面错误
    interrupts: float = 0.0          # 中断数
    load_avg: float = 0.0            # 负载平均值
    process_count: float = 0.0       # 进程数
    thread_count: float = 0.0        # 线程数
```

### 2. **特征列定义**

```python
# 定义特征列名（20个特征）
FEATURE_COLUMNS = [
    'cpu_util', 'mem_util', 'disk_io', 'net_io', 'gpu_util', 'gpu_mem',
    'temp_cpu', 'temp_gpu', 'fan_speed', 'voltage', 'current', 'frequency',
    'cache_miss', 'cache_hit', 'context_switch', 'page_fault', 'interrupts',
    'load_avg', 'process_count', 'thread_count'
]

# 目标列名
TARGET_COLUMN = 'power_consumption'

# 元数据列名
METADATA_COLUMNS = ['time', 'node_id']
```

### 3. **数据预处理流程**

#### 3.1 **缺失值处理**

```python
def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
    # 1. 前向填充
    df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(method='ffill')

    # 2. 后向填充
    df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(method='bfill')

    # 3. 均值填充
    df[FEATURE_COLUMNS + [TARGET_COLUMN]] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].fillna(
        df[FEATURE_COLUMNS + [TARGET_COLUMN]].mean()
    )
```

#### 3.2 **特征工程**

```python
def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
    # 1. 时间特征
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 2. 滞后特征
    for col in FEATURE_COLUMNS[:5]:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)

    # 3. 滚动统计特征
    for col in FEATURE_COLUMNS[:3]:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std()

    # 4. 交互特征
    df['cpu_mem_interaction'] = df['cpu_util'] * df['mem_util']
    df['gpu_util_mem_interaction'] = df['gpu_util'] * df['gpu_mem']
```

#### 3.3 **异常值处理**

```python
def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    # 使用IQR方法检测和处理异常值
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 将异常值限制在边界内
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

#### 3.4 **数据标准化**

```python
def _fit_scaler(self, df: pd.DataFrame) -> StandardScaler:
    # 获取所有数值特征列
    feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS + ['timestamp']]

    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler

def _transform_data(self, df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    # 获取所有数值特征列
    feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS + ['timestamp']]

    # 转换特征
    scaled_features = scaler.transform(df[feature_cols])

    # 创建新的DataFrame
    result_df = df.copy()
    result_df[feature_cols] = scaled_features
    return result_df
```

### 4. **EMSHAP输入准备**

在训练脚本中，数据被转换为EMSHAP的输入格式：

```python
def load_and_preprocess_data(data_path: str) -> tuple:
    """Load and preprocess data"""
    logger.info(f"Loading data: {data_path}")

    # 读取数据
    df = pd.read_parquet(data_path)

    # 提取特征和标签
    features = df[FEATURE_COLUMNS].values  # 20维特征向量
    labels = df[TARGET_COLUMN].values.reshape(-1, 1)  # 功耗标签

    logger.info(f"Data shape: features {features.shape}, labels {labels.shape}")
    return features, labels
```

### 5. **数据生成过程**

#### 5.1 **模拟数据生成**

```python
def generate_data_point(self, timestamp: datetime) -> FeatureVector:
    # 1. 确定工作负载模式
    mode = self._get_workload_mode(timestamp)

    # 2. 生成基础指标
    metrics = self._generate_workload_metrics(mode)

    # 3. 计算衍生指标
    metrics.update(self._calculate_derived_metrics(metrics))

    # 4. 计算功耗
    power_consumption = self._calculate_power_consumption(metrics)

    # 5. 创建特征向量
    feature_vector = FeatureVector()
    feature_vector.time = int(timestamp.timestamp())
    feature_vector.node_id = self.node_id
    feature_vector.power_consumption = power_consumption

    # 6. 填充所有特征
    for key, value in metrics.items():
        setattr(feature_vector, key, float(value))

    return feature_vector
```

### 6. **最终EMSHAP输入格式**

经过预处理后，EMSHAP接收的输入格式为：

```python
# 输入特征矩阵: (batch_size, 20)
features = df[FEATURE_COLUMNS].values

# 目标标签: (batch_size, 1)  
labels = df[TARGET_COLUMN].values.reshape(-1, 1)

# EMSHAP模型输入
model_input = torch.FloatTensor(features)  # 20维特征向量
```

## 7. 特征向量转换总结

### **转换步骤**：

1. **原始数据收集** → 22个字段的性能指标
2. **数据清洗** → 处理缺失值和异常值
3. **特征工程** → 添加时间特征、滞后特征、交互特征
4. **数据标准化** → 使用StandardScaler标准化
5. **特征选择** → 提取20个核心特征
6. **格式转换** → 转换为numpy数组/torch张量

### **最终特征向量**：

- **维度**: 20维
- **特征类型**: 系统性能指标（CPU、内存、GPU、温度、风扇等）
- **数据格式**: 标准化后的浮点数
- **时间序列**: 支持时间序列建模

这种设计确保了EMSHAP模型能够有效学习系统性能指标与功耗之间的复杂关系，并通过Shapley值分析每个特征对功耗预测的贡献度。
