
## 实际运行该项目的训练数据集建议

### 1. **系统监控指标数据集**

根据项目的 `FeatureVector` 定义，该项目需要包含以下22个特征的系统监控数据：

**核心性能指标：**

* `cpu_util` - CPU使用率
* `mem_util` - 内存使用率
* `gpu_util` - GPU使用率
* `gpu_mem` - GPU内存使用率
* `disk_io` - 磁盘I/O
* `net_io` - 网络I/O

**硬件状态指标：**

* `temp_cpu` - CPU温度
* `temp_gpu` - GPU温度
* `fan_speed` - 风扇转速
* `voltage` - 电压
* `current` - 电流
* `frequency` - 频率

**系统性能指标：**

* `cache_miss` - 缓存未命中率
* `cache_hit` - 缓存命中率
* `context_switch` - 上下文切换次数
* `page_fault` - 页面错误次数
* `interrupts` - 中断次数
* `load_avg` - 平均负载
* `process_count` - 进程数量
* `thread_count` - 线程数量

**目标变量：**

* `power_consumption` - 功耗消耗

### 2. **推荐的数据集来源**

#### **A. 公开数据集**

1. **Google Cluster Data**
  
  * 包含Google数据中心的资源使用数据
  * 适合训练功耗预测模型
  * 数据量大，时间序列完整
2. **UCI Machine Learning Repository**
  
  * Individual Household Electric Power Consumption Dataset
  * 包含家庭用电数据，可用于功耗预测
3. **Kaggle数据集**
  
  * 服务器性能监控数据集
  * 数据中心能耗数据集

#### **B. 自建数据集**

1. **实时数据收集**
  
      # 使用项目的数据管道收集实时数据
      python data_pipeline/consumer.py
  
2. **模拟数据生成**
  
  * 基于真实硬件规格生成模拟数据
  * 使用负载测试工具生成不同工作负载下的数据

### 3. **数据质量要求**

#### **数据量要求**

* **最小数据量**: 10,000个样本（根据配置中的 `max_samples`）
* **推荐数据量**: 50,000-100,000个样本
* **时间跨度**: 至少1-2周，包含不同工作负载模式

#### **数据质量**

* **完整性**: 所有22个特征字段不能有缺失值
* **准确性**: 监控数据需要准确反映系统状态
* **一致性**: 数据格式和时间戳需要一致

#### **数据分布**

* **工作负载多样性**: 包含空闲、中等负载、高负载状态
* **时间模式**: 包含工作日、周末、不同时段的模式
* **异常情况**: 包含系统异常、重启等特殊情况

### 4. **数据预处理建议**

#### **数据清洗**

    # 移除异常值
    df = df[(df['power_consumption'] > 0) & (df['power_consumption'] < max_power)]
    df = df[(df['cpu_util'] >= 0) & (df['cpu_util'] <= 100)]
    
    # 处理缺失值
    df = df.fillna(method='ffill')  # 前向填充

#### **特征工程**

    # 添加时间特征
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    df['day_of_week'] = pd.to_datetime(df['time'], unit='s').dt.dayofweek
    
    # 添加滞后特征
    df['cpu_util_lag1'] = df['cpu_util'].shift(1)
    df['power_consumption_lag1'] = df['power_consumption'].shift(1)

#### **数据标准化**

    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col not in ['time', 'node_id', 'power_consumption']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

### 5. **实际运行步骤**

#### **步骤1: 准备数据**

    # 创建数据目录
    mkdir -p data/processed
    
    # 将数据集放入data目录
    # 支持格式: CSV, Parquet, JSON

#### **步骤2: 数据预处理**

    # 运行数据管道进行预处理
    python data_pipeline/consumer.py

#### **步骤3: 训练模型**

    # 训练功耗预测模型
    python train_power_model.py
    
    # 训练EMSHAP模型
    python train_emshap_model.py

#### **步骤4: 评估模型**

    # 评估模型性能
    python evaluate.py

### 6. **数据集获取建议**

#### **立即可用的选项**

1. **使用模拟数据**: 基于硬件规格生成模拟数据集
2. **公开数据集**: 从UCI或Kaggle下载相关数据集
3. **小规模测试**: 使用少量真实数据进行概念验证

#### **长期方案**

1. **建立数据收集系统**: 部署监控代理收集真实数据
2. **数据标注**: 手动标注部分数据用于验证
3. **持续优化**: 根据模型表现调整数据收集策略

这个项目主要针对**服务器/数据中心的功耗预测**，因此最适合的数据集应该包含服务器在各种工作负载下的性能指标和对应的功耗数据。