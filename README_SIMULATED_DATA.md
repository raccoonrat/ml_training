# 模拟数据生成器使用说明

本项目提供了完整的模拟数据生成器，用于生成符合机器学习训练项目要求的系统监控数据。

## 功能特性

### 🎯 核心功能
- **基于硬件规格生成数据**: 支持不同配置的CPU、GPU、内存等硬件
- **真实的工作负载模式**: 模拟空闲、办公、计算、图形、混合等不同工作负载
- **时间模式模拟**: 工作日、周末、夜间等不同时间段的使用模式
- **物理关系建模**: 确保生成的数据符合真实的物理规律
- **多种数据格式**: 支持Parquet格式，兼容项目的数据管道

### 📊 生成的数据特征
- **22个系统监控指标**: 包含CPU、GPU、内存、温度、功耗等
- **时间序列数据**: 支持自定义时间间隔和范围
- **多节点支持**: 可生成多个不同配置的服务器数据
- **数据质量保证**: 包含合理的噪声和异常值

## 快速开始

### 1. 生成快速测试数据

```bash
# 生成一周的测试数据（推荐用于快速测试）
python generate_sample_data.py
```

这将生成：
- `data/processed/simulated_data_test.parquet` - 测试数据文件
- `data/processed/config_test_server_001.json` - 配置信息

### 2. 生成多个数据集

```bash
# 生成多个不同配置的数据集
python generate_sample_data.py multiple
```

这将生成三个不同配置的数据集：
- `simulated_data_high_perf.parquet` - 高性能服务器数据
- `simulated_data_mid_perf.parquet` - 中等性能服务器数据  
- `simulated_data_low_power.parquet` - 低功耗服务器数据

### 3. 使用完整的数据生成器

```python
from simulated_data_generator import SimulatedDataGenerator, HardwareSpec
from datetime import datetime, timedelta

# 创建自定义硬件配置
hardware_spec = HardwareSpec(
    cpu_cores=16,
    cpu_base_freq=3.0,
    cpu_max_freq=4.0,
    cpu_tdp=125.0,
    gpu_memory=8192,
    gpu_tdp=150.0
)

# 创建数据生成器
generator = SimulatedDataGenerator(
    hardware_spec=hardware_spec,
    node_id="custom_server_001"
)

# 生成数据
df = generator.generate_dataset(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    interval_minutes=5,
    save_path="data/processed/custom_data.parquet"
)
```

## 数据可视化

### 生成可视化图表

```bash
# 生成数据质量分析图表
python visualize_simulated_data.py
```

这将生成以下图表：
- `time_series.png` - 时间序列图
- `correlation_matrix.png` - 特征相关性矩阵
- `power_analysis.png` - 功耗分析图
- `workload_patterns.png` - 工作负载模式分析
- `system_health.png` - 系统健康状态分析
- `data_summary.png` - 数据摘要统计

## 硬件配置说明

### 默认硬件规格

```python
HardwareSpec(
    cpu_cores=8,           # CPU核心数
    cpu_base_freq=3.2,     # 基础频率 (GHz)
    cpu_max_freq=4.5,      # 最大频率 (GHz)
    cpu_tdp=95.0,          # 热设计功耗 (W)
    gpu_memory=8192,       # GPU显存 (MB)
    gpu_base_freq=1500,    # GPU基础频率 (MHz)
    gpu_max_freq=1800,     # GPU最大频率 (MHz)
    gpu_tdp=150.0,         # GPU热设计功耗 (W)
    memory_size=32768,     # 内存大小 (MB)
    memory_freq=3200,      # 内存频率 (MHz)
    storage_type="SSD",    # 存储类型
    storage_speed=3500,    # 存储速度 (MB/s)
    network_speed=1000,    # 网络速度 (Mbps)
    fan_max_speed=3000,    # 最大风扇转速 (RPM)
    ambient_temp=25.0      # 环境温度 (°C)
)
```

### 预定义配置

1. **高性能服务器** (`high_perf`)
   - 32核CPU, 180W TDP
   - 24GB GPU显存, 300W TDP
   - 128GB内存, 4800MHz
   - NVMe存储, 10Gbps网络

2. **中等性能服务器** (`mid_perf`)
   - 16核CPU, 125W TDP
   - 8GB GPU显存, 150W TDP
   - 64GB内存, 3200MHz
   - SSD存储, 2.5Gbps网络

3. **低功耗服务器** (`low_power`)
   - 8核CPU, 65W TDP
   - 4GB GPU显存, 75W TDP
   - 32GB内存, 2666MHz
   - SSD存储, 1Gbps网络

## 工作负载模式

### 支持的工作负载类型

1. **空闲模式** (`idle`)
   - CPU使用率: 5-15%
   - GPU使用率: 0-5%
   - 内存使用率: 20-40%

2. **办公模式** (`office`)
   - CPU使用率: 20-50%
   - GPU使用率: 5-15%
   - 内存使用率: 40-70%

3. **计算模式** (`compute`)
   - CPU使用率: 60-95%
   - GPU使用率: 10-30%
   - 内存使用率: 60-90%

4. **图形模式** (`graphics`)
   - CPU使用率: 30-70%
   - GPU使用率: 70-95%
   - 内存使用率: 50-85%

5. **混合模式** (`mixed`)
   - CPU使用率: 50-85%
   - GPU使用率: 40-80%
   - 内存使用率: 60-90%

### 时间模式

- **工作日** (周一至周五): 9:00-18:00高负载概率70%
- **周末** (周六、周日): 全天高负载概率30%
- **夜间** (23:00-6:00): 低负载概率90%

## 生成的数据字段

### 元数据字段
- `time`: 时间戳 (Unix时间戳)
- `node_id`: 节点ID
- `timestamp`: 可读时间格式

### 目标变量
- `power_consumption`: 功耗消耗 (W)

### 系统性能指标
- `cpu_util`: CPU使用率 (%)
- `mem_util`: 内存使用率 (%)
- `gpu_util`: GPU使用率 (%)
- `gpu_mem`: GPU内存使用率 (%)
- `disk_io`: 磁盘I/O (MB/s)
- `net_io`: 网络I/O (MB/s)

### 硬件状态指标
- `temp_cpu`: CPU温度 (°C)
- `temp_gpu`: GPU温度 (°C)
- `fan_speed`: 风扇转速 (RPM)
- `voltage`: 电压 (V)
- `current`: 电流 (A)
- `frequency`: 频率 (MHz)

### 系统性能指标
- `cache_miss`: 缓存未命中率 (%)
- `cache_hit`: 缓存命中率 (%)
- `context_switch`: 上下文切换次数
- `page_fault`: 页面错误次数
- `interrupts`: 中断次数
- `load_avg`: 平均负载
- `process_count`: 进程数量
- `thread_count`: 线程数量

## 数据质量保证

### 物理关系建模
- **功耗计算**: 基于CPU、GPU使用率和温度
- **温度计算**: 基于使用率和散热效果
- **风扇转速**: 基于温度自动调节
- **缓存性能**: 与CPU使用率相关
- **系统指标**: 与负载相关

### 数据连续性
- 时间序列平滑过渡
- 避免数据突变
- 合理的随机噪声

### 数据完整性
- 无缺失值
- 数值范围合理
- 时间戳连续

## 使用示例

### 训练机器学习模型

```python
# 加载生成的数据
import pandas as pd
from train_power_model import PowerModelTrainer

# 加载数据
df = pd.read_parquet("data/processed/simulated_data_test.parquet")

# 训练模型
trainer = PowerModelTrainer()
trainer.load_data("data/processed/simulated_data_test.parquet")
trainer.train()
```

### 数据验证

```python
# 验证数据质量
df = pd.read_parquet("data/processed/simulated_data_test.parquet")

print(f"数据点数量: {len(df)}")
print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
print(f"功耗范围: {df['power_consumption'].min():.2f}W 到 {df['power_consumption'].max():.2f}W")
print(f"CPU使用率范围: {df['cpu_util'].min():.2f}% 到 {df['cpu_util'].max():.2f}%")
```

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```bash
   # 确保先生成数据
   python generate_sample_data.py
   ```

2. **可视化图表生成失败**
   ```bash
   # 安装依赖
   pip install matplotlib seaborn
   ```

3. **内存不足**
   ```python
   # 减少数据量
   generator.generate_dataset(
       start_time=datetime.now() - timedelta(days=1),  # 只生成一天数据
       interval_minutes=10  # 增加时间间隔
   )
   ```

### 性能优化

- 对于大数据集，建议分批生成
- 可以调整时间间隔来平衡数据量和生成速度
- 使用SSD存储可以提高I/O性能

## 扩展功能

### 自定义工作负载模式

```python
from simulated_data_generator import WorkloadPattern

# 自定义工作负载模式
custom_pattern = WorkloadPattern(
    idle_cpu_range=(10.0, 25.0),
    office_cpu_range=(30.0, 60.0),
    compute_cpu_range=(70.0, 98.0)
)

generator = SimulatedDataGenerator(
    workload_pattern=custom_pattern
)
```

### 自定义时间模式

```python
from simulated_data_generator import TimePattern

# 自定义时间模式
custom_time = TimePattern(
    workday_start_hour=8,
    workday_end_hour=20,
    workday_high_load_prob=0.8
)

generator = SimulatedDataGenerator(
    time_pattern=custom_time
)
```

## 贡献指南

欢迎提交Issue和Pull Request来改进数据生成器：

1. 添加新的硬件配置
2. 改进物理关系建模
3. 优化数据生成性能
4. 增加新的可视化功能

## 许可证

本项目采用MIT许可证。
