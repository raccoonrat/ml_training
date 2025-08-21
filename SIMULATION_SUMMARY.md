# 模拟数据生成器完成总结

## 项目概述

成功为机器学习训练项目创建了一个完整的模拟数据生成系统，用于生成系统监控数据来测试EMSHAP和Power Predictor模型。

## 完成的功能

### 1. 核心数据生成器 (`simulated_data_generator.py`)
- **硬件规格配置**: 支持CPU、GPU、内存、存储、网络等硬件参数配置
- **工作负载模式**: 模拟空闲、办公、计算、图形、混合等不同工作负载
- **时间模式**: 模拟工作日、周末、夜间等时间模式
- **物理关系建模**: 模拟CPU/GPU使用率、温度、风扇转速、功耗之间的物理关系
- **数据连续性**: 确保时间序列数据的连续性和平滑性

### 2. 简化的FeatureVector类 (`data_pipeline/feature_vector.py`)
- 替代了有问题的protobuf实现
- 包含22个系统监控指标
- 支持与DataFrame的相互转换

### 3. 数据生成脚本 (`generate_sample_data.py`)
- **快速测试模式**: 生成一周的测试数据
- **多数据集模式**: 生成高性能、中等性能、低功耗三种配置的数据集
- **配置保存**: 自动保存硬件配置到JSON文件

### 4. 数据可视化工具 (`visualize_simulated_data.py`)
- **时间序列图**: 显示关键指标随时间的变化
- **相关性矩阵**: 分析各指标之间的相关性
- **功耗分析**: 分析功耗与其他指标的关系
- **工作负载模式**: 分析不同时间段的负载分布
- **系统健康状态**: 分析温度、风扇、缓存等系统健康指标
- **数据摘要**: 生成统计摘要和可视化图表

### 5. 配置修复 (`config.py`)
- 修复了dataclass中可变对象默认值的问题
- 使用`field(default_factory=...)`替代直接实例化

## 生成的数据集

### 1. 快速测试数据
- **文件**: `data/processed/quick_test_data.parquet`
- **数据点**: 13个（1小时，每5分钟一个）
- **用途**: 快速验证系统功能

### 2. 测试服务器数据
- **文件**: `data/processed/simulated_data_test.parquet`
- **数据点**: 2,017个（一周，每5分钟一个）
- **配置**: 中等性能服务器配置

### 3. 多配置数据集（两周数据）
- **高性能服务器**: `simulated_data_high_perf.parquet` (4,033个数据点)
- **中等性能服务器**: `simulated_data_mid_perf.parquet` (4,033个数据点)
- **低功耗服务器**: `simulated_data_low_power.parquet` (4,033个数据点)

## 数据特征

### 硬件配置差异
- **高性能**: 32核CPU, 24GB GPU, 128GB内存, 300W TDP
- **中等性能**: 16核CPU, 8GB GPU, 64GB内存, 150W TDP
- **低功耗**: 8核CPU, 4GB GPU, 32GB内存, 75W TDP

### 监控指标
包含22个系统监控指标：
- 基础指标: CPU/GPU使用率、内存使用率、磁盘/网络I/O
- 温度指标: CPU/GPU温度、风扇转速
- 电气指标: 电压、电流、频率
- 性能指标: 缓存命中/未命中、上下文切换、页面错误
- 系统指标: 中断、负载平均值、进程/线程数

### 数据质量
- **时间连续性**: 数据点间隔5分钟，时间序列连续
- **物理合理性**: 功耗与使用率、温度相关，符合物理规律
- **工作负载变化**: 模拟真实的工作负载模式（工作时间高负载，夜间低负载）
- **噪声添加**: 添加适当的随机噪声，模拟真实环境

## 可视化结果

生成了7个可视化图表：
1. `time_series.png` - 时间序列图
2. `correlation_matrix.png` - 相关性矩阵
3. `power_analysis.png` - 功耗分析
4. `workload_patterns.png` - 工作负载模式
5. `system_health.png` - 系统健康状态
6. `data_summary.png` - 数据摘要图
7. `data_summary.csv` - 统计摘要数据

## 使用方法

### 快速开始
```bash
# 生成快速测试数据
python generate_sample_data.py

# 生成多个数据集
python generate_sample_data.py multiple

# 生成可视化图表
python visualize_simulated_data.py
```

### 自定义配置
可以通过修改`HardwareSpec`、`WorkloadPattern`、`TimePattern`类来自定义：
- 硬件规格（CPU核心数、GPU内存、TDP等）
- 工作负载模式（不同场景的使用率范围）
- 时间模式（工作时间、周末模式等）

## 技术特点

1. **模块化设计**: 各组件独立，易于维护和扩展
2. **配置驱动**: 通过配置文件控制数据生成参数
3. **物理建模**: 基于真实硬件特性建模，数据更真实
4. **可视化支持**: 提供完整的数据分析和可视化工具
5. **格式兼容**: 生成的Parquet格式与现有ML训练流程兼容

## 下一步建议

1. **模型训练**: 使用生成的数据集训练EMSHAP和Power Predictor模型
2. **数据验证**: 与真实数据进行对比，验证模拟数据的质量
3. **参数调优**: 根据训练结果调整数据生成参数
4. **扩展功能**: 添加更多硬件配置和工作负载模式

## 文件结构

```
ml_training/
├── simulated_data_generator.py      # 核心数据生成器
├── generate_sample_data.py          # 数据生成脚本
├── visualize_simulated_data.py      # 数据可视化工具
├── data_pipeline/
│   ├── feature_vector.py           # FeatureVector类定义
│   └── ...
├── data/
│   ├── processed/                  # 生成的数据文件
│   └── visualizations/             # 可视化图表
├── config.py                       # 配置修复
└── README_SIMULATED_DATA.md        # 详细使用说明
```

## 总结

成功创建了一个完整的模拟数据生成系统，能够生成高质量的系统监控数据用于机器学习模型训练。系统具有良好的可配置性、可扩展性和易用性，为项目的后续开发提供了坚实的数据基础。
