# Google Cluster Data EMSHAP增强模型（改进版本）

本项目提供了改进版本的Google Cluster Data加载器和EMSHAP增强模型训练脚本，能够正确处理原始数据格式并自动映射到标准列名。

## 文件说明

### 核心文件

1. **`google_cluster_data_loader_improved.py`** - 改进版本的Google Cluster Data加载器
   - 自动检测和处理原始数据格式
   - 智能列名映射
   - 数据合并和预处理
   - 可视化功能

2. **`train_emshap_enhanced_cluster_improved.py`** - 改进版本的训练脚本
   - 使用改进版本的数据加载器
   - 完整的训练、评估和可视化流程
   - 自动保存结果

3. **`test_google_cluster_improved.py`** - 快速测试脚本
   - 验证数据加载器功能
   - 测试EMSHAP模型导入
   - 确保所有组件正常工作

## 快速开始

### 1. 测试环境

首先运行测试脚本确保所有组件正常工作：

```bash
python test_google_cluster_improved.py
```

### 2. 运行训练

使用改进版本的训练脚本：

```bash
# 基本训练
python train_emshap_enhanced_cluster_improved.py

# 自定义参数
python train_emshap_enhanced_cluster_improved.py \
    --data-types task_usage task_events machine_events \
    --input-dim 64 \
    --context-dim 64 \
    --target-column cpu_rate \
    --output-dir evaluation_results_cluster \
    --device auto
```

### 3. 参数说明

- `--data-types`: 要加载的数据类型（task_events, task_usage, machine_events）
- `--input-dim`: 输入特征维度（默认64）
- `--context-dim`: 上下文向量维度（默认64）
- `--target-column`: 目标列名（默认cpu_rate）
- `--output-dir`: 输出目录（默认evaluation_results_cluster）
- `--device`: 设备选择（auto/cpu/cuda）

## 功能特性

### 数据加载器改进

1. **智能列名映射**
   - 自动检测原始数据格式
   - 应用标准列名映射
   - 处理列数不匹配的情况

2. **数据合并优化**
   - 智能合并多个数据源
   - 处理缺失值
   - 保留重要特征

3. **特征工程**
   - 自动特征选择
   - 相关性分析
   - 维度调整

4. **可视化支持**
   - 数据分布图
   - 相关性热图
   - 特征重要性图

### 训练脚本改进

1. **快速训练**
   - 减少训练轮数（20轮）
   - 减少Shapley样本数（50个）
   - 优化批次大小

2. **完整评估**
   - Shapley值计算
   - 特征重要性分析
   - 预测性能评估

3. **结果保存**
   - 模型检查点
   - 评估结果
   - 可视化图表

## 输出文件

训练完成后，会在输出目录中生成以下文件：

```
evaluation_results_cluster/
├── cluster_emshap_enhanced_model.pth          # 训练好的模型
├── cluster_shapley_values.csv                 # Shapley值
├── cluster_feature_importance.csv             # 特征重要性
├── cluster_evaluation_summary.json            # 评估摘要
├── cluster_training_history.png               # 训练历史图
├── cluster_feature_importance.png             # 特征重要性图
├── cluster_shapley_distribution.png           # Shapley值分布图
├── cluster_feature_correlation.png            # 特征相关性热图
└── cluster_cpu_prediction.png                 # CPU使用率预测图
```

## 数据格式

### 支持的Google Cluster Data格式

1. **task_events.csv** - 任务事件数据
   - 包含任务提交、开始、完成等事件
   - 资源请求信息

2. **task_usage.csv** - 任务使用数据
   - CPU、内存、磁盘使用情况
   - 性能指标

3. **machine_events.csv** - 机器事件数据
   - 机器状态变化
   - 硬件配置信息

### 自动生成模拟数据

如果原始数据文件不存在，系统会自动生成符合Google Cluster Data格式的模拟数据，确保训练脚本能够正常运行。

## 性能优化

### 训练优化

- 使用较小的批次大小（64）
- 减少训练轮数（20轮）
- 早停机制（8轮耐心值）

### 评估优化

- 减少Shapley样本数（50个）
- 只计算最重要的特征
- 并行处理支持

## 故障排除

### 常见问题

1. **数据文件不存在**
   - 系统会自动生成模拟数据
   - 检查日志确认数据加载状态

2. **内存不足**
   - 减少批次大小
   - 减少输入维度
   - 使用CPU训练

3. **训练时间过长**
   - 减少训练轮数
   - 减少Shapley样本数
   - 使用更小的模型

### 日志查看

所有操作都会记录详细日志，可以通过以下方式查看：

```python
from loguru import logger
logger.info("查看训练进度")
```

## 扩展功能

### 自定义数据源

可以通过修改`google_cluster_data_loader_improved.py`来支持其他数据源：

```python
def load_custom_data(self, file_path: str) -> pd.DataFrame:
    """加载自定义数据源"""
    # 实现自定义数据加载逻辑
    pass
```

### 自定义特征工程

可以在`preprocess_cluster_data_for_emshap`方法中添加自定义特征：

```python
def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """添加自定义特征"""
    # 实现自定义特征工程
    return df
```

## 版本历史

- **v1.0** - 初始版本，基本功能
- **v1.1** - 改进版本，智能列名映射和错误处理
- **v1.2** - 当前版本，完整的数据处理和训练流程

## 贡献

欢迎提交问题和改进建议！

## 许可证

本项目采用MIT许可证。
