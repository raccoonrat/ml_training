"""
模拟数据可视化脚本
用于验证生成的模拟数据质量和特征关系
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from loguru import logger

# 设置中文字体
def setup_chinese_font():
    """设置中文字体支持"""
    import platform
    import matplotlib.font_manager as fm
    
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统字体
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        # macOS系统字体
        font_names = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体
        font_names = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Liberation Sans']
    
    # 查找可用的中文字体
    available_fonts = []
    for font_name in font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path != fm.rcParams['font.sans-serif']:
                available_fonts.append(font_name)
                break
        except:
            continue
    
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans']
        logger.info(f"使用中文字体: {available_fonts[0]}")
    else:
        # 如果没有找到中文字体，使用默认字体并给出警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        logger.warning("未找到中文字体，图表中的中文可能无法正确显示")
    
    plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
setup_chinese_font()

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_simulated_data(data_path: str) -> pd.DataFrame:
    """加载模拟数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    return df


def plot_time_series(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """绘制时间序列图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择主要指标进行可视化
    main_metrics = ['power_consumption', 'cpu_util', 'gpu_util', 'mem_util', 'temp_cpu', 'temp_gpu']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('系统监控指标时间序列', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(main_metrics):
        row = i // 2
        col = i % 2
        
        axes[row, col].plot(df['timestamp'], df[metric], linewidth=1, alpha=0.8)
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].set_xlabel('时间')
        axes[row, col].set_ylabel(metric.replace("_", " ").title())
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"时间序列图已保存到: {save_dir}/time_series.png")


def plot_correlation_matrix(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """绘制相关性矩阵"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择数值型特征
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col not in ['time']]
    
    # 计算相关性矩阵
    corr_matrix = df[numeric_columns].corr()
    
    # 绘制热力图
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('特征相关性矩阵', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"相关性矩阵已保存到: {save_dir}/correlation_matrix.png")


def plot_power_analysis(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """分析功耗与其他指标的关系"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择与功耗相关的指标
    power_related = ['cpu_util', 'gpu_util', 'mem_util', 'temp_cpu', 'temp_gpu', 'fan_speed']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('功耗与其他指标的关系分析', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(power_related):
        row = i // 3
        col = i % 3
        
        axes[row, col].scatter(df[metric], df['power_consumption'], alpha=0.6, s=20)
        axes[row, col].set_xlabel(metric.replace("_", " ").title())
        axes[row, col].set_ylabel('Power Consumption (W)')
        axes[row, col].set_title(f'Power vs {metric.replace("_", " ").title()}')
        axes[row, col].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(df[metric], df['power_consumption'], 1)
        p = np.poly1d(z)
        axes[row, col].plot(df[metric], p(df[metric]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/power_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"功耗分析图已保存到: {save_dir}/power_analysis.png")


def plot_workload_patterns(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """分析工作负载模式"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 添加时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('工作负载模式分析', fontsize=16, fontweight='bold')
    
    # 按小时分析
    hourly_avg = df.groupby('hour')[['cpu_util', 'gpu_util', 'power_consumption']].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg['cpu_util'], label='CPU使用率', marker='o')
    axes[0, 0].plot(hourly_avg.index, hourly_avg['gpu_util'], label='GPU使用率', marker='s')
    axes[0, 0].set_xlabel('小时')
    axes[0, 0].set_ylabel('使用率 (%)')
    axes[0, 0].set_title('24小时使用率模式')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 按星期分析
    daily_avg = df.groupby('day_of_week')[['cpu_util', 'gpu_util', 'power_consumption']].mean()
    axes[0, 1].plot(daily_avg.index, daily_avg['cpu_util'], label='CPU使用率', marker='o')
    axes[0, 1].plot(daily_avg.index, daily_avg['gpu_util'], label='GPU使用率', marker='s')
    axes[0, 1].set_xlabel('星期')
    axes[0, 1].set_ylabel('使用率 (%)')
    axes[0, 1].set_title('一周使用率模式')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 功耗分布
    axes[1, 0].hist(df['power_consumption'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('功耗 (W)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('功耗分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 工作日vs周末对比
    weekend_data = df[df['is_weekend']]
    weekday_data = df[~df['is_weekend']]
    
    axes[1, 1].boxplot([weekday_data['power_consumption'], weekend_data['power_consumption']], 
                       labels=['工作日', '周末'])
    axes[1, 1].set_ylabel('功耗 (W)')
    axes[1, 1].set_title('工作日vs周末功耗对比')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/workload_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"工作负载模式分析图已保存到: {save_dir}/workload_patterns.png")


def plot_system_health(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """系统健康状态分析"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('系统健康状态分析', fontsize=16, fontweight='bold')
    
    # 温度分析
    axes[0, 0].scatter(df['temp_cpu'], df['temp_gpu'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('CPU温度 (°C)')
    axes[0, 0].set_ylabel('GPU温度 (°C)')
    axes[0, 0].set_title('CPU vs GPU温度关系')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 风扇转速与温度关系
    axes[0, 1].scatter(df['temp_cpu'], df['fan_speed'], alpha=0.6, s=20, label='CPU温度')
    axes[0, 1].scatter(df['temp_gpu'], df['fan_speed'], alpha=0.6, s=20, label='GPU温度')
    axes[0, 1].set_xlabel('温度 (°C)')
    axes[0, 1].set_ylabel('风扇转速 (RPM)')
    axes[0, 1].set_title('温度与风扇转速关系')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 缓存性能
    axes[1, 0].scatter(df['cache_hit'], df['cache_miss'], alpha=0.6, s=20)
    axes[1, 0].set_xlabel('缓存命中率 (%)')
    axes[1, 0].set_ylabel('缓存未命中率 (%)')
    axes[1, 0].set_title('缓存性能分析')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 系统负载
    axes[1, 1].scatter(df['load_avg'], df['process_count'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('平均负载')
    axes[1, 1].set_ylabel('进程数量')
    axes[1, 1].set_title('系统负载分析')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/system_health.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"系统健康状态分析图已保存到: {save_dir}/system_health.png")


def generate_data_summary(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """生成数据摘要报告"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 基本统计信息
    summary = df.describe()
    
    # 保存到文件
    summary.to_csv(f'{save_dir}/data_summary.csv')
    
    # 创建摘要图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('数据摘要统计', fontsize=16, fontweight='bold')
    
    # 主要指标分布
    main_metrics = ['power_consumption', 'cpu_util', 'gpu_util', 'mem_util']
    summary_data = [df[metric] for metric in main_metrics]
    
    axes[0, 0].boxplot(summary_data, labels=[m.replace('_', ' ').title() for m in main_metrics])
    axes[0, 0].set_ylabel('数值')
    axes[0, 0].set_title('主要指标分布')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 数据量统计
    total_points = len(df)
    time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600  # 小时
    
    axes[0, 1].text(0.1, 0.8, f'总数据点: {total_points:,}', fontsize=14, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.6, f'时间跨度: {time_span:.1f} 小时', fontsize=14, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.4, f'数据间隔: 5分钟', fontsize=14, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.2, f'节点ID: {df["node_id"].iloc[0]}', fontsize=14, transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('数据集信息')
    axes[0, 1].axis('off')
    
    # 功耗统计
    axes[1, 0].hist(df['power_consumption'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(df['power_consumption'].mean(), color='red', linestyle='--', label=f'平均值: {df["power_consumption"].mean():.1f}W')
    axes[1, 0].set_xlabel('功耗 (W)')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('功耗分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 使用率统计
    usage_metrics = ['cpu_util', 'gpu_util', 'mem_util']
    usage_means = [df[metric].mean() for metric in usage_metrics]
    usage_labels = ['CPU', 'GPU', '内存']
    
    axes[1, 1].bar(usage_labels, usage_means, alpha=0.7)
    axes[1, 1].set_ylabel('平均使用率 (%)')
    axes[1, 1].set_title('平均使用率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/data_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"数据摘要已保存到: {save_dir}/data_summary.csv")
    logger.info(f"数据摘要图已保存到: {save_dir}/data_summary.png")


def main():
    """主函数"""
    # 查找数据文件
    data_dir = "data/processed"
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not data_files:
        logger.error("未找到数据文件，请先生成模拟数据")
        return
    
    # 使用第一个数据文件
    data_file = data_files[0]
    data_path = os.path.join(data_dir, data_file)
    
    logger.info(f"加载数据文件: {data_path}")
    
    # 加载数据
    df = load_simulated_data(data_path)
    
    logger.info(f"数据加载完成，共 {len(df)} 个数据点")
    logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 创建可视化目录
    viz_dir = "data/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 生成各种可视化图表
    logger.info("生成时间序列图...")
    plot_time_series(df, viz_dir)
    
    logger.info("生成相关性矩阵...")
    plot_correlation_matrix(df, viz_dir)
    
    logger.info("生成功耗分析图...")
    plot_power_analysis(df, viz_dir)
    
    logger.info("生成工作负载模式分析...")
    plot_workload_patterns(df, viz_dir)
    
    logger.info("生成系统健康状态分析...")
    plot_system_health(df, viz_dir)
    
    logger.info("生成数据摘要...")
    generate_data_summary(df, viz_dir)
    
    logger.info("所有可视化图表生成完成！")
    logger.info(f"图表保存在: {viz_dir}")


if __name__ == "__main__":
    main()
