"""
English Version of Simulated Data Visualization Script
For verifying the quality of generated simulated data and feature relationships
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from loguru import logger

# Set chart style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_simulated_data(data_path: str) -> pd.DataFrame:
    """Load simulated data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    return df


def plot_time_series(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Plot time series charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select main metrics for visualization
    main_metrics = ['power_consumption', 'cpu_util', 'gpu_util', 'mem_util', 'temp_cpu', 'temp_gpu']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('System Monitoring Metrics Time Series', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(main_metrics):
        row = i // 2
        col = i % 2
        
        axes[row, col].plot(df['timestamp'], df[metric], linewidth=1, alpha=0.8)
        axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel(metric.replace("_", " ").title())
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/time_series_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Time series chart saved to: {save_dir}/time_series_english.png")


def plot_correlation_matrix(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Plot correlation matrix"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select numeric features
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col not in ['time']]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrix_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Correlation matrix saved to: {save_dir}/correlation_matrix_english.png")


def plot_power_analysis(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Analyze power consumption vs other metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select power-related metrics
    power_related = ['cpu_util', 'gpu_util', 'mem_util', 'temp_cpu', 'temp_gpu', 'fan_speed']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Power Consumption vs Other Metrics Analysis', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(power_related):
        row = i // 3
        col = i % 3
        
        axes[row, col].scatter(df[metric], df['power_consumption'], alpha=0.6, s=20)
        axes[row, col].set_xlabel(metric.replace('_', ' ').title())
        axes[row, col].set_ylabel('Power Consumption (W)')
        axes[row, col].set_title(f'Power vs {metric.replace("_", " ").title()}')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df[metric], df['power_consumption'], 1)
        p = np.poly1d(z)
        axes[row, col].plot(df[metric], p(df[metric]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/power_analysis_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Power analysis chart saved to: {save_dir}/power_analysis_english.png")


def plot_workload_patterns(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Analyze workload patterns"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract hour and weekday
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Separate weekday and weekend data
    weekday_data = df[df['weekday'] < 5]  # Monday to Friday
    weekend_data = df[df['weekday'] >= 5]  # Saturday and Sunday
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Workload Pattern Analysis', fontsize=16, fontweight='bold')
    
    # Hourly power consumption pattern
    hourly_power = df.groupby('hour')['power_consumption'].mean()
    axes[0, 0].plot(hourly_power.index, hourly_power.values, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Power Consumption (W)')
    axes[0, 0].set_title('Hourly Power Consumption Pattern')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power distribution
    axes[0, 1].hist(df['power_consumption'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Power Consumption (W)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Power Consumption Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weekday vs Weekend comparison
    weekday_hours = weekday_data.groupby('hour')['power_consumption'].mean()
    weekend_hours = weekend_data.groupby('hour')['power_consumption'].mean()
    
    axes[1, 0].plot(weekday_hours.index, weekday_hours.values, marker='o', label='Weekday', linewidth=2)
    axes[1, 0].plot(weekend_hours.index, weekend_hours.values, marker='s', label='Weekend', linewidth=2)
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Average Power Consumption (W)')
    axes[1, 0].set_title('Weekday vs Weekend Power Pattern')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    axes[1, 1].boxplot([weekday_data['power_consumption'], weekend_data['power_consumption']],
                      tick_labels=['Weekday', 'Weekend'])
    axes[1, 1].set_ylabel('Power Consumption (W)')
    axes[1, 1].set_title('Power Consumption: Weekday vs Weekend')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/workload_patterns_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Workload patterns chart saved to: {save_dir}/workload_patterns_english.png")


def plot_system_health(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Analyze system health metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select system health metrics
    health_metrics = ['temp_cpu', 'temp_gpu', 'fan_speed', 'cache_hit', 'cache_miss', 'load_avg']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('System Health Analysis', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(health_metrics):
        row = i // 3
        col = i % 3
        
        axes[row, col].scatter(df['power_consumption'], df[metric], alpha=0.6, s=20)
        axes[row, col].set_xlabel('Power Consumption (W)')
        axes[row, col].set_ylabel(metric.replace('_', ' ').title())
        axes[row, col].set_title(f'Power vs {metric.replace("_", " ").title()}')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/system_health_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"System health chart saved to: {save_dir}/system_health_english.png")


def generate_data_summary(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Generate data summary and visualization"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select main metrics for summary
    main_metrics = ['power_consumption', 'cpu_util', 'gpu_util', 'mem_util', 'temp_cpu', 'temp_gpu']
    
    # Calculate summary statistics
    summary_stats = df[main_metrics].describe()
    
    # Save summary to CSV
    summary_path = f'{save_dir}/data_summary_english.csv'
    summary_stats.to_csv(summary_path)
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Summary', fontsize=16, fontweight='bold')
    
    # Box plot of main metrics
    summary_data = [df[metric] for metric in main_metrics]
    axes[0, 0].boxplot(summary_data, labels=[m.replace('_', ' ').title() for m in main_metrics])
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Main Metrics Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap of main metrics
    corr_matrix = df[main_metrics].corr()
    im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[0, 1].set_xticks(range(len(main_metrics)))
    axes[0, 1].set_yticks(range(len(main_metrics)))
    axes[0, 1].set_xticklabels([m.replace('_', ' ').title() for m in main_metrics], rotation=45)
    axes[0, 1].set_yticklabels([m.replace('_', ' ').title() for m in main_metrics])
    axes[0, 1].set_title('Main Metrics Correlation')
    
    # Add correlation values
    for i in range(len(main_metrics)):
        for j in range(len(main_metrics)):
            text = axes[0, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black")
    
    # Time series of main metrics (first 100 points)
    sample_size = min(100, len(df))
    sample_df = df.head(sample_size)
    
    for metric in main_metrics:
        axes[1, 0].plot(sample_df['timestamp'], sample_df[metric], label=metric.replace('_', ' ').title(), alpha=0.7)
    
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Main Metrics Time Series (First 100 Points)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    # Create table data
    table_data = []
    for metric in main_metrics:
        table_data.append([
            metric.replace('_', ' ').title(),
            f"{df[metric].mean():.2f}",
            f"{df[metric].std():.2f}",
            f"{df[metric].min():.2f}",
            f"{df[metric].max():.2f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 1].set_title('Statistical Summary')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/data_summary_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Data summary saved to: {save_dir}/data_summary_english.csv")
    logger.info(f"Data summary chart saved to: {save_dir}/data_summary_english.png")


def main():
    """Main function"""
    logger.info("Starting English version of simulated data visualization...")
    
    # Find the latest data file
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
        logger.error(f"Processed data directory not found: {processed_dir}")
        return
    
    data_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
    if not data_files:
        logger.error(f"No data files found in {processed_dir}")
        return
    
    # Use the largest file (most data points)
    data_files.sort(key=lambda x: os.path.getsize(os.path.join(processed_dir, x)), reverse=True)
    data_path = os.path.join(processed_dir, data_files[0])
    
    logger.info(f"Loading data file: {data_path}")
    
    try:
        # Load data
        df = load_simulated_data(data_path)
        logger.info(f"Data loaded successfully, {len(df)} data points")
        logger.info(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Generate visualizations
        logger.info("Generating time series chart...")
        plot_time_series(df)
        
        logger.info("Generating correlation matrix...")
        plot_correlation_matrix(df)
        
        logger.info("Generating power analysis chart...")
        plot_power_analysis(df)
        
        logger.info("Generating workload pattern analysis...")
        plot_workload_patterns(df)
        
        logger.info("Generating system health analysis...")
        plot_system_health(df)
        
        logger.info("Generating data summary...")
        generate_data_summary(df)
        
        logger.info("All English visualization charts generated successfully!")
        logger.info(f"Charts saved in: data/visualizations")
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
