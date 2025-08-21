"""
English Version Data Visualization Script
For visualizing simulated data and model results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from loguru import logger

# Set English font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from file"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    
    logger.info(f"Data loaded successfully: {len(df)} records")
    return df


def plot_time_series(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Plot time series charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select main metrics for visualization
    main_metrics = ['power_consumption', 'cpu_util', 'gpu_util', 'mem_util', 'temp_cpu', 'temp_gpu']
    available_metrics = [m for m in main_metrics if m in df.columns]
    
    if not available_metrics:
        logger.warning("No main metrics found in data")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('System Monitoring Metrics Time Series', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(available_metrics):
        if 'timestamp' in df.columns:
            axes[i].plot(df['timestamp'], df[metric], linewidth=1, alpha=0.8)
            axes[i].set_xlabel('Time')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            axes[i].plot(df[metric], linewidth=1, alpha=0.8)
            axes[i].set_xlabel('Sample Index')
        
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
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
    
    if len(numeric_columns) < 2:
        logger.warning("Not enough numeric columns for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_matrix_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Correlation matrix saved to: {save_dir}/correlation_matrix_english.png")


def plot_feature_importance(importance_data: dict, save_dir: str = "evaluation_results"):
    """Plot feature importance"""
    os.makedirs(save_dir, exist_ok=True)
    
    features = importance_data.get('feature_names', [])
    importance = importance_data.get('feature_importance', [])
    
    if not features or not importance:
        logger.warning("No feature importance data available")
        return
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[-15:]  # Top 15 features
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), np.array(importance)[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance chart saved to: {save_dir}/feature_importance_english.png")


def generate_data_summary(df: pd.DataFrame, save_dir: str = "data/visualizations"):
    """Generate data summary"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col not in ['time']]
    
    if not numeric_columns:
        logger.warning("No numeric columns found for summary")
        return
    
    # Calculate summary statistics
    summary_stats = df[numeric_columns].describe()
    
    # Save summary to CSV
    summary_path = f'{save_dir}/data_summary_english.csv'
    summary_stats.to_csv(summary_path)
    
    logger.info(f"Data summary saved to: {summary_path}")
    
    # Create summary visualization
    n_metrics = min(6, len(numeric_columns))  # Limit to 6 for visualization
    selected_metrics = numeric_columns[:n_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Summary', fontsize=16, fontweight='bold')
    
    # Box plot of main metrics
    summary_data = [df[metric] for metric in selected_metrics]
    axes[0, 0].boxplot(summary_data, labels=[m.replace('_', ' ').title() for m in selected_metrics])
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Main Metrics Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of first metric
    if selected_metrics:
        axes[0, 1].hist(df[selected_metrics[0]], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel(selected_metrics[0].replace('_', ' ').title())
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{selected_metrics[0].replace("_", " ").title()} Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics table
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    
    # Create table data
    table_data = []
    for metric in selected_metrics:
        table_data.append([
            metric.replace('_', ' ').title(),
            f"{df[metric].mean():.2f}",
            f"{df[metric].std():.2f}",
            f"{df[metric].min():.2f}",
            f"{df[metric].max():.2f}"
        ])
    
    if table_data:
        table = axes[1, 0].table(cellText=table_data,
                                colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 0].set_title('Statistical Summary')
    
    # Correlation heatmap (simplified)
    if len(selected_metrics) > 1:
        corr_matrix = df[selected_metrics].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_xticks(range(len(selected_metrics)))
        axes[1, 1].set_yticks(range(len(selected_metrics)))
        axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in selected_metrics], rotation=45)
        axes[1, 1].set_yticklabels([m.replace('_', ' ').title() for m in selected_metrics])
        axes[1, 1].set_title('Metrics Correlation')
        
        # Add correlation values
        for i in range(len(selected_metrics)):
            for j in range(len(selected_metrics)):
                text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/data_summary_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Data summary chart saved to: {save_dir}/data_summary_english.png")


def main():
    """Main function"""
    logger.info("Starting English data visualization...")
    
    # Try to find data files
    data_paths = [
        "data/processed/simulated_data_high_perf.parquet",
        "evaluation_results/shapley_values.csv"
    ]
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            logger.info(f"Found data file: {data_path}")
            
            try:
                if data_path.endswith('.parquet'):
                    df = load_data(data_path)
                    
                    # Generate visualizations
                    logger.info("Generating time series chart...")
                    plot_time_series(df)
                    
                    logger.info("Generating correlation matrix...")
                    plot_correlation_matrix(df)
                    
                    logger.info("Generating data summary...")
                    generate_data_summary(df)
                    
                elif data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                    logger.info(f"CSV data loaded: {len(df)} records")
                    
                    # Generate basic visualization for CSV data
                    generate_data_summary(df, "evaluation_results")
                
                logger.info(f"Visualization completed for: {data_path}")
                
            except Exception as e:
                logger.error(f"Error processing {data_path}: {e}")
                continue
    
    logger.info("English data visualization completed!")


if __name__ == "__main__":
    main()
