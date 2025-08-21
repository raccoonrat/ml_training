"""
快速生成示例数据脚本
用于测试机器学习训练项目
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

from simulated_data_generator import SimulatedDataGenerator, HardwareSpec


def generate_quick_test_data():
    """生成快速测试数据"""
    
    # 创建数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 创建中等性能服务器的数据生成器
    hardware_spec = HardwareSpec(
        cpu_cores=16,
        cpu_base_freq=3.0,
        cpu_max_freq=4.0,
        cpu_tdp=125.0,
        gpu_memory=8192,
        gpu_base_freq=1500,
        gpu_max_freq=1800,
        gpu_tdp=150.0,
        memory_size=65536,
        memory_freq=3200,
        storage_type="SSD",
        storage_speed=3500,
        network_speed=2500
    )
    
    generator = SimulatedDataGenerator(
        hardware_spec=hardware_spec,
        node_id="test_server_001"
    )
    
    # 生成一周的数据，每5分钟一个数据点
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    logger.info("开始生成测试数据...")
    
    df = generator.generate_dataset(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=5,
        save_path="data/processed/simulated_data_test.parquet"
    )
    
    # 保存配置
    generator.save_config("data/processed/config_test_server_001.json")
    
    logger.info(f"测试数据生成完成！")
    logger.info(f"数据点数量: {len(df)}")
    logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    logger.info(f"功耗范围: {df['power_consumption'].min():.2f}W 到 {df['power_consumption'].max():.2f}W")
    logger.info(f"CPU使用率范围: {df['cpu_util'].min():.2f}% 到 {df['cpu_util'].max():.2f}%")
    logger.info(f"GPU使用率范围: {df['gpu_util'].min():.2f}% 到 {df['gpu_util'].max():.2f}%")
    
    return df


def generate_multiple_datasets():
    """生成多个不同配置的数据集"""
    
    # 创建数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 定义不同的硬件配置
    configs = [
        {
            "name": "high_perf",
            "spec": HardwareSpec(
                cpu_cores=32,
                cpu_base_freq=2.8,
                cpu_max_freq=4.2,
                cpu_tdp=180.0,
                gpu_memory=24576,
                gpu_base_freq=1200,
                gpu_max_freq=2000,
                gpu_tdp=300.0,
                memory_size=131072,
                memory_freq=4800,
                storage_type="NVMe",
                storage_speed=7000,
                network_speed=10000
            )
        },
        {
            "name": "mid_perf", 
            "spec": HardwareSpec(
                cpu_cores=16,
                cpu_base_freq=3.0,
                cpu_max_freq=4.0,
                cpu_tdp=125.0,
                gpu_memory=8192,
                gpu_base_freq=1500,
                gpu_max_freq=1800,
                gpu_tdp=150.0,
                memory_size=65536,
                memory_freq=3200,
                storage_type="SSD",
                storage_speed=3500,
                network_speed=2500
            )
        },
        {
            "name": "low_power",
            "spec": HardwareSpec(
                cpu_cores=8,
                cpu_base_freq=2.5,
                cpu_max_freq=3.5,
                cpu_tdp=65.0,
                gpu_memory=4096,
                gpu_base_freq=1000,
                gpu_max_freq=1400,
                gpu_tdp=75.0,
                memory_size=32768,
                memory_freq=2666,
                storage_type="SSD",
                storage_speed=2000,
                network_speed=1000
            )
        }
    ]
    
    # 生成时间范围
    start_time = datetime.now() - timedelta(days=14)  # 两周数据
    end_time = datetime.now()
    
    for i, config in enumerate(configs):
        logger.info(f"生成第 {i+1}/{len(configs)} 个数据集: {config['name']}")
        
        generator = SimulatedDataGenerator(
            hardware_spec=config['spec'],
            node_id=f"{config['name']}_server_001"
        )
        
        df = generator.generate_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=5,
            save_path=f"data/processed/simulated_data_{config['name']}.parquet"
        )
        
        # 保存配置
        generator.save_config(f"data/processed/config_{config['name']}_server_001.json")
        
        logger.info(f"数据集 {config['name']} 生成完成，共 {len(df)} 个数据点")
    
    logger.info("所有数据集生成完成！")


if __name__ == "__main__":
    # 设置日志
    logger.add(sys.stderr, level="INFO")
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "multiple":
        logger.info("生成多个数据集...")
        generate_multiple_datasets()
    else:
        logger.info("生成快速测试数据...")
        generate_quick_test_data()
