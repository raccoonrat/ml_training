"""
测试数据生成器
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulated_data_generator import SimulatedDataGenerator, HardwareSpec


def test_data_generator():
    """测试数据生成器"""
    
    logger.info("开始测试数据生成器...")
    
    # 创建简单的硬件配置
    hardware_spec = HardwareSpec(
        cpu_cores=8,
        cpu_base_freq=3.0,
        cpu_max_freq=4.0,
        cpu_tdp=95.0,
        gpu_memory=4096,
        gpu_base_freq=1200,
        gpu_max_freq=1600,
        gpu_tdp=75.0,
        memory_size=16384,
        memory_freq=2666,
        storage_type="SSD",
        storage_speed=2000,
        network_speed=1000,
        fan_max_speed=2000,
        ambient_temp=25.0
    )
    
    # 创建数据生成器
    generator = SimulatedDataGenerator(
        hardware_spec=hardware_spec,
        node_id="test_server_001"
    )
    
    # 生成少量数据点进行测试
    start_time = datetime.now() - timedelta(hours=2)  # 只生成2小时的数据
    end_time = datetime.now()
    
    logger.info(f"生成测试数据: {start_time} 到 {end_time}")
    
    # 生成数据
    df = generator.generate_dataset(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=10,  # 每10分钟一个数据点
        save_path="data/processed/test_data.parquet"
    )
    
    logger.info(f"测试数据生成完成！")
    logger.info(f"数据点数量: {len(df)}")
    logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    logger.info(f"功耗范围: {df['power_consumption'].min():.2f}W 到 {df['power_consumption'].max():.2f}W")
    logger.info(f"CPU使用率范围: {df['cpu_util'].min():.2f}% 到 {df['cpu_util'].max():.2f}%")
    logger.info(f"GPU使用率范围: {df['gpu_util'].min():.2f}% 到 {df['gpu_util'].max():.2f}%")
    
    # 显示前几行数据
    logger.info("前5行数据:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    # 设置日志
    logger.add(sys.stderr, level="INFO")
    
    # 创建数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 运行测试
    test_data_generator()
