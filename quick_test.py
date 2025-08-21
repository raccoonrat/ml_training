"""
快速测试脚本
生成少量数据用于测试
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_quick_data():
    """生成快速测试数据"""
    
    print("开始生成快速测试数据...")
    
    # 创建数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 生成时间序列（1小时，每5分钟一个数据点）
    start_time = datetime.now() - timedelta(hours=1)
    timestamps = []
    current_time = start_time
    while current_time <= datetime.now():
        timestamps.append(current_time)
        current_time += timedelta(minutes=5)
    
    print(f"生成 {len(timestamps)} 个数据点")
    
    # 生成数据
    data_points = []
    for i, timestamp in enumerate(timestamps):
        # 模拟CPU使用率（随时间变化）
        hour = timestamp.hour
        if 9 <= hour <= 18:  # 工作时间
            cpu_util = np.random.uniform(30, 80)
        else:  # 非工作时间
            cpu_util = np.random.uniform(5, 30)
        
        # 模拟GPU使用率
        gpu_util = np.random.uniform(0, 50)
        
        # 模拟内存使用率
        mem_util = np.random.uniform(40, 80)
        
        # 计算功耗（简化模型）
        power_consumption = 20 + (cpu_util / 100) * 95 + (gpu_util / 100) * 150 + np.random.normal(0, 2)
        
        # 计算温度
        temp_cpu = 25 + (cpu_util / 100) * 40 + np.random.normal(0, 2)
        temp_gpu = 30 + (gpu_util / 100) * 50 + np.random.normal(0, 3)
        
        # 创建数据点
        data_point = {
            'time': int(timestamp.timestamp()),
            'node_id': 'test_server_001',
            'power_consumption': max(0, power_consumption),
            'cpu_util': min(100, max(0, cpu_util)),
            'gpu_util': min(100, max(0, gpu_util)),
            'mem_util': min(100, max(0, mem_util)),
            'temp_cpu': max(20, temp_cpu),
            'temp_gpu': max(25, temp_gpu),
            'disk_io': np.random.uniform(0, 100),
            'net_io': np.random.uniform(0, 50),
            'gpu_mem': gpu_util * 0.8 + np.random.uniform(0, 20),
            'fan_speed': 800 + (temp_cpu + temp_gpu) / 2 * 20 + np.random.normal(0, 100),
            'voltage': 12.0 + np.random.normal(0, 0.5),
            'current': (cpu_util + gpu_util) / 100 * 10 + np.random.normal(0, 1),
            'frequency': 3000 + cpu_util * 15 + np.random.normal(0, 100),
            'cache_miss': np.random.uniform(0, 30),
            'cache_hit': np.random.uniform(70, 100),
            'context_switch': cpu_util * 2 + np.random.uniform(0, 100),
            'page_fault': mem_util * 0.1 + np.random.uniform(0, 10),
            'interrupts': cpu_util * 1.5 + np.random.uniform(0, 50),
            'load_avg': cpu_util / 100 * 8 + np.random.uniform(0, 2),
            'process_count': 50 + cpu_util * 0.5 + np.random.uniform(0, 30),
            'thread_count': 100 + cpu_util * 1.0 + np.random.uniform(0, 50)
        }
        
        data_points.append(data_point)
    
    # 转换为DataFrame
    df = pd.DataFrame(data_points)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    
    # 保存数据
    save_path = "data/processed/quick_test_data.parquet"
    df.to_parquet(save_path, index=False)
    
    print(f"数据生成完成！")
    print(f"数据点数量: {len(df)}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"功耗范围: {df['power_consumption'].min():.2f}W 到 {df['power_consumption'].max():.2f}W")
    print(f"CPU使用率范围: {df['cpu_util'].min():.2f}% 到 {df['cpu_util'].max():.2f}%")
    print(f"数据已保存到: {save_path}")
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df[['timestamp', 'power_consumption', 'cpu_util', 'gpu_util', 'mem_util']].head())
    
    return df

if __name__ == "__main__":
    generate_quick_data()
