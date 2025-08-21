"""
独立的简单数据生成器
不依赖其他模块，用于快速生成测试数据
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SimpleHardwareSpec:
    """简化的硬件规格配置"""
    cpu_cores: int = 8
    cpu_tdp: float = 95.0  # W
    gpu_tdp: float = 150.0  # W
    ambient_temp: float = 25.0  # °C


class SimpleDataGenerator:
    """简化的数据生成器"""
    
    def __init__(self, hardware_spec: SimpleHardwareSpec = None, node_id: str = "test_server_001"):
        self.hardware_spec = hardware_spec or SimpleHardwareSpec()
        self.node_id = node_id
        
        # 设置随机种子
        np.random.seed(42)
        random.seed(42)
        
        print(f"初始化数据生成器，节点ID: {node_id}")
    
    def _calculate_power_consumption(self, cpu_util: float, gpu_util: float, mem_util: float) -> float:
        """计算功耗消耗"""
        # 基础功耗
        base_power = 20.0
        
        # CPU功耗
        cpu_power = (cpu_util / 100.0) * self.hardware_spec.cpu_tdp
        
        # GPU功耗
        gpu_power = (gpu_util / 100.0) * self.hardware_spec.gpu_tdp
        
        # 内存功耗
        memory_power = (mem_util / 100.0) * 15.0
        
        total_power = base_power + cpu_power + gpu_power + memory_power
        
        # 添加随机噪声
        noise = np.random.normal(0, 2.0)
        total_power += noise
        
        return max(0, total_power)
    
    def _calculate_temperature(self, cpu_util: float, gpu_util: float) -> Tuple[float, float]:
        """计算CPU和GPU温度"""
        # CPU温度
        cpu_temp = self.hardware_spec.ambient_temp + 20 + (cpu_util / 100.0) * 40 + np.random.normal(0, 2)
        
        # GPU温度
        gpu_temp = self.hardware_spec.ambient_temp + 25 + (gpu_util / 100.0) * 50 + np.random.normal(0, 3)
        
        return max(self.hardware_spec.ambient_temp, cpu_temp), max(self.hardware_spec.ambient_temp, gpu_temp)
    
    def _get_workload_mode(self, timestamp: datetime) -> str:
        """根据时间戳确定工作负载模式"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # 夜间模式 (23:00-6:00)
        if hour >= 23 or hour <= 6:
            if np.random.random() < 0.9:
                return "idle"
        
        # 工作日模式 (周一到周五, 9:00-18:00)
        if weekday < 5 and 9 <= hour <= 18:
            if np.random.random() < 0.7:
                return np.random.choice(["office", "compute", "graphics", "mixed"], p=[0.4, 0.2, 0.2, 0.2])
        
        # 周末模式
        if weekday >= 5:
            if np.random.random() < 0.3:
                return np.random.choice(["office", "graphics", "mixed"], p=[0.5, 0.3, 0.2])
        
        # 默认空闲模式
        return "idle"
    
    def _generate_workload_metrics(self, mode: str) -> Dict[str, float]:
        """根据工作负载模式生成指标"""
        if mode == "idle":
            cpu_util = np.random.uniform(5.0, 15.0)
            gpu_util = np.random.uniform(0.0, 5.0)
            mem_util = np.random.uniform(20.0, 40.0)
        elif mode == "office":
            cpu_util = np.random.uniform(20.0, 50.0)
            gpu_util = np.random.uniform(5.0, 15.0)
            mem_util = np.random.uniform(40.0, 70.0)
        elif mode == "compute":
            cpu_util = np.random.uniform(60.0, 95.0)
            gpu_util = np.random.uniform(10.0, 30.0)
            mem_util = np.random.uniform(60.0, 90.0)
        elif mode == "graphics":
            cpu_util = np.random.uniform(30.0, 70.0)
            gpu_util = np.random.uniform(70.0, 95.0)
            mem_util = np.random.uniform(50.0, 85.0)
        elif mode == "mixed":
            cpu_util = np.random.uniform(50.0, 85.0)
            gpu_util = np.random.uniform(40.0, 80.0)
            mem_util = np.random.uniform(60.0, 90.0)
        else:
            cpu_util = np.random.uniform(10, 30)
            gpu_util = np.random.uniform(0, 20)
            mem_util = np.random.uniform(30, 60)
        
        # 添加时间连续性
        if hasattr(self, '_last_cpu_util'):
            cpu_util = 0.8 * self._last_cpu_util + 0.2 * cpu_util
            gpu_util = 0.8 * self._last_gpu_util + 0.2 * gpu_util
            mem_util = 0.9 * self._last_mem_util + 0.1 * mem_util
        
        self._last_cpu_util = cpu_util
        self._last_gpu_util = gpu_util
        self._last_mem_util = mem_util
        
        # 生成相关指标
        disk_io = cpu_util * 0.3 + np.random.uniform(0, 50)
        net_io = cpu_util * 0.2 + np.random.uniform(0, 30)
        gpu_mem = gpu_util * 0.8 + np.random.uniform(0, 20)
        
        # 电压和电流
        voltage = 12.0 + np.random.normal(0, 0.5)
        current = (cpu_util + gpu_util) / 100.0 * 10 + np.random.normal(0, 1)
        
        # 频率
        frequency = 3000 + (cpu_util / 100.0) * 1500 + np.random.normal(0, 100)
        
        # 缓存性能
        cache_hit_rate = max(0.5, 1.0 - (cpu_util / 100.0) * 0.3)
        cache_hit = cache_hit_rate * 100
        cache_miss = (1.0 - cache_hit_rate) * 100
        
        # 系统指标
        context_switch = cpu_util * 2 + np.random.uniform(0, 100)
        page_fault = mem_util * 0.1 + np.random.uniform(0, 10)
        interrupts = cpu_util * 1.5 + np.random.uniform(0, 50)
        load_avg = cpu_util / 100.0 * self.hardware_spec.cpu_cores + np.random.uniform(0, 2)
        process_count = 50 + cpu_util * 0.5 + np.random.uniform(0, 30)
        thread_count = process_count * 2 + np.random.uniform(0, 50)
        
        return {
            'cpu_util': min(100, max(0, cpu_util)),
            'mem_util': min(100, max(0, mem_util)),
            'disk_io': max(0, disk_io),
            'net_io': max(0, net_io),
            'gpu_util': min(100, max(0, gpu_util)),
            'gpu_mem': min(100, max(0, gpu_mem)),
            'voltage': max(0, voltage),
            'current': max(0, current),
            'frequency': max(0, frequency),
            'cache_miss': max(0, cache_miss),
            'cache_hit': max(0, cache_hit),
            'context_switch': max(0, context_switch),
            'page_fault': max(0, page_fault),
            'interrupts': max(0, interrupts),
            'load_avg': max(0, load_avg),
            'process_count': max(0, process_count),
            'thread_count': max(0, thread_count)
        }
    
    def generate_data_point(self, timestamp: datetime) -> Dict[str, any]:
        """生成单个数据点"""
        # 确定工作负载模式
        mode = self._get_workload_mode(timestamp)
        
        # 生成基础指标
        metrics = self._generate_workload_metrics(mode)
        
        # 计算温度和风扇转速
        temp_cpu, temp_gpu = self._calculate_temperature(metrics['cpu_util'], metrics['gpu_util'])
        
        # 计算风扇转速
        avg_temp = (temp_cpu + temp_gpu) / 2
        if avg_temp < 40:
            fan_speed = 800 + np.random.normal(0, 100)
        elif avg_temp < 60:
            fan_speed = 1500 + np.random.normal(0, 100)
        elif avg_temp < 80:
            fan_speed = 2200 + np.random.normal(0, 100)
        else:
            fan_speed = 3000 + np.random.normal(0, 100)
        fan_speed = max(0, min(3000, fan_speed))
        
        # 计算功耗
        power_consumption = self._calculate_power_consumption(
            metrics['cpu_util'], metrics['gpu_util'], metrics['mem_util']
        )
        
        # 创建数据点
        data_point = {
            'time': int(timestamp.timestamp()),
            'node_id': self.node_id,
            'power_consumption': float(power_consumption),
            'cpu_util': float(metrics['cpu_util']),
            'mem_util': float(metrics['mem_util']),
            'disk_io': float(metrics['disk_io']),
            'net_io': float(metrics['net_io']),
            'gpu_util': float(metrics['gpu_util']),
            'gpu_mem': float(metrics['gpu_mem']),
            'temp_cpu': float(temp_cpu),
            'temp_gpu': float(temp_gpu),
            'fan_speed': float(fan_speed),
            'voltage': float(metrics['voltage']),
            'current': float(metrics['current']),
            'frequency': float(metrics['frequency']),
            'cache_miss': float(metrics['cache_miss']),
            'cache_hit': float(metrics['cache_hit']),
            'context_switch': float(metrics['context_switch']),
            'page_fault': float(metrics['page_fault']),
            'interrupts': float(metrics['interrupts']),
            'load_avg': float(metrics['load_avg']),
            'process_count': float(metrics['process_count']),
            'thread_count': float(metrics['thread_count'])
        }
        
        return data_point
    
    def generate_dataset(self, 
                        start_time: datetime = None,
                        end_time: datetime = None,
                        interval_minutes: int = 5,
                        save_path: str = None) -> pd.DataFrame:
        """生成完整的数据集"""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.now()
        
        print(f"生成数据集: {start_time} 到 {end_time}, 间隔: {interval_minutes}分钟")
        
        # 生成时间序列
        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        # 生成数据点
        data_points = []
        for i, timestamp in enumerate(timestamps):
            if i % 100 == 0:
                print(f"已生成 {i}/{len(timestamps)} 个数据点")
            
            data_point = self.generate_data_point(timestamp)
            data_points.append(data_point)
        
        # 转换为DataFrame
        df = pd.DataFrame(data_points)
        
        # 添加时间列
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"数据集生成完成，共 {len(df)} 个数据点")
        
        # 保存数据
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_parquet(save_path, index=False)
            print(f"数据已保存到: {save_path}")
        
        return df


def main():
    """主函数"""
    # 创建数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 创建硬件配置
    hardware_spec = SimpleHardwareSpec(
        cpu_cores=16,
        cpu_tdp=125.0,
        gpu_tdp=150.0,
        ambient_temp=25.0
    )
    
    # 创建数据生成器
    generator = SimpleDataGenerator(
        hardware_spec=hardware_spec,
        node_id="test_server_001"
    )
    
    # 生成一周的数据
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    print("开始生成测试数据...")
    
    df = generator.generate_dataset(
        start_time=start_time,
        end_time=end_time,
        interval_minutes=5,
        save_path="data/processed/simulated_data_test.parquet"
    )
    
    print(f"测试数据生成完成！")
    print(f"数据点数量: {len(df)}")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"功耗范围: {df['power_consumption'].min():.2f}W 到 {df['power_consumption'].max():.2f}W")
    print(f"CPU使用率范围: {df['cpu_util'].min():.2f}% 到 {df['cpu_util'].max():.2f}%")
    print(f"GPU使用率范围: {df['gpu_util'].min():.2f}% 到 {df['gpu_util'].max():.2f}%")
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    main()
