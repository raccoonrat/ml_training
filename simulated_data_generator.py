"""
模拟数据生成器
基于硬件规格生成系统监控数据，用于测试机器学习训练项目
"""

import os
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from loguru import logger

from data_pipeline.feature_vector import FeatureVector, feature_vector_to_dict


@dataclass
class HardwareSpec:
    """硬件规格配置"""
    # CPU配置
    cpu_cores: int = 8
    cpu_base_freq: float = 3.2  # GHz
    cpu_max_freq: float = 4.5   # GHz
    cpu_tdp: float = 95.0       # W
    
    # GPU配置
    gpu_memory: int = 8192      # MB
    gpu_base_freq: float = 1500 # MHz
    gpu_max_freq: float = 1800  # MHz
    gpu_tdp: float = 150.0      # W
    
    # 内存配置
    memory_size: int = 32768    # MB
    memory_freq: float = 3200   # MHz
    
    # 存储配置
    storage_type: str = "SSD"   # SSD or HDD
    storage_speed: float = 3500 # MB/s
    
    # 网络配置
    network_speed: float = 1000 # Mbps
    
    # 散热配置
    fan_max_speed: float = 3000 # RPM
    ambient_temp: float = 25.0  # °C


@dataclass
class WorkloadPattern:
    """工作负载模式配置"""
    # 空闲模式
    idle_cpu_range: Tuple[float, float] = (5.0, 15.0)
    idle_gpu_range: Tuple[float, float] = (0.0, 5.0)
    idle_memory_range: Tuple[float, float] = (20.0, 40.0)
    
    # 办公模式
    office_cpu_range: Tuple[float, float] = (20.0, 50.0)
    office_gpu_range: Tuple[float, float] = (5.0, 15.0)
    office_memory_range: Tuple[float, float] = (40.0, 70.0)
    
    # 计算模式
    compute_cpu_range: Tuple[float, float] = (60.0, 95.0)
    compute_gpu_range: Tuple[float, float] = (10.0, 30.0)
    compute_memory_range: Tuple[float, float] = (60.0, 90.0)
    
    # 图形模式
    graphics_cpu_range: Tuple[float, float] = (30.0, 70.0)
    graphics_gpu_range: Tuple[float, float] = (70.0, 95.0)
    graphics_memory_range: Tuple[float, float] = (50.0, 85.0)
    
    # 混合模式
    mixed_cpu_range: Tuple[float, float] = (50.0, 85.0)
    mixed_gpu_range: Tuple[float, float] = (40.0, 80.0)
    mixed_memory_range: Tuple[float, float] = (60.0, 90.0)


@dataclass
class TimePattern:
    """时间模式配置"""
    # 工作日模式
    workday_start_hour: int = 9
    workday_end_hour: int = 18
    workday_high_load_prob: float = 0.7
    
    # 周末模式
    weekend_high_load_prob: float = 0.3
    
    # 夜间模式
    night_start_hour: int = 23
    night_end_hour: int = 6
    night_low_load_prob: float = 0.9
    
    # 峰值模式
    peak_load_prob: float = 0.05
    peak_duration_range: Tuple[int, int] = (5, 30)  # 分钟


class SimulatedDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, 
                 hardware_spec: HardwareSpec = None,
                 workload_pattern: WorkloadPattern = None,
                 time_pattern: TimePattern = None,
                 node_id: str = "simulated_node_001"):
        """
        初始化数据生成器
        
        Args:
            hardware_spec: 硬件规格配置
            workload_pattern: 工作负载模式配置
            time_pattern: 时间模式配置
            node_id: 节点ID
        """
        self.hardware_spec = hardware_spec or HardwareSpec()
        self.workload_pattern = workload_pattern or WorkloadPattern()
        self.time_pattern = time_pattern or TimePattern()
        self.node_id = node_id
        
        # 设置随机种子
        np.random.seed(42)
        random.seed(42)
        
        logger.info(f"初始化模拟数据生成器，节点ID: {node_id}")
    
    def _calculate_power_consumption(self, cpu_util: float, gpu_util: float, 
                                   mem_util: float, disk_io: float, 
                                   net_io: float, temp_cpu: float, temp_gpu: float) -> float:
        """
        计算功耗消耗
        
        Args:
            cpu_util: CPU使用率
            gpu_util: GPU使用率
            mem_util: 内存使用率
            disk_io: 磁盘I/O
            net_io: 网络I/O
            temp_cpu: CPU温度
            temp_gpu: GPU温度
            
        Returns:
            功耗消耗（瓦特）
        """
        # 基础功耗（待机状态）
        base_power = 20.0
        
        # CPU功耗（与使用率和温度相关）
        cpu_power = (cpu_util / 100.0) * self.hardware_spec.cpu_tdp * (1 + 0.01 * (temp_cpu - 25))
        
        # GPU功耗（与使用率和温度相关）
        gpu_power = (gpu_util / 100.0) * self.hardware_spec.gpu_tdp * (1 + 0.01 * (temp_gpu - 25))
        
        # 内存功耗
        memory_power = (mem_util / 100.0) * 15.0
        
        # 存储功耗
        storage_power = (disk_io / 1000.0) * 5.0
        
        # 网络功耗
        network_power = (net_io / 1000.0) * 2.0
        
        # 散热功耗（风扇）
        fan_power = max(0, (temp_cpu + temp_gpu) / 2 - 25) * 0.5
        
        total_power = base_power + cpu_power + gpu_power + memory_power + storage_power + network_power + fan_power
        
        # 添加随机噪声
        noise = np.random.normal(0, 2.0)
        total_power += noise
        
        return max(0, total_power)
    
    def _calculate_temperature(self, cpu_util: float, gpu_util: float, 
                             fan_speed: float, ambient_temp: float) -> Tuple[float, float]:
        """
        计算CPU和GPU温度
        
        Args:
            cpu_util: CPU使用率
            gpu_util: GPU使用率
            fan_speed: 风扇转速
            ambient_temp: 环境温度
            
        Returns:
            (CPU温度, GPU温度)
        """
        # CPU温度计算
        cpu_temp_base = ambient_temp + 20
        cpu_temp_load = (cpu_util / 100.0) * 40  # 负载增加的温度
        cpu_temp_cooling = max(0, (3000 - fan_speed) / 3000) * 10  # 散热效果
        cpu_temp = cpu_temp_base + cpu_temp_load - cpu_temp_cooling + np.random.normal(0, 2)
        
        # GPU温度计算
        gpu_temp_base = ambient_temp + 25
        gpu_temp_load = (gpu_util / 100.0) * 50  # GPU负载增加的温度
        gpu_temp_cooling = max(0, (3000 - fan_speed) / 3000) * 15  # 散热效果
        gpu_temp = gpu_temp_base + gpu_temp_load - gpu_temp_cooling + np.random.normal(0, 3)
        
        return max(ambient_temp, cpu_temp), max(ambient_temp, gpu_temp)
    
    def _calculate_fan_speed(self, temp_cpu: float, temp_gpu: float) -> float:
        """
        计算风扇转速
        
        Args:
            temp_cpu: CPU温度
            temp_gpu: GPU温度
            
        Returns:
            风扇转速（RPM）
        """
        avg_temp = (temp_cpu + temp_gpu) / 2
        if avg_temp < 40:
            base_speed = 800
        elif avg_temp < 60:
            base_speed = 1500
        elif avg_temp < 80:
            base_speed = 2200
        else:
            base_speed = 3000
        
        # 添加随机变化
        speed_variation = np.random.normal(0, 100)
        fan_speed = base_speed + speed_variation
        
        return max(0, min(self.hardware_spec.fan_max_speed, fan_speed))
    
    def _get_workload_mode(self, timestamp: datetime) -> str:
        """
        根据时间戳确定工作负载模式
        
        Args:
            timestamp: 时间戳
            
        Returns:
            工作负载模式
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # 夜间模式
        if hour >= self.time_pattern.night_start_hour or hour <= self.time_pattern.night_end_hour:
            if np.random.random() < self.time_pattern.night_low_load_prob:
                return "idle"
        
        # 工作日模式
        if weekday < 5:  # 周一到周五
            if self.time_pattern.workday_start_hour <= hour <= self.time_pattern.workday_end_hour:
                if np.random.random() < self.time_pattern.workday_high_load_prob:
                    return np.random.choice(["office", "compute", "graphics", "mixed"], p=[0.4, 0.2, 0.2, 0.2])
        
        # 周末模式
        else:
            if np.random.random() < self.time_pattern.weekend_high_load_prob:
                return np.random.choice(["office", "graphics", "mixed"], p=[0.5, 0.3, 0.2])
        
        # 默认空闲模式
        return "idle"
    
    def _generate_workload_metrics(self, mode: str) -> Dict[str, float]:
        """
        根据工作负载模式生成指标
        
        Args:
            mode: 工作负载模式
            
        Returns:
            指标字典
        """
        if mode == "idle":
            cpu_util = np.random.uniform(*self.workload_pattern.idle_cpu_range)
            gpu_util = np.random.uniform(*self.workload_pattern.idle_gpu_range)
            mem_util = np.random.uniform(*self.workload_pattern.idle_memory_range)
        elif mode == "office":
            cpu_util = np.random.uniform(*self.workload_pattern.office_cpu_range)
            gpu_util = np.random.uniform(*self.workload_pattern.office_gpu_range)
            mem_util = np.random.uniform(*self.workload_pattern.office_memory_range)
        elif mode == "compute":
            cpu_util = np.random.uniform(*self.workload_pattern.compute_cpu_range)
            gpu_util = np.random.uniform(*self.workload_pattern.compute_gpu_range)
            mem_util = np.random.uniform(*self.workload_pattern.compute_memory_range)
        elif mode == "graphics":
            cpu_util = np.random.uniform(*self.workload_pattern.graphics_cpu_range)
            gpu_util = np.random.uniform(*self.workload_pattern.graphics_gpu_range)
            mem_util = np.random.uniform(*self.workload_pattern.graphics_memory_range)
        elif mode == "mixed":
            cpu_util = np.random.uniform(*self.workload_pattern.mixed_cpu_range)
            gpu_util = np.random.uniform(*self.workload_pattern.mixed_gpu_range)
            mem_util = np.random.uniform(*self.workload_pattern.mixed_memory_range)
        else:
            cpu_util = np.random.uniform(10, 30)
            gpu_util = np.random.uniform(0, 20)
            mem_util = np.random.uniform(30, 60)
        
        # 添加时间连续性（避免突变）
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
        
        # 电压和电流（与功耗相关）
        voltage = 12.0 + np.random.normal(0, 0.5)
        current = (cpu_util + gpu_util) / 100.0 * 10 + np.random.normal(0, 1)
        
        # 频率（与使用率相关）
        cpu_freq = self.hardware_spec.cpu_base_freq + (cpu_util / 100.0) * (self.hardware_spec.cpu_max_freq - self.hardware_spec.cpu_base_freq)
        gpu_freq = self.hardware_spec.gpu_base_freq + (gpu_util / 100.0) * (self.hardware_spec.gpu_max_freq - self.hardware_spec.gpu_base_freq)
        frequency = (cpu_freq + gpu_freq) / 2
        
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
    
    def generate_data_point(self, timestamp: datetime) -> FeatureVector:
        """
        生成单个数据点
        
        Args:
            timestamp: 时间戳
            
        Returns:
            FeatureVector对象
        """
        # 确定工作负载模式
        mode = self._get_workload_mode(timestamp)
        
        # 生成基础指标
        metrics = self._generate_workload_metrics(mode)
        
        # 计算温度和风扇转速
        temp_cpu, temp_gpu = self._calculate_temperature(
            metrics['cpu_util'], metrics['gpu_util'], 
            self.hardware_spec.fan_max_speed / 2, 
            self.hardware_spec.ambient_temp
        )
        
        fan_speed = self._calculate_fan_speed(temp_cpu, temp_gpu)
        
        # 计算功耗
        power_consumption = self._calculate_power_consumption(
            metrics['cpu_util'], metrics['gpu_util'], metrics['mem_util'],
            metrics['disk_io'], metrics['net_io'], temp_cpu, temp_gpu
        )
        
        # 创建FeatureVector
        feature_vector = FeatureVector()
        feature_vector.time = int(timestamp.timestamp())
        feature_vector.node_id = self.node_id
        feature_vector.power_consumption = float(power_consumption)
        feature_vector.cpu_util = float(metrics['cpu_util'])
        feature_vector.mem_util = float(metrics['mem_util'])
        feature_vector.disk_io = float(metrics['disk_io'])
        feature_vector.net_io = float(metrics['net_io'])
        feature_vector.gpu_util = float(metrics['gpu_util'])
        feature_vector.gpu_mem = float(metrics['gpu_mem'])
        feature_vector.temp_cpu = float(temp_cpu)
        feature_vector.temp_gpu = float(temp_gpu)
        feature_vector.fan_speed = float(fan_speed)
        feature_vector.voltage = float(metrics['voltage'])
        feature_vector.current = float(metrics['current'])
        feature_vector.frequency = float(metrics['frequency'])
        feature_vector.cache_miss = float(metrics['cache_miss'])
        feature_vector.cache_hit = float(metrics['cache_hit'])
        feature_vector.context_switch = float(metrics['context_switch'])
        feature_vector.page_fault = float(metrics['page_fault'])
        feature_vector.interrupts = float(metrics['interrupts'])
        feature_vector.load_avg = float(metrics['load_avg'])
        feature_vector.process_count = float(metrics['process_count'])
        feature_vector.thread_count = float(metrics['thread_count'])
        
        return feature_vector
    
    def generate_dataset(self, 
                        start_time: datetime = None,
                        end_time: datetime = None,
                        interval_minutes: int = 5,
                        save_path: str = None) -> pd.DataFrame:
        """
        生成完整的数据集
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            interval_minutes: 数据间隔（分钟）
            save_path: 保存路径
            
        Returns:
            DataFrame格式的数据集
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.now()
        
        logger.info(f"生成数据集: {start_time} 到 {end_time}, 间隔: {interval_minutes}分钟")
        
        # 生成时间序列
        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        # 生成数据点
        data_points = []
        for i, timestamp in enumerate(timestamps):
            if i % 1000 == 0:
                logger.info(f"已生成 {i}/{len(timestamps)} 个数据点")
            
            feature_vector = self.generate_data_point(timestamp)
            data_dict = feature_vector_to_dict(feature_vector)
            data_points.append(data_dict)
        
        # 转换为DataFrame
        df = pd.DataFrame(data_points)
        
        # 添加时间列
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(f"数据集生成完成，共 {len(df)} 个数据点")
        
        # 保存数据
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_parquet(save_path, index=False)
            logger.info(f"数据已保存到: {save_path}")
        
        return df
    
    def save_config(self, config_path: str):
        """保存配置到文件"""
        config = {
            'hardware_spec': asdict(self.hardware_spec),
            'workload_pattern': asdict(self.workload_pattern),
            'time_pattern': asdict(self.time_pattern),
            'node_id': self.node_id
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {config_path}")


def create_sample_generators() -> List[SimulatedDataGenerator]:
    """创建示例数据生成器"""
    
    # 高性能服务器
    high_perf_spec = HardwareSpec(
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
    
    # 中等性能服务器
    mid_perf_spec = HardwareSpec(
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
    
    # 低功耗服务器
    low_power_spec = HardwareSpec(
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
    
    generators = [
        SimulatedDataGenerator(high_perf_spec, node_id="high_perf_server_001"),
        SimulatedDataGenerator(mid_perf_spec, node_id="mid_perf_server_001"),
        SimulatedDataGenerator(low_power_spec, node_id="low_power_server_001")
    ]
    
    return generators


if __name__ == "__main__":
    # 创建数据目录
    os.makedirs("data/processed", exist_ok=True)
    
    # 创建示例生成器
    generators = create_sample_generators()
    
    # 生成数据集
    start_time = datetime.now() - timedelta(days=14)  # 两周的数据
    end_time = datetime.now()
    
    for i, generator in enumerate(generators):
        logger.info(f"生成第 {i+1} 个数据集")
        
        # 生成数据
        df = generator.generate_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=5,
            save_path=f"data/processed/simulated_data_{generator.node_id}.parquet"
        )
        
        # 保存配置
        generator.save_config(f"data/processed/config_{generator.node_id}.json")
        
        logger.info(f"数据集 {generator.node_id} 生成完成，共 {len(df)} 个数据点")
    
    logger.info("所有模拟数据集生成完成！")
