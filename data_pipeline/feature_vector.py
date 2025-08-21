"""
简化的FeatureVector类
避免使用protobuf，直接使用Python数据类
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FeatureVector:
    """特征向量数据类"""
    time: int = 0
    node_id: str = ""
    power_consumption: float = 0.0
    cpu_util: float = 0.0
    mem_util: float = 0.0
    disk_io: float = 0.0
    net_io: float = 0.0
    gpu_util: float = 0.0
    gpu_mem: float = 0.0
    temp_cpu: float = 0.0
    temp_gpu: float = 0.0
    fan_speed: float = 0.0
    voltage: float = 0.0
    current: float = 0.0
    frequency: float = 0.0
    cache_miss: float = 0.0
    cache_hit: float = 0.0
    context_switch: float = 0.0
    page_fault: float = 0.0
    interrupts: float = 0.0
    load_avg: float = 0.0
    process_count: float = 0.0
    thread_count: float = 0.0


def feature_vector_to_dict(feature_vector: FeatureVector) -> Dict[str, Any]:
    """
    将FeatureVector转换为字典
    
    Args:
        feature_vector: FeatureVector对象
        
    Returns:
        包含所有字段的字典
    """
    return {
        'time': feature_vector.time,
        'node_id': feature_vector.node_id,
        'power_consumption': feature_vector.power_consumption,
        'cpu_util': feature_vector.cpu_util,
        'mem_util': feature_vector.mem_util,
        'disk_io': feature_vector.disk_io,
        'net_io': feature_vector.net_io,
        'gpu_util': feature_vector.gpu_util,
        'gpu_mem': feature_vector.gpu_mem,
        'temp_cpu': feature_vector.temp_cpu,
        'temp_gpu': feature_vector.temp_gpu,
        'fan_speed': feature_vector.fan_speed,
        'voltage': feature_vector.voltage,
        'current': feature_vector.current,
        'frequency': feature_vector.frequency,
        'cache_miss': feature_vector.cache_miss,
        'cache_hit': feature_vector.cache_hit,
        'context_switch': feature_vector.context_switch,
        'page_fault': feature_vector.page_fault,
        'interrupts': feature_vector.interrupts,
        'load_avg': feature_vector.load_avg,
        'process_count': feature_vector.process_count,
        'thread_count': feature_vector.thread_count,
    }


def dict_to_feature_vector(data_dict: Dict[str, Any]) -> FeatureVector:
    """
    将字典转换为FeatureVector
    
    Args:
        data_dict: 包含字段的字典
        
    Returns:
        FeatureVector对象
    """
    return FeatureVector(**data_dict)


# 定义特征列名
FEATURE_COLUMNS = [
    'cpu_util', 'mem_util', 'disk_io', 'net_io', 'gpu_util', 'gpu_mem',
    'temp_cpu', 'temp_gpu', 'fan_speed', 'voltage', 'current', 'frequency',
    'cache_miss', 'cache_hit', 'context_switch', 'page_fault', 'interrupts',
    'load_avg', 'process_count', 'thread_count'
]

# 目标列名
TARGET_COLUMN = 'power_consumption'

# 元数据列名
METADATA_COLUMNS = ['time', 'node_id']
