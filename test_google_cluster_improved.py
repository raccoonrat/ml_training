"""
快速测试脚本：验证改进版本的Google Cluster Data加载器
"""

import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
from utils import setup_logging
setup_logging()

from google_cluster_data_loader_improved import GoogleClusterDataLoaderImproved


def test_google_cluster_loader():
    """测试Google Cluster Data加载器"""
    
    logger.info("开始测试改进版本的Google Cluster Data加载器...")
    
    try:
        # 1. 初始化加载器
        loader = GoogleClusterDataLoaderImproved()
        logger.info("✓ 加载器初始化成功")
        
        # 2. 加载数据
        cluster_data = loader.load_cluster_data(['task_usage'])  # 只测试一个数据类型
        logger.info("✓ 数据加载成功")
        
        # 3. 检查数据
        for data_type, df in cluster_data.items():
            logger.info(f"✓ {data_type}数据形状: {df.shape}")
            logger.info(f"✓ {data_type}列名: {list(df.columns)}")
        
        # 4. 合并数据
        merged_data = loader.merge_cluster_data(cluster_data)
        logger.info(f"✓ 数据合并成功，形状: {merged_data.shape}")
        
        # 5. 预处理数据
        features, targets = loader.preprocess_cluster_data_for_emshap(
            merged_data, 
            input_dim=32,  # 使用较小的维度进行测试
            target_column='cpu_rate'
        )
        logger.info(f"✓ 数据预处理成功，特征形状: {features.shape}, 目标形状: {targets.shape}")
        
        # 6. 创建EMSHAP数据集
        feature_tensor, mask_tensor, target_tensor = loader.create_emshap_dataset(
            features, targets, sequence_length=5
        )
        logger.info(f"✓ EMSHAP数据集创建成功，特征张量形状: {feature_tensor.shape}")
        
        # 7. 保存数据
        loader.save_processed_data(features, targets, "data/test_google_cluster_processed.parquet")
        logger.info("✓ 数据保存成功")
        
        # 8. 可视化数据
        loader.visualize_cluster_data(merged_data, "data/test_visualizations")
        logger.info("✓ 数据可视化成功")
        
        logger.info("🎉 所有测试通过！改进版本的Google Cluster Data加载器工作正常。")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False


def test_emshap_model_import():
    """测试EMSHAP模型导入"""
    
    logger.info("测试EMSHAP模型导入...")
    
    try:
        from models.emshap_enhanced import EMSHAPEnhanced
        from models.emshap_trainer import EMSHAPTrainer
        
        # 创建模型
        model = EMSHAPEnhanced(input_dim=32, gru_hidden_dim=64, context_dim=32)
        logger.info("✓ EMSHAP模型创建成功")
        
        # 创建训练器
        trainer = EMSHAPTrainer(model, learning_rate=1e-3)
        logger.info("✓ EMSHAP训练器创建成功")
        
        logger.info("🎉 EMSHAP模型导入测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"❌ EMSHAP模型导入测试失败: {e}")
        return False


def main():
    """主测试函数"""
    
    logger.info("=" * 60)
    logger.info("开始Google Cluster Data改进版本测试")
    logger.info("=" * 60)
    
    # 测试1: 数据加载器
    test1_passed = test_google_cluster_loader()
    
    # 测试2: EMSHAP模型导入
    test2_passed = test_emshap_model_import()
    
    # 总结
    logger.info("=" * 60)
    logger.info("测试总结:")
    logger.info(f"数据加载器测试: {'✓ 通过' if test1_passed else '❌ 失败'}")
    logger.info(f"EMSHAP模型导入测试: {'✓ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 所有测试通过！可以开始使用改进版本的Google Cluster Data训练EMSHAP模型。")
        logger.info("运行命令: python train_emshap_enhanced_cluster_improved.py")
    else:
        logger.error("❌ 部分测试失败，请检查错误信息。")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
