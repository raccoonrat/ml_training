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
    """Test Google Cluster Data loader"""
    
    logger.info("Starting test of improved Google Cluster Data loader...")
    
    try:
        # 1. Initialize loader
        loader = GoogleClusterDataLoaderImproved()
        logger.info("✓ Loader initialized successfully")
        
        # 2. Load data
        cluster_data = loader.load_cluster_data(['task_usage'])  # Only test one data type
        logger.info("✓ Data loaded successfully")
        
        # 3. Check data
        for data_type, df in cluster_data.items():
            logger.info(f"✓ {data_type} data shape: {df.shape}")
            logger.info(f"✓ {data_type} column names: {list(df.columns)}")
        
        # 4. Merge data
        merged_data = loader.merge_cluster_data(cluster_data)
        logger.info(f"✓ Data merged successfully, shape: {merged_data.shape}")
        
        # 5. Preprocess data
        features, targets = loader.preprocess_cluster_data_for_emshap(
            merged_data, 
            input_dim=32,  # Use smaller dimension for testing
            target_column='cpu_rate'
        )
        logger.info(f"✓ Data preprocessing successful, feature shape: {features.shape}, target shape: {targets.shape}")
        
        # 6. Create EMSHAP dataset
        feature_tensor, mask_tensor, target_tensor = loader.create_emshap_dataset(
            features, targets, sequence_length=5
        )
        logger.info(f"✓ EMSHAP dataset created successfully, feature tensor shape: {feature_tensor.shape}")
        
        # 7. Save data
        loader.save_processed_data(features, targets, "data/test_google_cluster_processed.parquet")
        logger.info("✓ Data saved successfully")
        
        # 8. Visualize data
        loader.visualize_cluster_data(merged_data, "data/test_visualizations")
        logger.info("✓ Data visualization successful")
        
        logger.info("🎉 All tests passed! Improved Google Cluster Data loader is working properly.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False


def test_emshap_model_import():
    """Test EMSHAP model import"""
    
    logger.info("Testing EMSHAP model import...")
    
    try:
        from models.emshap_enhanced import EMSHAPEnhanced
        from models.emshap_trainer import EMSHAPTrainer
        
        # Create model
        model = EMSHAPEnhanced(input_dim=32, gru_hidden_dim=64, context_dim=32)
        logger.info("✓ EMSHAP model created successfully")
        
        # Create trainer
        trainer = EMSHAPTrainer(model, learning_rate=1e-3)
        logger.info("✓ EMSHAP trainer created successfully")
        
        logger.info("🎉 EMSHAP model import test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ EMSHAP模型导入测试失败: {e}")
        return False


def main():
    """Main test function"""
    
    logger.info("=" * 60)
    logger.info("Starting Google Cluster Data improved version test")
    logger.info("=" * 60)
    
    # Test 1: Data loader
    test1_passed = test_google_cluster_loader()
    
    # Test 2: EMSHAP model import
    test2_passed = test_emshap_model_import()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"Data loader test: {'✓ Passed' if test1_passed else '❌ Failed'}")
    logger.info(f"EMSHAP model import test: {'✓ Passed' if test2_passed else '❌ Failed'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! You can now start using the improved Google Cluster Data to train EMSHAP models.")
        logger.info("Run command: python train_emshap_enhanced_cluster_improved.py")
    else:
        logger.error("❌ Some tests failed, please check error messages.")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
