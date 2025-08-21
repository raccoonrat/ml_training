"""
项目设置测试脚本
验证项目结构和依赖是否正确配置
"""

import os
import sys
import importlib
from pathlib import Path

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    modules_to_test = [
        'torch',
        'numpy',
        'pandas',
        'sklearn',
        'loguru',
        'confluent_kafka',
        'protobuf',
        'onnx',
        'onnxruntime'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n警告: 以下模块导入失败: {failed_imports}")
        print("请运行: pip install -r requirements.txt")
    else:
        print("\n✓ 所有依赖模块导入成功")
    
    return len(failed_imports) == 0

def test_project_modules():
    """测试项目内部模块"""
    print("\n测试项目模块...")
    
    project_modules = [
        'config',
        'utils',
        'models',
        'models.emshap',
        'models.power_predictor',
        'data_pipeline',
        'data_pipeline.consumer',
        'data_pipeline.feature_vector_pb2'
    ]
    
    failed_modules = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n警告: 以下项目模块导入失败: {failed_modules}")
    else:
        print("\n✓ 所有项目模块导入成功")
    
    return len(failed_modules) == 0

def test_directories():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    required_dirs = [
        'data',
        'data/processed',
        'models',
        'logs',
        'checkpoints',
        'evaluation_results'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (不存在)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n创建缺失的目录...")
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ 创建目录: {dir_path}")
    
    return len(missing_dirs) == 0

def test_config():
    """测试配置加载"""
    print("\n测试配置加载...")
    
    try:
        from config import config
        print("✓ 配置加载成功")
        
        # 测试配置属性
        print(f"  - Kafka服务器: {config.kafka.bootstrap_servers}")
        print(f"  - 批次大小: {config.model.batch_size}")
        print(f"  - 学习率: {config.model.learning_rate}")
        print(f"  - 设备: {config.training.device}")
        
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False

def test_models():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        import torch
        from models.emshap import create_emshap_model
        from models.power_predictor import create_power_predictor
        
        # 测试EMSHAP模型
        emshap_model = create_emshap_model(input_dim=64)
        print(f"✓ EMSHAP模型创建成功，参数数量: {sum(p.numel() for p in emshap_model.parameters())}")
        
        # 测试功耗预测模型
        power_model = create_power_predictor(input_dim=64)
        print(f"✓ 功耗预测模型创建成功，参数数量: {sum(p.numel() for p in power_model.parameters())}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_utils():
    """测试工具函数"""
    print("\n测试工具函数...")
    
    try:
        from utils import get_device, set_seed, count_parameters
        
        # 测试设备检测
        device = get_device()
        print(f"✓ 设备检测: {device}")
        
        # 测试随机种子设置
        set_seed(42)
        print("✓ 随机种子设置成功")
        
        # 测试参数计数
        import torch.nn as nn
        test_model = nn.Linear(10, 1)
        param_count = count_parameters(test_model)
        print(f"✓ 参数计数: {param_count}")
        
        return True
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("EMSHAP模型训练项目 - 设置测试")
    print("=" * 50)
    
    tests = [
        ("依赖模块导入", test_imports),
        ("项目模块导入", test_project_modules),
        ("目录结构", test_directories),
        ("配置加载", test_config),
        ("模型创建", test_models),
        ("工具函数", test_utils)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！项目设置正确。")
        print("\n下一步:")
        print("1. 配置Kafka连接信息")
        print("2. 运行数据管道: python data_pipeline/consumer.py")
        print("3. 训练模型: python train_power_model.py")
        print("4. 评估模型: python evaluate.py")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        print("\n常见解决方案:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 检查Python版本 (需要3.9+)")
        print("3. 检查项目文件是否完整")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
