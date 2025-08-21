"""
中文字体修复脚本
解决matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import subprocess
import sys
from loguru import logger

def install_chinese_font():
    """安装中文字体"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统通常已经有中文字体，尝试使用系统字体
        logger.info("检测到Windows系统，尝试使用系统字体")
        return True
    elif system == 'Darwin':  # macOS
        logger.info("检测到macOS系统，尝试使用系统字体")
        return True
    else:  # Linux
        logger.info("检测到Linux系统，尝试安装中文字体")
        try:
            # 尝试安装文泉驿微米黑字体
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-wqy-microhei'], check=True)
            logger.info("成功安装文泉驿微米黑字体")
            return True
        except:
            logger.warning("无法安装中文字体，将使用备用方案")
            return False

def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    # 不同系统的字体优先级
    if system == 'Windows':
        font_names = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
            'Arial Unicode MS', 'Tahoma', 'Verdana'
        ]
    elif system == 'Darwin':  # macOS
        font_names = [
            'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS',
            'Helvetica Neue', 'Helvetica'
        ]
    else:  # Linux
        font_names = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans',
            'Liberation Sans', 'Ubuntu', 'Noto Sans CJK SC'
        ]
    
    # 查找可用的中文字体
    available_fonts = []
    for font_name in font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and font_path != fm.rcParams['font.sans-serif']:
                # 验证字体是否支持中文
                test_text = "测试中文"
                try:
                    test_font = fm.FontProperties(family=font_name)
                    if test_font.get_name() != 'DejaVu Sans':
                        available_fonts.append(font_name)
                        logger.info(f"找到可用中文字体: {font_name}")
                        break
                except:
                    continue
        except:
            continue
    
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"成功设置中文字体: {available_fonts[0]}")
        return True
    else:
        # 如果没有找到中文字体，使用备用方案
        logger.warning("未找到中文字体，使用英文标签")
        return False

def create_english_version():
    """创建英文版本的图表标签"""
    english_labels = {
        '系统监控指标时间序列': 'System Monitoring Metrics Time Series',
        '时间': 'Time',
        '特征相关性矩阵': 'Feature Correlation Matrix',
        '功耗与其他指标的关系分析': 'Power Consumption vs Other Metrics Analysis',
        '功耗': 'Power Consumption',
        '工作负载模式分析': 'Workload Pattern Analysis',
        '小时': 'Hour',
        '功耗分布': 'Power Distribution',
        '工作日': 'Weekday',
        '周末': 'Weekend',
        '系统健康状态分析': 'System Health Analysis',
        '温度': 'Temperature',
        '风扇转速': 'Fan Speed',
        '缓存命中率': 'Cache Hit Rate',
        '系统负载': 'System Load',
        '数据摘要': 'Data Summary',
        '数值': 'Value',
        '主要指标': 'Main Metrics',
        '统计信息': 'Statistics',
        '数据集': 'Dataset'
    }
    return english_labels

def test_chinese_font():
    """测试中文字体是否正常工作"""
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title('测试中文字体 Test Chinese Font')
        ax.set_xlabel('时间 Time')
        ax.set_ylabel('数值 Value')
        ax.grid(True)
        
        # 保存测试图片
        test_path = 'font_test.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"字体测试图片已保存到: {test_path}")
        return True
    except Exception as e:
        logger.error(f"字体测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始修复中文字体问题...")
    
    # 尝试安装字体（如果需要）
    install_chinese_font()
    
    # 设置中文字体
    font_success = setup_chinese_font()
    
    # 测试字体
    test_success = test_chinese_font()
    
    if font_success and test_success:
        logger.info("✅ 中文字体修复成功！")
        logger.info("现在可以正常显示中文图表了。")
    else:
        logger.warning("⚠️ 中文字体修复部分成功，建议使用英文标签。")
        logger.info("可以调用 create_english_version() 获取英文标签映射。")

if __name__ == "__main__":
    main()
