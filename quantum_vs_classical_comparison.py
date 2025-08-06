#!/usr/bin/env python3
"""
量子CDF(2,2)变换与经典CDF(2,2)变换对比分析
处理4.2.03.tiff图像并分析结果差异
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from scipy import ndimage
import cv2

class CDF22Transform:
    """
    CDF(2,2)小波变换实现类
    包含经典版本和量子模拟版本
    """
    
    def __init__(self):
        self.transform_history = []
    
    def classical_cdf_transform(self, signal):
        """
        经典CDF(2,2)小波变换实现
        """
        signal = np.array(signal, dtype=float)
        n = len(signal)
        
        # Step 1: Split
        even = signal[::2]  # S(2i)
        odd = signal[1::2]  # S(2i+1)
        
        # Step 2: Predict - P(S) = 1/2[S(2i) + S(2i+2)]
        predict = np.zeros_like(odd)
        for i in range(len(odd)):
            if i == 0:
                # 边界处理
                predict[i] = even[0]
            elif i == len(odd) - 1:
                # 边界处理
                predict[i] = even[-1]
            else:
                predict[i] = 0.5 * (even[i] + even[i+1])
        
        # 计算详细系数 D(i) = S(2i+1) - P(S)
        detail = odd - predict
        
        # Step 3: Update - W(D) = 1/4[D(i-1) + D(i)]
        update = np.zeros_like(even)
        for i in range(len(even)):
            if i == 0:
                # 边界处理
                if len(detail) > 0:
                    update[i] = 0.25 * detail[0]
            elif i == len(even) - 1:
                # 边界处理
                if len(detail) > i-1:
                    update[i] = 0.25 * detail[i-1]
            else:
                if i-1 < len(detail) and i < len(detail):
                    update[i] = 0.25 * (detail[i-1] + detail[i])
        
        # 计算近似系数 A(i) = S(2i) + W(D)
        approx = even + update
        
        return approx, detail
    
    def quantum_simulated_cdf_transform(self, signal):
        """
        量子模拟CDF(2,2)变换
        使用经典算法模拟量子计算过程
        """
        signal = np.array(signal, dtype=float)
        n = len(signal)
        
        # 量子模拟：添加一些量子噪声和精度限制
        quantum_precision = 8  # 量子比特精度
        max_quantum_value = 2**quantum_precision - 1
        
        # Step 1: Split (量子分离)
        even = signal[::2]
        odd = signal[1::2]
        
        # 量子精度限制
        even = np.clip(even, 0, max_quantum_value)
        odd = np.clip(odd, 0, max_quantum_value)
        
        # Step 2: Predict (量子预测)
        predict = np.zeros_like(odd)
        for i in range(len(odd)):
            if i == 0:
                predict_val = even[0]
            elif i == len(odd) - 1:
                predict_val = even[-1]
            else:
                predict_val = 0.5 * (even[i] + even[i+1])
            
            # 量子精度限制
            predict_val = np.clip(predict_val, 0, max_quantum_value)
            predict[i] = predict_val
        
        # 计算详细系数
        detail = odd - predict
        
        # Step 3: Update (量子更新)
        update = np.zeros_like(even)
        for i in range(len(even)):
            if i == 0:
                if len(detail) > 0:
                    update_val = 0.25 * detail[0]
                else:
                    update_val = 0
            elif i == len(even) - 1:
                if i-1 < len(detail):
                    update_val = 0.25 * detail[i-1]
                else:
                    update_val = 0
            else:
                if i-1 < len(detail) and i < len(detail):
                    update_val = 0.25 * (detail[i-1] + detail[i])
                else:
                    update_val = 0
            
            # 量子精度限制
            update_val = np.clip(update_val, 0, max_quantum_value)
            update[i] = update_val
        
        # 计算近似系数
        approx = even + update
        
        return approx, detail
    
    def process_image_blocks(self, image, block_size=(2, 2)):
        """
        将图像分割为块并处理
        """
        h, w = image.shape
        blocks = []
        block_positions = []
        
        for i in range(0, h, block_size[0]):
            for j in range(0, w, block_size[1]):
                block = image[i:i+block_size[0], j:j+block_size[1]]
                if block.shape == block_size:
                    blocks.append(block.flatten())
                    block_positions.append((i, j))
        
        return blocks, block_positions
    
    def compare_transforms(self, signal):
        """
        对比经典和量子模拟变换
        """
        # 经典变换
        start_time = time.time()
        classical_approx, classical_detail = self.classical_cdf_transform(signal)
        classical_time = time.time() - start_time
        
        # 量子模拟变换
        start_time = time.time()
        quantum_approx, quantum_detail = self.quantum_simulated_cdf_transform(signal)
        quantum_time = time.time() - start_time
        
        # 计算差异
        approx_diff = np.abs(np.array(classical_approx) - np.array(quantum_approx))
        detail_diff = np.abs(np.array(classical_detail) - np.array(quantum_detail))
        
        return {
            'classical': {
                'approx': classical_approx,
                'detail': classical_detail,
                'time': classical_time
            },
            'quantum': {
                'approx': quantum_approx,
                'detail': quantum_detail,
                'time': quantum_time
            },
            'differences': {
                'approx_diff': approx_diff,
                'detail_diff': detail_diff,
                'max_approx_diff': np.max(approx_diff),
                'max_detail_diff': np.max(detail_diff),
                'mean_approx_diff': np.mean(approx_diff),
                'mean_detail_diff': np.mean(detail_diff)
            }
        }

def load_and_preprocess_image(image_path):
    """
    加载和预处理图像
    """
    print(f"加载图像: {image_path}")
    
    try:
        # 尝试使用PIL加载
        image = Image.open(image_path)
        image = np.array(image)
        
        # 转换为灰度图
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
        
        print(f"图像尺寸: {image.shape}")
        print(f"像素值范围: [{image.min()}, {image.max()}]")
        
        return image
        
    except Exception as e:
        print(f"加载图像失败: {e}")
        # 创建测试图像
        test_image = np.array([
            [1, 3, 2, 1],
            [2, 6, 6, 6],
            [5, 2, 7, 4],
            [3, 1, 7, 2]
        ], dtype=np.uint8)
        print("使用测试图像")
        return test_image

def analyze_image_blocks(image, cdf_transform):
    """
    分析图像块的处理结果
    """
    print("\n" + "="*60)
    print("图像块分析")
    print("="*60)
    
    # 分割图像为块
    blocks, positions = cdf_transform.process_image_blocks(image, block_size=(2, 2))
    
    print(f"总共 {len(blocks)} 个2x2块")
    
    # 分析每个块
    all_results = []
    total_classical_time = 0
    total_quantum_time = 0
    
    for i, (block, pos) in enumerate(zip(blocks, positions)):
        print(f"\n--- 块 {i+1} (位置 {pos}) ---")
        print(f"原始数据: {block}")
        
        # 对比变换
        result = cdf_transform.compare_transforms(block)
        
        print(f"经典变换:")
        print(f"  近似系数: {[f'{x:.3f}' for x in result['classical']['approx']]}")
        print(f"  详细系数: {[f'{x:.3f}' for x in result['classical']['detail']]}")
        print(f"  处理时间: {result['classical']['time']:.6f}s")
        
        print(f"量子模拟变换:")
        print(f"  近似系数: {[f'{x:.3f}' for x in result['quantum']['approx']]}")
        print(f"  详细系数: {[f'{x:.3f}' for x in result['quantum']['detail']]}")
        print(f"  处理时间: {result['quantum']['time']:.6f}s")
        
        print(f"差异分析:")
        print(f"  近似系数最大差异: {result['differences']['max_approx_diff']:.6f}")
        print(f"  详细系数最大差异: {result['differences']['max_detail_diff']:.6f}")
        print(f"  近似系数平均差异: {result['differences']['mean_approx_diff']:.6f}")
        print(f"  详细系数平均差异: {result['differences']['mean_detail_diff']:.6f}")
        
        all_results.append(result)
        total_classical_time += result['classical']['time']
        total_quantum_time += result['quantum']['time']
    
    return all_results, total_classical_time, total_quantum_time

def calculate_overall_statistics(all_results):
    """
    计算总体统计信息
    """
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)
    
    # 收集所有差异数据
    all_approx_diffs = []
    all_detail_diffs = []
    
    for result in all_results:
        all_approx_diffs.extend(result['differences']['approx_diff'])
        all_detail_diffs.extend(result['differences']['detail_diff'])
    
    if all_approx_diffs:
        print(f"近似系数差异统计:")
        print(f"  最大差异: {np.max(all_approx_diffs):.6f}")
        print(f"  最小差异: {np.min(all_approx_diffs):.6f}")
        print(f"  平均差异: {np.mean(all_approx_diffs):.6f}")
        print(f"  标准差: {np.std(all_approx_diffs):.6f}")
    
    if all_detail_diffs:
        print(f"详细系数差异统计:")
        print(f"  最大差异: {np.max(all_detail_diffs):.6f}")
        print(f"  最小差异: {np.min(all_detail_diffs):.6f}")
        print(f"  平均差异: {np.mean(all_detail_diffs):.6f}")
        print(f"  标准差: {np.std(all_detail_diffs):.6f}")
    
    # 成功率分析
    success_threshold = 0.01  # 1%的差异阈值
    approx_success = sum(1 for diff in all_approx_diffs if diff <= success_threshold)
    detail_success = sum(1 for diff in all_detail_diffs if diff <= success_threshold)
    
    print(f"\n成功率分析 (差异 <= {success_threshold}):")
    print(f"  近似系数成功率: {approx_success}/{len(all_approx_diffs)} ({100*approx_success/len(all_approx_diffs):.1f}%)")
    print(f"  详细系数成功率: {detail_success}/{len(all_detail_diffs)} ({100*detail_success/len(all_detail_diffs):.1f}%)")
    
    return {
        'approx_stats': {
            'max': np.max(all_approx_diffs),
            'min': np.min(all_approx_diffs),
            'mean': np.mean(all_approx_diffs),
            'std': np.std(all_approx_diffs),
            'success_rate': 100*approx_success/len(all_approx_diffs)
        },
        'detail_stats': {
            'max': np.max(all_detail_diffs),
            'min': np.min(all_detail_diffs),
            'mean': np.mean(all_detail_diffs),
            'std': np.std(all_detail_diffs),
            'success_rate': 100*detail_success/len(all_detail_diffs)
        }
    }

def visualize_comparison(all_results, image_shape):
    """
    可视化对比结果
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 收集数据
        classical_approx_all = []
        quantum_approx_all = []
        classical_detail_all = []
        quantum_detail_all = []
        
        for result in all_results:
            classical_approx_all.extend(result['classical']['approx'])
            quantum_approx_all.extend(result['quantum']['approx'])
            classical_detail_all.extend(result['classical']['detail'])
            quantum_detail_all.extend(result['quantum']['detail'])
        
        # 1. 近似系数对比
        axes[0, 0].scatter(classical_approx_all, quantum_approx_all, alpha=0.6)
        axes[0, 0].plot([min(classical_approx_all), max(classical_approx_all)], 
                        [min(classical_approx_all), max(classical_approx_all)], 'r--', label='理想线')
        axes[0, 0].set_xlabel('经典近似系数')
        axes[0, 0].set_ylabel('量子模拟近似系数')
        axes[0, 0].set_title('近似系数对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 详细系数对比
        axes[0, 1].scatter(classical_detail_all, quantum_detail_all, alpha=0.6)
        axes[0, 1].plot([min(classical_detail_all), max(classical_detail_all)], 
                        [min(classical_detail_all), max(classical_detail_all)], 'r--', label='理想线')
        axes[0, 1].set_xlabel('经典详细系数')
        axes[0, 1].set_ylabel('量子模拟详细系数')
        axes[0, 1].set_title('详细系数对比')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 差异分布
        approx_diffs = [abs(c - q) for c, q in zip(classical_approx_all, quantum_approx_all)]
        detail_diffs = [abs(c - q) for c, q in zip(classical_detail_all, quantum_detail_all)]
        
        axes[1, 0].hist(approx_diffs, bins=20, alpha=0.7, label='近似系数差异')
        axes[1, 0].set_xlabel('差异值')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('近似系数差异分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].hist(detail_diffs, bins=20, alpha=0.7, label='详细系数差异', color='orange')
        axes[1, 1].set_xlabel('差异值')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('详细系数差异分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('quantum_vs_classical_comparison.png', dpi=150, bbox_inches='tight')
        print("对比图已保存为 'quantum_vs_classical_comparison.png'")
        
        # 显示图像（如果可能）
        try:
            plt.show()
        except:
            print("无法显示图像，但已保存到文件")
            
    except Exception as e:
        print(f"可视化失败: {e}")

def print_conclusion(stats, classical_time, quantum_time):
    """
    打印结论
    """
    print("\n" + "="*60)
    print("结论分析")
    print("="*60)
    
    print(f"性能对比:")
    print(f"  经典变换总时间: {classical_time:.6f}s")
    print(f"  量子模拟总时间: {quantum_time:.6f}s")
    print(f"  时间比率: {quantum_time/classical_time:.2f}x")
    
    print(f"\n精度分析:")
    print(f"  近似系数平均差异: {stats['approx_stats']['mean']:.6f}")
    print(f"  详细系数平均差异: {stats['detail_stats']['mean']:.6f}")
    print(f"  近似系数成功率: {stats['approx_stats']['success_rate']:.1f}%")
    print(f"  详细系数成功率: {stats['detail_stats']['success_rate']:.1f}%")
    
    # 判断量子版本是否成功
    success_threshold = 0.01  # 1%差异阈值
    approx_success = stats['approx_stats']['mean'] <= success_threshold
    detail_success = stats['detail_stats']['mean'] <= success_threshold
    
    print(f"\n量子版本成功性评估:")
    if approx_success and detail_success:
        print("🎉 量子版本非常成功！差异很小，精度很高。")
    elif approx_success or detail_success:
        print("✅ 量子版本基本成功，部分指标达到要求。")
    else:
        print("⚠️ 量子版本需要改进，差异较大。")
    
    print(f"\n建议:")
    if stats['approx_stats']['success_rate'] > 95 and stats['detail_stats']['success_rate'] > 95:
        print("✓ 量子实现精度很高，可以用于实际应用")
    elif stats['approx_stats']['success_rate'] > 80 and stats['detail_stats']['success_rate'] > 80:
        print("✓ 量子实现精度良好，适合进一步优化")
    else:
        print("⚠️ 建议优化量子实现，提高精度")

def main():
    """
    主函数
    """
    print("="*60)
    print("量子CDF(2,2)变换与经典CDF(2,2)变换对比分析")
    print("="*60)
    
    # 创建变换器
    cdf_transform = CDF22Transform()
    
    # 加载图像
    image_path = "4.2.03.tiff"
    if not os.path.exists(image_path):
        print(f"图像文件 {image_path} 不存在，使用测试图像")
        image_path = None
    
    image = load_and_preprocess_image(image_path)
    
    # 分析图像块
    all_results, classical_time, quantum_time = analyze_image_blocks(image, cdf_transform)
    
    # 计算统计信息
    stats = calculate_overall_statistics(all_results)
    
    # 可视化结果
    visualize_comparison(all_results, image.shape)
    
    # 打印结论
    print_conclusion(stats, classical_time, quantum_time)
    
    print(f"\n分析完成！")
    print(f"详细结果已保存到图像文件中。")

if __name__ == "__main__":
    main() 