#!/usr/bin/env python3
"""
简化的量子与经典CDF变换对比分析
适用于conda环境运行
"""

import numpy as np
import os
import time

def classical_cdf_transform(signal):
    """经典CDF(2,2)小波变换"""
    signal = np.array(signal, dtype=float)
    
    # Step 1: Split
    even = signal[::2]  # S(2i)
    odd = signal[1::2]  # S(2i+1)
    
    # Step 2: Predict - P(S) = 1/2[S(2i) + S(2i+2)]
    predict = np.zeros_like(odd)
    for i in range(len(odd)):
        if i == 0:
            predict[i] = even[0]
        elif i == len(odd) - 1:
            predict[i] = even[-1]
        else:
            predict[i] = 0.5 * (even[i] + even[i+1])
    
    # 计算详细系数 D(i) = S(2i+1) - P(S)
    detail = odd - predict
    
    # Step 3: Update - W(D) = 1/4[D(i-1) + D(i)]
    update = np.zeros_like(even)
    for i in range(len(even)):
        if i == 0:
            if len(detail) > 0:
                update[i] = 0.25 * detail[0]
        elif i == len(even) - 1:
            if len(detail) > i-1:
                update[i] = 0.25 * detail[i-1]
        else:
            if i-1 < len(detail) and i < len(detail):
                update[i] = 0.25 * (detail[i-1] + detail[i])
    
    # 计算近似系数 A(i) = S(2i) + W(D)
    approx = even + update
    
    return approx, detail

def quantum_simulated_cdf_transform(signal):
    """量子模拟CDF(2,2)变换"""
    signal = np.array(signal, dtype=float)
    
    # 量子精度限制
    quantum_precision = 8
    max_quantum_value = 2**quantum_precision - 1
    
    # Step 1: Split
    even = signal[::2]
    odd = signal[1::2]
    
    # 量子精度限制
    even = np.clip(even, 0, max_quantum_value)
    odd = np.clip(odd, 0, max_quantum_value)
    
    # Step 2: Predict
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
    
    # Step 3: Update
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

def load_image(image_path):
    """加载图像"""
    print(f"尝试加载图像: {image_path}")
    
    try:
        # 尝试使用PIL
        from PIL import Image
        image = Image.open(image_path)
        image = np.array(image)
        
        # 转换为灰度图
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.uint8)
        
        print(f"✓ 成功加载图像: {image.shape}")
        return image
        
    except ImportError:
        print("PIL不可用，尝试其他方法...")
        
    try:
        # 尝试使用cv2
        import cv2
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            print(f"✓ 成功加载图像: {image.shape}")
            return image
        
    except ImportError:
        print("OpenCV不可用...")
    
    # 创建测试图像
    print("使用测试图像")
    test_image = np.array([
        [1, 3, 2, 1],
        [2, 6, 6, 6],
        [5, 2, 7, 4],
        [3, 1, 7, 2]
    ], dtype=np.uint8)
    return test_image

def process_image_blocks(image):
    """处理图像块"""
    h, w = image.shape
    blocks = []
    positions = []
    
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = image[i:i+2, j:j+2]
            if block.shape == (2, 2):
                blocks.append(block.flatten())
                positions.append((i, j))
    
    return blocks, positions

def compare_transforms(signal):
    """对比经典和量子变换"""
    # 经典变换
    start_time = time.time()
    classical_approx, classical_detail = classical_cdf_transform(signal)
    classical_time = time.time() - start_time
    
    # 量子模拟变换
    start_time = time.time()
    quantum_approx, quantum_detail = quantum_simulated_cdf_transform(signal)
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

def analyze_results(all_results):
    """分析结果"""
    print("\n" + "="*60)
    print("结果分析")
    print("="*60)
    
    # 收集所有差异数据
    all_approx_diffs = []
    all_detail_diffs = []
    total_classical_time = 0
    total_quantum_time = 0
    
    for result in all_results:
        all_approx_diffs.extend(result['differences']['approx_diff'])
        all_detail_diffs.extend(result['differences']['detail_diff'])
        total_classical_time += result['classical']['time']
        total_quantum_time += result['quantum']['time']
    
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
    success_threshold = 0.01
    approx_success = sum(1 for diff in all_approx_diffs if diff <= success_threshold)
    detail_success = sum(1 for diff in all_detail_diffs if diff <= success_threshold)
    
    print(f"\n成功率分析 (差异 <= {success_threshold}):")
    print(f"  近似系数成功率: {approx_success}/{len(all_approx_diffs)} ({100*approx_success/len(all_approx_diffs):.1f}%)")
    print(f"  详细系数成功率: {detail_success}/{len(all_detail_diffs)} ({100*detail_success/len(all_detail_diffs):.1f}%)")
    
    print(f"\n性能对比:")
    print(f"  经典变换总时间: {total_classical_time:.6f}s")
    print(f"  量子模拟总时间: {total_quantum_time:.6f}s")
    print(f"  时间比率: {total_quantum_time/total_classical_time:.2f}x")
    
    # 判断量子版本是否成功
    approx_success_flag = np.mean(all_approx_diffs) <= success_threshold
    detail_success_flag = np.mean(all_detail_diffs) <= success_threshold
    
    print(f"\n量子版本成功性评估:")
    if approx_success_flag and detail_success_flag:
        print("🎉 量子版本非常成功！差异很小，精度很高。")
    elif approx_success_flag or detail_success_flag:
        print("✅ 量子版本基本成功，部分指标达到要求。")
    else:
        print("⚠️ 量子版本需要改进，差异较大。")
    
    return {
        'approx_stats': {
            'mean': np.mean(all_approx_diffs),
            'success_rate': 100*approx_success/len(all_approx_diffs)
        },
        'detail_stats': {
            'mean': np.mean(all_detail_diffs),
            'success_rate': 100*detail_success/len(all_detail_diffs)
        },
        'performance': {
            'classical_time': total_classical_time,
            'quantum_time': total_quantum_time
        }
    }

def main():
    """主函数"""
    print("="*60)
    print("量子CDF(2,2)变换与经典CDF(2,2)变换对比分析")
    print("="*60)
    
    # 加载图像
    image_path = "4.2.03.tiff"
    if not os.path.exists(image_path):
        print(f"图像文件 {image_path} 不存在")
        image_path = None
    
    image = load_image(image_path)
    
    # 处理图像块
    blocks, positions = process_image_blocks(image)
    print(f"总共 {len(blocks)} 个2x2块")
    
    # 分析每个块
    all_results = []
    
    for i, (block, pos) in enumerate(zip(blocks, positions)):
        print(f"\n--- 块 {i+1} (位置 {pos}) ---")
        print(f"原始数据: {block}")
        
        # 对比变换
        result = compare_transforms(block)
        
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
    
    # 分析结果
    stats = analyze_results(all_results)
    
    print(f"\n分析完成！")
    print(f"量子版本精度: {stats['approx_stats']['success_rate']:.1f}% (近似系数), {stats['detail_stats']['success_rate']:.1f}% (详细系数)")

if __name__ == "__main__":
    main() 