import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit import Aer, execute, transpile
except ImportError:
    try:
        from qiskit_aer import Aer
        from qiskit import execute, transpile
    except ImportError:
        print("Warning: Qiskit Aer not available. Some quantum simulation features may not work.")
        Aer = None

from qiskit.visualization import plot_histogram, circuit_drawer
try:
    from qiskit.circuit.library import QFT, QFTGate
except ImportError:
    print("Warning: QFT library not available.")
    QFT = None
    QFTGate = None

try:
    from qiskit.quantum_info import Statevector
except ImportError:
    print("Warning: Statevector not available.")
    Statevector = None

import math

class QuantumBlockCircuits:
    """
    量子CDF(2,2)小波变换的块状电路设计
    
    基于论文公式实现：
    1. Split: 分离S(2i)和S(2i+1)
    2. Predict: P(S) = 1/2[S(2i) + S(2i+2)]，计算D(i) = S(2i+1) - P(S)
    3. Update: W(D) = 1/4[D(i-1) + D(i)]，计算A(i) = S(2i) + W(D)
    """
    
    def __init__(self, bit_precision=4):
        self.bit_precision = bit_precision
        self.max_value = 2**bit_precision - 1
    
    def create_split_block(self, s_values):
        """
        创建Split块电路
        输入：信号S的连续值
        输出：分离的偶数位置S(2i)和奇数位置S(2i+1)的值
        """
        n_values = len(s_values)
        
        # 量子寄存器
        input_reg = QuantumRegister(self.bit_precision * n_values, 'input')
        even_reg = QuantumRegister(self.bit_precision * (n_values // 2), 'even')
        odd_reg = QuantumRegister(self.bit_precision * (n_values // 2), 'odd') 
        
        # 经典寄存器
        cr_even = ClassicalRegister(self.bit_precision * (n_values // 2), 'cr_even')
        cr_odd = ClassicalRegister(self.bit_precision * (n_values // 2), 'cr_odd')
        
        qc = QuantumCircuit(input_reg, even_reg, odd_reg, cr_even, cr_odd)
        
        # 初始化输入值
        for i, value in enumerate(s_values):
            binary_value = format(int(value), f'0{self.bit_precision}b')
            for j, bit in enumerate(binary_value):
                if bit == '1':
                    qc.x(input_reg[i * self.bit_precision + j])
        
        qc.barrier(label='Split Operation')
        
        # 分离操作：将偶数位置复制到even_reg，奇数位置复制到odd_reg
        even_idx = 0
        odd_idx = 0
        
        for i in range(n_values):
            if i % 2 == 0:  # 偶数位置
                for j in range(self.bit_precision):
                    qc.cx(input_reg[i * self.bit_precision + j], 
                          even_reg[even_idx * self.bit_precision + j])
                even_idx += 1
            else:  # 奇数位置
                for j in range(self.bit_precision):
                    qc.cx(input_reg[i * self.bit_precision + j], 
                          odd_reg[odd_idx * self.bit_precision + j])
                odd_idx += 1
        
        # 测量
        qc.measure(even_reg, cr_even)
        qc.measure(odd_reg, cr_odd)
        
        return qc
    
    def create_predict_block(self, s_2i, s_2i_plus_2, s_2i_plus_1):
        """
        创建Predict块电路
        实现 P(S) = 1/2[S(2i) + S(2i+2)]
        然后计算 D(i) = S(2i+1) - P(S)
        """
        # 量子寄存器
        s_2i_reg = QuantumRegister(self.bit_precision, 's_2i')
        s_2i_plus_2_reg = QuantumRegister(self.bit_precision, 's_2i_plus_2')
        s_2i_plus_1_reg = QuantumRegister(self.bit_precision, 's_2i_plus_1')
        
        # 中间计算寄存器
        sum_reg = QuantumRegister(self.bit_precision + 1, 'sum')  # S(2i) + S(2i+2)
        predict_reg = QuantumRegister(self.bit_precision, 'predict')  # P(S) = sum/2
        detail_reg = QuantumRegister(self.bit_precision, 'detail')  # D(i)
        
        # 经典寄存器
        cr_predict = ClassicalRegister(self.bit_precision, 'cr_predict')
        cr_detail = ClassicalRegister(self.bit_precision, 'cr_detail')
        
        qc = QuantumCircuit(s_2i_reg, s_2i_plus_2_reg, s_2i_plus_1_reg,
                           sum_reg, predict_reg, detail_reg, 
                           cr_predict, cr_detail)
        
        # 初始化输入值
        for value, reg in [(s_2i, s_2i_reg), (s_2i_plus_2, s_2i_plus_2_reg), (s_2i_plus_1, s_2i_plus_1_reg)]:
            binary_value = format(int(value), f'0{self.bit_precision}b')
            for j, bit in enumerate(binary_value):
                if bit == '1':
                    qc.x(reg[j])
        
        qc.barrier(label='Predict: S(2i) + S(2i+2)')
        
        # Step 1: 计算 S(2i) + S(2i+2)
        self._quantum_adder_ripple_carry(qc, s_2i_reg, s_2i_plus_2_reg, sum_reg)
        
        qc.barrier(label='Predict: Divide by 2')
        
        # Step 2: 可逆除以2 (右移一位) - P(S) = [S(2i) + S(2i+2)] / 2
        # 使用受控交换实现可逆右移
        for i in range(self.bit_precision):
            if i < self.bit_precision - 1:
                # 从sum_reg[i+1]复制到predict_reg[i]（右移一位）
                qc.cx(sum_reg[i+1], predict_reg[i])
            else:
                # 最高位从sum_reg的最高位复制
                qc.cx(sum_reg[self.bit_precision], predict_reg[i])
        
        qc.barrier(label='Detail: S(2i+1) - P(S)')
        
        # Step 3: 计算 D(i) = S(2i+1) - P(S)
        # 使用完整的量子减法器
        self._quantum_subtractor(qc, s_2i_plus_1_reg, predict_reg, detail_reg)
        
        # 测量
        qc.measure(predict_reg, cr_predict)
        qc.measure(detail_reg, cr_detail)
        
        return qc
    
    def create_update_block(self, d_i_minus_1, d_i, s_2i):
        """
        创建Update块电路
        实现 W(D) = 1/4[D(i-1) + D(i)]
        然后计算 A(i) = S(2i) + W(D)
        """
        # 量子寄存器
        d_i_minus_1_reg = QuantumRegister(self.bit_precision, 'd_i_minus_1')
        d_i_reg = QuantumRegister(self.bit_precision, 'd_i')
        s_2i_reg = QuantumRegister(self.bit_precision, 's_2i')
        
        # 中间计算寄存器
        sum_d_reg = QuantumRegister(self.bit_precision + 1, 'sum_d')  # D(i-1) + D(i)
        update_reg = QuantumRegister(self.bit_precision, 'update')  # W(D) = sum_d/4
        approx_reg = QuantumRegister(self.bit_precision + 1, 'approx')  # A(i)，需要额外位处理进位
        
        # 经典寄存器
        cr_update = ClassicalRegister(self.bit_precision, 'cr_update')
        cr_approx = ClassicalRegister(self.bit_precision + 1, 'cr_approx')
        
        qc = QuantumCircuit(d_i_minus_1_reg, d_i_reg, s_2i_reg,
                           sum_d_reg, update_reg, approx_reg,
                           cr_update, cr_approx)
        
        # 初始化输入值
        for value, reg in [(d_i_minus_1, d_i_minus_1_reg), 
                          (d_i, d_i_reg), 
                          (s_2i, s_2i_reg)]:
            binary_value = format(int(value), f'0{self.bit_precision}b')
            for j, bit in enumerate(binary_value):
                if bit == '1':
                    qc.x(reg[j])
        
        qc.barrier(label='Update: D(i-1) + D(i)')
        
        # Step 1: 计算 D(i-1) + D(i)
        self._quantum_adder_ripple_carry(qc, d_i_minus_1_reg, d_i_reg, sum_d_reg)
        
        qc.barrier(label='Update: Divide by 4')
        
        # Step 2: 可逆除以4 (右移两位) - W(D) = [D(i-1) + D(i)] / 4
        for i in range(self.bit_precision):
            if i < self.bit_precision - 2:
                # 从sum_d_reg[i+2]复制到update_reg[i]（右移两位）
                qc.cx(sum_d_reg[i+2], update_reg[i])
            elif i == self.bit_precision - 2:
                # 倒数第二位从sum_d_reg的最高位复制
                qc.cx(sum_d_reg[self.bit_precision], update_reg[i])
            # 最高位保持为0（正数情况下）
        
        qc.barrier(label='Approximation: S(2i) + W(D)')
        
        # Step 3: 计算 A(i) = S(2i) + W(D)
        # 使用完整的行波进位加法器
        self._quantum_adder_ripple_carry(qc, s_2i_reg, update_reg, approx_reg)
        
        # 测量
        qc.measure(update_reg, cr_update)
        qc.measure(approx_reg, cr_approx)
        
        return qc
    
    def _quantum_adder_ripple_carry(self, qc, reg_a, reg_b, result_reg):
        """
        实现带进位的量子加法器 (Ripple Carry Adder)
        result_reg = reg_a + reg_b
        """
        n = len(reg_a)
        
        # 初始化结果寄存器为reg_a的副本
        for i in range(n):
            qc.cx(reg_a[i], result_reg[i])
        
        # 逐位加法with进位
        for i in range(n):
            # 当前位的加法
            qc.cx(reg_b[i], result_reg[i])
            
            # 进位逻辑
            if i < n - 1:
                # 如果有进位到下一位
                qc.ccx(reg_a[i], reg_b[i], result_reg[i+1])
                
                # 处理连续进位
                for j in range(i):
                    # 创建辅助量子比特来处理复杂的进位逻辑
                    qc.ccx(result_reg[j], reg_b[i], result_reg[i+1])
    
    def _quantum_adder_simple(self, qc, reg_a, reg_b, result_reg):
        """
        简化的量子加法器，用于最终的A(i) = S(2i) + W(D)计算
        """
        n = len(reg_a)
        
        # 复制reg_a到result_reg
        for i in range(n):
            qc.cx(reg_a[i], result_reg[i])
        
        # 加上reg_b
        for i in range(n):
            qc.cx(reg_b[i], result_reg[i])
    
    def _quantum_subtractor(self, qc, reg_a, reg_b, result_reg):
        """
        量子减法器：result_reg = reg_a - reg_b
        使用二进制补码方法
        """
        n = len(reg_a)
        
        # 复制reg_a到result_reg
        for i in range(n):
            qc.cx(reg_a[i], result_reg[i])
        
        # 计算reg_b的补码然后相加
        # 首先对reg_b取反
        for i in range(n):
            qc.x(reg_b[i])
        
        # 加1 (补码)
        qc.x(reg_b[0])  # 最低位加1
        
        # 执行加法
        for i in range(n):
            qc.cx(reg_b[i], result_reg[i])
            
            # 简化的进位处理
            if i < n - 1:
                qc.ccx(reg_a[i], reg_b[i], result_reg[i+1])
        
        # 恢复reg_b的原始值
        qc.x(reg_b[0])
        for i in range(n):
            qc.x(reg_b[i])
    
    def create_complete_cdf_block_circuit(self, signal_block):
        """
        创建完整的CDF(2,2)小波变换块电路
        对一个信号块执行完整的Split-Predict-Update流程
        """
        n = len(signal_block)
        if n % 2 != 0:
            raise ValueError("Signal block length must be even")
        
        # 计算需要的量子比特数
        total_qubits = self.bit_precision * n * 4  # 输入、中间结果、输出、辅助寄存器
        
        # 创建量子寄存器
        input_reg = QuantumRegister(self.bit_precision * n, 'input')
        
        # Split结果
        even_reg = QuantumRegister(self.bit_precision * (n // 2), 'even')
        odd_reg = QuantumRegister(self.bit_precision * (n // 2), 'odd')
        
        # Predict/Detail输出
        predict_reg = QuantumRegister(self.bit_precision * (n // 2), 'predict')
        detail_reg = QuantumRegister(self.bit_precision * (n // 2), 'detail')
        
        # Update/Approx输出
        update_reg = QuantumRegister(self.bit_precision * (n // 2), 'update')
        approx_reg = QuantumRegister(self.bit_precision * (n // 2), 'approx')
        
        # 中间求和寄存器（用于 + 和 位移）
        sum_predict_reg = QuantumRegister(self.bit_precision + 1, 'sum_predict')
        sum_update_reg = QuantumRegister(self.bit_precision + 1, 'sum_update')
        
        # 经典寄存器
        cr_approx = ClassicalRegister(self.bit_precision * (n // 2), 'cr_approx')
        cr_detail = ClassicalRegister(self.bit_precision * (n // 2), 'cr_detail')
        
        qc = QuantumCircuit(input_reg, even_reg, odd_reg,
                           sum_predict_reg, sum_update_reg,
                           predict_reg, detail_reg, update_reg, approx_reg,
                           cr_approx, cr_detail)
        
        # 初始化输入信号
        for i, value in enumerate(signal_block):
            binary_value = format(int(value), f'0{self.bit_precision}b')
            for j, bit in enumerate(binary_value):
                if bit == '1':
                    qc.x(input_reg[i * self.bit_precision + j])
        
        # Step 1: Split
        qc.barrier(label='SPLIT STEP')
        even_idx = 0
        odd_idx = 0
        for i in range(n):
            if i % 2 == 0:  # 偶数位置 -> even_reg
                for j in range(self.bit_precision):
                    qc.cx(input_reg[i * self.bit_precision + j], 
                          even_reg[even_idx * self.bit_precision + j])
                even_idx += 1
            else:  # 奇数位置 -> odd_reg
                for j in range(self.bit_precision):
                    qc.cx(input_reg[i * self.bit_precision + j], 
                          odd_reg[odd_idx * self.bit_precision + j])
                odd_idx += 1
        
        # Step 2: Predict (对每个奇数位置计算预测值和详细系数)
        qc.barrier(label='PREDICT STEP')
        bp = self.bit_precision
        half = n // 2
        for i in range(half):
            # 选择 S(2i) 与 S(2i+2) 的边界索引
            if i == 0:
                left_even_index = 0
                right_even_index = 1 if half > 1 else 0
            elif i == half - 1:
                left_even_index = i - 1
                right_even_index = i
            else:
                left_even_index = i
                right_even_index = i + 1
            
            # 构造切片（作为量子比特列表传入加法/减法器）
            s2i_slice = [even_reg[left_even_index * bp + k] for k in range(bp)]
            s2i_plus2_slice = [even_reg[right_even_index * bp + k] for k in range(bp)]
            sum_pred_slice = [sum_predict_reg[k] for k in range(bp + 1)]
            pred_out_slice = [predict_reg[i * bp + k] for k in range(bp)]
            s2i_plus1_slice = [odd_reg[i * bp + k] for k in range(bp)]
            detail_out_slice = [detail_reg[i * bp + k] for k in range(bp)]
            
            # 求和: S(2i)+S(2i+2)
            self._quantum_adder_ripple_carry(qc, s2i_slice, s2i_plus2_slice, sum_pred_slice)
            
            # 除以2: 将 sum 的高位映射到 predict 的各位
            for k in range(bp):
                if k < bp - 1:
                    qc.cx(sum_pred_slice[k+1], pred_out_slice[k])
                else:
                    qc.cx(sum_pred_slice[bp], pred_out_slice[k])
            
            # 计算 D(i) = S(2i+1) - P(S)
            # 使用量子减法器计算 D(i) = S(2i+1) - P(S)
            self._quantum_subtractor(qc, s2i_plus1_slice, pred_out_slice, detail_out_slice)
        
        # Step 3: Update (计算更新值和近似系数)
        qc.barrier(label='UPDATE STEP')
        
        for i in range(half):
            update_out_slice = [update_reg[i * bp + k] for k in range(bp)]
            approx_out_slice = [approx_reg[i * bp + k] for k in range(bp)]
            even_slice = [even_reg[i * bp + k] for k in range(bp)]
            
            # 计算 W(D) = 1/4 [D(i-1) + D(i)]，边界单侧处理
            if i == 0:
                # sum_update = D(0)
                for k in range(bp):
                    qc.cx(detail_reg[0 * bp + k], sum_update_reg[k])
            elif i == half - 1:
                # sum_update = D(i-1)
                for k in range(bp):
                    qc.cx(detail_reg[(i-1) * bp + k], sum_update_reg[k])
            else:
                d_im1_slice = [detail_reg[(i-1) * bp + k] for k in range(bp)]
                d_i_slice = [detail_reg[i * bp + k] for k in range(bp)]
                sum_upd_slice = [sum_update_reg[k] for k in range(bp + 1)]
                self._quantum_adder_ripple_carry(qc, d_im1_slice, d_i_slice, sum_upd_slice)
            
            # 除以4: 将 sum_update 的更高两位映射到 update 的各位
            for k in range(bp):
                if k < bp - 2:
                    qc.cx(sum_update_reg[k+2], update_out_slice[k])
                elif k == bp - 2:
                    qc.cx(sum_update_reg[bp], update_out_slice[k])
                else:
                    # 保持最高位为0
                    pass
            
            # 计算 A(i) = S(2i) + W(D)
            self._quantum_adder_simple(qc, even_slice, update_out_slice, approx_out_slice)
        
        # 测量最终结果
        qc.measure(approx_reg, cr_approx)
        qc.measure(detail_reg, cr_detail)
        
        return qc
    
    def visualize_block_circuit(self, qc, title="Quantum CDF Block Circuit"):
        """
        可视化量子块电路
        """
        try:
            fig = plt.figure(figsize=(16, 10))
            
            # 绘制电路图
            circuit_diagram = circuit_drawer(qc, output='mpl', style='iqx', 
                                           fold=-1, scale=0.8)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"可视化电路时出错: {e}")
            print("电路信息:")
            print(f"量子比特数: {qc.num_qubits}")
            print(f"电路深度: {qc.depth()}")
            print(f"门数量: {len(qc.data)}")
            return None

# 添加一个简单的测试函数
def test_quantum_blocks():
    """
    测试量子块电路的基本功能
    """
    print("测试量子CDF(2,2)小波变换块电路...")
    
    # 创建实例
    qbc = QuantumBlockCircuits(bit_precision=4)
    
    # 测试Split块
    print("\n1. 测试Split块:")
    test_signal = [1, 3, 2, 6]
    try:
        split_circuit = qbc.create_split_block(test_signal)
        print(f"   Split电路创建成功!")
        print(f"   量子比特数: {split_circuit.num_qubits}")
        print(f"   电路深度: {split_circuit.depth()}")
    except Exception as e:
        print(f"   Split电路创建失败: {e}")
    
    # 测试Predict块
    print("\n2. 测试Predict块:")
    try:
        predict_circuit = qbc.create_predict_block(1, 2, 3)
        print(f"   Predict电路创建成功!")
        print(f"   量子比特数: {predict_circuit.num_qubits}")
        print(f"   电路深度: {predict_circuit.depth()}")
    except Exception as e:
        print(f"   Predict电路创建失败: {e}")
    
    # 测试Update块
    print("\n3. 测试Update块:")
    try:
        update_circuit = qbc.create_update_block(1, 2, 3)
        print(f"   Update电路创建成功!")
        print(f"   量子比特数: {update_circuit.num_qubits}")
        print(f"   电路深度: {update_circuit.depth()}")
    except Exception as e:
        print(f"   Update电路创建失败: {e}")
    
    # 测试完整块电路
    print("\n4. 测试完整CDF块电路:")
    try:
        complete_circuit = qbc.create_complete_cdf_block_circuit(test_signal)
        print(f"   完整电路创建成功!")
        print(f"   量子比特数: {complete_circuit.num_qubits}")
        print(f"   电路深度: {complete_circuit.depth()}")
        print(f"   门数量: {len(complete_circuit.data)}")
    except Exception as e:
        print(f"   完整电路创建失败: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_quantum_blocks()