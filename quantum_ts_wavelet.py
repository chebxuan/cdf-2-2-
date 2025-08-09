import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit import Aer, execute, transpile
except ImportError:
    try:
        from qiskit_aer import Aer
        from qiskit import execute, transpile
    except ImportError:
        Aer = None


class QuantumTSWavelet:
    """
    一维整数 TS 变换的量子版本（提升方案骨架）

    对应用户图示公式（用提升表示）：
      - Predict-like: P_i = (S_{2i+1} - S_{2i-1}) / 2  (用高位映射实现/2)
      - Update-even:  A_i = S_{2i} + alpha * P_i
      - Smooth-even:  W_i = (A_i + A_{i+1}) / 2
      - Update-odd:   D_i = S_{2i+1} - beta * W_i

    注意：/2 通过位重映射来近似实现，保持电路可逆（需要辅助寄存器保存丢失位时方可严格可逆）。
    """

    def __init__(self, bit_precision: int = 8, alpha: int = 1, beta: int = 1):
        self.bit_precision = bit_precision
        self.alpha = alpha
        self.beta = beta

    def create_split(self, n_pairs: int):
        even = QuantumRegister(self.bit_precision * n_pairs, 'even')
        odd = QuantumRegister(self.bit_precision * n_pairs, 'odd')
        cr_e = ClassicalRegister(self.bit_precision * n_pairs, 'cr_even')
        cr_o = ClassicalRegister(self.bit_precision * n_pairs, 'cr_odd')
        qc = QuantumCircuit(even, odd, cr_e, cr_o)
        # 此处假定外部已写入 S(2i)/S(2i+1) 到 even/odd
        qc.barrier(label='Split (provided)')
        return qc

    def create_predict(self, s_im1_reg, s_ip1_reg):
        """计算 P_i = (S_{2i+1} - S_{2i-1})/2 到寄存器 predict。
        s_im1_reg: S_{2i-1} 的量子寄存器
        s_ip1_reg: S_{2i+1} 的量子寄存器
        """
        predict = QuantumRegister(self.bit_precision, 'predict')
        sum_reg = QuantumRegister(self.bit_precision + 1, 'diff')
        cr = ClassicalRegister(self.bit_precision, 'cr_predict')
        qc = QuantumCircuit(s_im1_reg, s_ip1_reg, sum_reg, predict, cr)

        # diff = s_ip1 - s_im1（用加法器与求补实现，这里用复制近似占位）
        for i in range(self.bit_precision):
            qc.cx(s_ip1_reg[i], sum_reg[i])
            qc.cx(s_im1_reg[i], sum_reg[i])

        # 除以2（右移一位到 predict）
        for i in range(self.bit_precision - 1):
            qc.cx(sum_reg[i + 1], predict[i])
        # 最高位丢弃，若需完全可逆应保存在垃圾位并在反算阶段清理

        qc.measure(predict, cr)
        return qc

    def create_update_even(self, s_even_reg, predict_reg):
        """A_i = S_{2i} + alpha * P_i（alpha 默认为 1）"""
        a_reg = QuantumRegister(self.bit_precision + 1, 'A')
        cr = ClassicalRegister(self.bit_precision + 1, 'cr_A')
        qc = QuantumCircuit(s_even_reg, predict_reg, a_reg, cr)

        # 复制 S(2i)
        for i in range(self.bit_precision):
            qc.cx(s_even_reg[i], a_reg[i])
        # 加上 alpha*P_i（简单重复加法 alpha 次）
        for _ in range(max(1, self.alpha)):
            for i in range(self.bit_precision):
                qc.cx(predict_reg[i], a_reg[i])
        qc.measure(a_reg, cr)
        return qc

    def create_smooth_and_update_odd(self, a_i_reg, a_ip1_reg, s_odd_reg):
        """W_i = (A_i + A_{i+1})/2; D_i = S_{2i+1} - beta * W_i"""
        sum_reg = QuantumRegister(self.bit_precision + 1, 'sumA')
        w_reg = QuantumRegister(self.bit_precision, 'W')
        d_reg = QuantumRegister(self.bit_precision + 1, 'D')
        cr_w = ClassicalRegister(self.bit_precision, 'cr_W')
        cr_d = ClassicalRegister(self.bit_precision + 1, 'cr_D')
        qc = QuantumCircuit(a_i_reg, a_ip1_reg, s_odd_reg, sum_reg, w_reg, d_reg, cr_w, cr_d)

        # 求和 A_i + A_{i+1}（近似复制）
        for i in range(self.bit_precision):
            qc.cx(a_i_reg[i], sum_reg[i])
            qc.cx(a_ip1_reg[i], sum_reg[i])
        # /2 -> w_reg（右移一位）
        for i in range(self.bit_precision - 1):
            qc.cx(sum_reg[i + 1], w_reg[i])

        # D = S_{2i+1} - beta*W（这里以异或近似；严格应使用量子减法器）
        for _ in range(max(1, self.beta)):
            for i in range(self.bit_precision):
                qc.cx(w_reg[i], d_reg[i])
        for i in range(self.bit_precision):
            qc.cx(s_odd_reg[i], d_reg[i])

        qc.measure(w_reg, cr_w)
        qc.measure(d_reg, cr_d)
        return qc