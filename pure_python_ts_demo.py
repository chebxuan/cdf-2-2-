#!/usr/bin/env python3
"""
纯Python实现的 TS 整数小波变换（基于提升方案）

从用户提供的原理图抽取的前向公式（索引以偶/奇交替的一维信号表示）：
- P_i = round_half_to_even((S_{2i+1} - S_{2i-1}) / 2)
- A_i = S_{2i} + alpha * P_i
- W_i = round_half_to_even((A_i + A_{i+1}) / 2)
- D_i = S_{2i+1} - beta * W_i

其中 alpha, beta 为提升系数（整数或 2 的幂分母时配合相同舍入规则可保持可逆）。
边界采用复制延拓：
- i = 0 处使用 S_{-1} := S_{1}；i = N-1 处使用 A_{N} := A_{N-1}

逆变换：
- 先还原奇样本：S_{2i+1} = D_i + beta * W_i
- 复算 P_i = round_half_to_even((S_{2i+1} - S_{2i-1}) / 2)
- 再还原偶样本：S_{2i} = A_i - alpha * P_i

该流程是标准的“先更新偶样本，再用偶样本去更新奇样本”的两步提升结构，满足逐点可逆。
"""

from typing import List, Tuple


def _round_div2(x: int) -> int:
    """对整数 x 执行四舍六入五留双的 /2 舍入（round half to even）。
    该规则在正负数下都满足可逆性需求，比简单地向下取整更对称。
    """
    # Python 的 round 是银行家舍入，但对负数的行为合适；确保返回 int
    return int(round(x / 2.0))


def _round_avg(a: int, b: int) -> int:
    """两数平均并做银行家舍入。"""
    return _round_div2(a + b)


def ts_integer_forward(signal: List[int], alpha: int = 1, beta: int = 1) -> Tuple[List[int], List[int]]:
    """前向 TS 整数变换（提升实现）。

    Args:
        signal: 一维整数信号，长度为 2N
        alpha: 偶样本更新的提升系数
        beta: 奇样本更新的提升系数
    Returns:
        (A, D): A 为近似/低频（偶位置），D 为细节/高频（奇位置）
    """
    n = len(signal)
    if n % 2 != 0:
        raise ValueError("信号长度必须为偶数")

    even = signal[::2]  # S_{2i}
    odd = signal[1::2]  # S_{2i+1}
    N = len(even)

    # Step 1: Predict-like（由奇样本差分来更新偶样本）
    # P_i = (S_{2i+1} - S_{2i-1})/2 (round)
    P = []
    for i in range(N):
        left_odd = odd[i - 1] if i - 1 >= 0 else odd[0]  # 复制延拓 S_{-1} := S_1
        right_odd = odd[i] if i < len(odd) else odd[-1]
        P_i = _round_div2(right_odd - left_odd)
        P.append(P_i)
    A = [int(even[i] + alpha * P[i]) for i in range(N)]

    # Step 2: Update-like（由 A 的平滑去更新奇样本为 D）
    W = []
    for i in range(N):
        a_i = A[i]
        a_ip1 = A[i + 1] if i + 1 < N else A[-1]  # 复制延拓 A_N := A_{N-1}
        W_i = _round_avg(a_i, a_ip1)
        W.append(W_i)
    D = [int(odd[i] - beta * W[i]) for i in range(N)]

    return A, D


def ts_integer_inverse(A: List[int], D: List[int], alpha: int = 1, beta: int = 1) -> List[int]:
    """逆向 TS 整数变换。

    Returns:
        还原的一维整数信号
    """
    N = len(A)
    if len(D) != N:
        raise ValueError("A 与 D 长度必须相同")

    # 先利用 A 计算平滑 W，并还原奇样本
    W = []
    for i in range(N):
        a_i = A[i]
        a_ip1 = A[i + 1] if i + 1 < N else A[-1]
        W_i = _round_avg(a_i, a_ip1)
        W.append(W_i)
    odd = [int(D[i] + beta * W[i]) for i in range(N)]

    # 再复算 P 并还原偶样本
    even = []
    for i in range(N):
        left_odd = odd[i - 1] if i - 1 >= 0 else odd[0]
        right_odd = odd[i]
        P_i = _round_div2(right_odd - left_odd)
        even.append(int(A[i] - alpha * P_i))

    # 交错合成
    out = []
    for i in range(N):
        out.append(even[i])
        out.append(odd[i])
    return out


def _demo():
    print("=" * 80)
    print("TS 整数小波变换（提升方案） - 纯Python演示")
    print("=" * 80)

    signal = [1, 3, 2, 6]
    print(f"输入信号: {signal}")

    A, D = ts_integer_forward(signal, alpha=1, beta=1)
    print(f"前向：A={A}, D={D}")

    recon = ts_integer_inverse(A, D, alpha=1, beta=1)
    print(f"重构：{recon}")

    ok = recon == signal
    print(f"可逆性检查: {'通过' if ok else '失败'}")


if __name__ == "__main__":
    _demo()