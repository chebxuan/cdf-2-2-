# 量子CDF(2,2)小波变换实现

本项目实现了基于论文中CDF(2,2)小波变换提升方案的量子电路设计与经典模拟实现。

## 项目概述

本项目实现了基于论文中CDF(2,2)小波变换提升方案的量子电路设计，包含三个主要步骤：

1. **Split**: 分离奇偶样本
2. **Predict**: P(S) = 1/2[S(2i) + S(2i+2)]，计算D(i) = S(2i+1) - P(S)
3. **Update**: W(D) = 1/4[D(i-1) + D(i)]，计算A(i) = S(2i) + W(D)

## 更新说明

- 调整了块状电路使其与 figure5 一致：
  - 修正 Predict 子块，显式接入 `S(2i+1)`，用行波进位加法器实现 `S(2i)+S(2i+2)`，通过位映射实现“/2”，再用减法器得到 `D(i)`。
  - 完整块电路 `create_complete_cdf_block_circuit` 现按 Split→Predict→Update 贯通实现，不再使用复制占位；新增中间求和寄存器 `sum_predict_reg`、`sum_update_reg` 支撑加法与位移。
- 增强 MATLAB 脚本以满足 figure6：
  - 新增 `classical_cdf_transform_multilevel(signal, levels)` 与 `quantum_simulated_cdf_transform_multilevel(signal, levels)`，并在主程序演示了一维信号的三层分解 `A3, D3, D2, D1`。

提示：为简洁起见，Predict/Update 中的除以2/4使用高位映射（右移）实现，属于可运行的近似可逆构造；若需完全可逆且无垃圾比特，可在此基础上添加“反算”阶段清理辅助位。

## 新增：TS 整数小波变换（提升方案）

基于你提供的原理图与公式，我们给出一维整数 TS 变换的提升实现（经典与量子骨架）：

- **图示公式归纳**（以 `S(2i), S(2i+1)` 表示偶/奇序列）：
  - 预测/差分：`P_i = 1/2 · [ S(2i+1) − S(2i−1) ]`（边界复制，/2 采用银行家舍入）
  - 偶更新：`A_i = S(2i) + α · P_i`
  - 平滑：`W_i = 1/2 · [ A_{i+1} + A_i ]`（边界复制，/2 银行家舍入）
  - 奇更新：`D_i = S(2i+1) − β · W_i`
  - 逆向：先用 `W_i` 还原奇样本，再复算 `P_i` 还原偶样本，保证可逆。

- **文件**：
  - `pure_python_ts_demo.py`：纯 Python 的整数 TS 变换（含前向/逆向与可逆性自测）
  - `quantum_ts_wavelet.py`：量子 TS 变换骨架（Split / Predict-like / Update-even / Smooth+Update-odd），复用右移位映射，可按需替换为严格的量子加减法器以得到完全可逆实现

- **快速试用**：
  ```bash
  python3 pure_python_ts_demo.py
  ```

- 与 CDF 相同，量子版本的除法通过高位重映射实现。如需严格可逆，请在分子寄存器保留被移出的位并在反向阶段清理。

## 文件结构

```
├── quantum_cdf_wavelet.py      # 量子CDF小波变换核心实现（含经典对比）
├── quantum_block_circuits.py   # 量子块状电路（Split/Predict/Update 按公式实现）
├── quantum_cdf_demo.py         # 完整演示程序（需要qiskit）
├── pure_python_cdf_demo.py     # 纯Python演示（无外部依赖，经典验证）
├── pure_python_ts_demo.py      # 新增：TS 整数小波变换（提升方案，纯Python）
├── quantum_ts_wavelet.py       # 新增：TS 量子骨架
├── quantum_vs_classical_matlab.m # MATLAB 对比与三层分解演示（figure6）
├── run_demo.py                 # 简化运行脚本（经典步骤展示）
├── test_quantum_blocks.py      # 基本电路与算法自检
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明
```

## 快速开始

### 方法1：纯Python演示（推荐）

无需安装任何外部库，直接运行：

```bash
python3 pure_python_cdf_demo.py
python3 pure_python_ts_demo.py
```

### 方法2：查看并测试量子块电路（figure5）

需要安装依赖：

```bash
pip install -r requirements.txt
python3 test_quantum_blocks.py         # 自检：算法与电路基础
python3 quantum_block_circuits.py      # 构建并查看块状电路
```

### 方法3：完整量子演示

```bash
pip install -r requirements.txt
python3 quantum_cdf_demo.py
```

### 方法4：MATLAB 三层分解演示（figure6）

在 MATLAB 中打开并运行：

- `quantum_vs_classical_matlab.m`
  - 前半部分：对图像分块的经典/量子（模拟）一层分解与对比
  - 末尾新增：一维信号的三层分解演示，输出 `A3, D3, D2, D1`

## 核心算法

### CDF(2,2)小波变换三步骤

#### 1. Split步骤
```
输入: S = [S(0), S(1), S(2), S(3)]
输出: Even = [S(0), S(2)], Odd = [S(1), S(3)]
```

#### 2. Predict步骤
```
对每个奇数位置i：
P(S) = 1/2[S(2i) + S(2i+2)]
D(i) = S(2i+1) - P(S)
```

#### 3. Update步骤
```
对每个偶数位置i：
W(D) = 1/4[D(i-1) + D(i)]
A(i) = S(2i) + W(D)
```

## 量子电路设计

- **Split量子电路块**：基于索引奇偶性分离，使用 CNOT 实现条件复制
- **Predict量子电路块**：
  - 行波进位加法器计算 `S(2i) + S(2i+2)`
  - 通过高位映射实现除以2（右移）得到 `P(S)`
  - 量子减法器计算 `D(i) = S(2i+1) - P(S)`
- **Update量子电路块**：
  - 行波进位加法器计算 `D(i-1) + D(i)`（边界用单侧 detail）
  - 高位映射实现除以4（右移两位）得到 `W(D)`
  - 简化量子加法器计算 `A(i) = S(2i) + W(D)`

## 演示结果

运行演示程序会处理一个4x4测试图像：

```
测试图像 (4x4):
[1, 3, 2, 1]
[2, 6, 6, 6]
[5, 2, 7, 4]
[3, 1, 7, 2]
```

分割为4个2x2块，每个块应用CDF(2,2)变换，生成近似系数和详细系数。

## 与图示一致性

- **figure5（块状电路）**：`quantum_block_circuits.py` 中 Split/Predict/Update 现按提升公式计算，Predict 显式使用 `S(2i), S(2i+2), S(2i+1)`；完整块电路贯通三步。
- **figure6（三层分解）**：`quantum_vs_classical_matlab.m` 新增多层分解函数与演示，输出一维信号的三层分解结果。

## 技术特点

✓ **精确实现**: 严格按照论文公式实现CDF(2,2)变换；新增 TS 变换采用整数提升保证可逆
✓ **模块化设计**: 三个独立的量子电路块可组合使用
✓ **边界处理**: 完整的边界条件处理
✓ **并行处理**: 支持多个图像块的并行处理
✓ **经典验证**: 提供经典实现作为对比验证
✓ **可扩展性**: 可扩展到更大图像和更高精度

## 量子优势

- **并行计算**: 量子叠加态可同时处理多种输入值
- **量子纠缠**: 利用量子纠缠实现高效的相关性计算
- **量子算法**: 量子加法器和减法器的固有优势
- **可逆计算**: 量子计算的可逆性质适合信号处理

## 未来扩展

- 支持更大的图像尺寸
- 实现更高精度的量子比特编码
- 集成量子错误校正
- 优化量子电路深度和门数量
- 实现真实量子硬件上的运行

## 贡献

本项目基于用户提供的CDF(2,2)小波变换量子提升方案论文实现，并扩展了 TS 整数变换版本。欢迎提交问题和改进建议。