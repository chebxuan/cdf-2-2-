
# Qiskit兼容性包装器
# 将此代码添加到您的quantum_block_circuits.py文件开头

try:
    from qiskit import Aer, execute, transpile
except ImportError:
    try:
        from qiskit_aer import Aer
        from qiskit import execute, transpile
    except ImportError:
        print("警告: Qiskit Aer不可用，某些功能可能无法使用")
        Aer = None
        
def safe_get_backend(backend_name='qasm_simulator'):
    """安全获取后端"""
    if Aer is None:
        raise ImportError("Aer模拟器不可用")
    return Aer.get_backend(backend_name)

def safe_execute(circuit, backend, shots=1024):
    """安全执行量子电路"""
    if Aer is None:
        raise ImportError("Aer模拟器不可用")
    compiled = transpile(circuit, backend)
    job = execute(compiled, backend, shots=shots)
    return job.result()
