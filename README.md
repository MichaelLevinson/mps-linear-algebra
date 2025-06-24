# 🚀 MPS Linear Algebra for PyTorch

**Native linear algebra operations for Apple's Metal Performance Shaders (MPS) backend**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MPS](https://img.shields.io/badge/MPS-Native-green.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Problem Solved

PyTorch's MPS backend doesn't implement many essential linear algebra operations (`torch.linalg.svd`, `torch.linalg.pinv`, etc.), forcing expensive CPU fallbacks in scientific computing applications. This package provides **native MPS implementations** that maintain numerical accuracy while staying on the GPU.

### Before (CPU Fallbacks):
```python
# ❌ Falls back to CPU, breaking your pipeline
A_mps = torch.randn(1000, 500, device='mps')
U, S, Vt = torch.linalg.svd(A_mps)  # NotImplementedError or silent CPU fallback
```

### After (Native MPS):
```python
# ✅ Stays on MPS, maintains performance
from mps_linalg import pinv, solve
A_mps = torch.randn(1000, 500, device='mps')
A_pinv = pinv(A_mps)  # Native MPS computation
```

## 🌟 Key Features

- **🔥 Native MPS Operations**: QR decomposition, pseudoinverse, linear solving
- **📊 Numerical Stability**: Modified Gram-Schmidt with reorthogonalization  
- **🎯 Drop-in Replacement**: Compatible with `torch.linalg` interface
- **⚡ Performance**: No CPU transfers, full MPS utilization
- **🧪 Thoroughly Tested**: Comprehensive test suite with accuracy validation
- **🔬 Scientific Computing Ready**: Designed for ML/physics applications

## 📦 Installation

```bash
git clone https://github.com/michaellevinson/mps-linear-algebra.git
cd mps-linear-algebra
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
import torch
from mps_linalg import pinv, solve, lstsq, matrix_rank

# Ensure MPS is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Create test matrices
A = torch.randn(100, 50, device=device)
b = torch.randn(100, device=device)

# Pseudoinverse computation
A_pinv = pinv(A)
print(f"Pseudoinverse shape: {A_pinv.shape}")

# Linear system solving  
x = solve(A, b)  # Least squares solution
print(f"Solution error: {torch.norm(A @ x - b):.2e}")

# Matrix rank estimation
rank = matrix_rank(A)
print(f"Estimated rank: {rank}")
```

## 🔬 Core Functions

### `QR_mps(A, reorth_steps=2, shift=1e-8)`
Numerically stable QR decomposition using Modified Gram-Schmidt with reorthogonalization.

```python
from qr_decomp import QR_mps

A = torch.randn(200, 100, device='mps')
Q, R = QR_mps(A)

# Verify decomposition
reconstruction_error = torch.norm(A - Q @ R)
orthogonality_error = torch.norm(Q.T @ Q - torch.eye(Q.shape[1], device='mps'))
```

### `pinv(A, rcond=1e-15)`
Pseudoinverse computation via QR decomposition.

```python
from mps_linalg import pinv

A = torch.randn(150, 80, device='mps')
A_pinv = pinv(A)

# Verify pseudoinverse properties
identity_check = torch.norm(A @ A_pinv @ A - A)  # Should be ~1e-5
```

### `solve(A, b)` / `lstsq(A, b)`
Linear system solving with automatic regularization.

```python
from mps_linalg import solve, lstsq

# Square system
A = torch.randn(100, 100, device='mps')
b = torch.randn(100, device='mps')
x = solve(A, b)

# Overdetermined system (least squares)
A_over = torch.randn(150, 100, device='mps')
b_over = torch.randn(150, device='mps')
x_ls = lstsq(A_over, b_over)
```

## 📊 Performance & Accuracy

### Numerical Accuracy (vs CPU reference):
- **QR reconstruction error**: < 1e-5
- **Pseudoinverse identity error**: < 2e-5  
- **Linear solve residual**: < 1e-4
- **Orthogonality error**: < 1e-6

### Performance Benefits:
- ✅ **100% MPS utilization** (no CPU transfers)
- ✅ **Memory efficient** (operations stay on GPU)
- ✅ **Scalable** (tested up to 1000x500 matrices)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_mps_compatibility.py
```

Expected output:
```
🚀 COMPREHENSIVE MPS COMPATIBILITY TESTING
✅ QR Decomposition: PASSED
✅ Pseudoinverse: PASSED  
✅ Linear System Solving: PASSED
✅ Condition Number Estimation: PASSED
✅ SINDy Integration: PASSED
✅ Performance Benchmark: PASSED
✅ Numerical Stability: PASSED

Overall: 7/7 tests passed (100.0%)
🌟 EXCELLENT: Full MPS compatibility achieved!
```

## 💡 Use Cases

### Scientific Computing
```python
# Physics simulations, PDE solving
from mps_linalg import solve
coefficient_matrix = build_finite_difference_matrix()
solution = solve(coefficient_matrix, rhs_vector)
```

### Machine Learning
```python
# Feature preprocessing, dimensionality reduction
from mps_linalg import pinv
X_plus = pinv(feature_matrix)
coefficients = X_plus @ target_vector
```

### Computer Vision
```python
# Camera calibration, fundamental matrix estimation
from mps_linalg import lstsq
F_matrix = lstsq(correspondence_matrix, zero_vector)
```

## 🔧 Advanced Usage

### Custom Regularization
```python
from qr_decomp import QR_mps
from mps_linalg import MPSLinearAlgebra

# Custom solver with specific regularization
mps_solver = MPSLinearAlgebra(default_shift=1e-6)
A_pinv = mps_solver.pinv(ill_conditioned_matrix)
```

### Condition Number Monitoring
```python
from mps_linalg import cond, MPSLinearAlgebraContext

with MPSLinearAlgebraContext(default_shift=1e-8) as mps_linalg:
    condition_num = mps_linalg.cond(matrix)
    if condition_num > 1e12:
        print("⚠️ Matrix is severely ill-conditioned")
    
    solution = mps_linalg.solve(matrix, rhs)
```

## 🏗️ Architecture

```
mps-linear-algebra/
├── qr_decomp.py              # Core QR decomposition implementation
├── mps_linalg.py            # High-level interface and utilities
├── test_mps_compatibility.py # Comprehensive testing suite
└── examples/                # Usage examples and integrations
```

### Core Components:

1. **Modified Gram-Schmidt**: Numerically stable orthogonalization
2. **Adaptive Regularization**: Condition-number aware stabilization  
3. **Device Management**: Automatic MPS/CPU selection and validation
4. **Error Handling**: Graceful degradation for edge cases

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Eigenvalue decomposition** for symmetric matrices
- **Singular Value Decomposition** approximation algorithms
- **Sparse matrix** support
- **Batch operations** optimization
- **Extended precision** modes

## 📚 Background

This package was developed to solve MPS compatibility issues in advanced scientific computing applications, specifically for:

- **Sparse Identification of Nonlinear Dynamics (SINDy)**
- **Physics-Informed Neural Networks (PINNs)**  
- **Koopman Operator Theory**
- **Computer Vision Pipelines**

The implementation uses **Modified Gram-Schmidt with reorthogonalization** for numerical stability and **QR-based algorithms** as drop-in replacements for SVD-dependent operations.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by numerical linear algebra best practices from LAPACK
- Developed for advanced physics discovery and machine learning applications
- Tested on Apple M-series hardware with PyTorch MPS backend

---

**⭐ If this package helps your research or projects, please consider starring the repository!**

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/mps-linear-algebra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mps-linear-algebra/discussions)
- **Documentation**: See examples/ directory for detailed usage patterns