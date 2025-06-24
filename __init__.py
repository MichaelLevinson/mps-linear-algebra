"""
MPS Linear Algebra - Native linear algebra operations for PyTorch MPS backend

This package provides numerically stable linear algebra operations that work
natively on Apple's Metal Performance Shaders (MPS) backend, eliminating 
the need for expensive CPU fallbacks in scientific computing applications.

Main modules:
- qr_decomp: Core QR decomposition implementation
- mps_linalg: High-level linear algebra interface

Key functions:
- QR_mps: QR decomposition with Modified Gram-Schmidt
- pinv: Pseudoinverse computation via QR
- solve: Linear system solving
- lstsq: Least squares regression
- matrix_rank: Matrix rank estimation
- cond: Condition number estimation
"""

__version__ = "1.0.0"
__author__ = "Advanced Scientific Computing"
__license__ = "MIT"

# Import main functions for convenient access
from .qr_decomp import QR_mps, pinv_via_qr, solve_via_qr, condition_number_estimate
from .mps_linalg import (
    MPSLinearAlgebra, 
    MPSLinearAlgebraContext,
    pinv, 
    solve, 
    lstsq, 
    matrix_rank, 
    cond,
    mps_linalg
)

__all__ = [
    # Core QR functions
    'QR_mps',
    'pinv_via_qr', 
    'solve_via_qr',
    'condition_number_estimate',
    
    # High-level interface
    'MPSLinearAlgebra',
    'MPSLinearAlgebraContext', 
    'mps_linalg',
    
    # Convenience functions
    'pinv',
    'solve',
    'lstsq', 
    'matrix_rank',
    'cond',
    
    # Package metadata
    '__version__',
    '__author__',
    '__license__'
]