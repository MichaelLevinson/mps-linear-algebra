"""
MPS-Compatible Linear Algebra Utilities for SINDy Training

This module provides MPS-native implementations of essential linear algebra operations
that are not supported by PyTorch's MPS backend, specifically designed for the
integrated Latent-SINDy + FiLM-LSTM training pipeline.

Key Features:
1. QR-based pseudoinverse computation
2. Regularized linear system solving  
3. Numerical stability monitoring
4. Automatic fallback strategies
5. Full MPS device compatibility

Author: Michael Levinson
"""

import torch
import torch.nn as nn
import warnings
from typing import Tuple, Optional, Union
from qr_decomp import QR_mps, pinv_via_qr, solve_via_qr, condition_number_estimate


class MPSLinearAlgebra:
    """
    MPS-compatible linear algebra operations for SINDy training.
    
    Provides drop-in replacements for torch.linalg operations that are
    not supported on MPS devices, specifically optimized for the numerical
    stability requirements of physics discovery and neural forecasting.
    """
    
    def __init__(self, default_shift: float = 1e-8, condition_threshold: float = 1e12):
        self.default_shift = default_shift
        self.condition_threshold = condition_threshold
        self.device_warnings_shown = set()
    
    def pinv(self, A: torch.Tensor, rcond: float = 1e-15) -> torch.Tensor:
        """
        MPS-compatible pseudoinverse using QR decomposition.
        
        Args:
            A: Input matrix (m, n)
            rcond: Relative condition number threshold (for compatibility)
            
        Returns:
            Pseudoinverse of A
        """
        if A.device.type != 'mps' and 'pinv_cpu_warning' not in self.device_warnings_shown:
            warnings.warn("pinv called on non-MPS device - consider using torch.linalg.pinv")
            self.device_warnings_shown.add('pinv_cpu_warning')
        
        # Estimate condition number and adjust shift
        cond_est = condition_number_estimate(A)
        adaptive_shift = max(self.default_shift, 1e-16 * cond_est.item())
        
        return pinv_via_qr(A, shift=adaptive_shift)
    
    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        MPS-compatible linear system solver using QR decomposition.
        
        Args:
            A: Coefficient matrix (n, n) or (m, n) for least squares
            b: Right-hand side vector(s)
            
        Returns:
            Solution x such that Ax = b (or least squares solution)
        """
        if A.device.type != 'mps' and 'solve_cpu_warning' not in self.device_warnings_shown:
            warnings.warn("solve called on non-MPS device - consider using torch.linalg.solve")
            self.device_warnings_shown.add('solve_cpu_warning')
        
        # Adaptive regularization based on condition number
        cond_est = condition_number_estimate(A)
        adaptive_shift = max(self.default_shift, 1e-16 * cond_est.item())
        
        return solve_via_qr(A, b, shift=adaptive_shift)
    
    def lstsq(self, A: torch.Tensor, b: torch.Tensor, rcond: Optional[float] = None) -> torch.Tensor:
        """
        MPS-compatible least squares solver.
        
        Args:
            A: Coefficient matrix (m, n)
            b: Right-hand side (m, k) or (m,)
            rcond: Condition number cutoff (for compatibility)
            
        Returns:
            Least squares solution
        """
        # For overdetermined systems, QR gives least squares solution directly
        if A.shape[0] >= A.shape[1]:
            return self.solve(A, b)
        else:
            # For underdetermined systems, use pseudoinverse
            return self.pinv(A) @ b
    
    def svd_substitute(self, A: torch.Tensor, full_matrices: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SVD substitute using QR decomposition for specific SINDy use cases.
        
        This is NOT a full SVD replacement but provides the essential functionality
        needed for pseudoinverse computation and dimensionality reduction in SINDy.
        
        Args:
            A: Input matrix (m, n)
            full_matrices: Compatibility parameter (ignored)
            
        Returns:
            Tuple approximating (U, S, V^T) for pseudoinverse computation
        """
        warnings.warn(
            "svd_substitute provides limited SVD functionality for pseudoinverse computation only. "
            "For full SVD, consider CPU fallback or alternative algorithms."
        )
        
        # Use QR-based approach for essential SVD functionality
        Q, R = QR_mps(A, shift=self.default_shift)
        
        # For pseudoinverse purposes, we need something that approximates U*S*V^T = A
        # This is a simplified approximation - not mathematically equivalent to SVD
        m, n = A.shape
        
        if m >= n:
            # Return Q, diagonal of R, and identity-like V^T
            U = Q
            S = torch.abs(torch.diag(R))  # Approximate singular values
            Vt = torch.eye(n, device=A.device, dtype=A.dtype)
        else:
            # For wide matrices, transpose approach
            Q_t, R_t = QR_mps(A.T, shift=self.default_shift)
            U = torch.eye(m, device=A.device, dtype=A.dtype)
            S = torch.abs(torch.diag(R_t))
            Vt = Q_t.T
        
        return U, S, Vt
    
    def matrix_rank(self, A: torch.Tensor, tol: Optional[float] = None) -> torch.Tensor:
        """
        Estimate matrix rank using QR decomposition.
        
        Args:
            A: Input matrix
            tol: Tolerance for rank determination
            
        Returns:
            Estimated rank
        """
        Q, R = QR_mps(A, shift=0)  # No shift for rank estimation
        diag_R = torch.abs(torch.diag(R))
        
        if tol is None:
            tol = max(A.shape) * torch.finfo(A.dtype).eps * torch.max(diag_R)
        
        return torch.sum(diag_R > tol)
    
    def cond(self, A: torch.Tensor) -> torch.Tensor:
        """
        Estimate condition number using QR decomposition.
        
        Args:
            A: Input matrix
            
        Returns:
            Estimated condition number
        """
        return condition_number_estimate(A)


class MPSLinearAlgebraContext:
    """
    Context manager for automatic MPS linear algebra operations.
    
    Usage:
        with MPSLinearAlgebraContext() as mps_linalg:
            pinv_A = mps_linalg.pinv(A)
            solution = mps_linalg.solve(B, c)
    """
    
    def __init__(self, default_shift: float = 1e-8):
        self.mps_linalg = MPSLinearAlgebra(default_shift=default_shift)
    
    def __enter__(self):
        return self.mps_linalg
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def validate_mps_tensor(tensor: torch.Tensor, operation_name: str = "operation") -> torch.Tensor:
    """
    Validate tensor for MPS operations.
    
    Args:
        tensor: Input tensor
        operation_name: Name of operation for error messages
        
    Returns:
        Validated tensor
        
    Raises:
        RuntimeError: If tensor is not suitable for MPS operations
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{operation_name} requires torch.Tensor input")
    
    if torch.isnan(tensor).any():
        raise ValueError(f"{operation_name} input contains NaN values")
    
    if torch.isinf(tensor).any():
        raise ValueError(f"{operation_name} input contains infinite values")
    
    if tensor.device.type == 'mps' and tensor.dtype not in [torch.float32, torch.float16]:
        warnings.warn(f"{operation_name}: MPS works best with float32/float16, got {tensor.dtype}")
    
    return tensor


def adaptive_shift_selection(A: torch.Tensor, base_shift: float = 1e-8) -> float:
    """
    Select adaptive regularization shift based on matrix properties.
    
    Args:
        A: Input matrix
        base_shift: Base regularization amount
        
    Returns:
        Recommended shift value
    """
    # Estimate condition number
    cond_est = condition_number_estimate(A)
    
    # Adaptive shift based on condition number
    if cond_est > 1e12:
        return max(base_shift, 1e-12 * cond_est.item())
    elif cond_est > 1e8:
        return max(base_shift, 1e-14 * cond_est.item())
    else:
        return base_shift


# Global instance for convenience
mps_linalg = MPSLinearAlgebra()

# Convenience functions that match torch.linalg interface
def pinv(A: torch.Tensor, rcond: float = 1e-15) -> torch.Tensor:
    """Drop-in replacement for torch.linalg.pinv on MPS."""
    return mps_linalg.pinv(A, rcond)

def solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.linalg.solve on MPS."""
    return mps_linalg.solve(A, b)

def lstsq(A: torch.Tensor, b: torch.Tensor, rcond: Optional[float] = None) -> torch.Tensor:
    """Drop-in replacement for torch.linalg.lstsq on MPS."""
    return mps_linalg.lstsq(A, b, rcond)

def matrix_rank(A: torch.Tensor, tol: Optional[float] = None) -> torch.Tensor:
    """Drop-in replacement for torch.linalg.matrix_rank on MPS."""
    return mps_linalg.matrix_rank(A, tol)

def cond(A: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.linalg.cond on MPS."""
    return mps_linalg.cond(A)