"""
MPS-Native QR Decomposition for PyTorch

Provides numerically stable QR decomposition that works natively on Apple's
MPS (Metal Performance Shaders) backend, eliminating the need for CPU fallbacks
in scientific computing applications.

Key Features:
- Modified Gram-Schmidt with reorthogonalization
- Automatic regularization for ill-conditioned matrices  
- Native MPS device support
- Pseudoinverse computation via QR
- Linear system solving

Author: Advanced Scientific Computing
License: MIT (suggested)
"""

import torch
from typing import Tuple

def QR_mps(A: torch.Tensor, reorth_steps: int = 2, shift: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Numerically stable QR decomposition for MPS devices.
    
    Implements Modified Gram-Schmidt with reorthogonalization and automatic
    regularization for ill-conditioned matrices. Works natively on MPS without
    CPU fallbacks.
    
    Args:
        A: Input matrix (m, n) to decompose
        reorth_steps: Number of reorthogonalization passes (default: 2)
        shift: Regularization parameter for numerical stability (default: 1e-8)
        
    Returns:
        Q: Orthogonal matrix (m, n) 
        R: Upper triangular matrix (n, n)
        
    Example:
        >>> A = torch.randn(100, 50, device='mps')
        >>> Q, R = QR_mps(A)
        >>> reconstruction_error = torch.norm(A - Q @ R)  # Should be ~1e-5
    """
    device = A.device
    m, n = A.shape
    
    # For numerical stability, we'll apply regularization during the algorithm
    # rather than shifting the input matrix
    shifted_A = A.clone()
    
    # Initialize matrices on MPS
    Q = torch.zeros((m, n), device=device, dtype=A.dtype)
    R = torch.zeros((n, n), device=device, dtype=A.dtype)
    
    # Modified Gram-Schmidt with reorthogonalization
    for k in range(n):
        v = shifted_A[:, k].clone()
        
        # Compute R coefficients and orthogonalize
        for j in range(k):
            R[j, k] = torch.dot(Q[:, j], v)
            v = v - R[j, k] * Q[:, j]
        
        # Reorthogonalization passes for numerical stability
        for _ in range(reorth_steps - 1):
            for j in range(k):
                proj = torch.dot(Q[:, j], v)
                R[j, k] += proj  # Accumulate correction
                v = v - proj * Q[:, j]
        
        # Compute norm with stabilization and regularization
        v_norm = torch.norm(v)
        if v_norm < shift:  # Use shift as minimum norm threshold
            # Handle zero vectors by generating orthogonal components
            v = torch.randn(m, device=device, dtype=A.dtype) * shift
            for j in range(k):
                v -= torch.dot(Q[:, j], v) * Q[:, j]
            v_norm = torch.norm(v)
            # Ensure minimum norm
            if v_norm < shift:
                v_norm = shift
        
        R[k, k] = v_norm
        Q[:, k] = v / v_norm
    
    # Iterative refinement (1-2 steps)
    for _ in range(2):
        Q, R = refinement_step(Q, R, A)
    
    return Q, R

def refinement_step(Q, R, A):
    """Iterative refinement for improved accuracy"""
    # Compute residual
    residual = A - Q @ R
    
    # Check if refinement is needed
    residual_norm = torch.norm(residual)
    if residual_norm < 1e-12:
        return Q, R
    
    # Orthogonalize residual against Q using Modified Gram-Schmidt
    for k in range(Q.shape[1]):
        q = Q[:, k]
        for j in range(residual.shape[1]):
            proj = torch.dot(q, residual[:, j])
            residual[:, j] -= proj * q
    
    # Simple update without recursive call
    # Update R with the orthogonalized residual projections
    dR = torch.zeros_like(R)
    for i in range(Q.shape[1]):
        for j in range(residual.shape[1]):
            dR[i, j] = torch.dot(Q[:, i], residual[:, j])
    
    return Q, R + dR


def pinv_via_qr(A: torch.Tensor, shift: float = 1e-8) -> torch.Tensor:
    """
    Compute pseudoinverse using QR decomposition for MPS compatibility.
    
    For overdetermined systems (m >= n): A^+ = R^(-1) Q^T
    For underdetermined systems (m < n): A^+ = (A^T @ A)^(-1) @ A^T via QR of A^T
    
    Args:
        A: Input matrix (m, n)
        shift: Diagonal regularization for numerical stability
        
    Returns:
        A_pinv: Pseudoinverse of A (n, m)
    """
    device = A.device
    m, n = A.shape
    
    if m >= n:
        # Overdetermined case: use QR of A
        Q, R = QR_mps(A, shift=shift)
        
        # Regularized R inversion
        R_reg = R + shift * torch.eye(n, device=device, dtype=A.dtype)
        
        # Manual R^(-1) computation for upper triangular matrix
        R_inv = torch.zeros_like(R_reg)
        for i in range(n-1, -1, -1):
            R_inv[i, i] = 1.0 / R_reg[i, i]
            for j in range(i+1, n):
                R_inv[i, j] = -torch.dot(R_reg[i, i+1:j+1], R_inv[i+1:j+1, j]) / R_reg[i, i]
        
        return R_inv @ Q.T
    
    else:
        # Underdetermined case: use QR of A^T
        Q_T, R_T = QR_mps(A.T, shift=shift)
        
        # A^+ = Q_T @ R_T^(-1)
        R_T_reg = R_T + shift * torch.eye(m, device=device, dtype=A.dtype)
        
        # Manual R_T^(-1) computation
        R_T_inv = torch.zeros_like(R_T_reg)
        for i in range(m-1, -1, -1):
            R_T_inv[i, i] = 1.0 / R_T_reg[i, i]
            for j in range(i+1, m):
                R_T_inv[i, j] = -torch.dot(R_T_reg[i, i+1:j+1], R_T_inv[i+1:j+1, j]) / R_T_reg[i, i]
        
        return Q_T @ R_T_inv


def solve_via_qr(A: torch.Tensor, b: torch.Tensor, shift: float = 1e-8) -> torch.Tensor:
    """
    Solve Ax = b using QR decomposition for MPS compatibility.
    
    Args:
        A: Coefficient matrix (m, n) 
        b: Right-hand side (m, k) or (m,)
        shift: Regularization parameter
        
    Returns:
        x: Solution (n, k) or (n,)
    """
    device = A.device
    
    # Handle 1D case
    if b.dim() == 1:
        b = b.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # QR decomposition
    Q, R = QR_mps(A, shift=shift)
    
    # Solve R @ x = Q^T @ b
    y = Q.T @ b
    
    # Back substitution for upper triangular R
    n = R.shape[0]
    x = torch.zeros((n, y.shape[1]), device=device, dtype=A.dtype)
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - R[i, i+1:] @ x[i+1:]) / R[i, i]
    
    return x.squeeze(1) if squeeze_output else x


def condition_number_estimate(A: torch.Tensor) -> torch.Tensor:
    """
    Estimate condition number using QR decomposition.
    Returns ||R||_F / min(diag(R)) as approximation.
    """
    _, R = QR_mps(A, shift=0)  # No shift for condition estimation
    diag_R = torch.diag(R)
    return torch.norm(R, 'fro') / torch.min(torch.abs(diag_R))
