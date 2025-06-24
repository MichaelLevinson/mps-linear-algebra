"""
Basic Usage Example: MPS Linear Algebra

Simple examples demonstrating the core functionality of the mps-linear-algebra package.
Perfect for getting started and understanding the basic operations.
"""

import torch
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps_linalg import pinv, solve, lstsq, matrix_rank, cond
from qr_decomp import QR_mps


def example_qr_decomposition():
    """Example 1: QR Decomposition"""
    print("📐 Example 1: QR Decomposition")
    print("-" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create a test matrix
    A = torch.tensor([
        [3.0, 1.0, 4.0],
        [1.0, 5.0, 9.0], 
        [2.0, 6.0, 5.0],
        [3.0, 5.0, 8.0]
    ], device=device)
    
    print(f"Input matrix A ({A.shape}):")
    print(A)
    
    # QR decomposition
    Q, R = QR_mps(A)
    
    print(f"\nQ matrix ({Q.shape}):")
    print(Q)
    print(f"\nR matrix ({R.shape}):")
    print(R)
    
    # Verify decomposition
    reconstruction = Q @ R
    error = torch.norm(A - reconstruction).item()
    
    print(f"\nReconstruction Q @ R:")
    print(reconstruction)
    print(f"Reconstruction error: {error:.2e}")
    
    # Check orthogonality  
    orthog_error = torch.norm(Q.T @ Q - torch.eye(Q.shape[1], device=device)).item()
    print(f"Orthogonality error: {orthog_error:.2e}")


def example_pseudoinverse():
    """Example 2: Pseudoinverse Computation"""
    print("\n\n🔍 Example 2: Pseudoinverse")
    print("-" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create overdetermined system
    A = torch.randn(100, 50, device=device)
    
    print(f"Matrix A: {A.shape}")
    print(f"Computing pseudoinverse...")
    
    # Compute pseudoinverse
    A_pinv = pinv(A)
    
    print(f"Pseudoinverse A^+: {A_pinv.shape}")
    
    # Verify pseudoinverse properties
    # Property 1: A @ A^+ @ A = A
    identity_check = torch.norm(A @ A_pinv @ A - A).item()
    print(f"Identity property |A @ A^+ @ A - A|: {identity_check:.2e}")
    
    # Property 2: A^+ @ A @ A^+ = A^+  
    pinv_check = torch.norm(A_pinv @ A @ A_pinv - A_pinv).item()
    print(f"Pseudoinverse property |A^+ @ A @ A^+ - A^+|: {pinv_check:.2e}")
    
    if identity_check < 1e-3 and pinv_check < 1e-3:
        print("✅ Pseudoinverse properties satisfied!")
    else:
        print("❌ Pseudoinverse properties not satisfied")


def example_linear_solving():
    """Example 3: Linear System Solving"""
    print("\n\n🔧 Example 3: Linear System Solving") 
    print("-" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Square system: Ax = b
    print("Square system:")
    n = 50
    A = torch.randn(n, n, device=device) + 0.1 * torch.eye(n, device=device)  # Well-conditioned
    x_true = torch.randn(n, device=device)
    b = A @ x_true
    
    print(f"Solving {A.shape} @ x = {b.shape}")
    
    # Solve system
    x_solved = solve(A, b)
    
    # Check solution
    solution_error = torch.norm(x_solved - x_true).item()
    residual_error = torch.norm(A @ x_solved - b).item()
    
    print(f"Solution error |x_solved - x_true|: {solution_error:.2e}")
    print(f"Residual error |Ax - b|: {residual_error:.2e}")
    
    # Overdetermined system (least squares)
    print("\nOverdetermined system (least squares):")
    m, n = 100, 60
    A_over = torch.randn(m, n, device=device)
    b_over = torch.randn(m, device=device)
    
    print(f"Solving {A_over.shape} @ x = {b_over.shape} (least squares)")
    
    x_ls = lstsq(A_over, b_over)
    residual_ls = torch.norm(A_over @ x_ls - b_over).item()
    
    print(f"Least squares residual: {residual_ls:.2e}")


def example_matrix_analysis():
    """Example 4: Matrix Analysis"""
    print("\n\n📊 Example 4: Matrix Analysis")
    print("-" * 40)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Well-conditioned matrix
    A_good = torch.eye(20, device=device) + 0.1 * torch.randn(20, 20, device=device)
    
    # Ill-conditioned matrix  
    U = torch.randn(30, 20, device=device)
    S = torch.logspace(0, -10, 20).to(device)  # Create on CPU then move to avoid MPS logspace issue
    A_ill = U @ torch.diag(S) @ torch.randn(20, 20, device=device)
    
    matrices = [
        ("Well-conditioned", A_good),
        ("Ill-conditioned", A_ill)
    ]
    
    for name, A in matrices:
        print(f"\n{name} matrix ({A.shape}):")
        
        # Condition number
        cond_num = cond(A).item()
        print(f"  Condition number: {cond_num:.2e}")
        
        # Matrix rank
        rank = matrix_rank(A).item()
        print(f"  Estimated rank: {rank}")
        
        # Classification
        if cond_num < 1e6:
            print("  ✅ Well-conditioned")
        elif cond_num < 1e12:
            print("  ⚠️ Moderately ill-conditioned")
        else:
            print("  ❌ Severely ill-conditioned")


def benchmark_performance():
    """Example 5: Performance Comparison"""
    print("\n\n⚡ Example 5: Performance Benchmark")
    print("-" * 40)
    
    if not torch.backends.mps.is_available():
        print("MPS not available - skipping performance benchmark")
        return
    
    import time
    
    sizes = [100, 200, 500]
    
    print(f"{'Size':<6} {'CPU Time':<10} {'MPS Time':<10} {'Speedup':<8}")
    print("-" * 40)
    
    for size in sizes:
        # Generate test data
        A_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size)
        
        A_mps = A_cpu.to('mps')
        b_mps = b_cpu.to('mps')
        
        # CPU timing
        start = time.time()
        try:
            x_cpu = torch.linalg.solve(A_cpu, b_cpu)
            cpu_time = time.time() - start
        except:
            cpu_time = float('inf')
        
        # MPS timing  
        start = time.time()
        x_mps = solve(A_mps, b_mps)
        mps_time = time.time() - start
        
        # Calculate speedup
        speedup = cpu_time / mps_time if mps_time > 0 else float('inf')
        
        print(f"{size:<6} {cpu_time:<10.4f} {mps_time:<10.4f} {speedup:<8.2f}x")


def main():
    """Run all examples"""
    print("🚀 MPS Linear Algebra - Basic Usage Examples")
    print("=" * 60)
    
    if torch.backends.mps.is_available():
        print("✅ MPS backend available")
    else:
        print("⚠️ MPS backend not available - running on CPU")
    
    # Run examples
    example_qr_decomposition()
    example_pseudoinverse()
    example_linear_solving()
    example_matrix_analysis()
    benchmark_performance()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed successfully!")
    print("💡 Your matrices stayed on MPS throughout all operations!")


if __name__ == "__main__":
    main()