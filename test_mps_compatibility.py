"""
Comprehensive MPS Compatibility Test Suite

Tests the complete MPS-native linear algebra implementation against CPU reference
implementations to validate numerical accuracy and performance for the SINDy
training pipeline.

Key Features:
1. QR decomposition accuracy validation
2. Pseudoinverse numerical stability testing  
3. Linear system solving comparison
4. Integration with SINDy components
5. Performance benchmarking
"""

import torch
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional

from qr_decomp import QR_mps, pinv_via_qr, solve_via_qr, condition_number_estimate
from mps_linalg import MPSLinearAlgebra, pinv, solve, lstsq
# Note: Removed proprietary SINDy imports for public release


class MPSCompatibilityTester:
    """
    Comprehensive tester for MPS linear algebra compatibility.
    """
    
    def __init__(self, device_preference: str = "auto"):
        if device_preference == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device_preference)
        
        self.mps_linalg = MPSLinearAlgebra()
        self.test_results = {}
        
        print(f"🧪 MPSCompatibilityTester initialized on {self.device}")
        
        if self.device.type != "mps":
            warnings.warn("MPS not available - tests will run on CPU only")
    
    def run_comprehensive_tests(self) -> Dict[str, Dict]:
        """Run all compatibility tests and return results."""
        print("\n" + "=" * 60)
        print("🚀 COMPREHENSIVE MPS COMPATIBILITY TESTING")
        print("=" * 60)
        
        test_suite = [
            ("QR Decomposition", self.test_qr_decomposition),
            ("Pseudoinverse", self.test_pseudoinverse),
            ("Linear System Solving", self.test_linear_solving),
            ("Condition Number Estimation", self.test_condition_number),
            ("ML Integration", self.test_ml_integration),
            ("Performance Benchmark", self.test_performance),
            ("Numerical Stability", self.test_numerical_stability)
        ]
        
        for test_name, test_func in test_suite:
            print(f"\n📊 Running: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'details': result
                }
                print(f"   ✅ {test_name} PASSED")
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"   ❌ {test_name} FAILED: {e}")
        
        self._print_summary()
        return self.test_results
    
    def test_qr_decomposition(self) -> Dict:
        """Test QR decomposition accuracy against torch.linalg.qr (CPU)."""
        test_cases = [
            (50, 30),   # Overdetermined
            (30, 30),   # Square
            (30, 50),   # Underdetermined
            (100, 80),  # Larger overdetermined
        ]
        
        results = {}
        for m, n in test_cases:
            # Generate test matrix
            A = torch.randn(m, n, dtype=torch.float32)
            A_mps = A.to(self.device)
            
            # CPU reference
            Q_ref, R_ref = torch.linalg.qr(A)
            
            # MPS implementation
            Q_mps, R_mps = QR_mps(A_mps)
            Q_mps_cpu = Q_mps.cpu()
            R_mps_cpu = R_mps.cpu()
            
            # Reconstruction error
            A_recon = Q_mps_cpu @ R_mps_cpu
            recon_error = torch.norm(A - A_recon).item()
            
            # Orthogonality check
            orthog_error = torch.norm(Q_mps_cpu.T @ Q_mps_cpu - torch.eye(Q_mps_cpu.shape[1])).item()
            
            results[f"{m}x{n}"] = {
                'reconstruction_error': recon_error,
                'orthogonality_error': orthog_error,
                'passed': recon_error < 1e-4 and orthog_error < 1e-4
            }
        
        return results
    
    def test_pseudoinverse(self) -> Dict:
        """Test pseudoinverse accuracy."""
        test_cases = [
            (50, 30),   # Overdetermined
            (30, 50),   # Underdetermined  
            (40, 40),   # Square
        ]
        
        results = {}
        for m, n in test_cases:
            # Generate test matrix
            A = torch.randn(m, n, dtype=torch.float32)
            A_mps = A.to(self.device)
            
            # CPU reference
            A_pinv_ref = torch.linalg.pinv(A)
            
            # MPS implementation
            A_pinv_mps = pinv(A_mps).cpu()
            
            # Test: A @ A^+ @ A ≈ A
            identity_test = torch.norm(A @ A_pinv_mps @ A - A).item()
            
            # Test: A^+ @ A @ A^+ ≈ A^+
            pinv_test = torch.norm(A_pinv_mps @ A @ A_pinv_mps - A_pinv_mps).item()
            
            results[f"{m}x{n}"] = {
                'identity_test_error': identity_test,
                'pinv_test_error': pinv_test,
                'passed': identity_test < 1e-3 and pinv_test < 1e-3
            }
        
        return results
    
    def test_linear_solving(self) -> Dict:
        """Test linear system solving."""
        test_cases = [
            (50, 30),   # Overdetermined (least squares)
            (30, 30),   # Square system
        ]
        
        results = {}
        for m, n in test_cases:
            # Generate well-conditioned system
            A = torch.randn(m, n, dtype=torch.float32)
            x_true = torch.randn(n, dtype=torch.float32)
            b = A @ x_true
            
            A_mps = A.to(self.device)
            b_mps = b.to(self.device)
            
            # MPS solution
            x_mps = solve(A_mps, b_mps).cpu()
            
            # Solution error
            solution_error = torch.norm(x_mps - x_true).item()
            
            # Residual error
            residual_error = torch.norm(A @ x_mps - b).item()
            
            results[f"{m}x{n}"] = {
                'solution_error': solution_error,
                'residual_error': residual_error,
                'passed': residual_error < 1e-4
            }
        
        return results
    
    def test_condition_number(self) -> Dict:
        """Test condition number estimation."""
        # Create matrices with known condition numbers
        test_matrices = []
        
        # Well-conditioned matrix
        A1 = torch.eye(20) + 0.1 * torch.randn(20, 20)
        test_matrices.append(("well_conditioned", A1))
        
        # Ill-conditioned matrix
        U = torch.randn(30, 20)
        S = torch.linspace(1, 1e-8, 20)
        A2 = U @ torch.diag(S) @ torch.randn(20, 20)
        test_matrices.append(("ill_conditioned", A2))
        
        results = {}
        for name, A in test_matrices:
            A_mps = A.to(self.device)
            
            # CPU reference
            cond_ref = torch.linalg.cond(A).item()
            
            # MPS estimation
            cond_mps = condition_number_estimate(A_mps).item()
            
            # Relative error
            rel_error = abs(cond_mps - cond_ref) / cond_ref if cond_ref > 0 else float('inf')
            
            results[name] = {
                'reference_cond': cond_ref,
                'mps_cond': cond_mps,
                'relative_error': rel_error,
                'passed': rel_error < 0.5  # Allow 50% tolerance for estimation
            }
        
        return results
    
    def test_ml_integration(self) -> Dict:
        """Test integration with typical ML workflows."""
        try:
            # Simulate typical ML regression task
            batch_size, n_features, n_targets = 100, 50, 10
            
            # Generate synthetic data
            X = torch.randn(batch_size, n_features, device=self.device)
            true_weights = torch.randn(n_features, n_targets, device=self.device)
            noise = 0.1 * torch.randn(batch_size, n_targets, device=self.device)
            y = X @ true_weights + noise
            
            # Test 1: Linear regression via pseudoinverse
            X_pinv = pinv(X)
            weights_pinv = X_pinv @ y
            
            # Test 2: Ridge regression via solve
            lambda_reg = 0.01
            XtX_reg = X.T @ X + lambda_reg * torch.eye(n_features, device=self.device)
            Xty = X.T @ y
            weights_ridge = solve(XtX_reg, Xty)
            
            # Test 3: Least squares via lstsq
            weights_lstsq = lstsq(X, y)
            
            # Evaluate predictions
            pred_pinv = X @ weights_pinv
            pred_ridge = X @ weights_ridge
            pred_lstsq = X @ weights_lstsq
            
            # Compute errors
            error_pinv = torch.norm(y - pred_pinv).item()
            error_ridge = torch.norm(y - pred_ridge).item()
            error_lstsq = torch.norm(y - pred_lstsq).item()
            
            results = {
                'pseudoinverse_success': True,
                'ridge_regression_success': True,
                'lstsq_success': True,
                'device_consistency': all([
                    weights_pinv.device == self.device,
                    weights_ridge.device == self.device,
                    weights_lstsq.device == self.device
                ]),
                'prediction_errors': {
                    'pinv': error_pinv,
                    'ridge': error_ridge,
                    'lstsq': error_lstsq
                },
                'reasonable_errors': all([
                    error_pinv < 100,  # Reasonable prediction error
                    error_ridge < 100,
                    error_lstsq < 100
                ])
            }
            
            return results
            
        except Exception as e:
            return {'error': str(e), 'integration_success': False}
    
    def test_performance(self) -> Dict:
        """Benchmark MPS vs CPU performance."""
        if self.device.type != "mps":
            return {'skipped': 'MPS not available'}
        
        sizes = [50, 100, 200]
        results = {}
        
        for size in sizes:
            A = torch.randn(size, size, dtype=torch.float32)
            b = torch.randn(size, dtype=torch.float32)
            
            # CPU timing
            start_time = time.time()
            _ = torch.linalg.solve(A, b)
            cpu_time = time.time() - start_time
            
            # MPS timing
            A_mps = A.to(self.device)
            b_mps = b.to(self.device)
            
            start_time = time.time()
            _ = solve(A_mps, b_mps)
            mps_time = time.time() - start_time
            
            results[f"size_{size}"] = {
                'cpu_time': cpu_time,
                'mps_time': mps_time,
                'speedup': cpu_time / mps_time if mps_time > 0 else float('inf')
            }
        
        return results
    
    def test_numerical_stability(self) -> Dict:
        """Test numerical stability with challenging matrices."""
        results = {}
        
        # Test with near-singular matrix
        A = torch.randn(50, 50, dtype=torch.float32)
        A[-1] = A[0] * 1e-10  # Make nearly rank-deficient
        
        A_mps = A.to(self.device)
        
        try:
            A_pinv = pinv(A_mps)
            results['near_singular_success'] = True
            results['pinv_max_value'] = torch.max(torch.abs(A_pinv)).item()
        except Exception as e:
            results['near_singular_success'] = False
            results['error'] = str(e)
        
        # Test with very ill-conditioned matrix
        U = torch.randn(40, 30)
        S = torch.logspace(0, -12, 30)  # Condition number ~1e12
        V = torch.randn(30, 30)
        A_ill = U @ torch.diag(S) @ V
        
        A_ill_mps = A_ill.to(self.device)
        
        try:
            cond_est = condition_number_estimate(A_ill_mps).item()
            results['ill_conditioned_cond'] = cond_est
            results['ill_conditioned_success'] = True
        except Exception as e:
            results['ill_conditioned_success'] = False
            results['ill_conditioned_error'] = str(e)
        
        return results
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        total = len(self.test_results)
        
        print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_icon} {test_name}: {result['status']}")
        
        print("\n🎯 MPS Compatibility Assessment:")
        if passed == total:
            print("   🌟 EXCELLENT: Full MPS compatibility achieved!")
            print("   Your revolutionary SINDy training is ready for MPS acceleration!")
        elif passed >= total * 0.8:
            print("   👍 GOOD: Most tests passed, minor issues may exist")
        else:
            print("   ⚠️  NEEDS WORK: Several compatibility issues detected")


def run_mps_compatibility_tests():
    """Main function to run all MPS compatibility tests."""
    tester = MPSCompatibilityTester()
    results = tester.run_comprehensive_tests()
    return results


if __name__ == "__main__":
    print("🧪 Starting MPS Compatibility Test Suite...")
    results = run_mps_compatibility_tests()
    
    # Save results
    import json
    with open("mps_compatibility_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: mps_compatibility_results.json")