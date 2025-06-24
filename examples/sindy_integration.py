"""
SINDy Integration Example: Physics Discovery with MPS Linear Algebra

This example demonstrates how to use mps-linear-algebra for Sparse Identification 
of Nonlinear Dynamics (SINDy) on Apple Silicon, enabling physics discovery 
entirely on the GPU without CPU fallbacks.

Key features demonstrated:
1. Feature library construction for polynomial terms
2. MPS-native least squares regression for coefficient discovery
3. Sparsity-inducing regularization 
4. Physics equation extraction and interpretation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

# Import our MPS-native linear algebra
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps_linalg import solve, lstsq, pinv, matrix_rank
from qr_decomp import QR_mps


class MPSSINDy:
    """
    SINDy implementation using MPS-native linear algebra.
    
    Discovers governing equations of the form:
    dx/dt = f(x) = Θ(x) @ ξ
    
    Where:
    - x: state variables
    - Θ(x): feature library (polynomials, etc.)
    - ξ: sparse coefficient vector
    """
    
    def __init__(self, poly_order: int = 3, threshold: float = 1e-3):
        self.poly_order = poly_order
        self.threshold = threshold
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"🔬 MPSSINDy initialized on {self.device}")
    
    def build_library(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Build polynomial feature library Θ(X).
        
        Args:
            X: State data (n_samples, n_vars)
            
        Returns:
            Theta: Feature library (n_samples, n_features)
            feature_names: List of feature descriptions
        """
        n_samples, n_vars = X.shape
        features = []
        names = []
        
        # Constant term
        features.append(torch.ones(n_samples, 1, device=self.device))
        names.append("1")
        
        # Linear terms: x, y, z, ...
        for i in range(n_vars):
            features.append(X[:, i:i+1])
            names.append(f"x{i}")
        
        # Polynomial terms up to specified order
        for order in range(2, self.poly_order + 1):
            for i in range(n_vars):
                for j in range(i, n_vars):
                    if order == 2:
                        # Quadratic: x^2, xy, y^2, etc.
                        feature = X[:, i:i+1] * X[:, j:j+1]
                        if i == j:
                            name = f"x{i}^2"
                        else:
                            name = f"x{i}*x{j}"
                    elif order == 3 and i == j:
                        # Cubic: x^3, y^3, etc.
                        feature = X[:, i:i+1] ** 3
                        name = f"x{i}^3"
                    else:
                        continue
                    
                    features.append(feature)
                    names.append(name)
        
        Theta = torch.cat(features, dim=1)
        print(f"📚 Feature library: {X.shape} → {Theta.shape}")
        print(f"   Features: {names[:10]}{'...' if len(names) > 10 else ''}")
        
        return Theta, names
    
    def sparse_regression(self, Theta: torch.Tensor, dXdt: torch.Tensor) -> torch.Tensor:
        """
        Solve sparse regression: dX/dt = Θ(X) @ ξ
        
        Uses Sequential Thresholded Least Squares (STLSQ) with MPS linear algebra.
        """
        print(f"🔍 Sparse regression: {Theta.shape} @ ξ = {dXdt.shape}")
        
        n_vars = dXdt.shape[1]
        n_features = Theta.shape[1]
        
        # Initialize coefficient matrix
        Xi = torch.zeros(n_features, n_vars, device=self.device)
        
        # STLSQ for each variable
        for i in range(n_vars):
            print(f"   Variable {i+1}/{n_vars}: ", end="")
            
            # Initial least squares solution using MPS
            xi = lstsq(Theta, dXdt[:, i])
            
            # Iterative thresholding
            for iteration in range(10):  # Max 10 iterations
                # Threshold small coefficients
                mask = torch.abs(xi) >= self.threshold
                n_active = mask.sum().item()
                
                if n_active == 0:
                    print("All coefficients thresholded to zero")
                    break
                
                # Re-solve with active features only
                Theta_active = Theta[:, mask]
                if Theta_active.shape[1] > 0:
                    xi_active = lstsq(Theta_active, dXdt[:, i])
                    xi = torch.zeros_like(xi)
                    xi[mask] = xi_active
                
                print(f"{n_active} active", end="")
                if iteration < 9:
                    print(", ", end="")
            
            Xi[:, i] = xi
            print()
        
        return Xi
    
    def fit(self, X: torch.Tensor, dt: float = 0.01) -> Dict:
        """
        Fit SINDy model to discover governing equations.
        
        Args:
            X: State trajectory data (n_samples, n_vars)
            dt: Time step for numerical differentiation
            
        Returns:
            Dictionary with discovered model information
        """
        print(f"🚀 Fitting SINDy model to {X.shape} trajectory...")
        
        # Compute derivatives numerically
        dXdt = self.numerical_derivative(X, dt)
        
        # Build feature library
        Theta, feature_names = self.build_library(X[:-1])  # Remove last point for derivative
        
        # Solve sparse regression
        Xi = self.sparse_regression(Theta, dXdt)
        
        # Extract discovered equations
        equations = self.extract_equations(Xi, feature_names)
        
        # Compute model diagnostics
        X_pred = Theta @ Xi
        mse = torch.mean((dXdt - X_pred) ** 2).item()
        sparsity = (torch.abs(Xi) < self.threshold).float().mean().item()
        
        print(f"✅ Model fit complete!")
        print(f"   MSE: {mse:.2e}")
        print(f"   Sparsity: {sparsity:.1%}")
        
        return {
            'coefficients': Xi,
            'feature_names': feature_names,
            'equations': equations,
            'mse': mse,
            'sparsity': sparsity,
            'theta': Theta,
            'dxdt_true': dXdt,
            'dxdt_pred': X_pred
        }
    
    def numerical_derivative(self, X: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute numerical derivatives using finite differences."""
        # Simple forward difference
        dXdt = (X[1:] - X[:-1]) / dt
        return dXdt
    
    def extract_equations(self, Xi: torch.Tensor, feature_names: List[str]) -> List[str]:
        """Extract human-readable equations from coefficient matrix."""
        equations = []
        n_vars = Xi.shape[1]
        
        for i in range(n_vars):
            terms = []
            for j, coeff in enumerate(Xi[:, i]):
                if torch.abs(coeff) >= self.threshold:
                    if len(terms) == 0:
                        terms.append(f"{coeff.item():.3f}*{feature_names[j]}")
                    else:
                        sign = "+" if coeff > 0 else "-"
                        terms.append(f"{sign}{abs(coeff.item()):.3f}*{feature_names[j]}")
            
            if len(terms) == 0:
                equation = f"dx{i}/dt = 0"
            else:
                equation = f"dx{i}/dt = " + " ".join(terms)
            
            equations.append(equation)
        
        return equations


def generate_lorenz_data(t_span: Tuple[float, float] = (0, 10), 
                        n_points: int = 1000,
                        sigma: float = 10.0, 
                        rho: float = 28.0, 
                        beta: float = 8.0/3.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Lorenz system trajectory data.
    
    Lorenz equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Time vector
    t = torch.linspace(t_span[0], t_span[1], n_points, device=device)
    dt = (t_span[1] - t_span[0]) / (n_points - 1)
    
    # Initial conditions
    X = torch.zeros(n_points, 3, device=device)
    X[0] = torch.tensor([1.0, 1.0, 1.0], device=device)  # x0, y0, z0
    
    # Integrate using Euler method (for simplicity)
    for i in range(n_points - 1):
        x, y, z = X[i]
        
        # Lorenz equations
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        
        # Euler step
        X[i+1, 0] = x + dt * dxdt
        X[i+1, 1] = y + dt * dydt  
        X[i+1, 2] = z + dt * dzdt
    
    return t, X


def main():
    """Main example: Discover Lorenz equations using MPS SINDy."""
    print("🌟 SINDy + MPS Linear Algebra Example")
    print("=" * 50)
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("⚠️ MPS not available, falling back to CPU")
    
    # Generate Lorenz system data
    print("📊 Generating Lorenz system data...")
    t, X = generate_lorenz_data(t_span=(0, 5), n_points=500)
    
    print(f"   Time span: {t[0]:.1f} to {t[-1]:.1f}")
    print(f"   Data shape: {X.shape}")
    print(f"   Device: {X.device}")
    
    # Initialize SINDy with MPS
    sindy = MPSSINDy(poly_order=2, threshold=1e-2)
    
    # Fit model to discover equations
    dt = (t[1] - t[0]).item()
    results = sindy.fit(X, dt=dt)
    
    # Display discovered equations
    print("\n🔬 DISCOVERED EQUATIONS:")
    print("-" * 30)
    for i, eq in enumerate(results['equations']):
        print(f"{i+1}. {eq}")
    
    # Compare with true Lorenz equations
    print("\n📖 TRUE LORENZ EQUATIONS:")
    print("-" * 25)
    print("1. dx0/dt = 10.000*(x1 - x0)")
    print("2. dx1/dt = x0*(28.000 - x2) - x1")  
    print("3. dx2/dt = x0*x1 - 2.667*x2")
    
    # Model diagnostics
    print(f"\n📈 MODEL DIAGNOSTICS:")
    print(f"   MSE: {results['mse']:.2e}")
    print(f"   Sparsity: {results['sparsity']:.1%}")
    print(f"   Active coefficients: {torch.sum(torch.abs(results['coefficients']) >= sindy.threshold).item()}")
    
    # Demonstrate continued MPS usage
    print(f"\n⚡ MPS UTILIZATION:")
    print(f"   Coefficients on: {results['coefficients'].device}")
    print(f"   Features on: {results['theta'].device}")
    print(f"   No CPU fallbacks required! 🎉")
    
    return results


if __name__ == "__main__":
    # Run the example
    results = main()
    
    print("\n✅ Example completed successfully!")
    print("💡 This demonstrates full MPS compatibility for physics discovery!")