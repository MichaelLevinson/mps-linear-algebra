"""
Setup script for mps-linear-algebra package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="mps-linear-algebra",
    version="1.0.0",
    author="Advanced Scientific Computing",
    author_email="your.email@example.com",
    description="Native linear algebra operations for PyTorch MPS backend",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mps-linear-algebra",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.4.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "scientific": [
            "scipy>=1.7.0",
            "matplotlib>=3.3.0",
        ],
    },
    keywords=[
        "pytorch", "mps", "linear-algebra", "apple-silicon", "gpu-computing",
        "scientific-computing", "machine-learning", "physics", "sindy"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mps-linear-algebra/issues",
        "Source": "https://github.com/yourusername/mps-linear-algebra",
        "Documentation": "https://github.com/yourusername/mps-linear-algebra/blob/main/README.md",
    },
)