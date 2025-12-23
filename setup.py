"""
Setup script for PrecisionProject - A comprehensive precision testing framework for PyTorch and vLLM models.
"""

from setuptools import setup, find_packages
import os

# Read the version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'PrecisionProject', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '1.0.0'

# Read README if it exists
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A comprehensive precision testing framework for PyTorch and vLLM models"

# Read requirements from requirements.txt
def get_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="PrecisionProject",
    version=get_version(),
    author="PrecisionProject Team",
    author_email="contact@precisionproject.dev",
    description="A comprehensive precision testing framework for PyTorch and vLLM models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/precisionproject/precisionproject",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "precisionproject=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "PrecisionProject": [
            "*.yaml",
            "*.json",
            "*.txt",
        ],
    },
    keywords="pytorch, vllm, precision, testing, machine-learning, model-validation",
    project_urls={
        "Bug Reports": "https://github.com/precisionproject/precisionproject/issues",
        "Source": "https://github.com/precisionproject/precisionproject",
        "Documentation": "https://precisionproject.readthedocs.io/",
    },
)