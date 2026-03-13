from setuptools import setup, find_packages

setup(
    name="smart-alarm-triage",
    version="0.1.0",
    description="ML-based alarm event classification using CICIDS2017",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov"]
    },
)
