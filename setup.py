from setuptools import find_packages, setup

setup(
    name="moa",
    packages=find_packages(),
    version="0.1.0",
    description="Develop an efficient algorithm for classifying drugs based on their biological activity",
    url="https://github.com/oleksandrsirenko/mechanisms-of-action-moa-prediction",
    author="Oleksandr Sirenko",
    author_email="oleksandr.sirenko2@nure.ua",
    license="MIT",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "tabnet",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
