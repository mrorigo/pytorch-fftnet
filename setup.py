from setuptools import setup, find_packages

setup(
    name="fftnet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "tiktoken",
        "matplotlib",
        "numpy",
        "tqdm",
    ],
    author="FFTNet Implementation",
    description="PyTorch implementation of FFTNet: An Efficient Alternative to Self-Attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
