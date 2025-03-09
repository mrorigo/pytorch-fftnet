from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="fftnet",
    version="0.1.0",
    packages=find_packages(where='fftnet'),
    package_dir={'': 'fftnet'},
    install_requires=[
        "torch>=1.10.0",
        "tiktoken",
        "matplotlib",
        "numpy",
        "tqdm",
    ],

    author="FFTNet Implementation Team",
    author_email="mrorigo@gmail.com",
    description="PyTorch implementation of FFTNet: An Efficient Alternative to Self-Attention",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mrorigo/fftnet",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Framework :: PyTorch", # Add framework classifier
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords="fft, fourier, pytorch, attention, nlp, deep learning, efficient, transformer",
    project_urls={
        # "Documentation": "https://your-documentation-link.com",
        "Source Code": "https://github.com/morigo/fftnet",
        "Bug Tracker": "https://github.com/mrorigo/fftnet/issues",
    },
    python_requires=">=3.7", # Specify minimum Python version

    # Package Data (if you have non-python files to include, like datasets - OPTIONAL for now)
    # package_data={
    #     'fftnet': ['tinyshakespeare.txt'], # Example: Include the dataset (if desired to package it)
    # },
    include_package_data=True, # To make package_data work, and include other files defined in MANIFEST.in (if you have one)

)
