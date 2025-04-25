from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural_network_lib",
    version="1.1.0",
    author="Rufus Pearce",
    author_email="rufuspearce1@gmail.com",
    description="A comprehensive neural network library with Hebbian learning, neurogenesis, and backpropagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural_network_lib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
    extras_require={
        'visualization': ['PyQt5>=5.15.0'],
        'examples': ['matplotlib>=3.3.0'],
        'dev': ['pytest>=6.0.0', 'flake8>=3.8.0', 'twine>=3.2.0'],
    },
)