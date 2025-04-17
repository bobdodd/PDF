from setuptools import setup, find_packages

setup(
    name="italic-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "pymupdf>=1.18.0",
        "skl2onnx>=1.9.0",
        "onnx>=1.10.0",
        "onnxruntime>=1.8.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "italic-detector=src.main:main",
        ],
    },
)