from setuptools import setup, find_packages

setup(
    name="signature-detection",  # You can change this name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "certifi>=2025.1.31",
        "dropbox>=12.0.2",
        "easyocr>=1.7.2",
        "numpy>=2.1.2",
        "opencv-python>=4.10.0.84",
        "pandas>=2.2.3",
        "pdf2image>=1.17.0",
        "pillow>=11.0.0",
        "pytesseract>=0.3.13",
        "requests>=2.32.3",
        "torch>=2.5.0",
        "torchvision>=0.20.0",
    ],
    python_requires=">=3.8",
    author="Your Name",  # Add your name
    author_email="your.email@example.com",  # Add your email
    description="A tool for detecting signatures in documents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 