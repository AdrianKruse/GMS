from setuptools import setup, find_packages

setup(
    name="arrow_block_demo",
    version="0.1.0",
    description="A terminal-based game with a block that moves across a grid",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "arrow-block-demo=arrow_block_demo.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 