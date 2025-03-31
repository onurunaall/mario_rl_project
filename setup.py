#!/usr/bin/env python
"""
setup.py
--------
Setup script for the mario_rl_project package.
"""

from setuptools import setup, find_packages

# Read the README file for the long description.
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mario_rl_project",
    version="0.1.0",
    description="A Deep Reinforcement Learning project to train a Mario agent using DDQN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    url="https://github.com/yourusername/mario_rl_project",  # Update URL accordingly
    packages=find_packages(),
    install_requires=[
        "torch",
        "gym",
        "gym-super-mario-bros",
        "nes_py",
        "torchvision",
        "tensordict",
        "torchrl",
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "mario_train=train:main",  # Allows running the training loop via command line.
        ],
    },
)
