#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="cellx-predict",
    version="0.1",
    description="CellX-Predict libraries",
    author="Alan R. Lowe, Christopher Soelistyo",
    author_email="a.lowe@ucl.ac.uk",
    url="https://github.com/lowe-lab-ucl/cellx-predict",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)
