from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
import os

with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name="pylidar2d",
    description='Python tools for lidar 2d scans',
    author='Daniel Dugas',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/danieldugas/pylidar2d",
    version='0.1',
    packages=find_packages(),
    ext_modules = cythonize("clib_clustering/lidar_clustering.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)
