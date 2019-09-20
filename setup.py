from setuptools import setup, find_packages
from Cython.Build import cythonize
import os



setup(
    name="pylidar2d",
    description='Python tools for lidar 2d scans',
    author='Daniel Dugas',
    version='0.0',
    packages=find_packages(),
    ext_modules = cythonize("clib_clustering/lidar_clustering.pyx", annotate=True),
)
