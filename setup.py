#!/usr/bin/env python3
"""
Caliby - High-Performance Disk-Aware Vector Search Library

This setup.py builds caliby using CMake and pybind11.
"""

import os
import sys
import subprocess
import multiprocessing
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building native code."""
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command using CMake."""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Set Python_EXECUTABLE for cmake to use
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_STRIP=/bin/true",
            "-DCALIBY_BUILD_PYTHON=ON",
            "-DCMAKE_CXX_FLAGS_DEBUG=-O1 -g -fno-omit-frame-pointer -march=native -DCALICO_SPECIALIZATION_CALICO",
            "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -fno-omit-frame-pointer -march=native -DNDEBUG -DCALICO_SPECIALIZATION_CALICO",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -g -fno-omit-frame-pointer -march=native -DNDEBUG -DCALICO_SPECIALIZATION_CALICO",
        ]

        build_args = ["--config", "Release"]
        
        # Use all available CPUs for compilation
        build_args += ["-j", str(multiprocessing.cpu_count())]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True
        )


# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="caliby",
    version="0.1.1",
    author="Caliby Contributors",
    author_email="caliby@example.com",
    description="High-Performance Disk-Aware Vector Search Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caliby/caliby",
    project_urls={
        "Bug Tracker": "https://github.com/caliby/caliby/issues",
        "Documentation": "https://github.com/caliby/caliby#readme",
        "Source Code": "https://github.com/caliby/caliby",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
    ],
    keywords=[
        "vector-search",
        "approximate-nearest-neighbors",
        "hnsw",
        "diskann",
        "similarity-search",
        "embeddings",
        "machine-learning",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
            "numpy>=1.19.0",
        ],
    },
    ext_modules=[CMakeExtension("caliby")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
