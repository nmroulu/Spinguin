"""
This script is required for compiling the sparse_dot() and intersect_indices()
Cython functions.
"""

from setuptools import Extension, setup
from Cython.Build import cythonize
import sys
import numpy as np

# Platform-specific compiler and linker settings
if sys.platform == "win32":
    extra_compile_args = ['/openmp', '/O2', '/arch:SSE2', '/GS-']
    extra_link_args = []
elif sys.platform == "linux":
    extra_compile_args = ['-fopenmp', '-Ofast', '-march=native']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "spinguin.core.sparse_dot",
        ["src/spinguin/core/sparse_dot.pyx"],
        [np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
    Extension(
        "spinguin.core.intersect_indices",
        ["src/spinguin/core/intersect_indices.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True)
)