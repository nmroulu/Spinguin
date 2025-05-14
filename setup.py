"""
This script is required for compiling the sparse_dot() and intersect_indices()
Cython functions.
"""

from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

# Platform-specific compiler and linker settings
if sys.platform == "win32":
    extra_compile_args = ['/openmp', '/O2', '/arch:SSE2', '/GS-']
    extra_link_args = []
elif sys.platform == "linux":
    extra_compile_args = ['-fopenmp', '-Ofast', '-march=native']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "spinguin.utils.sparse_dot",
        ["src/spinguin/utils/sparse_dot.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
    Extension(
        "spinguin.utils.intersect_indices",
        ["src/spinguin/utils/intersect_indices.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True)
)