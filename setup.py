"""
Necessary for compiling the sparse_dot() Cython function.
"""

from setuptools import Extension, setup
from Cython.Build import cythonize
import sys

# Platform-specific settings
if sys.platform == "win32":
    extra_compile_args = ['/openmp', '/O2', '/arch:SSE2', '/GS-']
    extra_link_args = []
elif sys.platform == "linux":
    extra_compile_args = ['-fopenmp', '-Ofast', '-march=native']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "spinguin.sparse_dot",
        ["src/spinguin/sparse_dot.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)