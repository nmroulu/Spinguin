"""
Build the Cython extension modules used by the Spinguin core.

This script configures the platform-dependent compiler and linker flags for
the `spinguin._core._sparse_dot` and `spinguin._core._intersect_indices`
extension modules.
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def _get_openmp_flags() -> tuple[list[str], list[str]]:
    """
    Return the platform-dependent compiler and linker flags.

    Returns
    -------
    tuple of list of str
        Compiler flags and linker flags, in this order.

    Raises
    ------
    RuntimeError
        If the current platform is not supported by this build script.
    """

    # Define the compiler and linker flags for Microsoft compilers.
    if sys.platform == "win32":
        extra_compile_args = ["/openmp", "/O2", "/arch:SSE2", "/GS-"]
        extra_link_args = []

    # Define the compiler and linker flags for Linux compilers.
    elif sys.platform.startswith("linux"):
        extra_compile_args = ["-fopenmp", "-Ofast", "-march=native"]
        extra_link_args = ["-fopenmp"]

    # Stop with a clear error on unsupported platforms.
    else:
        raise RuntimeError(
            "This build script currently supports only Windows and Linux."
        )

    return extra_compile_args, extra_link_args


# Resolve the platform-dependent build flags.
extra_compile_args, extra_link_args = _get_openmp_flags()


# Define the extension modules to be compiled.
ext_modules = [
    Extension(
        name="spinguin._core._sparse_dot",
        sources=["src/spinguin/_core/_sparse_dot.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="spinguin._core._intersect_indices",
        sources=["src/spinguin/_core/_intersect_indices.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]


# Build the extension modules with Cython annotation enabled.
setup(
    ext_modules=cythonize(ext_modules, annotate=True),
)