#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

# Uncomment to compile with clang instea dof gcc.
# It seems we are faster with gcc, which is the default.
#import os
#os.environ['LDSHARED'] = 'clang -shared -flto'
#os.environ['CC'] = 'clang'
#os.environ['CXX'] = 'clang++'

compile_args = [
    "-O3",
    "-march=native",
    "-ffast-math",
    "-Wno-unused-function",
    "-fprefetch-loop-arrays",
            #'-unroll-count=4',
            #"-flto",
            #'-mprefetchwt1',
            #"-m64",
]

extensions = [
    Extension(
        "tinyknn._fast_pq",
        sources=["tinyknn/_fast_pq.pyx"],
        extra_compile_args=compile_args + [
            #"-flto",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "tinyknn._fast_pq_avx",
        sources=["tinyknn/_fast_pq_256.pyx"],
        extra_compile_args=compile_args + [
            "-mavx",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

Options.boundscheck = False
Options.wraparound = False
Options.cdivision = True
Options.initializedcheck = False
Options.docstrings = False
Options.nonecheck = False
Options.overflowcheck = False

setup(
    packages=["tinyknn"],
    ext_modules=cythonize(extensions, annotate=True),
)
