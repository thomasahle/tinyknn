#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "tinyknn._fast_pq",
        sources=["tinyknn/_fast_pq.pyx"],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-Wno-deprecated-declarations",
            "-Wno-deprecated-api",
            "-m64",
        ],
        language="c++",
        include_dirs=[np.get_include()],
    ),
    Extension(
        "tinyknn._fast_pq_avx",
        sources=["tinyknn/_fast_pq_256.pyx"],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            # '-unroll-count=4',
            "-Wno-deprecated-declarations",
            "-Wno-deprecated-api",
            "-mavx",
            "-m64",
            # '-mprefetchwt1'
        ],
        language="c++",
        include_dirs=[np.get_include()],
    ),
]

setup(
    packages=["tinyknn"],
    ext_modules=cythonize(extensions, annotate=True),
)
