#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    packages=["tinyknn"],
    ext_modules=cythonize(
        Extension(
            "tinyknn._fast_pq",
            sources=["tinyknn/_fast_pq.pyx"],
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
        ),
        annotate=True,
    ),
)
