from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="_fast_pq",
    ext_modules=cythonize(
        Extension(
            "_fast_pq",
            sources=["_fast_pq.pyx"],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-ffast-math",
                #'-unroll-count=4',
                "-Wno-deprecated-declarations",
                "-Wno-deprecated-api",
                "-mavx",
                #'-mprefetchwt1'
            ],
            language="c++",
        ),
        annotate=True,
    ),
)
