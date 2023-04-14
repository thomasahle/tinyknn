#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, Extension
from Cython.Build import cythonize

def read_requirements():
    with open("requirements.txt", "r") as file:
        requirements = file.readlines()
    return [req.strip() for req in requirements if req.strip()]

setup(
    name="fast_pq",
    version='0.2',
    description='Approximate Nearest Neighbors in Python.',
    packages=["fast_pq"],
    author="Thomas Dybdahl Ahle",
    author_email="lobais@gmail.com",
    license='GNU Affero General Public License v3.0',
    url='https://github.com/thomasahle/fast_pq/',
    ext_modules=cythonize(
        Extension(
            "fast_pq._fast_pq",
            sources=["fast_pq/_fast_pq.pyx"],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-ffast-math",
                #'-unroll-count=4',
                "-Wno-deprecated-declarations",
                "-Wno-deprecated-api",
                "-mavx",
                "-m64",
                #'-mprefetchwt1'
            ],
            language="c++",
        ),
        annotate=True,
    ),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='nns, approximate nearest neighbor search',
    install_requires=read_requirements(),
)
