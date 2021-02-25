# Fast PQ
Simple SIMD based Product Quantization in Python

# Example
```
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
$ python example.py
n=16000, d=128, queries=1000, dims_per_block=2
Sampling
Computing true neighbours
Fitting PQ
Querying

Median place of true nearest neighbor: 1.0
90% quantile: 19.0
Queries/second: 7101.262693814528

Total time spent on preprocess: 0.09025406837463379
Total time spent on search: 0.05056595802307129
Scipy speed for comparison: 2.249645948410034
```
In this example Fast PQ is about 16 times faster than optmized scipy/numpy.
The reason is that Fast PQ uses a trick called [Accelerated Nearest Neighbor Search with Qick ADC](https://dl.acm.org/doi/abs/10.1145/3078971.3078992)
with which SIMD instructions are used to perform 16 inner product operations in a single instruction.

[![build](https://github.com/thomasahle/fast_pq/actions/workflows/testing.yml/badge.svg)](https://github.com/thomasahle/fast_pq/actions/workflows/testing.yml)
