# Fast PQ
FastPQ is a lightweight Python Vector Database specifically designed to offer high performance and be easy to read.
The main ingredient is an optimized implementation of 4bit Product Quantization (PQ) which enables fast approximate distance computations.

![Queries / Recall](https://raw.githubusercontent.com/thomasahle/fast_pq/main/plot.png)

<p align="center">
(Performance tradeoff on <a href="http://ann-benchmarks.com/glove-100-angular_10_angular.html">ANN Benchmarks</a>. Up and to the right is better.)
</p>

## Features

FastPQ is a very lightweight index, much smaller than the dataset itself.
It is also quick to build, requiring only a single pass of sklearn's KMeans over the dataset.

## Example

Let's generate some random data

```python
import numpy as np
X = np.random.rand(10000, 128)
queries = np.random.rand(100, 128)
```

We can use the `FastPQ` class to quantize the data and perform nearest neighbor search.

```python
from fast_pq import FastPQ
# Initialize PQ with 64 columns of 2 dimensions each
pq = FastPQ(dims_per_block=2)
X_compressed = pq.fit_transform(X)

# Compute exact k-nearest neighbors, accelerated by PQ
for q in queries:
    distance_table = pq.distance_table(q)
    est8 = distance_table.estimate_distances(X_compressed)
    print("Probably nearest neighbor:", est8.argmin())

```

This will return the 10 nearest neighbors for each query, as well as the distances to those neighbors.
Note that we want a quite low `dims_per_block` compared to other Product Quantization methods, since we only use 4 bits per block.
This again can be attributed to the SSE instructions only allowing 4 bit table lookups.

We can use the `IVF` class to perform approximate nearest neighbor search with Inverted File Indexing.

```python
from fast_pq import IVF, FastPQ

ivf = IVF("euclidean", cl=100, pq=FastPQ(dims_per_block=2))
ivf.fit(X).build(X)

distances, neighbors = ivf.query(queries, k=10, n_probes=10)
```

This will perform approximate nearest neighbor search using Inverted File Indexing, with 10 probes for each query. Note that we first have to call the `fit` method to build the codebook, and then the `build` method to populate the inverted file.

See also `examples/` for more detailed examples of usage.

## Installing

You need to build `fast_pq` before you can run it, as it contains Cython code.

```bash
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
```

To test it we can run an example:

```bash
$ python -m examples.example
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

## Benchmarking
To benchmark on the GloVe dataset, first download and preprocess it using
```
examples/glove/prepare-dataset.sh
```
Then run
```
$ python3 -m examples.bench examples/glove/dataset/glove.twitter.27B.100d.npy --metric angular
Loading and shuffling...
num_points=1183514, num_dims=100, num_queries=10000, dims_per_block=2, num_clusters=1087
...
Querying
Recall10@10: 0.37403000000000003
Queries/second: 4727.144941521318
Recall10@10: 0.5021399999999999
Queries/second: 3965.6137940410995
...
```
