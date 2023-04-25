#!/usr/bin/env python3

import time
import numpy as np
import tqdm
import argparse

from tinyknn import FastPQ, knn_brute

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=160_000, help="Number of samples")
parser.add_argument("--d", type=int, default=128, help="Dimension of each sample")
parser.add_argument(
    "--k", type=int, default=1_000, help="Number of nearest neighbors to find"
)
parser.add_argument("--dpb", type=int, default=2, help="Dimensions per block")
parser.add_argument(
    "--unsigned", action="store_true", help="Use unsigned distance quantization"
)
args = parser.parse_args()
n, d, k, dpb, signed = args.n, args.d, args.k, args.dpb, not args.unsigned
print(f"{n=}, {d=}, queries={k}, dims_per_block={dpb}")

print("Sampling")
X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(k, d).astype(np.float32)

print("Computing true neighbours")
start = time.time()
trus = knn_brute(qs, X, k=1)[:, 0]
t0 = time.time() - start

print("Fitting PQ")
start = time.time()
pq = FastPQ(dims_per_block=dpb, use_kmeans=True)
data = pq.fit_transform(X)
print("Took:", time.time() - start)

print("Querying")
t1, t2 = 0, 0
sat_up, sat_down, total = 0, 0, 0
places = []
for q, tru in zip(qs, tqdm.tqdm(trus)):
    start = time.time()
    dtable = pq.distance_table(q) if signed else pq.udistance_table(q)
    t1 += time.time() - start

    start = time.time()
    est8 = dtable.estimate_distances(data)
    t2 += time.time() - start

    sat_up += np.sum(est8 == 127)
    sat_down += np.sum(est8 == -128)
    total += est8.size

    place = list(est8.argsort()).index(tru)
    places.append(place)

print()
print("Median place of true nearest neighbor:", np.median(places))
for q in [0.5, 0.75, 0.9, 0.99]:
    print(f"{q:.2%} quantile:", np.quantile(places, q))
print("Queries/second:", k / (t1 + t2))
print()
print("Total time spent on preprocess:", t1)
print("Total time spent on search:", t2)
print("Numpy knn_brute force speed for comparison:", t0)
# Number of times we reached the max/min values of the int8
print(f"Saturation degree: up: {sat_up}/{total}, down: {sat_down}/{total}")
