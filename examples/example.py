#!/usr/bin/env python3

import time
import numpy as np
import tqdm
import argparse
import re

from tinyknn import FastPQ, knn_brute, utils

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="random-10000-128", help="Input file or random-n-d")
parser.add_argument(
    "--k", type=int, default=1_000, help="Number of nearest neighbors to find"
)
parser.add_argument("--dpb", type=int, default=2, help="Dimensions per block")
parser.add_argument(
    "--unsigned", action="store_true", help="Use unsigned distance quantization"
)
args = parser.parse_args()


if match := re.match(r"random-(\d+)-(\d+)", args.input):
    n, d = map(int, match.groups())
    with utils.timer(True, f"Sampling {n=} vectors of dimension {d=}"):
        X = np.random.randn(n, d).astype(np.float32)
        qs = np.random.randn(args.k, d).astype(np.float32)
else:
    with utils.timer(True, f"Loading and shuffling {args.input}"):
        data = np.load(args.input)
        n, d = data.shape
        np.random.seed(10)

        from scipy.stats import ortho_group
        R = ortho_group.rvs(dim=d)
        R = R[:64]
        data = data @ R.T

        np.random.shuffle(data)
        qs = data[:args.k]
        X = data[args.k:]

k, dpb, signed = args.k, args.dpb, not args.unsigned
print(f"{n=}, {d=}, queries={k}, dims_per_block={dpb}")

with utils.timer(True, "Computing true neighbours"):
    trus = knn_brute(qs, X, k=1)[:, 0]

with utils.timer(True, "Fitting PQ"):
    pq = FastPQ(dims_per_block=dpb, use_kmeans=True)
    pq.fit(X[:10**5], verbose=True)

with utils.timer(True, "Transforming data"):
    data = pq.transform(X, verbose=True)

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
# print("Numpy knn_brute force speed for comparison:", t0)
# Number of times we reached the max/min values of the int8
print(f"Saturation degree: up: {sat_up}/{total}, down: {sat_down}/{total}")
