#!/usr/bin/env python3

import argparse
import time
import numpy as np
import os.path
import pickle
import tqdm
import re
import sklearn.metrics

from fast_pq import FastPQ
from fast_pq import IVF, brute

parser = argparse.ArgumentParser(
    description="Benchmark FastPQ and IVF on GloVe dataset"
)
parser.add_argument(
    "filename", help="Path to the GloVe .npy file (e.g., glove.840B.100d.npy)"
)
parser.add_argument(
    "--n-queries",
    type=int,
    default=10000,
    help="Number of random queries to run (default: 10000)",
)
parser.add_argument(
    "--dims-per-block",
    type=int,
    default=2,
    help="More dims-per-block is faster, but less precise (default: 2)",
)
parser.add_argument(
    "--k-neighbours",
    type=int,
    default=10,
    help="Number of neighbours in k-NN search (default: 10)",
)
parser.add_argument(
    "--metric",
    choices=["euclidean", "angular"],
    default="euclidean",
    help="Metric to use in IVF. Can be 'euclidean' or 'angular'. Default is 'euclidean'.",
)
parser.add_argument(
    "--a",
    type=float,
    default=1.0,
    help="Number of clusters will be int(a * sqrt(n))",
)
args = parser.parse_args()

num_queries = args.n_queries
dims_per_block = args.dims_per_block
k_neighbours = args.k_neighbours
metric = args.metric
simple_name = args.filename.split("/")[-1] if "/" in args.filename else args.filename

print("Loading and shuffling...")
if match := re.match(r"random-(\w+)-(\d+)", args.filename):
    size = {"xs": 10**5}[match.group(1)]
    dim = int(match.group(2))
    data = np.random.randn(10**5 + num_queries, dim)
else:
    data = np.load(args.filename)
    np.random.seed(10)
    np.random.shuffle(data)
data, queries = data[:-num_queries], data[-num_queries:]

num_points, num_dims = data.shape
num_clusters = int(args.a * num_points**0.5)
print(f"{num_points=}, {num_dims=}, {num_queries=}, {dims_per_block=}, {num_clusters=}")

true_neighbours_filename = (
    f"trus_{simple_name}_{k_neighbours=}_{num_queries=}_{metric=}.npy"
)
if os.path.isfile(true_neighbours_filename):
    print("Loading true neighbours from", true_neighbours_filename)
    true_neighbours = np.load(true_neighbours_filename)
    num_queries, k_neighbours = true_neighbours.shape
    print(f"Found {num_queries=}, {k_neighbours=}")
else:
    print("Computing true neighbours...")
    start = time.time()
    true_neighbours = brute(queries, data, k_neighbours, metric=args.metric)
    print(f"Took {time.time() - start:.1f} seconds.")
    np.save(true_neighbours_filename, true_neighbours)

ivf_filename = (
    f"ivf_{simple_name}_{args.metric}_{num_clusters=}_{dims_per_block=}.pickle"
)
if os.path.isfile(ivf_filename):
    print("Loading Index from", ivf_filename)
    with open(ivf_filename, "rb") as file:
        pq, ivf = pickle.load(file)
else:
    print("Building Index...")
    pq = FastPQ(dims_per_block)
    ivf = IVF(args.metric, num_clusters, pq)
    start = time.time()
    ivf.fit(data, verbose=True)
    print(f"Took {time.time() - start:.1f} seconds.")
    print("Saving index to", ivf_filename)
    with open(ivf_filename, "wb") as file:
        pickle.dump((pq, ivf), file)

print("Now that we have the index, actually add the points to it.")
# ivf.build(data, n_probes=2, verbose=True)

n_max_build_probes = 10
for build_probes in range(1, n_max_build_probes):
    print(f"Adding each point to {build_probes} lists...")
    start = time.time()
    ivf.build(data, n_probes=build_probes, verbose=True)
    print(f"Took {time.time() - start:.1f} seconds.")

    print("Querying")
    recall = 0
    n_probes = 1
    qpss, recalls = [], []
    while recall < 0.9:
        start = time.time()
        found = 0

        with tqdm.tqdm(
            enumerate(zip(queries, true_neighbours)), total=num_queries, leave=False
        ) as pbar:
            pbar.set_description(
                f"Probing: {n_probes} out of {ivf.n_clusters} clusters"
            )
            for _, (query, true_neighbor) in pbar:
                guess = ivf.query(query, k=k_neighbours, n_probes=n_probes)
                # guess = ivf.query(query, k=k_neighbours, n_probes=n_probes,
                #                        pass_1 = int((build_probes / 2) * n_probes * k_neighbours))
                found += len(set(true_neighbor) & set(guess))

        qps = num_queries / (time.time() - start)
        recall = found / k_neighbours / num_queries
        qpss.append(qps)
        recalls.append(recall)

        print(f"Recall{k_neighbours}@{k_neighbours}:", recall)
        print("Queries/second:", qps)

        n_probes += int(n_probes**0.5)

    # We compute the area under the curve, but only for recall in [1/2, 1]
    qpss.append(0)
    recalls.append(1)
    recall0 = 1 / 2
    qps0 = np.interp(recall0, recalls, qpss)
    i = np.searchsorted(recalls, recall0)
    auc = sklearn.metrics.auc([recall0] + recalls[i:], [qps0] + qpss[i:])
    print(f"Area under the curve from {recall0} to 1: {auc:.1f}")
