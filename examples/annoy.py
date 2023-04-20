#!/usr/bin/env python3

import argparse
import time
import numpy as np
import os.path
import tqdm

from annoy import AnnoyIndex
from fast_pq import knn_brute

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
    "--query-method",
    choices=["query", "query2", "query3"],
    default="query",
    help="Select the query method to use: query, query2, or query3 (default: query)",
)
parser.add_argument(
    "--metric",
    choices=["euclidean", "angular"],
    default="euclidean",
    help="Metric to use in IVF. Can be 'euclidean' or 'angular'. Default is 'euclidean'.",
)
args = parser.parse_args()

num_queries, dims_per_block, k_neighbours = (
    args.n_queries,
    args.dims_per_block,
    args.k_neighbours,
)
print("Loading and shuffling...")
data = np.load(args.filename)
np.random.seed(10)
np.random.shuffle(data)
data, queries = data[:-num_queries], data[-num_queries:]

num_points, num_dims = data.shape
num_clusters = int(num_points**0.5)
print(f"{num_points=}, {num_dims=}, {num_queries=}, {dims_per_block=}, {num_clusters=}")

true_neighbours_filename = f"trus_{num_points}_{num_queries}_{args.metric}.npy"
# true_neighbours_filename = f"trus_{num_points}_{num_queries}.npy"
if os.path.isfile(true_neighbours_filename):
    print("Loading true neighbours from", true_neighbours_filename)
    true_neighbours = np.load(true_neighbours_filename)
    num_queries, k_neighbours = true_neighbours.shape
    print(f"Found {num_queries=}, {k_neighbours=}")
else:
    print("Computing true neighbours...")
    start = time.time()
    true_neighbours = knn_brute(queries, data, k_neighbours, metric=args.metric)
    print(f"Took {time.time() - start:.1f} seconds.")
    np.save(true_neighbours_filename, true_neighbours)


print("Now that we have the index, actually add the points to it.")
for build_probes in [100, 200, 400]:
    print("Creating new Annoy index...")
    start = time.time()
    ann = AnnoyIndex(data.shape[1], "angular")
    for i, v in enumerate(tqdm.tqdm(data)):
        ann.add_item(i, v)
    print(f"Took {time.time() - start:.1f} seconds.")

    print(f"Building {build_probes} trees...")
    start = time.time()
    ann.build(build_probes)
    print(f"Took {time.time() - start:.1f} seconds.")

    print("Querying")
    recall = 0
    for n_probes in [
        100,
        200,
        400,
        1000,
        2000,
        4000,
        10000,
        20000,
        40000,
        100000,
        200000,
        400000,
    ]:
        start = time.time()
        found = 0

        with tqdm.tqdm(
            enumerate(zip(queries, true_neighbours)), total=num_queries, leave=False
        ) as pbar:
            pbar.set_description(f"Probing: {n_probes} nodes")
            for _, (query, true_neighbor) in pbar:
                guess = ann.get_nns_by_vector(query, n=k_neighbours, search_k=n_probes)
                found += len(set(true_neighbor) & set(guess))

        query_time = time.time() - start
        recall = found / k_neighbours / num_queries

        print(f"Recall{k_neighbours}@{k_neighbours}:", recall)
        qps = num_queries / query_time
        print("Queries/second:", qps)
        if qps < 500:
            break
