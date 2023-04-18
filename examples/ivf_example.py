import argparse
import time
import numpy as np
import os.path
import sys
import tqdm

from fast_pq import FastPQ, DummyPQ
from fast_pq import IVF, cdist, brute


parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="random", help="Path to glove.6B.100d.npy")
parser.add_argument(
    "--n-queries", type=int, default=10000, help="Number of random queries to run"
)
parser.add_argument(
    "--dims-per-block",
    type=int,
    default=2,
    help="More dims-per-block is faster, but less precise",
)
args = parser.parse_args()

k, dpb = args.n_queries, args.dims_per_block
print("Sampling")
if args.filename == "random":
    X = np.random.randn(20000, 100)
else:
    X = np.load(args.filename)
np.random.seed(10)
np.random.shuffle(X)
X, qs = X[:-k], X[-k:]

n, d = X.shape
cl = int(n**0.5)
print(f"{n=}, {d=}, queries={k}, dims_per_block={dpb}, clusters={cl}")

print("Computing true neighbours")
fn = f"trus_{n}_{k}.npy"
if os.path.isfile(fn):
    trus = np.load(fn)
else:
    trus = brute(qs, X, 10)
    np.save(fn, trus)
print(trus.shape)

print("Building Index")
# We're fitting on a smaller sample for faster testing
sub_size = 10**5
if X.shape[0] > sub_size:
    subset = X[np.random.choice(X.shape[0], sub_size, replace=False)]
else:
    subset = X
ivf = IVF("euclidean", cl)
ivf.fit(subset)
ivf.build(X)

print("Querying")
m = 10
ts = [0] * m
recalls = [0] * m
for i, (q, tru) in tqdm.tqdm(enumerate(zip(qs, trus)), total=k):
    for prbs in range(m):
        start = time.time()
        guess = ivf.query(q, k=10, n_probes=prbs + 1)
        ts[prbs] += time.time() - start
        recalls[prbs] += len(set(tru) & set(guess))

print()
for t, recall in zip(ts, recalls):
    print("Recall10@10:", recall / 10 / k)
    print("Queries/second:", k / t)
