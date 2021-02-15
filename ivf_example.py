import time
import scipy as sp
import numpy as np

from fast_pq import FastPQ
from ivf import IVF

n, d, k, dpb = 16 * 1000, 128, 1000, 2
print(f'{n=}, {d=}, queries={k}, dims_per_block={dpb}')

print("Sampling")
X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(k, d).astype(np.float32)

print("Computing true neighbours")
start = time.time()
trus = sp.spatial.distance.cdist(qs, X).argmin(axis=1)
t0 = time.time() - start

print("Building Index")
pq = FastPQ(dims_per_block=dpb)
ivf = IVF('euclidean', int(n**.5), pq)
ivf.fit(X)

print("Querying")
t1, t2 = 0, 0
recall_at_10 = 0
for q, tru in zip(qs, trus):
    start = time.time()
    guess = ivf.query(q, k=10, n_probes=10)
    t1 += time.time() - start

    recall_at_10 += int(tru in guess)

print()
print("Recall@10:", recall_at_10 / k)
print("Queries/second:", k / (t1 + t2))
