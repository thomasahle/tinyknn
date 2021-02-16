import time
import scipy as sp
import numpy as np

from fast_pq import FastPQ, DummyPQ
from ivf import IVF

k, dpb = 1000, 2
print("Sampling")
#X = np.load('/Users/thdy/notes/tree_decoder/data/glove.6B.100d.npy')
#X = X[:10000]
X = np.random.randn(1000, 100)
n, d = X.shape
cl = int(n**.5)
print(f'{n=}, {d=}, queries={k}, dims_per_block={dpb}, clusters={cl}')
qs = np.random.randn(k, d).astype(np.float32)

print("Computing true neighbours")
start = time.time()
trus = sp.spatial.distance.cdist(qs, X).argpartition(axis=1, kth=10)[:,:10]
t0 = time.time() - start

print("Building Index")
pq = FastPQ(dims_per_block=dpb)
ivf = IVF('euclidean', cl, pq)
ivf.fit(X)

print("Querying")
m = 10
ts = [0]*m
recalls = [0]*m
for q, tru in zip(qs, trus):
    for prbs in range(m):
        start = time.time()
        guess = ivf.query(q, k=10, n_probes=prbs+1)
        ts[prbs] += time.time() - start
        recalls[prbs] += len(set(tru) & set(guess))

print()
for t, recall in zip(ts, recalls):
    print("Recall10@10:", recall / 10 / k)
    print("Queries/second:", k / t)
