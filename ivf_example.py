import time
import numpy as np
import os.path

from fast_pq import FastPQ, DummyPQ
from ivf import IVF, cdist, brute

k, dpb = 10000, 2
print("Sampling")
X = np.load('/mnt/large_storage/thdy/glove.6B.100d.npy')
# X = np.random.randn(20000, 100)
np.random.seed(10)
np.random.shuffle(X)
X, qs = X[:-k], X[-k:]

n, d = X.shape
cl = int(n ** 0.5)
print(f"{n=}, {d=}, queries={k}, dims_per_block={dpb}, clusters={cl}")

print("Computing true neighbours")
fn = f'trus_{n}_{k}.npy'
if os.path.isfile(fn):
    trus = np.load(fn)
else:
    trus = brute(qs, X, 10)
    np.save(fn, trus)
print(trus.shape)

print("Building Index")
pq = FastPQ(dims_per_block=dpb)
# pq = DummyPQ()
ivf = IVF("euclidean", cl, pq)
# We're fitting on a smaller sample for faster testing
ivf.fit(X[np.random.choice(X.shape[0], 10 ** 5, replace=False)])
ivf.build(X)

print("Querying")
m = 10
ts = [0] * m
recalls = [0] * m
for i, (q, tru) in enumerate(zip(qs, trus)):
    for prbs in range(m):
        start = time.time()
        guess = ivf.query(q, k=10, n_probes=prbs + 1)
        ts[prbs] += time.time() - start
        recalls[prbs] += len(set(tru) & set(guess))

print()
for t, recall in zip(ts, recalls):
    print("Recall10@10:", recall / 10 / k)
    print("Queries/second:", k / t)
