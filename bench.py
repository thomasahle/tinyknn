import time
import numpy as np
import os.path
import sys
import pickle

from fast_pq import FastPQ, DummyPQ
from ivf import IVF, cdist, brute

k, dpb = 10000, 2
print("Loading and shuffling")
X = np.load('/mnt/large_storage/thdy/glove.6B.100d.npy')
np.random.seed(10)
np.random.shuffle(X)
X, qs = X[:-k], X[-k:]

n, d = X.shape
cl = int(n ** 0.5)
print(f"{n=}, {d=}, queries={k}, dims_per_block={dpb}, clusters={cl}")

fn = f'trus_{n}_{k}.npy'
if os.path.isfile(fn):
    print("Loading true neighbours from", fn)
    trus = np.load(fn)
else:
    print("Computing true neighbours")
    trus = brute(qs, X, 10)
    np.save(fn, trus)
print(trus.shape)

fn = f'ivf_{n=}_{k=}_{dpb=}.pickle'
if os.path.isfile(fn):
    print("Loading Index from", fn)
    pq, ivf = pickle.load(open(fn, 'rb'))
else:
    print("Building Index")
    pq = FastPQ(dims_per_block=dpb)
    ivf = IVF("euclidean", cl, pq)
    ivf.fit(X, verbose=True)
    pickle.dump((pq, ivf), open(fn, 'wb'))

print('Adding the points')
ivf.build(X)

if len(sys.argv) == 1:
    query = ivf.query
elif sys.argv[1] == 'q2':
    query = ivf.query2
elif sys.argv[1] == 'q3':
    query = ivf.query3
print(f'Using {query}')

print("Querying")
m = 10
ts = [0] * m
recalls = [0] * m
for i, (q, tru) in enumerate(zip(qs, trus)):
    for prbs in range(m):
        start = time.time()
        guess = query(q, k=10, n_probes=prbs + 1)
        ts[prbs] += time.time() - start
        recalls[prbs] += len(set(tru) & set(guess))

print()
for t, recall in zip(ts, recalls):
    print("Recall10@10:", recall / 10 / k)
    print("Queries/second:", k / t)
