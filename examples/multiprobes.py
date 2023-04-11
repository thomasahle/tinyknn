import numpy as np
from fast_pq import IVF, FastPQ, cdist

np.random.seed(10)

n = 1000
d = 10
nq = 30
at = 10
dpb = 2

X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(nq, d).astype(np.float32)

def compute_recall(metric, build_probes, query_probes):
    if at < n:
        trus = cdist(qs, X).argpartition(axis=1, kth=at)[:, :at]
    else:
        trus = np.broadcast_to(np.arange(n), (nq, n))
    pq = FastPQ(dims_per_block=dpb)
    ivf = IVF(metric, int(n**0.5), pq)
    ivf.fit(X).build(X, n_probes=build_probes)
    recall_at = 0
    for q, tru in zip(qs, trus):
        guess = ivf.query(q, k=at, n_probes=query_probes)
        recall_at += len(set(guess) & set(tru))
    return recall_at / nq / at

# Print header row
print(f"Recall {at}@{at} using build_probes=b and query_probes=q.")
print("b/q", end=' ')
for query_probes in range(1, 5):
    print(f"{query_probes:5}", end=' ')
print()

# Print table content
for build_probes in range(1, 5):
    print(f"{build_probes:4}", end=' ')
    for query_probes in range(1, 5):
        recall = compute_recall("euclidean", build_probes, query_probes)
        print(f"{recall:.2f}", end=", ")
    print()

