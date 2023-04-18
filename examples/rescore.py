import numpy as np
import time
from fast_pq import IVF, FastPQ, cdist

np.random.seed(10)

n = 1000
d = 10
nq = 30
at = 10
dpb = 2
max_probes = 10
metric = "euclidean"

X = np.random.randn(n, d).astype(np.float32)
ivf = IVF(metric, int(n**0.5 + 1))
ivf.fit(X).build(X, n_probes=2)

qs = np.random.randn(nq, d).astype(np.float32)
trus = cdist(qs, X).argpartition(axis=1, kth=at)[:, :at]


def compute_recall(pass_1, query_probes):
    start = time.time()
    recall_at = 0
    for q, tru in zip(qs, trus):
        guess = ivf.query(q, k=at, n_probes=query_probes, pass_1=pass_1)
        recall_at += len(set(guess) & set(tru))
    elasped = time.time() - start

    return recall_at / nq / at, elasped


# Print header row
print(f"Recall {at}@{at} using pass_1=p and query_probes=q.")
print("p/q", end=" ")
for query_probes in range(1, max_probes + 1):
    print(f"{query_probes:5}", end=" ")
print()

# Print table content
total_query_time = 0
for pass_1 in range(1, max_probes + 1):
    print(f"{pass_1:4}", end=" ")
    for query_probes in range(1, max_probes + 1):
        recall, query_time = compute_recall(pass_1 * at, query_probes)
        total_query_time += query_time
        print(f"{recall:.2f}", end=", ")
    print()

print(f"Total query time: {total_query_time:.1f}s")
