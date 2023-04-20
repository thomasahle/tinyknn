import numpy as np
import time
from fast_pq import IVF, brute

np.random.seed(10)

n = 1000
d = 10
nq = 30
at = 10
dpb = 2
max_probes = 10
metric = "euclidean"

X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(nq, d).astype(np.float32)


def compute_recall(metric, build_probes, query_probes):
    if at <= n:
        trus = brute(qs, X, at, metric=metric)
    else:
        trus = np.broadcast_to(np.arange(n), (nq, n))
    ivf = IVF(metric, int((n * build_probes) ** 0.5))
    ivf.fit(X).build(X, n_probes=build_probes)

    start = time.time()
    recall_at = 0
    for q, tru in zip(qs, trus):
        guess = ivf.query(q, k=at, n_probes=query_probes)
        recall_at += len(set(guess) & set(tru))
    elasped = time.time() - start

    return recall_at / nq / at, elasped


# Print header row
print(f"Recall {at}@{at} using build_probes=b and query_probes=q.")
print("b/q", end=" ")
for query_probes in range(1, max_probes + 1):
    print(f"{query_probes:5}", end=" ")
print()

# Print table content
total_query_time = 0
for build_probes in range(1, max_probes + 1):
    print(f"{build_probes:4}", end=" ")
    for query_probes in range(1, max_probes + 1):
        recall, query_time = compute_recall(metric, build_probes, query_probes)
        total_query_time += query_time
        print(f"{recall:.2f}", end=", ")
    print()

print(f"Total query time: {total_query_time:.1f}s")
