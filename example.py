from fast_pq import PQ
import scipy as sp
import numpy as np

n, d, k = 16 * 1000, 128, 1000

print("Sampling")
X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(k, d).astype(np.float32)

print("Computing true neighbours")
trus = sp.spatial.distance.cdist(qs, X).argsort(axis=1)
trus = map(list, trus)

print("Fitting PQ")
pq = PQ(dims_per_block=2)
data = pq.fit_transform(X)

print("Querying")
t1, t2, t3 = 0, 0, 0
totwhere = 0
for q, tru in zip(qs, trus):
    start = time.time()
    # Right now transforming is way too slow.
    tables, scale = pq.transform_query(q)
    t1 += time.time() - start
    # print('Scale:', scale)
    start = time.time()
    est8 = pq.distances(data, tables)
    t2 += time.time() - start
    # print('Saturation degree:', np.sum(est8 == 255)/est8.size)
    # print('Non saturated:', np.sum(est8 != 255))
    # est = est8.astype(np.float32) * scale
    # tru = ((X - q)**2).sum(axis=1)
    # print('MSE:', ((est - tru)**2).mean())
    start = time.time()
    where = tru.index(est8.argmin())
    t3 += time.time() - start
    totwhere += where
    # print('Place:', where)
    # print()
print("Avg place:", totwhere / k)
print("Queries/second:", k / (t1 + t2))
print(t1, t2, t3)
