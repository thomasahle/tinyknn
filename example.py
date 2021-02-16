import time
import scipy as sp
import numpy as np

from fast_pq import FastPQ

n, d, k, dpb = 16 * 1000, 128, 1000, 2
print(f"{n=}, {d=}, queries={k}, dims_per_block={dpb}")

print("Sampling")
X = np.random.randn(n, d).astype(np.float32)
qs = np.random.randn(k, d).astype(np.float32)

print("Computing true neighbours")
start = time.time()
trus = sp.spatial.distance.cdist(qs, X).argmin(axis=1)
t0 = time.time() - start

print("Fitting PQ")
pq = FastPQ(dims_per_block=dpb)
data = pq.fit_transform(X)

print("Querying")
t1, t2 = 0, 0
sat_up, sat_down = 0, 0
places = []
for q, tru in zip(qs, trus):
    start = time.time()
    dtable = pq.distance_table(q)
    t1 += time.time() - start

    start = time.time()
    est8 = dtable.estimate_distances(data)
    t2 += time.time() - start

    sat_up += np.sum(est8 == 127) / est8.size
    sat_down += np.sum(est8 == -128) / est8.size
    # print('Non saturated:', np.sum(est8 != 255))
    # est = est8.astype(np.float32) * scale
    # tru = ((X - q)**2).sum(axis=1)
    # print('MSE:', ((est - tru)**2).mean())

    place = list(est8.argsort()).index(tru)
    places.append(place)

print()
print("Median place of true nearest neighbor:", np.median(places))
print("90% quantile:", np.quantile(places, 0.9))
print("Queries/second:", k / (t1 + t2))
print()
print("Total time spent on preprocess:", t1)
print("Total time spent on search:", t2)
print("Scipy speed for comparison:", t0)
print("Saturation degree:", sat_up / k, sat_down / k)
