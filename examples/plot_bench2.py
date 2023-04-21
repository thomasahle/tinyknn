import re
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

sns.set_theme()

files = sys.argv[1:]

# Extract recall and queries/second values from the data

fast_pq_series = defaultdict(list)
cur_area = []

for file in files:
    cur_area = []
    match = re.search(r"bench_(\d)_a(\d)", file)
    if not match:
        continue
    r, a = map(int, match.groups())
    fast_pq_series[a].append(cur_area)
    for line in open(file):
        if match := re.search(r"Area under the curve from 0.5 to 1: ([\d\.]+)", line):
            cur_area.append(float(match.group(1)))

# Plot the data
for a, series in fast_pq_series.items():
    # Pad curves ended early
    longest = max(series, key=len)
    for ys in series:
        ys.extend(longest[len(ys):])
    # Find min/max
    ar = np.array(series)
    ar.sort(axis=0)
    # Plot
    xs = range(1, ar.shape[1]+1)
    plt.fill_between(xs, ar[0], ar[1], alpha=0.3)
    plt.plot(xs, ar.mean(0), marker="o", label=f"skew={a}")

plt.ylabel("Area under the qps curve from recall 0.5 to 1")
plt.xlabel("Build probes")
plt.legend()
plt.show()
