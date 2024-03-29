import re
import sys
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

data = sys.stdin.read()

# Extract recall and queries/second values from the data
fast_pq_series = []
annoy_series = []

for line in data.split("\n"):
    if match := re.search(r"Adding each point to (\d+) lists", line):
        recall, queries_per_second = [], []
        fast_pq_series.append((int(match.group(1)), recall, queries_per_second))
    if match := re.search(r"Building (\d+) trees", line):
        recall, queries_per_second = [], []
        annoy_series.append((int(match.group(1)), recall, queries_per_second))
    if match := re.search(r"Recall10@10: (\d+\.\d+)", line):
        recall_value = float(match.group(1))
        recall.append(recall_value)
    elif match := re.search(r"Queries/second: (\d+\.\d+)", line):
        qps_value = float(match.group(1))
        queries_per_second.append(qps_value)

# Plot the data
for n, recall, queries_per_second in fast_pq_series:
    plt.plot(recall, queries_per_second, marker="o", label=f"FastPQ {n} build probes")
for n, recall, queries_per_second in annoy_series:
    plt.plot(recall, queries_per_second, marker="o", label=f"Annoy {n} trees")
plt.xlabel("Recall10@10")
plt.xlim([0.3, 0.93])
plt.ylim([100, 6000])
plt.ylabel("Queries/second")
plt.title("Queries/second vs Recall on GloVe 100d angular")
plt.legend()
plt.show()
