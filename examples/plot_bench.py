import re, sys
import matplotlib.pyplot as plt

data = sys.stdin.read()

# Extract recall and queries/second values from the data
series = []

for line in data.split("\n"):
    if match := re.match(r"Adding each point to (\d) lists", line):
        recall = []
        queries_per_second = []
        series.append((int(match.group(1)), recall, queries_per_second))
    if (match := re.search(r"Recall10@10: (\d+\.\d+)", line)):
        recall_value = float(match.group(1))
        recall.append(recall_value)
    elif (match := re.search(r"Queries/second: (\d+\.\d+)", line)):
        qps_value = float(match.group(1))
        queries_per_second.append(qps_value)

# Plot the data
for l, recall, queries_per_second in series[1:2]:
    plt.plot(recall, queries_per_second, marker='o', label=f"{l} build probes")
plt.xlabel("Recall10@10")
plt.ylabel("Queries/second")
plt.title("Queries/second vs Recall on GloVe 100d angular")
#plt.legend()
plt.show()
