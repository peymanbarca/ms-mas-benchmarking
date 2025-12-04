import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open("./ex1/result/exp1_results.json", "r") as f:
    results1 = json.load(f)

avg_latency1 = [x['final_summary']['avg_latency'] for x in results1]
std_latency1 = [x['final_summary']['std_latency'] for x in results1]
consistency_failure1 = [x['final_summary']['failure_rate'] for x in results1]

with open("./ex2/result/exp2_results.json", "r") as f:
    results2 = json.load(f)

avg_latency2 = [x['final_summary']['avg_latency'] for x in results2]
std_latency2 = [x['final_summary']['std_latency'] for x in results2]
consistency_failure2 = [x['final_summary']['failure_rate'] for x in results2]

with open("./ex3/result/exp3_results.json", "r") as f:
    results3 = json.load(f)

avg_latency3 = [x['final_summary']['avg_latency'] for x in results3]
std_latency3 = [x['final_summary']['std_latency'] for x in results3]
consistency_failure3 = [x['final_summary']['failure_rate'] for x in results3]

with open("./ex4/result/exp4_results.json", "r") as f:
    results4 = json.load(f)

avg_latency4 = [x['final_summary']['avg_latency'] for x in results4]
std_latency4 = [x['final_summary']['std_latency'] for x in results4]
consistency_failure4 = [x['final_summary']['failure_rate'] for x in results4]

with open("./ex5/result/exp5_results.json", "r") as f:
    results5 = json.load(f)

avg_latency5 = [x['final_summary']['avg_latency'] for x in results5]
std_latency5 = [x['final_summary']['std_latency'] for x in results5]
consistency_failure5 = [x['final_summary']['failure_rate'] for x in results5]


# X-axis = experiment index
x1 = list(range(1, len(avg_latency1) + 1))
x2 = list(range(1, len(avg_latency2) + 1))
x4 = list(range(1, len(avg_latency4) + 1))
x5 = list(range(1, len(avg_latency5) + 1))


# plt.figure(figsize=(14, 10))
#
# # --- 1. Avg Latency ---
# plt.subplot(2, 1, 1)
# plt.plot(x1, avg_latency1, '-*', label="Exp1 Avg Latency", linewidth=2)
# plt.plot(x2, avg_latency2, '-*', label="Exp2 Avg Latency", linewidth=2)
#
# # Std shading
# plt.fill_between(x1,
#                  [a - s for a, s in zip(avg_latency1, std_latency1)],
#                  [a + s for a, s in zip(avg_latency1, std_latency1)],
#                  alpha=0.2)
# plt.fill_between(x2,
#                  [a - s for a, s in zip(avg_latency2, std_latency2)],
#                  [a + s for a, s in zip(avg_latency2, std_latency2)],
#                  alpha=0.2)
#
# plt.title("Average Latency")
# plt.ylabel("Latency (ms)")
# plt.legend()
# plt.grid(True)
#
#
# # --- 2. Consistency Failure Rate ---
# plt.subplot(2, 1, 2)
# plt.plot(x1, consistency_failure1, '-*', label="Exp1 Failure Rate")
# plt.plot(x2, consistency_failure2, '-*', label="Exp2 Failure Rate")
# plt.title("Consistency Failure Rate")
# plt.xlabel("Trial")
# plt.ylabel("Failure Rate (%)")
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()


# ====================================== Error bar plot
avg1 = np.array(avg_latency1)
std1 = np.array(std_latency1)
fail1 = np.array(consistency_failure1)

avg2 = np.array(avg_latency2)
std2 = np.array(std_latency2)
fail2 = np.array(consistency_failure2)

avg3 = np.array(avg_latency3)
std3 = np.array(std_latency3)
fail3 = np.array(consistency_failure3)

avg4 = np.array(avg_latency4)
std4 = np.array(std_latency4)
fail4 = np.array(consistency_failure4)

avg5 = np.array(avg_latency5)
std5 = np.array(std_latency5)
fail5 = np.array(consistency_failure5)

fig, axes = plt.subplots(3, 2, figsize=(14, 6), sharey=True)

# ------------------------
# Subplot 1 – Experiment 1
# ------------------------
axes[0][0].errorbar(
    avg1,
    fail1,
    xerr=std1,
    fmt='o',
    capsize=4,
    label="Experiment 1",
    color='red'
)
axes[0][0].set_title("Experiment 1: Latency vs Consistency Failure Rate")
axes[0][0].set_xlabel("Avg ± Std Latency (s)")
axes[0][0].set_ylabel("Consistency Failure Rate")
axes[0][0].set_ylim(-5, 60)
axes[0][0].grid(True)
axes[0][0].legend()

# ------------------------
# Subplot 2 – Experiment 2
# ------------------------
axes[0][1].errorbar(
    avg2,
    fail2,
    xerr=std2,
    fmt='o',
    capsize=4,
    label="Experiment 2",
    color='blue'
)
axes[0][1].set_title("Experiment 2: Latency vs Consistency Failure Rate")
axes[0][1].set_xlabel("Avg ± Std Latency (ms)")
axes[0][1].set_ylabel("Consistency Failure Rate")
axes[0][1].grid(True)
axes[0][1].legend()

# ------------------------
# Subplot 3 – Experiment 3
# ------------------------
axes[1][0].errorbar(
    avg3,
    fail3,
    xerr=std3,
    fmt='o',
    capsize=4,
    label="Experiment 3",
    color='green'
)
axes[1][0].set_title("Experiment 3: Latency vs Consistency Failure Rate")
axes[1][0].set_xlabel("Avg ± Std Latency (s)")
axes[1][0].set_ylabel("Consistency Failure Rate")
axes[1][0].set_ylim(-5, 60)
axes[1][0].grid(True)
axes[1][0].legend()

# ------------------------
# Subplot 4 – Experiment 4
# ------------------------
axes[1][1].errorbar(
    avg4,
    fail4,
    xerr=std4,
    fmt='o',
    capsize=4,
    label="Experiment 4",
    color='brown'
)
axes[1][1].set_title("Experiment 4: Latency vs Consistency Failure Rate")
axes[1][1].set_xlabel("Avg ± Std Latency (s)")
axes[1][1].set_ylabel("Consistency Failure Rate")
axes[1][1].grid(True)
axes[1][1].legend()

# ------------------------
# Subplot 5 – Experiment 5
# ------------------------
fig.delaxes(axes[2][0])
fig.delaxes(axes[2][1])

# === create a new subplot spanning the whole last row ===
ax_center = fig.add_subplot(3, 1, 3)   # row 3, full-width

# ============= your last plot goes here =============
ax_center.errorbar(
    avg5,
    fail5,
    xerr=std5,
    fmt='o',
    capsize=4,
    label="Experiment 5",
    color='violet'
)

ax_center.set_title("Experiment 5: Latency vs Consistency Failure Rate")
ax_center.set_xlabel("Avg ± Std Latency (s)")
ax_center.set_ylabel("Consistency Failure Rate")
ax_center.set_ylim(-5, 60)
ax_center.grid(True)
ax_center.legend()

plt.tight_layout()
plt.savefig('Latency_vs_Consistency_Failure_Rate.png', dpi=400)
plt.show()

