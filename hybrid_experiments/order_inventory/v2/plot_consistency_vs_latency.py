import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

delay = 5.0
drop = 10

with open(f"./ex1/result/exp1_results_delay_{delay}_drop_{drop}.json", "r") as f:
    results1 = json.load(f)

avg_latency1 = [x['final_summary']['avg_latency'] for x in results1]
std_latency1 = [x['final_summary']['std_latency'] for x in results1]
consistency_failure1 = [x['final_summary']['failure_rate'] for x in results1]

with open(f"./ex2/result/exp2_results_delay_{delay}_drop_{drop}.json", "r") as f:
    results2 = json.load(f)

avg_latency2 = [x['final_summary']['avg_latency'] for x in results2]
std_latency2 = [x['final_summary']['std_latency'] for x in results2]
consistency_failure2 = [x['final_summary']['failure_rate'] for x in results2]

with open(f"./ex3/result/exp3_results_delay_{delay}_drop_{drop}.json", "r") as f:
    results3 = json.load(f)

avg_latency3 = [x['final_summary']['avg_latency'] for x in results3]
std_latency3 = [x['final_summary']['std_latency'] for x in results3]
consistency_failure3 = [x['final_summary']['failure_rate'] for x in results3]

with open(f"./ex4/result/exp4_results_delay_{delay}_drop_{drop}.json", "r") as f:
    results4 = json.load(f)

avg_latency4 = [x['final_summary']['avg_latency'] for x in results4]
std_latency4 = [x['final_summary']['std_latency'] for x in results4]
consistency_failure4 = [x['final_summary']['failure_rate'] for x in results4]

with open(f"./ex5/result/exp5_results_delay_{delay}_drop_{drop}.json", "r") as f:
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
avg1 = np.array(avg_latency1[0])
std1 = np.array(std_latency1[0])
fail1 = np.array(consistency_failure1[0])

avg2 = np.array(avg_latency2[0])
std2 = np.array(std_latency2[0])
fail2 = np.array(consistency_failure2[0])

avg3 = np.array(avg_latency3[0])
std3 = np.array(std_latency3[0])
fail3 = np.array(consistency_failure3[0])

avg4 = np.array(avg_latency4[0])
std4 = np.array(std_latency4[0])
fail4 = np.array(consistency_failure4[0])

avg5 = np.array(avg_latency5[1])
std5 = np.array(std_latency5[1])
fail5 = np.array(consistency_failure5[1])

fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharey=False)

ylims = [
    (-5, 80),
    (-5, 30),
    (-5, 30),
    (-5, 30),
    (-5, 30),
]

# ------------------------
# Subplot 1 – Experiment 1
# ------------------------
axes[0].errorbar(
    avg1,
    fail1,
    xerr=std1,
    fmt='o',
    capsize=4,
    label="Experiment 1",
    color='red'
)
axes[0].set_title(f"Latency vs Consistency Failure Rate \n N=20, m=5, Delay Injected: {delay}s, "
                     f"Message Drop Rate Injected: {drop}%")
axes[0].set_xlabel("Avg ± Std Latency (s)")
axes[0].set_ylabel("Consistency Failure Rate\n %")
# axes[0].set_ylim(-5, 100)
axes[0].grid(True)
axes[0].legend()

# ------------------------
# Subplot 2 – Experiment 2
# ------------------------
axes[1].errorbar(
    avg2,
    fail2,
    xerr=std2,
    fmt='o',
    capsize=4,
    label="Experiment 2",
    color='blue'
)
# axes[1].set_title("Experiment 2")
axes[1].set_xlabel("Avg ± Std Latency (s)")
axes[1].set_ylabel("Consistency Failure Rate\n %")
# axes[1].set_ylim(-5, 50)
axes[1].grid(True)
axes[1].legend()

# ------------------------
# Subplot 3 – Experiment 3
# ------------------------
axes[2].errorbar(
    avg3,
    fail3,
    xerr=std3,
    fmt='o',
    capsize=4,
    label="Experiment 3",
    color='green'
)
# axes[2].set_title("Experiment 3")
axes[2].set_xlabel("Avg ± Std Latency (s)")
axes[2].set_ylabel("Consistency Failure Rate\n %")
# axes[2].set_ylim(-5, 50)
axes[2].grid(True)
axes[2].legend()

# ------------------------
# Subplot 4 – Experiment 4
# ------------------------
axes[3].errorbar(
    avg4,
    fail4,
    xerr=std4,
    fmt='o',
    capsize=4,
    label="Experiment 4",
    color='brown'
)
# axes[3].set_title("Experiment 4")
axes[3].set_xlabel("Avg ± Std Latency (s)")
axes[3].set_ylabel("Consistency Failure Rate\n %")
axes[3].grid(True)
# axes[3].set_ylim(-5, 50)
axes[3].legend()

# ------------------------
# Subplot 5 – Experiment 5
# ------------------------
axes[4].errorbar(
    avg5,
    fail5,
    xerr=std5,
    fmt='o',
    capsize=4,
    label="Experiment 5",
    color='purple'
)
# axes[4].set_title("Experiment 5")
axes[4].set_xlabel("Avg ± Std Latency (s)")
axes[4].set_ylabel("Consistency Failure Rate\n %")
# axes[4].set_ylim(-5, 50)
axes[4].grid(True)
axes[4].legend()

for ax, lim in zip(axes, ylims):
    ax.set_ylim(lim)

plt.tight_layout()
plt.savefig(f'Latency_vs_Consistency_Failure_Rate_delay_{delay}_drop_{drop}.png', dpi=400)
plt.show()

