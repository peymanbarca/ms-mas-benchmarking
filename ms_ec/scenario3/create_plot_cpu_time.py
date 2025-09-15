import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = list(range(1, 100))
y1 = []
y2 = []


with open('ms_sc3_parallel.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            cpu_time = float(str(line.split('CPU time: ')[1]).split(',')[0])
            # print(cpu_time * 1000)
            if cnt == 101:
                break
            if cnt >= 2:
                y1.append(cpu_time * 1000)
            cnt += 1
        except:
            pass

with open('../scenario1/ms_sc1_parallel.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            cpu_time = float(str(line.split('CPU time: ')[1]).split(',')[0])
            # print(cpu_time * 1000)
            if cnt == 101:
                break
            if cnt >= 2:
                y2.append(cpu_time * 1000)
            cnt += 1
        except:
            pass

fig, axs = plt.subplots(1, 1, figsize=(12, 6))


axs.plot(x, y1, color='blue', label=f'gRPC, Avg: {round(sum(y1) / len(y1), 1)} ms', marker='.', alpha=0.3)
axs.plot(x, y2, color='red', label=f'REST, Avg: {round(sum(y2) / len(y2), 1)} ms', marker='o', alpha=0.5)
axs.set_title(f"Parallel Order Placement")
axs.set_xlabel('Trial # (Parallel)')
axs.set_ylabel('CPU Usage Time (ms)')
axs.grid(True)
axs.legend()

fig.suptitle('CPU Usage Time in Synchronous Communication using gRPC vs REST')
plt.grid(True)
# plt.show()
plt.savefig('Synchronous_Communication_gRPC_vs_REST_cpu_time.png', dpi=400)
