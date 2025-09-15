import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = list(range(1, 100))
y1 = []
y2 = []

y3 = []
y4 = []

with open('ms_sc3_seq.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            if cnt == 101:
                break
            if cnt >= 2:
                y1.append(res_time * 1000)
            cnt += 1
        except:
            pass

with open('../scenario1/ms_sc1_sequential.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            if cnt == 101:
                break
            if cnt >= 2:
                y3.append(res_time * 1000)
            cnt += 1
        except:
            pass

with open('ms_sc3_parallel.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            if cnt == 101:
                break
            if cnt >= 2:
                y2.append(res_time * 1000)
            cnt += 1
        except:
            pass

with open('../scenario1/ms_sc1_parallel.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            if cnt == 101:
                break
            if cnt >= 2:
                y4.append(res_time * 1000)
            cnt += 1
        except:
            pass

fig, axs = plt.subplots(1, 2, figsize=(12, 6))


axs[0].plot(x, y1, color='blue', label=f'gRPC, Avg: {round(sum(y1) / len(y1), 1)} ms')
axs[0].plot(x, y3, color='red', label=f'REST, Avg: {round(sum(y3) / len(y3), 1)} ms')
axs[0].set_title(f"Sequential Order Placement")
axs[0].set_xlabel('Trial # (Sequential)')
axs[0].set_ylabel('Response Time (ms)')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(x, y2, color='blue', label=f'gRPC, Avg: {round(sum(y2) / len(y2), 1)} ms')
axs[1].plot(x, y4, color='red', label=f'REST, Avg: {round(sum(y4) / len(y4), 1)} ms')
axs[1].set_title(f"Parallel Order Placement")
axs[1].set_xlabel('Trial # (Parallel)')
axs[1].set_ylabel('Response Time (ms)')
axs[1].grid(True)
axs[1].legend()

fig.suptitle('Response Time in Synchronous Communication using gRPC vs REST')
plt.grid(True)
# plt.show()
plt.savefig('Synchronous_Communication_gRPC_vs_REST_response_time.png', dpi=400)
