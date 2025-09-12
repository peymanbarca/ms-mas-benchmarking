import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = list(range(1, 101))
y1 = []
y2 = []

with open('ms_sc1_sequential.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            if cnt == 101:
                break
            cnt += 1
            y1.append(res_time * 1000)
        except:
            pass

with open('ms_sc1_parallel.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            if cnt == 101:
                break
            cnt += 1
            y2.append(res_time * 1000)
        except:
            pass


fig, axs = plt.subplots(1, 2, figsize=(12, 6))


axs[0].plot(x, y1, color='blue')
axs[0].set_title(f"Sequential Order Placement, \n Avg = {sum(y1) / len(y1)} ms")
axs[0].set_xlabel('Trial # (Sequential)')
axs[0].set_ylabel('Response Time (ms)')
axs[0].grid(True)

axs[1].plot(x, y2, color='red')
axs[1].set_title(f"Parallel Order Placement, \n Avg = {sum(y2) / len(y2)} ms")
axs[1].set_xlabel('Trial # (Parallel)')
axs[1].set_ylabel('Response Time (ms)')
axs[1].grid(True)

fig.suptitle('Response Time in Synchronous Communication using REST')
plt.grid(True)
# plt.show()
plt.savefig('Synchronous_Communication_REST.png', dpi=400)
