import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x = list(range(1, 101))
y1 = []


with open('mcp_parallel_call.txt', 'r') as f:
    lines = f.readlines()
    cnt = 1
    for line in lines:
        try:
            res_time = float(str(line.split('Total Tool Call Response Took: ')[1]).split(',')[0])
            # print(res_time * 1000)
            y1.append(res_time * 1000)
            cnt += 1
        except:
            pass


plt.plot(x, y1)
plt.title(f'Response Time in Synchronous Communication using MCP, \n Avg: {round(sum(y1) / len(y1), 1)} ms')
plt.grid(True)
# plt.show()
plt.savefig('Synchronous_Communication_MCP_response_time.png', dpi=400)
