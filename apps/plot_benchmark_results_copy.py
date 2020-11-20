import json
from matplotlib import pyplot as plt
import numpy as np
import itertools
from math import log2

# filename = 'benchmark_fft_results_202310_122759.json'
filename = '/share/gnuradio/benchmark-dnn/gr-trt/benchmark_copy_results_202710_101513.json'

# colors = ['blue','green','red','gray','fuchsia','gold', 'maroon', 'silver']
colors = plt.get_cmap("tab10")

# Opening JSON file 
f = open(filename,'r')
data = json.load(f) 
data = data['results']

# Make a plot with x axis fft_size, y axis, time
# data series will be batch_size and mem_model

mem_models = np.unique([d['memmodel'] for d in data])
num_blocks = np.unique([d['nblocks'] for d in data])

plt.figure()
plt.title('Copy Blocks')
plt.xlabel('log2(Batch Size)')
plt.ylabel('Throughput (MSPS)')

coloridx = 0
lgnd = []
for n in num_blocks:

    d_filt_0 = [d for d in data if d['memmodel'] == 0 and d['nblocks'] == n]
    d_filt_1 = [d for d in data if d['memmodel'] == 1 and d['nblocks'] == n]
    batch_sizes = np.unique([d['batchsize'] for d in data])

    
    

    x1 = [log2(d['batchsize']) for d in d_filt_0]
    y1 = [d['tput']/1e6 for d in d_filt_0]

    x2 = [log2(d['batchsize']) for d in d_filt_1]
    y2 = [d['tput']/1e6 for d in d_filt_1]

    plt.plot(x1,y1, color=colors(coloridx), marker='o')
    plt.plot(x2,y2, color=colors(coloridx), marker='o', linestyle='dashed')
    lgnd.append(f'Nblocks={n}, Host/Device')
    lgnd.append(f'Nblocks={n}, Pinned')
    coloridx += 1

    # plt.legend([f'Batch Size={b}' for b in batch_sizes])
plt.legend(lgnd)

plt.show()
