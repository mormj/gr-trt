import json
from matplotlib import pyplot as plt
import numpy as np
import itertools

filename = 'benchmark_fft_results_202310_122759.json'

# Opening JSON file 
f = open(filename,'r')
data = json.load(f) 

# Make a plot with x axis fft_size, y axis, time
# data series will be batch_size and mem_model

mem_models = np.unique([d['mem_model'] for d in data])
num_blocks = np.unique([d['num_blocks'] for d in data])


for m,n in itertools.product(mem_models, num_blocks):
    plt.figure()
    plt.title(f'{n} FFT Blocks with Mem Model {m}')
    plt.xlabel('FFT Size')
    plt.ylabel('Throughput (MSPS)')
    d_filt = [d for d in data if d['mem_model'] == m and d['num_blocks'] == n]
    batch_sizes = np.unique([d['batch_size'] for d in d_filt])
    for b in batch_sizes:
        d_filt2 = [d for d in d_filt if d['batch_size'] == b]
        x = [d['fft_size'] for d in d_filt2]
        y = [d['tput']/1e6 for d in d_filt2]
        plt.plot(x,y)

    plt.legend([f'Batch Size={b}' for b in batch_sizes])

    plt.show()
