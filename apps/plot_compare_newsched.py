import json
from matplotlib import pyplot as plt
import numpy as np
import itertools

colors = ['blue','green','red','gray','fuchsia','gold']

filename_gr39 = '/share/gnuradio/benchmark-dnn/gr-trt/benchmark_fft_results_202610_120210.json'
filename_newsched = '/share/gnuradio/newsched/benchmark_newsched_bm_cuda_fft_results_202610_110518.json'

# Opening JSON file 
f = open(filename_gr39,'r')
data_gr39 = json.load(f) 
data_gr39 = data_gr39['results']

f = open(filename_newsched,'r')
data_newsched = json.load(f) 
data_newsched = data_newsched['results']

# Make a plot with x axis fft_size, y axis, time
# data series will be batch_size and mem_model

mem_models = np.unique([d['memmodel'] for d in data_gr39])
num_blocks = np.unique([d['nblocks'] for d in data_gr39])


for m,n in itertools.product(mem_models, num_blocks):
    plt.figure()
    plt.title(f'{n} FFT Blocks with Mem Model {m}')
    plt.xlabel('FFT Size')
    plt.ylabel('Throughput (MSPS)')
    d_filt_39 = [d for d in data_gr39 if d['memmodel'] == m and d['nblocks'] == n]
    d_filt_ns = [d for d in data_newsched if d['memmodel'] == m and d['nblocks'] == n]
    batch_sizes = np.unique([d['batchsize'] for d in d_filt_39])

    lgnd = []
    coloridx = 0
    for b in batch_sizes:
        d_filt2_39 = [d for d in d_filt_39 if d['batchsize'] == b]
        d_filt2_ns = [d for d in d_filt_ns if d['batchsize'] == b]
        x1 = [d['fftsize'] for d in d_filt2_39]
        y1 = [d['tput']/1e6 for d in d_filt2_39]

        x2 = [d['fftsize'] for d in d_filt2_ns]
        y2 = [d['tput']/1e6 for d in d_filt2_ns]

        plt.plot(x1,y1, color=colors[coloridx])
        plt.plot(x2,y2, color=colors[coloridx], linestyle='dashed')
        lgnd.append(f'(GR39) Batch Size={b}')
        lgnd.append(f'(newsched) Batch Size={b}')
        coloridx += 1

    # plt.legend([f'Batch Size={b}' for b in batch_sizes])
    plt.legend(lgnd)

    plt.show()
