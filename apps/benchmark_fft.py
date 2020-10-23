#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.0.0-git

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.fft import window
import time
import trt
import json
import datetime
import itertools

class benchmark_fft(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args['num_samples']
        fft_size = args['fft_size']
        batch_size = args['batch_size']
        self.actual_samples = actual_samples = (
            fft_size * batch_size) * int(nsamples / (fft_size * batch_size))
        num_blocks = args['num_blocks']
        mem_model = args['mem_model']

        ##################################################
        # Blocks
        ##################################################
        fft_blocks = []
        for i in range(num_blocks):
            fft_blocks.append(
                trt.fft(
                    fft_size, True, False, batch_size, mem_model)
            )

        self.blocks_null_source_0 = blocks.null_source(
            gr.sizeof_gr_complex*fft_size)
        self.blocks_null_sink_0 = blocks.null_sink(
            gr.sizeof_gr_complex*fft_size)
        self.blocks_head_0 = blocks.head(
            gr.sizeof_gr_complex*fft_size, int(nsamples / fft_size))

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (fft_blocks[0], 0))
        self.connect((self.blocks_null_source_0, 0), (self.blocks_head_0, 0))

        for i in range(1,num_blocks):
            self.connect((fft_blocks[i-1], 0), (fft_blocks[i], 0))
        
        self.connect((fft_blocks[num_blocks-1], 0), (self.blocks_null_sink_0, 0))


def main(top_block_cls=benchmark_fft, options=None):

    num_samples = 100e6
    dtstr = datetime.datetime.today()
    results_filename = 'benchmark_fft_results_{:%y%d%m_%H%M%S}.json'.format(dtstr)
    res = []

    # mem_mode_iter = [trt.memory_model_t.TRADITIONAL, trt.memory_model_t.PINNED, trt.memory_model_t.UNIFIED]
    # num_blocks_iter = [1, 2, 3, 4]
    # fft_size_iter = [4, 16, 64, 256, 1024]
    # batch_size_iter = [1, 4, 16, 64]

    mem_model_iter = [0,1]
    num_blocks_iter = [1,4]
    fft_size_iter = [64, 256, 1024]
    batch_size_iter = [1, 64]

    for mem_model, num_blocks, fft_size, batch_size in itertools.product(mem_model_iter, num_blocks_iter, fft_size_iter, batch_size_iter):

        args = {'num_samples': num_samples, 'mem_model': mem_model,
                'num_blocks': num_blocks, 'fft_size': fft_size, 'batch_size': batch_size}

        print(args)

        tb = top_block_cls(args)

        def sig_handler(sig=None, frame=None):
            tb.stop()
            tb.wait()
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        print("starting ...")
        startt = time.time()
        tb.start()

        tb.wait()
        endt = time.time()

        r = args
        r['time'] = (endt-startt)
        r['actual_samples'] = tb.actual_samples
        
        meas_tput = tb.actual_samples / (endt-startt)
        r['tput'] = meas_tput
        print("{} MSamps / sec, {}".format(meas_tput/1e6, endt-startt))

        res.append(r)
        with open(results_filename, 'w') as json_file:
            json.dump(res, json_file)


if __name__ == '__main__':
    main()
