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
        nsamples = args['samples']
        batch_size = args['batchsize']
        self.actual_samples = actual_samples = (batch_size) * int(nsamples /  batch_size)
        num_blocks = args['nblocks']
        mem_model = args['memmodel']

        ##################################################
        # Blocks
        ##################################################
        ptblocks = []
        for i in range(num_blocks):
            ptblocks.append(
                trt.copy(
                    batch_size, mem_model)
            )

        self.blocks_null_source_0 = blocks.null_source(
            gr.sizeof_gr_complex*1)
        self.blocks_null_sink_0 = blocks.null_sink(
            gr.sizeof_gr_complex*1)
        self.blocks_head_0 = blocks.head(
            gr.sizeof_gr_complex*1, actual_samples)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (ptblocks[0], 0))
        self.connect((self.blocks_null_source_0, 0), (self.blocks_head_0, 0))

        for i in range(1, num_blocks):
            self.connect((ptblocks[i-1], 0), (ptblocks[i], 0))

        self.connect((ptblocks[num_blocks-1], 0),
                     (self.blocks_null_sink_0, 0))


def main(top_block_cls=benchmark_fft, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

    
    dtstr = datetime.datetime.today()
    results_filename = 'benchmark_copy_results_{:%y%d%m_%H%M%S}.json'.format(
        dtstr)

    json_output = {}
    res = []
    json_output['params'] = {}
    json_output['results'] = res
    save_file = True

    num_samples = 1e9
    mem_model_iter = [0, 1]
    num_blocks_iter = [1,2,4,8]
    batch_size_iter = [1024*x for x in [1,2,4,8,16,32,64,128,256,1024]]

    
    # num_samples = 1e9
    # mem_model_iter = [0]
    # num_blocks_iter = [1]
    # batch_size_iter = [1024*16]
    # save_file = False


    for mem_model, num_blocks, batch_size in itertools.product(mem_model_iter, num_blocks_iter,  batch_size_iter):

        args = {'samples': num_samples, 'memmodel': mem_model,
                'nblocks': num_blocks, 'batchsize': batch_size}

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
        if save_file:
            with open(results_filename, 'w') as json_file:
                json.dump(json_output, json_file)


if __name__ == '__main__':
    main()
