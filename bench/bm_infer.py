#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.9.0.0-git

from gnuradio import gr, blocks
import sys
import signal
from argparse import ArgumentParser
from gnuradio.fft import window
import time
import trt

class benchmark_fft(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        nsamples = args.samples
        mem_model = args.memmodel

        ##################################################
        # Blocks
        ##################################################
        src = blocks.file_source(gr.sizeof_float*1, args.data_filename, True, 0, 0)
        infer = trt.infer(args.model_filename, gr.sizeof_float, mem_model, 1073741824, -1)
        snk = blocks.null_sink(
            gr.sizeof_float)
        hd = blocks.head(
            gr.sizeof_float, int(nsamples))


        ##################################################
        # Connections
        ##################################################
        self.connect((hd, 0), (infer, 0))
        self.connect((src, 0), (hd, 0))

        self.connect((infer, 0),
                     (snk, 0))


def main(top_block_cls=benchmark_fft, options=None):
    parser = ArgumentParser(description='Benchmark series of cuFFT blocks')
    parser.add_argument('--rt_prio', help='enable realtime scheduling', action='store_true')
    parser.add_argument('--samples', type=int, default=2e8)
    # parser.add_argument('--fftsize', type=int, default=1)
    # parser.add_argument('--batchsize', type=int, default=1) # batchsize is built into the onnx model
    parser.add_argument('--memmodel', type=int, default=0)
    parser.add_argument('--model_filename', type=str)
    parser.add_argument('--data_filename', type=str)
    # parser.add_argument('--output_len', type=int)

    args = parser.parse_args()
    print(args)

    if args.rt_prio and gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

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

    print(f'[PROFILE_TIME]{endt-startt}[PROFILE_TIME]')



if __name__ == '__main__':
    main()
