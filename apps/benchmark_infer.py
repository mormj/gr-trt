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

class benchmark_infer(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        num_samples = args['num_samples']
        batch_size = args['batch_size']
        filename = args['onnx_filename']
        input_dim = args['input_dim']

        self.actual_samples = actual_samples = (
            input_dim * batch_size) * int(num_samples / (input_dim * batch_size))
        mem_model = args['mem_model']



        ##################################################
        # Blocks
        ##################################################
        self.trt_infer_0 = trt.infer(f'{filename}_{batch_size}.onnx', gr.sizeof_float, mem_model, 1073741824, -1)
        self.blocks_vector_source_x_0 = blocks.vector_source_f(list(range(256)) * int(num_samples / 256), False, 1, [])
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*1)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_vector_source_x_0, 0), (self.trt_infer_0, 0))
        self.connect((self.trt_infer_0, 0), (self.blocks_null_sink_0, 0))


def main(top_block_cls=benchmark_infer, options=None):

    num_samples = 100e6
    onnx_filename = '/share/gnuradio/benchmark-dnn/FCNSMALL'
    input_dim = 256
    # output_dim = 11

    dtstr = datetime.datetime.today()
    results_filename = 'benchmark_infer_results_{:%y%d%m_%H%M%S}.json'.format(dtstr)
    res = []

    mem_model_iter = [0,1,2]
    batch_size_iter = [1,2,4,8,16,32]

    for mem_model, batch_size in itertools.product(mem_model_iter, batch_size_iter):

        args = {'input_dim':input_dim,'onnx_filename':onnx_filename, 'num_samples': num_samples, 'mem_model': mem_model,
                 'batch_size': batch_size}

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
