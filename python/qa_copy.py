#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Perspecta Labs, Inc..
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr, gr_unittest
from gnuradio import blocks
try:
    from trt import copy
except ImportError:
    import os
    import sys
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.append(os.path.join(dirname, "bindings"))
    from trt import copy

class qa_copy(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_copy(self):
        src_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        src = blocks.vector_source_b(src_data)
        op = blocks.copy(gr.sizeof_char)
        dst = blocks.vector_sink_b()
        self.tb.connect(src, op, dst)
        self.tb.run()
        dst_data = dst.data()
        self.assertEqual(expected_result, dst_data)

    # TODO: Add dropping into the CUDA copy block
    # def test_copy_drop(self):
    #     src_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     expected_result = []
    #     src = blocks.vector_source_b(src_data)
    #     op = blocks.copy(gr.sizeof_char)
    #     op.set_enabled(False)
    #     dst = blocks.vector_sink_b()
    #     self.tb.connect(src, op, dst)
    #     self.tb.run()
    #     dst_data = dst.data()
    #     self.assertEqual(expected_result, dst_data)


if __name__ == '__main__':
    gr_unittest.run(qa_copy)
