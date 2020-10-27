/* -*- c++ -*- */
/*
 * Copyright 2020 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TRT_PASSTHROUGH_IMPL_H
#define INCLUDED_TRT_PASSTHROUGH_IMPL_H

#include <trt/passthrough.h>
#include <cuComplex.h>

namespace gr {
namespace trt {

class passthrough_impl : public passthrough
{
private:
    int d_batch_size;
    int d_min_grid_size;
    int d_block_size;
    memory_model_t d_mem_model;
    cuFloatComplex *d_data;

public:
    passthrough_impl(int batch_size, memory_model_t mem_model);
    ~passthrough_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace trt
} // namespace gr

#endif /* INCLUDED_TRT_PASSTHROUGH_IMPL_H */
