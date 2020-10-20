/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TRT_FFT_IMPL_H
#define INCLUDED_TRT_FFT_IMPL_H

#include <trt/fft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

namespace gr {
namespace trt {

class fft_impl : public fft
{
private:
    size_t d_fft_size;
    bool d_forward;
    std::vector<float> d_window;
    bool d_shift;
    size_t d_batch_size;
    memory_model_t d_mem_model;

    cufftHandle d_plan;
    cufftComplex *d_data;
    cufftComplex *d_data2;
    float *d_window_dev;
    


public:
    fft_impl(const size_t fft_size,
                     const bool forward,
                     const std::vector<float>& window,
                     bool shift = false,
                     const size_t batch_size = 1,
                     const memory_model_t mem_model = memory_model_t::TRADITIONAL);
    ~fft_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace trt
} // namespace gr

#endif /* INCLUDED_TRT_FFT_IMPL_H */
