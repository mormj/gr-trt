/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_impl.h"
#include <gnuradio/io_signature.h>

#include <helper_cuda.h>

namespace gr {
namespace trt {

using input_type = gr_complex;
using output_type = gr_complex;
fft::sptr fft::make(const size_t fft_size,
                    const bool forward,
                    const std::vector<float>& window,
                    bool shift,
                    const size_t batch_size,
                    const memory_model_t mem_model)
{
    return gnuradio::make_block_sptr<fft_impl>(
        fft_size, forward, window, shift, batch_size, mem_model);
}

/*
 * The private constructor
 */
fft_impl::fft_impl(const size_t fft_size,
                   const bool forward,
                   const std::vector<float>& window,
                   bool shift,
                   const size_t batch_size,
                   const memory_model_t mem_model)
    : gr::sync_block("fft",
                     gr::io_signature::make(
                         1, 1 /* min, max nr of inputs */, fft_size * sizeof(input_type)),
                     gr::io_signature::make(1,
                                            1 /* min, max nr of outputs */,
                                            fft_size * sizeof(output_type))),
      d_fft_size(fft_size),
      d_forward(forward),
      d_window(window),
      d_shift(shift),
      d_batch_size(batch_size),
      d_mem_model(mem_model)

{

    if (d_mem_model == memory_model_t::TRADITIONAL) {

        checkCudaErrors(cudaMalloc((void**)&d_data,
                                   sizeof(cufftComplex) * d_fft_size * d_batch_size));


    } else if (d_mem_model == memory_model_t::PINNED) {
        checkCudaErrors(cudaHostAlloc(
            (void**)&d_data, sizeof(cufftComplex) * d_fft_size * d_batch_size, 0));

    } else // UNIFIED
    {
        checkCudaErrors(cudaMallocManaged(
            (void**)&d_data, sizeof(cufftComplex) * d_fft_size * d_batch_size));
    }

    size_t workSize;
    int fftSize = d_fft_size;

    checkCudaErrors(cufftCreate(&d_plan));

    checkCudaErrors(cufftMakePlanMany(
        d_plan, 1, &fftSize, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, d_batch_size, &workSize));
    printf("Temporary buffer size %li bytes\n", workSize);

    // checkCudaErrors(cufftPlan1d(&d_plan, d_fft_size, CUFFT_C2C, 1));

    set_output_multiple(d_batch_size);
}

/*
 * Our virtual destructor.
 */
fft_impl::~fft_impl()
{
    cufftDestroy(d_plan);
    cudaFree(d_data);
}

int fft_impl::work(int noutput_items,
                   gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items)
{
    const input_type* in = reinterpret_cast<const input_type*>(input_items[0]);
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);

    auto work_size = d_batch_size * d_fft_size; // number of samples
    auto nvecs = noutput_items / d_batch_size;
    auto mem_size = work_size * sizeof(gr_complex); // in bytes, for the memcpy

    for (auto s = 0; s < nvecs; s++) {

        if (d_mem_model == memory_model_t::TRADITIONAL) {
            checkCudaErrors(
                cudaMemcpy(d_data, in + s * work_size, mem_size, cudaMemcpyHostToDevice));
        } else {
            memcpy(d_data, in + s * work_size, mem_size);
        }

        if (d_forward) {
            checkCudaErrors(cufftExecC2C(d_plan, d_data, d_data, CUFFT_FORWARD));
        } else {
            checkCudaErrors(cufftExecC2C(d_plan, d_data, d_data, CUFFT_INVERSE));
        }

        cudaDeviceSynchronize();

        if (d_mem_model == memory_model_t::TRADITIONAL) {
            checkCudaErrors(cudaMemcpy(
                out + s * work_size, d_data, mem_size, cudaMemcpyDeviceToHost));
        } else {
            memcpy(out + s * work_size, d_data, mem_size);
        }
    }

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace trt */
} // namespace gr
