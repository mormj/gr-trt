/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "fft_impl.h"
#include <gnuradio/io_signature.h>

#include <helper_cuda.h>

extern void apply_window(cuFloatComplex* in, float* window, int fft_size, int batch_size);

namespace gr {
namespace trt {

using input_type = gr_complex;
using output_type = gr_complex;
fft::sptr fft::make(const size_t fft_size,
                    const bool forward,
                    const std::vector<float>& window,
                    bool shift,
                    const size_t batch_size)
{
    return gnuradio::make_block_sptr<fft_impl>(
        fft_size, forward, window, shift, batch_size);
}


/*
 * The private constructor
 */
fft_impl::fft_impl(const size_t fft_size,
                   const bool forward,
                   const std::vector<float>& window,
                   bool shift,
                   const size_t batch_size)
    : gr::sync_block(
          "fft",
          gr::io_signature::make(1, 1 /* min, max nr of inputs */, sizeof(input_type)),
          gr::io_signature::make(1, 1 /* min, max nr of outputs */, sizeof(output_type))),
      d_fft_size(fft_size),
      d_forward(forward),
      d_window(window),
      d_shift(shift),
      d_batch_size(batch_size)

{

    checkCudaErrors(
        cudaMalloc((void**)&d_data, sizeof(cufftComplex) * d_fft_size * d_batch_size));

    checkCudaErrors(
        cudaMalloc((void**)&d_window_dev, sizeof(float) * d_fft_size * d_batch_size));

    for (auto i = 0; i < d_batch_size; i++) {
        checkCudaErrors(cudaMemcpy(d_window_dev + i * d_fft_size,
                                   &window[0],
                                   d_fft_size * sizeof(float),
                                   cudaMemcpyHostToDevice));
    }

    size_t workSize;
    int fftSize = d_fft_size;

    checkCudaErrors(cufftCreate(&d_plan));

    checkCudaErrors(cufftMakePlanMany(
        d_plan, 1, &fftSize, NULL, 1, 1, NULL, 1, 1, CUFFT_C2C, d_batch_size, &workSize));
    printf("Temporary buffer size %li bytes\n", workSize);

    // checkCudaErrors(cufftPlan1d(&d_plan, d_fft_size, CUFFT_C2C, 1));

    set_output_multiple(d_fft_size * d_batch_size);
}

/*
 * Our virtual destructor.
 */
fft_impl::~fft_impl()
{
    cufftDestroy(d_plan);
    cudaFree(d_data);
    cudaFree(d_window_dev);
}

int fft_impl::work(int noutput_items,
                   gr_vector_const_void_star& input_items,
                   gr_vector_void_star& output_items)
{
    const input_type* in = reinterpret_cast<const input_type*>(input_items[0]);
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);

    auto work_size = d_batch_size * d_fft_size;
    auto nvecs = noutput_items / work_size;
    auto mem_size = work_size * sizeof(gr_complex);

    for (auto s = 0; s < nvecs; s++) {
        checkCudaErrors(
            cudaMemcpy(d_data, in + s * work_size, mem_size, cudaMemcpyHostToDevice));

        apply_window(d_data, d_window_dev, d_fft_size, d_batch_size);
        cudaDeviceSynchronize();


        if (d_forward) {
            cufftExecC2C(d_plan, d_data, d_data, CUFFT_FORWARD);
        } else {
            cufftExecC2C(d_plan, d_data, d_data, CUFFT_INVERSE);
        }

        cudaDeviceSynchronize();

        checkCudaErrors(
            cudaMemcpy(out + s * work_size, d_data, mem_size, cudaMemcpyDeviceToHost));
    }

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace trt */
} /* namespace gr */
