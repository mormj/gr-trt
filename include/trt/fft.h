/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TRT_FFT_H
#define INCLUDED_TRT_FFT_H

#include <gnuradio/sync_block.h>
#include <trt/api.h>

namespace gr {
namespace trt {

/*!
 * \brief <+description of block+>
 * \ingroup trt
 *
 */
class TRT_API fft : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<fft> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of trt::fft.
     *
     * To avoid accidental use of raw pointers, trt::fft's
     * constructor is in a private implementation
     * class. trt::fft::make is the public interface for
     * creating new instances.
     */
    static sptr make(const size_t fft_size,
                     const bool forward,
                     const std::vector<float>& window,
                     bool shift = false,
                     const size_t batch_size = 1);
};

} // namespace trt
} // namespace gr

#endif /* INCLUDED_TRT_FFT_H */
