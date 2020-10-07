/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TRT_INFER_H
#define INCLUDED_TRT_INFER_H

#include <gnuradio/sync_block.h>
#include <trt/api.h>

namespace gr {
namespace trt {

/*!
 * \brief <+description of block+>
 * \ingroup trt
 *
 */
class TRT_API infer : virtual public gr::block
{
public:
    typedef std::shared_ptr<infer> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of trt::infer.
     *
     * To avoid accidental use of raw pointers, trt::infer's
     * constructor is in a private implementation
     * class. trt::infer::make is the public interface for
     * creating new instances.
     */
    static sptr make(const std::string& onnx_pathname, size_t itemsize, size_t batch_size=1, uint64_t workspace_size=(1<<30), int dla_core = -1);
};

} // namespace trt
} // namespace gr

#endif /* INCLUDED_TRT_INFER_H */
