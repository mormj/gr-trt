/* -*- c++ -*- */
/*
 * Copyright 2020 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TRT_PASSTHROUGH_H
#define INCLUDED_TRT_PASSTHROUGH_H

#include <gnuradio/sync_block.h>
#include <trt/api.h>
#include <trt/memmodel.h>

namespace gr {
namespace trt {

/*!
 * \brief <+description of block+>
 * \ingroup trt
 *
 */
class TRT_API passthrough : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<passthrough> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of trt::passthrough.
     *
     * To avoid accidental use of raw pointers, trt::passthrough's
     * constructor is in a private implementation
     * class. trt::passthrough::make is the public interface for
     * creating new instances.
     */
    static sptr make(int batch_size, memory_model_t mem_model);
};

} // namespace trt
} // namespace gr

#endif /* INCLUDED_TRT_PASSTHROUGH_H */
