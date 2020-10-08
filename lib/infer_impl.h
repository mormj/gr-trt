/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_TRT_INFER_IMPL_H
#define INCLUDED_TRT_INFER_IMPL_H

#include <trt/infer.h>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

namespace gr {
namespace trt {

class infer_impl : public infer
{
private:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    nvinfer1::Dims d_input_dims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims d_output_dims; //!< The dimensions of the output to the network.
    bool d_int8{ false };         //!< Allow runnning the network in Int8 mode.
    bool d_fp16{ false };         //!< Allow running the network in FP16 mode.
    int32_t d_dla_core{ -1 };     //!< Specify the DLA core to run network on.

    std::shared_ptr<nvinfer1::ICudaEngine>
        d_engine; //!< The TensorRT engine used to run the network
    std::string d_onnx_pathname;
    int32_t d_batch_size;
    uint64_t d_workspace_size;
    memory_model_t d_memory_model;

    std::shared_ptr<samplesCommon::BufferManager> d_buffers;
    SampleUniquePtr<nvinfer1::IExecutionContext> d_context;

    std::vector<void*> d_device_bindings;

    int d_inputH, d_outputH, d_inputW, d_outputW;
    int d_input_vlen, d_output_vlen;

    bool build();
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                          SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser);

public:
    infer_impl(const std::string& onnx_pathname, size_t itemsize, size_t batch_size, memory_model_t memory_model, uint64_t workspace_size, int dla_core);
    ~infer_impl();

    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace trt
} // namespace gr

#endif /* INCLUDED_TRT_INFER_IMPL_H */
