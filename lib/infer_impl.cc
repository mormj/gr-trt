/* -*- c++ -*- */
/*
 * Copyright 2020 Perspecta Labs, Inc.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "infer_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/cuda/cuda_buffer.h>

namespace gr {
namespace trt {

using input_type = float;
using output_type = float;
infer::sptr infer::make(const std::string& onnx_pathname,
                        size_t itemsize,
                        uint64_t workspace_size,
                        int dla_core)
{
    return gnuradio::make_block_sptr<infer_impl>(
        onnx_pathname, itemsize, workspace_size, dla_core);
}


/*
 * The private constructor
 */
infer_impl::infer_impl(const std::string& onnx_pathname,
                       size_t itemsize,
                       uint64_t workspace_size,
                       int dla_core)
    : gr::block(
          "infer",
          gr::io_signature::make(1, 1, sizeof(input_type), cuda_buffer::type),
          gr::io_signature::make(1, 1, sizeof(output_type), cuda_buffer::type)),
      d_onnx_pathname(onnx_pathname),
      d_engine(nullptr),
      d_workspace_size(workspace_size),
      d_dla_core(dla_core)
{
    build();

    set_output_multiple(d_output_vlen);
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST
//! network
//!
//! \param builder Pointer to the engine builder
//!
bool infer_impl::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                  SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                                  SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                  SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed =
        parser->parseFromFile(d_onnx_pathname.c_str(),
                              static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    // config->setMaxWorkspaceSize(2_GiB);
    config->setMaxWorkspaceSize(d_workspace_size);
    if (d_fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (d_int8) {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), d_dla_core);

    return true;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and
//! builds
//!          the engine that will be used to run MNIST (d_engine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool infer_impl::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    const auto explicitBatch =
        1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }


    auto config =
        SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    d_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!d_engine) {
        return false;
    }

    std::cout << "Max Batch Size: " << d_engine->getMaxBatchSize() << std::endl;
    assert(network->getNbInputs() == 1);
    d_input_dims = network->getInput(0)->getDimensions();
    // assert(d_input_dims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    d_output_dims = network->getOutput(0)->getDimensions();
    // assert(d_output_dims.nbDims == 2);

    d_inputH = d_input_dims.d[0];
    d_inputW = d_input_dims.d[1];
    d_outputH = d_output_dims.d[0];
    d_outputW = d_output_dims.d[1];

    d_input_vlen = d_inputH * d_inputW;
    d_output_vlen = d_outputH * d_outputW;

    d_context =
        SampleUniquePtr<nvinfer1::IExecutionContext>(d_engine->createExecutionContext());
    if (!d_context) {
        return false;
    }

    auto nb = d_engine->getNbBindings();
    for (auto i = 0; i < nb; i++) {
        auto type = d_engine->getBindingDataType(i);
        auto dims = d_context->getBindingDimensions(i);
        int vecDim = d_engine->getBindingVectorizedDim(i);
        // size_t vol = d_context || !d_batch_size ? 1 :
        // static_cast<size_t>(d_batch_size);
        size_t vol = 1; // ONNX Parser only supports explicit batch which means it is
                        // baked into the model
        // size_t vol = d_batch_size;
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = d_engine->getBindingComponentsPerElement(i);
            dims.d[vecDim] = samplesCommon::divUp(dims.d[vecDim], scalarsPerVec);
            vol *= scalarsPerVec;
        }
        vol *= samplesCommon::volume(dims);

        d_device_bindings.resize(2);

    }
    return true;
}


/*
 * Our virtual destructor.
 */
infer_impl::~infer_impl() {}


void infer_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = ( noutput_items * d_input_vlen ) / d_output_vlen;

    // std::cout << "Forecast: " << noutput_items << " / " << ninput_items_required[0] << std::endl;
}

int infer_impl::general_work(int noutput_items,
                             gr_vector_int& ninput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    // std::cout << "work" << std::endl;
    const input_type* in = static_cast<const input_type*>(input_items[0]);
    output_type* out = static_cast<output_type*>(output_items[0]);

    int in_sz = d_input_vlen;   // * d_batch_size;
    int out_sz = d_output_vlen; // * d_batch_size;

    auto num_batches = noutput_items / out_sz;
    auto ni = num_batches * in_sz;

    for (auto b = 0; b < num_batches; b++) {

        d_device_bindings[0] = const_cast<input_type *>(in + b * in_sz);
        d_device_bindings[1] = const_cast<output_type *>(out + b * out_sz);

        bool status = d_context->executeV2(d_device_bindings.data());
        // bool status = d_context->execute(d_batch_size, d_device_bindings.data());
        if (!status) {
            return false;
        }
        cudaDeviceSynchronize();
    }

    // std::cout << "consumed " << ni << std::endl;
    consume_each(ni);

    // Tell runtime system how many output items we produced.
    // std::cout << "produced " << noutput_items << std::endl;
    return noutput_items;
}

} /* namespace trt */
} /* namespace gr */
