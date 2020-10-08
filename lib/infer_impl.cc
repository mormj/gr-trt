/* -*- c++ -*- */
/*
 * Copyright 2020 gr-trt author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "infer_impl.h"
#include <gnuradio/io_signature.h>


namespace gr {
namespace trt {

using input_type = float;
using output_type = float;
infer::sptr
infer::make(const std::string& onnx_pathname, size_t itemsize, memory_model_t memory_model, uint64_t workspace_size, int dla_core)
{
    return gnuradio::make_block_sptr<infer_impl>(onnx_pathname, itemsize, memory_model, workspace_size, dla_core);
}


/*
 * The private constructor
 */
infer_impl::infer_impl(const std::string& onnx_pathname,
                       size_t itemsize,
                       memory_model_t memory_model, 
                       uint64_t workspace_size,
                       int dla_core)
    : gr::block(
          "infer",
          gr::io_signature::make(1, 1 /* min, max nr of inputs */, sizeof(input_type)),
          gr::io_signature::make(1, 1 /* min, max nr of outputs */, sizeof(output_type))),
      d_onnx_pathname(onnx_pathname),
      d_engine(nullptr),
      d_memory_model(memory_model),
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
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
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
        // size_t vol = d_context || !d_batch_size ? 1 : static_cast<size_t>(d_batch_size);
        size_t vol = 1; // ONNX Parser only supports explicit batch which means it is baked into the model
        // size_t vol = d_batch_size;
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = d_engine->getBindingComponentsPerElement(i);
            dims.d[vecDim] = samplesCommon::divUp(dims.d[vecDim], scalarsPerVec);
            vol *= scalarsPerVec;
        }
        vol *= samplesCommon::volume(dims);

        void *ptr;
        switch(d_memory_model)
        {
            case memory_model_t::TRADITIONAL:
                // Input memory will be explicitly set to device memory
                
                if (!cudaMalloc(&ptr, vol * samplesCommon::getElementSize(type)) == cudaSuccess )
                    return false;
                if (!cudaMalloc(&ptr, vol * samplesCommon::getElementSize(type)) == cudaSuccess )
                    return false;

                d_device_bindings.emplace_back(ptr);
            break;

            case memory_model_t::PINNED:

                // Input memory will be copied into pinned shared memory

                if (!cudaHostAlloc(&ptr, vol * samplesCommon::getElementSize(type), 0) == cudaSuccess )
                    return false;
                if (!cudaHostAlloc(&ptr, vol * samplesCommon::getElementSize(type), 0) == cudaSuccess )
                    return false;

                d_device_bindings.emplace_back(ptr);

            break;

            case memory_model_t::UNIFIED:

                // Use unified memory constructs

                if (!cudaMallocManaged(&ptr, vol * samplesCommon::getElementSize(type)) == cudaSuccess )
                    return false;
                if (!cudaMallocManaged(&ptr, vol * samplesCommon::getElementSize(type)) == cudaSuccess )
                    return false;

                d_device_bindings.emplace_back(ptr);

            break;

            default:
                throw std::runtime_error("Invalid Memory Model Specified");
        }

    }
    return true;
}


/*
 * Our virtual destructor.
 */
infer_impl::~infer_impl() {}


void infer_impl::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
    int nb = noutput_items / d_output_vlen;
    ninput_items_required[0] = nb * d_input_vlen;
}

int infer_impl::general_work(int noutput_items,
                             gr_vector_int& ninput_items,
                             gr_vector_const_void_star& input_items,
                             gr_vector_void_star& output_items)
{
    const input_type* in = reinterpret_cast<const input_type*>(input_items[0]);
    output_type* out = reinterpret_cast<output_type*>(output_items[0]);

    int in_sz = d_input_vlen; // * d_batch_size;
    int out_sz = d_output_vlen; // * d_batch_size;

    auto num_batches = noutput_items / out_sz;
    auto ni = num_batches * in_sz;

    for (auto b = 0; b < num_batches; b++) {

        // memcpy(d_input_buffer, in, noutput_items * sizeof(float));

        // Memcpy from host input buffers to device input buffers
        // d_buffers->copyInputToDevice();
        cudaMemcpy(d_device_bindings[0],
                   in + b * in_sz,
                   in_sz * sizeof(float),
                   cudaMemcpyHostToDevice);

        bool status = d_context->executeV2(d_device_bindings.data());
        // bool status = d_context->execute(d_batch_size, d_device_bindings.data());
        if (!status) {
            return false;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(out + b * out_sz,
                   d_device_bindings[1],
                   out_sz * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
    }

    consume_each(ni);

    // Tell runtime system how many output items we produced.
    return noutput_items;
}

} /* namespace trt */
} /* namespace gr */
