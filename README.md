# gr-trt

GNU Radio wrapping of TensorRT, and other CUDA blocks

Infer block Based on the MNIST ONNX included with TensorRT (sample is included for reference)

Design similar to gr-wavelearner, but uses TensorRT API matching the sample code, and loading in of ONNX files

Developed to be used as a basis to benchmark improvements in GNU Radio buffering schemes and modular scheduling

## Dependencies

- CUDA Toolkit (tested with 11.2)
- cuDNN (tested with 11.0)
- TensorRT (tested with 8.0)

## Building

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTensorRT_ROOT="/share/opt/TensorRT-7.2.0.14
make install -j
