# gr-trt

GNU Radio wrapping of TensorRT 

Infer block Based on the MNIST ONNX included with TensorRT (sample is included for reference)

Design similar to gr-wavelearner, but uses TensorRT API matching the sample code, and loading in ONNX files

## Dependencies

- CUDA Toolkit
- cuDNN
- TensorRT

## Building

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTensorRT_ROOT="/share/opt/TensorRT-7.2.0.14
make install -j
