# gr-trt

GNU Radio wrapping of TensorRT

Infer block Based on the MNIST ONNX included with TensorRT (sample is included for reference)

Design similar to gr-wavelearner, but uses TensorRT API matching the sample code, and loading in of ONNX files

Developed to be used as a basis to benchmark improvements in GNU Radio buffering schemes and modular scheduling

## Dependencies

- CUDA Toolkit (tested with 11.5)
- cuDNN (tested with 8.3.2)
- TensorRT (tested with 8.2.3)

### Installing Dependencies

Installing CUDA drivers can be tricky, so be careful to follow the installation directions of the individual components and test after each step

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

Here are some of the things I had to go through to install on `Ubuntu 20.04` (definitely not an Install Guide):

- Download the nvidia Cuda Toolkit (https://developer.nvidia.com/cuda-toolkit-archive)
- Remove existing nvidia drivers
```bash
sudo apt remove nvidia-driver-450
sudo apt autoremove
```
- Blacklist the nouveau driver (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-nouveau)

Required booting to a non-graphical shell after this, and running the runfile script from there

- Install the CUDA Toolkit
- Add bin dir to $PATH in .bashrc
- Add lib dir to $LD_LIBRARY_PATH in .bashrc


## Building

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTensorRT_ROOT="/usr/local/TensorRT-8.2.3.0"
make -j
make install
make test
```

## Benchmarking

The `bench/` directory contains python flowgraph that are intended to be used with [gr-bench](https://github.com/mormj/gr-bench)

Each of these takes in command line parameters for the sensible variables to be modified, and runs the flowgraph under test and produces a printout of the total time elapsed.  
