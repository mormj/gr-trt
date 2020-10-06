#include <cuComplex.h>

__global__ void
apply_window_kernel(cuFloatComplex* in, float* window, int fft_size, int batch_size)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = fft_size * batch_size;
    if (i < n) {
        // int w = i % fft_size;
        in[i].x *= window[i];
        in[i].y *= window[i];
    }
}


void apply_window(cuFloatComplex* in, float* window, int fft_size, int batch_size)
{
    apply_window_kernel<<<batch_size, fft_size>>>(in, window, fft_size, batch_size);
}