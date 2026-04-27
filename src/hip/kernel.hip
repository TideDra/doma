#include <hip/hip_runtime.h>

extern "C" __global__ void busy_loop(float* data, int num_loops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[0] + (float)idx;
    for (int i = 0; i < num_loops; ++i) {
        val = val * 1.000001f + 0.000001f;
        if (val > 1.0e20f) {
            val = data[0];
        }
    }
    if (idx == 0) {
        data[0] = val;
    }
}
