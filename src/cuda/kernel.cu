extern "C" __global__ void busy_loop(float* data, int num_loops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[0] + (float)idx; // 避免被编译器完全优化掉
    for (int i = 0; i < num_loops; ++i) {
        val = val * 1.000001f / 1.000001f;
    }
    // 将结果写回，防止整个循环被优化掉
    if (idx == 0) {
        data[0] = val;
    }
}