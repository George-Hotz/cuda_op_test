#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cuda_fp16.h>
#include "softmax_fp16_kernel.h"  // 包含之前的 FP16 头文件

// CPU 参考实现: Softmax (FP32 计算，与 FP16 GPU 结果比较)
void softmax_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; ++i) {
        const float* x = inp + i * C;
        float* y = out + i * C;

        // Step 1: 计算最大值（数值稳定性）
        float maxval = -INFINITY;
        for (int j = 0; j < C; ++j) {
            if (x[j] > maxval) maxval = x[j];
        }

        // Step 2: 计算 exp(x - max) 和 sum
        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            y[j] = expf(x[j] - maxval);
            sum += y[j];
        }

        // Step 3: 归一化
        for (int j = 0; j < C; ++j) {
            y[j] /= sum;
        }
    }
}

// 生成随机数据 (FP16)
void generate_random_data(half* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; ++i) {
        float val = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
        data[i] = __float2half(val);
    }
}

// 比较 FP16 GPU 结果和 FP32 CPU 结果（允许更大误差）
bool compare_results(const half* a, const float* b, int size, float epsilon = 1e-2) {
    for (int i = 0; i < size; ++i) {
        float a_f32 = __half2float(a[i]);
        if (fabs(a_f32 - b[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << a_f32 << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 100;  // 行数
    const int C = 1024; // 列数
    const int total_elements = N * C;

    // 分配主机内存
    std::vector<half> h_inp(total_elements);
    std::vector<float> h_out_cpu(total_elements);  // CPU 结果仍用 FP32
    std::vector<half> h_out_gpu(total_elements);

    // 生成随机输入数据 (FP16)
    generate_random_data(h_inp.data(), total_elements);

    // 分配设备内存 (FP16)
    half *d_inp, *d_out;
    cudaMalloc(&d_inp, total_elements * sizeof(half));
    cudaMalloc(&d_out, total_elements * sizeof(half));

    // 拷贝数据到设备
    cudaMemcpy(d_inp, h_inp.data(), total_elements * sizeof(half), cudaMemcpyHostToDevice);

    // 为 CPU 计算创建 FP32 副本
    std::vector<float> h_inp_f32(total_elements);
    for (int i = 0; i < total_elements; ++i) {
        h_inp_f32[i] = __half2float(h_inp[i]);
    }

    // 调用 CPU 参考实现 (FP32)
    softmax_cpu(h_out_cpu.data(), h_inp_f32.data(), N, C);

    // 调用 GPU 核函数 (FP16)
    launch_softmax_kernel(d_out, d_inp, N, C);

    // 拷贝结果回主机
    cudaMemcpy(h_out_gpu.data(), d_out, total_elements * sizeof(half), cudaMemcpyDeviceToHost);

    // 验证结果 (允许更大误差)
    bool is_correct = compare_results(h_out_gpu.data(), h_out_cpu.data(), total_elements, 1e-2);
    if (is_correct) {
        std::cout << "Test passed! GPU FP16 results match CPU FP32 reference (within tolerance)." << std::endl;
    } else {
        std::cerr << "Test failed! GPU FP16 results do not match CPU FP32 reference." << std::endl;
    }

    // 性能测试
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_trials = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_trials; ++i) {
        launch_softmax_kernel(d_out, d_inp, N, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Average GPU FP16 time: " << milliseconds / num_trials << " ms" << std::endl;

    // 释放资源
    cudaFree(d_inp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return is_correct ? 0 : 1;
}