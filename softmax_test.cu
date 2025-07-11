#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include "softmax_kernel.h"  // 包含之前的头文件

// CPU 参考实现: Softmax (按行计算)
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

// 生成随机数据
void generate_random_data(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; ++i) {
        data[i] = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
    }
}

// 比较两个数组是否接近（允许误差）
bool compare_results(const float* a, const float* b, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
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
    std::vector<float> h_inp(total_elements);
    std::vector<float> h_out_cpu(total_elements);
    std::vector<float> h_out_gpu(total_elements);

    // 生成随机输入数据
    generate_random_data(h_inp.data(), total_elements);

    // 分配设备内存
    float *d_inp, *d_out;
    cudaMalloc(&d_inp, total_elements * sizeof(float));
    cudaMalloc(&d_out, total_elements * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_inp, h_inp.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 调用 CPU 参考实现
    softmax_cpu(h_out_cpu.data(), h_inp.data(), N, C);

    // 调用 GPU 核函数
    launch_softmax_kernel(d_out, d_inp, N, C);

    // 拷贝结果回主机
    cudaMemcpy(h_out_gpu.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool is_correct = compare_results(h_out_cpu.data(), h_out_gpu.data(), total_elements);
    if (is_correct) {
        std::cout << "Test passed! GPU results match CPU reference." << std::endl;
    } else {
        std::cerr << "Test failed! GPU results do not match CPU reference." << std::endl;
    }

    // 性能测试（可选）
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
    std::cout << "Average GPU time: " << milliseconds / num_trials << " ms" << std::endl;

    // 释放资源
    cudaFree(d_inp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return is_correct ? 0 : 1;
}