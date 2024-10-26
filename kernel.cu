#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// CUDA kernel for matrix multiplication without loop unrolling
__global__ void matrixMulBasic(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with loop unrolling factor of 2
__global__ void matrixMulUnrolled2(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k += 2) {
            sum += A[row * width + k] * B[k * width + col];
            if (k + 1 < width) {
                sum += A[row * width + k + 1] * B[k * width + col];
            }
        }
        C[row * width + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with loop unrolling factor of 4
__global__ void matrixMulUnrolled4(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k += 4) {
            sum += A[row * width + k] * B[k * width + col];
            if (k + 1 < width) sum += A[row * width + k + 1] * B[k * width + col];
            if (k + 2 < width) sum += A[row * width + k + 2] * B[k * width + col];
            if (k + 3 < width) sum += A[row * width + k + 3] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with loop unrolling factor of 8
__global__ void matrixMulUnrolled8(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k += 8) {
            sum += A[row * width + k] * B[k * width + col];
            if (k + 1 < width) sum += A[row * width + k + 1] * B[k * width + col];
            if (k + 2 < width) sum += A[row * width + k + 2] * B[k * width + col];
            if (k + 3 < width) sum += A[row * width + k + 3] * B[k * width + col];
            if (k + 4 < width) sum += A[row * width + k + 4] * B[k * width + col];
            if (k + 5 < width) sum += A[row * width + k + 5] * B[k * width + col];
            if (k + 6 < width) sum += A[row * width + k + 6] * B[k * width + col];
            if (k + 7 < width) sum += A[row * width + k + 7] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with loop unrolling factor of 16
__global__ void matrixMulUnrolled16(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; k += 16) {
            sum += A[row * width + k] * B[k * width + col];
            if (k + 1 < width) sum += A[row * width + k + 1] * B[k * width + col];
            if (k + 2 < width) sum += A[row * width + k + 2] * B[k * width + col];
            if (k + 3 < width) sum += A[row * width + k + 3] * B[k * width + col];
            if (k + 4 < width) sum += A[row * width + k + 4] * B[k * width + col];
            if (k + 5 < width) sum += A[row * width + k + 5] * B[k * width + col];
            if (k + 6 < width) sum += A[row * width + k + 6] * B[k * width + col];
            if (k + 7 < width) sum += A[row * width + k + 7] * B[k * width + col];
            if (k + 8 < width) sum += A[row * width + k + 8] * B[k * width + col];
            if (k + 9 < width) sum += A[row * width + k + 9] * B[k * width + col];
            if (k + 10 < width) sum += A[row * width + k + 10] * B[k * width + col];
            if (k + 11 < width) sum += A[row * width + k + 11] * B[k * width + col];
            if (k + 12 < width) sum += A[row * width + k + 12] * B[k * width + col];
            if (k + 13 < width) sum += A[row * width + k + 13] * B[k * width + col];
            if (k + 14 < width) sum += A[row * width + k + 14] * B[k * width + col];
            if (k + 15 < width) sum += A[row * width + k + 15] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}


void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() % 100;
    }
}

void printResults(const char* method, int matrixSize, int blockSize, float time) {
    printf("Method: %s, Matrix Size: %d, Block Size: %d, Time: %f ms\n", method, matrixSize, blockSize, time);
}

#define cudaCheckError() {                                 \
    cudaError_t e=cudaGetLastError();                      \
    if(e!=cudaSuccess) {                                   \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        return;                                            \
    }                                                      \
}

int main() {
    int matrixSizes[] = { 128, 256, 512, 1024, 2048, 4096 };
    int blockSizes[] = { 8, 16, 32 };
    int numMatrixSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    for (int i = 0; i < numMatrixSizes; i++) {
        int N = matrixSizes[i];
        int size = N * N;
        size_t bytes = size * sizeof(float);

        // Host matrices
        float* h_A, * h_B, * h_C, * h_D;
        h_A = (float*)malloc(bytes);
        h_B = (float*)malloc(bytes);
        h_C = (float*)malloc(bytes);
        h_D = (float*)malloc(bytes);

        // Initialize matrices
        initializeMatrix(h_A, size);
        initializeMatrix(h_B, size);

        // Device matrices
        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // Copy data to device
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        for (int j = 0; j < numBlockSizes; j++) {
            int blockSize = blockSizes[j];
            int gridSize = (N + blockSize - 1) / blockSize;
            dim3 dimBlock(blockSize, blockSize);
            dim3 dimGrid(gridSize, gridSize);

            // Kernel launch parameters
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Matrix multiplication without loop unrolling
            cudaEventRecord(start);
            matrixMulBasic << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            if (cudaPeekAtLastError() == cudaSuccess) {
                cudaEventElapsedTime(&milliseconds, start, stop);
                printResults("Basic", N, blockSize, milliseconds);
                cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
            }
            else {
                printf("Skipping Basic, Matrix Size: %d, Block Size: %d due to CUDA error.\n", N, blockSize);
            }

            // Matrix multiplication with loop unrolling factor of 2
            cudaEventRecord(start);
            matrixMulUnrolled2 << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if (cudaPeekAtLastError() == cudaSuccess) {
                cudaEventElapsedTime(&milliseconds, start, stop);
                printResults("Unrolled Factor 2", N, blockSize, milliseconds);
                cudaMemcpy(h_D, d_C, bytes, cudaMemcpyDeviceToHost);
            }
            else {
                printf("Skipping Unrolled Factor 2, Matrix Size: %d, Block Size: %d due to CUDA error.\n", N, blockSize);
            }

            // Matrix multiplication with loop unrolling factor of 4
            cudaEventRecord(start);
            matrixMulUnrolled4 << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if (cudaPeekAtLastError() == cudaSuccess) {
                cudaEventElapsedTime(&milliseconds, start, stop);
                printResults("Unrolled Factor 4", N, blockSize, milliseconds);
                cudaMemcpy(h_D, d_C, bytes, cudaMemcpyDeviceToHost);
            }
            else {
                printf("Skipping Unrolled Factor 4, Matrix Size: %d, Block Size: %d due to CUDA error.\n", N, blockSize);
            }

            // Matrix multiplication with loop unrolling factor of 8
            cudaEventRecord(start);
            matrixMulUnrolled8 << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if (cudaPeekAtLastError() == cudaSuccess) {
                cudaEventElapsedTime(&milliseconds, start, stop);
                printResults("Unrolled Factor 8", N, blockSize, milliseconds);
                cudaMemcpy(h_D, d_C, bytes, cudaMemcpyDeviceToHost);
            }
            else {
                printf("Skipping Unrolled Factor 8, Matrix Size: %d, Block Size: %d due to CUDA error.\n", N, blockSize);
            }

            // Matrix multiplication with loop unrolling factor of 16
            cudaEventRecord(start);
            matrixMulUnrolled16 << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if (cudaPeekAtLastError() == cudaSuccess) {
                cudaEventElapsedTime(&milliseconds, start, stop);
                printResults("Unrolled Factor 16", N, blockSize, milliseconds);
                cudaMemcpy(h_D, d_C, bytes, cudaMemcpyDeviceToHost);
            }
            else {
                printf("Skipping Unrolled Factor 16, Matrix Size: %d, Block Size: %d due to CUDA error.\n", N, blockSize);
            }


            // Clean up CUDA events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Clean up host and device memory
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_D);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}
