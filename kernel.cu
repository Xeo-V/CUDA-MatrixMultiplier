
// Optimized CUDA Matrix Multiplication Code
// Author: X30
// This code is optimized using techniques like tiling, shared memory, and efficient thread configurations.
// Feel free to modify and use it for your projects.
// Keep in mind if you are a college student and are planning on to show this porject as its the one you created, make sure to read the explaination.md provided with the project.
// I have added a lot of information, function usage, arguments information, etc.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Define tile size for tiling optimization
// You can change this value to see how it affects performance.
#define TILE_WIDTH 16  

// Function to print a matrix for debugging purposes
// Takes a 1D array representation of the matrix, the size N, and a name for the matrix.
void printMatrix(float* matrix, int N, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CUDA Kernel for optimized matrix multiplication
// This kernel takes four parameters:
// A, B are the input matrices, C is the output matrix, and N is the size of the matrices.
// A, B, and C are 1D array representations of the matrices.
__global__ void matrixMulOptimized(float* A, float* B, float* C, int N) {
    // Shared memory to store tiles of matrices A and B
    // This allows threads in the same block to reuse these values, reducing global memory accesses.
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Thread and block indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Calculate the row and column indices for this thread
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float val = 0;

    // Loop over tiles of the input matrices
    for (int t = 0; t < (N - 1) / TILE_WIDTH + 1; ++t) {
        // Load a tile of A and B into shared memory
        // If the row or column is out of bounds, load a zero.
        if (row < N && t * TILE_WIDTH + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_WIDTH + tx];
        }
        else {
            As[ty][tx] = 0;
        }

        if (col < N && t * TILE_WIDTH + ty < N) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        }
        else {
            Bs[ty][tx] = 0;
        }

        // Synchronize to make sure all threads have loaded their data into shared memory
        __syncthreads();

        // Perform the matrix multiplication for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            val += As[ty][k] * Bs[k][tx];
        }

        // Synchronize again to make sure computation is done before overwriting shared memory in the next iteration
        __syncthreads();
    }

    // Write the computed value to the output matrix C
    if (row < N && col < N) {
        C[row * N + col] = val;
    }
}

int main() {
    int N;  // Size of the matrix
    char choice;  // User choice for integer or float matrix

    // User input for matrix size and type
    printf("Enter the size of the matrix: ");
    scanf("%d", &N);
    printf("Do you want an integer matrix or a float matrix? (i/f): ");
    scanf(" %c", &choice);

    // Allocate and initialize host memory for matrices A, B, and C
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B with random values
    // If the user chose 'i', the matrices will have integer values.
    // Otherwise, they will have float values.
    for (int i = 0; i < N * N; ++i) {
        if (choice == 'i') {
            A[i] = (float)(rand() % 10);
            B[i] = (float)(rand() % 10);
        }
        else {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Print initial matrices for debugging
    printMatrix(A, N, "A");
    printMatrix(B, N, "B");

    // Allocate device memory for matrices A, B, and C
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the dimensions of the thread block and grid
    // These are optimized to ensure all tiles are covered
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N - 1) / TILE_WIDTH + 1, (N - 1) / TILE_WIDTH + 1);

    // Launch the optimized kernel
    matrixMulOptimized << <dimGrid, dimBlock >> > (d_A, d_B, d_C, N);

    // Copy the result back to host memory
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting matrix
    printMatrix(C, N, "C");

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    // Wait for user input before exiting
    printf("Press Enter to exit...");
    getchar();  // Consume the newline from previous input
    getchar();  // Wait for user to press Enter

    return 0;
}
