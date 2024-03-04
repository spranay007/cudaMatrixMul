#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <ctime>

//Function Declarations
void matrixMulHost(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB);
void AllocateDeviceMemory(float** d_A, float** d_B, float** d_C, int mem_size_A, int mem_size_B, int mem_size_C);
void CopyHostToDevice(float* d_A, float* d_B, float* d_C, float* h_A, float* h_B, int mem_size_A, int mem_size_B);
void CopyDeviceToHost(float* h_C, float* d_C, int mem_size_C);
void DeallocateDeviceMemory(float* d_A, float* d_B, float* d_C);
void matrixMulHost(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB);
int MatrixMultiply(int argc, char** argv, int block_size, const dim3& dimsA, const dim3& dimsB);
void PopulateMatrix(float* data, int matSize);
void printMatrix(float* matrix, int numRows, int numCols, const char* label);
void printMatrices(float* h_C_CPU, float* h_C_GPU, int numRows, int numCols);

// Kernel for tiled matrix multiplication
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int numRowsA, int numColsA, int numColsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread falls within the matrix dimensions
    if (row >= numRowsA || col >= numColsB) return;

    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0.0;

    // Loop over tiles
    for (int t = 0; t < (numColsA + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int globalCol = t * BLOCK_SIZE + threadIdx.x;

        // Load tiles into shared memory
        if (row < numRowsA && globalCol < numColsA) {
            As[threadIdx.y][threadIdx.x] = A[row * numColsA + globalCol];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (globalCol < numColsB && col < numColsB) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * numColsB + col];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply tiles and accumulate the result
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            if (row < numRowsA && col < numColsB) {
                Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        // Synchronize to make sure the multiplication is done before loading new tiles
        __syncthreads();
    }

    // Write the result to global memory
    if (row < numRowsA && col < numColsB) {
        C[row * numColsB + col] = Cvalue;
    }
}

void AllocateDeviceMemory(float** d_A, float** d_B, float** d_C, int mem_size_A, int mem_size_B, int mem_size_C) {
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(d_C), mem_size_C));
}

void CopyHostToDevice(float* d_A, float* d_B, float* d_C, float* h_A, float* h_B, int mem_size_A, int mem_size_B) {
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
}

void CopyDeviceToHost(float* h_C, float* d_C, int mem_size_C) {
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
}

void DeallocateDeviceMemory(float* d_A, float* d_B, float* d_C) {
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}

// Function to perform matrix multiplication on the host
void matrixMulHost(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB) {
    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numColsB; ++j) {
            float sum = 0.0;
            for (int k = 0; k < numColsA; ++k) {
                sum += A[i * numColsA + k] * B[k * numColsB + j];
            }
            C[i * numColsB + j] = sum;
        }
    }
}

void PopulateMatrix(float* data, int matSize)
{
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < matSize; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(float* matrix, int numRows, int numCols, const char* label) {
    printf("%s:\n", label);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            printf("%.4f\t", matrix[i * numCols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrices(float* h_C_CPU, float* h_C_GPU, int numRows, int numCols) {

    printMatrix(h_C_CPU, numRows, numCols, "Matrix C (CPU)");

    printMatrix(h_C_GPU, numRows, numCols, "Matrix C (GPU)");
}

int MatrixMultiply(int argc, char** argv, int block_size, const dim3& dimsA, const dim3& dimsB) {
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    PopulateMatrix(h_A, size_A);
    PopulateMatrix(h_B, size_B);

    float* d_A, * d_B, * d_C;

    dim3 dimsC(dimsA.x, dimsB.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }
    // memory for calcualtion of matrix on the cpu separately
    float* h_C_CPU = new float[dimsA.x * dimsB.y];

    AllocateDeviceMemory(&d_A, &d_B, &d_C, mem_size_A, mem_size_B, mem_size_C);

    CopyHostToDevice(d_A, d_B, d_C, h_A, h_B, mem_size_A, mem_size_B);

    dim3 threads(block_size, block_size);
    dim3 grid((dimsB.x + threads.x - 1) / threads.x, (dimsA.y + threads.y - 1) / threads.y);

    printf("Computing result using CUDA Kernel...\n");

    if (block_size == 16) {
        MatrixMulCUDA<16> <<<grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.y);
    }
    else {
        MatrixMulCUDA<32> <<<grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.y);
    }

    printf("done\n");
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16> <<<grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.y);
        }
        else {
            MatrixMulCUDA<32> <<<grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsA.y, dimsB.y);
        }
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
        static_cast<double>(dimsA.y) *
        static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
        " WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    CopyDeviceToHost(h_C, d_C, mem_size_C);

    // Perform matrix multiplication on the host for verification
    matrixMulHost(h_A, h_B, h_C_CPU, dimsA.x, dimsA.y, dimsB.y);


    //Time to print the matrices
    printMatrices(h_C_CPU, h_C, dimsA.x, dimsB.y);
    std::cout << " " << std::endl;
    printf("Checking computed result for correctness: ");
    bool correct = true;

    // Verify the results
    for (int i = 0; i < dimsA.x * dimsB.y; i++) {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5) {
            std::cerr << "Mismatch at element " << i << ": " << h_C[i] << " != " << h_C_CPU[i] << std::endl;
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result = PASS (GPU result matches with CPU results!)" : "Result = FAIL (Mismatch in results found!)");

    DeallocateDeviceMemory(d_A, d_B, d_C);
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));


    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}

int main(int argc, char** argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");
    printf("You have entered %d arguments:\n", argc);

    int dev = findCudaDevice(argc, (const char**)argv);

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    for (int i = 0; i < argc; i++)
        printf("\nargv[%d]: %s", i, argv[i]);

    if (argc != 5) {
        printf("\nIncorrect number of arguments! Usage: ./TiledMatrixMul -i <rowDimA> <colDimA> <rowDimB>\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "-i") != 0) {
        printf("\nInvalid option! Usage: ./TiledMatrixMul -i <rowDimA> <colDimA> <rowDimB>\n");
        exit(EXIT_FAILURE);
    }

    int rowDimA = atoi(argv[2]);
    int colDimA = atoi(argv[3]);
    int rowDimB = atoi(argv[4]);

    printf("\nrowDimA: %d", rowDimA);
    printf("\ncolDimA: %d", colDimA);
    printf("\nrowDimB: %d", rowDimB);
    
    dimsA.x = rowDimA;
    dimsA.y = colDimA;
    dimsB.x = rowDimB;
    
    checkCudaErrors(cudaProfilerStart());
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(matrix_result);
}
