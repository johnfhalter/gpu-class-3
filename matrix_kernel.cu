#ifndef _MATRIX_KERNEL_H_
#define _MATRIX_KERNEL_H_
__global__ void matAdd(float* C, const float* A, const float* B, int h, int w)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < h * w)
        C[i] = A[i] + B[i];
}

__global__ void matSub(float* C, const float* A, const float* B, int h, int w)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < h * w)
        C[i] = A[i] - B[i];
}

__global__ void matTranspose(float* B, const float* A, int h, int w)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < h * w)
	
        B[(i % w) * h + (i / w)] = A[i]; 
}

__global__ void matMult(float* C,const float* A, const float* B,  int Aw, int Bw)
{
    // Indexes
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int BLOCK_SIZE=16;	
    
    // Shared memory for A and B
    __shared__ float s_A[16][16];
    __shared__ float s_B[16][16];

    int A_start = Aw * BLOCK_SIZE * by;
    int A_stop   = A_start + Aw - 1;
    int A_step  = BLOCK_SIZE;

    int B_start = BLOCK_SIZE * bx;
    int B_step  = BLOCK_SIZE * Bw;

    float s_C = 0;

    for (int a = A_start, b = B_start;a <= A_stop;a += A_step, b += B_step) 
	{
        s_A[ty][tx] = A[a + Aw * ty + tx];
        s_B[tx][ty] = B[b + Bw * tx + ty];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
           s_C += s_A[ty][k] * s_B[k][tx];

        __syncthreads();
    }

    int c = Bw * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + Bw * ty + tx] = s_C;
	
}

__global__ void matCopy(float* B, const float* A, int h, int w)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < h * w)
        B[i] = A[i];
}

#endif // _MATRIX_KERNEL_H_

