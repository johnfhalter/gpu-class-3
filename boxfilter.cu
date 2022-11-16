#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <iomanip>
#include <iostream>

#define BLOCK_WIDTH 64
#define BLOCK_HEIGHT 16

const int kernelWidth = 5;
__constant__ float d_filter[kernelWidth * kernelWidth];
__constant__ int const d_kernelWidth = kernelWidth;
__constant__ int const d_halfKernelWidth = (kernelWidth - 1) / 2;
__constant__ int d_size[2];

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void makeLaplacianFilter(float** h_filter);
void makeHorizontalLineFilter(float** h_filter);
void makeSharpnessFilter(float** h_filter, int type);
void makeBlurFilter(float** h_filter, int type);
void makeBlur9_9(float** h_filter);

template <typename T>
void check(T err, const char* const func, const char* const file,
	const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
	return fmax(a, fmin(b, x));
}

__global__ void box_filter_shared_mem(const unsigned char* const inputChannel,
	unsigned char* const outputChannel)
{
	const int numRows = d_size[0];
	const int numCols = d_size[1];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	__shared__ unsigned char ds_inputChannel[BLOCK_HEIGHT + d_kernelWidth][BLOCK_WIDTH + d_kernelWidth];

	const unsigned int tx = threadIdx.x + d_halfKernelWidth;
	const unsigned int ty = threadIdx.y + d_halfKernelWidth;

	ds_inputChannel[ty][tx] = inputChannel[thread_1D_pos];

	// bottom border
	if (threadIdx.y == BLOCK_HEIGHT - 1)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			if (thread_2D_pos.y + i < numRows)
				ds_inputChannel[ty + i][tx] = inputChannel[(thread_2D_pos.y + i) * numCols + (thread_2D_pos.x)];
	}

	// top border
	if (threadIdx.y == 0)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			if (thread_2D_pos.y - i >= 0)
				ds_inputChannel[ty - i][tx] = inputChannel[(thread_2D_pos.y - i) * numCols + (thread_2D_pos.x)];
	}

	// right border
	if (threadIdx.x == BLOCK_WIDTH - 1)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			if (thread_2D_pos.x + i < numCols)
				ds_inputChannel[ty][tx + i] = inputChannel[(thread_2D_pos.y) * numCols + (thread_2D_pos.x + i)];
	}

	// left border
	if (threadIdx.x == 0)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			if (thread_2D_pos.x - i >= 0)
				ds_inputChannel[ty][tx - i] = inputChannel[(thread_2D_pos.y) * numCols + (thread_2D_pos.x - i)];
	}

	// bottom right corner
	if (threadIdx.y == BLOCK_HEIGHT - 1 && threadIdx.x == BLOCK_WIDTH - 1)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			for (int j = 1; j <= d_halfKernelWidth; ++j)
				if (thread_2D_pos.y + i < numRows && thread_2D_pos.x + j < numCols)
					ds_inputChannel[ty + i][tx + j] = inputChannel[(thread_2D_pos.y + i) * numCols + (thread_2D_pos.x + j)];
	}

	// top right corner
	if (threadIdx.y == 0 && threadIdx.x == BLOCK_WIDTH - 1)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			for (int j = 1; j <= d_halfKernelWidth; ++j)
				if (thread_2D_pos.y - i >= 0 && thread_2D_pos.x + j < numCols)
					ds_inputChannel[ty - i][tx + j] = inputChannel[(thread_2D_pos.y - i) * numCols + (thread_2D_pos.x + j)];
	}

	// bottom left corner
	if (threadIdx.y == BLOCK_HEIGHT - 1 && threadIdx.x == 0)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			for (int j = 1; j <= d_halfKernelWidth; ++j)
				if (thread_2D_pos.y + i < numRows && thread_2D_pos.x - j >= 0)
					ds_inputChannel[ty + i][tx - j] = inputChannel[(thread_2D_pos.y + i) * numCols + (thread_2D_pos.x - j)];
	}

	// top left corner
	if (threadIdx.y == 0 && threadIdx.x == 0)
	{
		for (int i = 1; i <= d_halfKernelWidth; ++i)
			for (int j = 1; j <= d_halfKernelWidth; ++j)
				if (thread_2D_pos.y - i >= 0 && thread_2D_pos.x - j >= 0)
					ds_inputChannel[ty - i][tx - j] = inputChannel[(thread_2D_pos.y - i) * numCols + (thread_2D_pos.x - j)];
	}

	__syncthreads();

	float convolutionSum = 0;
	for (int i = -d_halfKernelWidth; i <= d_halfKernelWidth; ++i)
	{
		for (int j = -d_halfKernelWidth; j <= d_halfKernelWidth; ++j)
		{
			if ((thread_2D_pos.x + j) >= numCols || (thread_2D_pos.x + j) < 0 || (thread_2D_pos.y + i) >= numRows || (thread_2D_pos.y + i) < 0)
				convolutionSum += 0;
			else
				convolutionSum += (float)ds_inputChannel[ty + i][tx + j] * d_filter[(i + d_halfKernelWidth) * d_kernelWidth + (j + d_halfKernelWidth)];
		}
	}

	convolutionSum = clamp(convolutionSum, 0, 255);

	outputChannel[thread_1D_pos] = convolutionSum;
}

__global__ void box_filter(const unsigned char* const inputChannel,
	unsigned char* const outputChannel)
{
	const int numRows = d_size[0];
	const int numCols = d_size[1];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	float convolutionSum = 0;
	for (int i = -d_halfKernelWidth; i <= d_halfKernelWidth; ++i)
	{
		for (int j = -d_halfKernelWidth; j <= d_halfKernelWidth; ++j)
		{
			if ((thread_2D_pos.x + j) >= numCols || (thread_2D_pos.x + j) < 0 || (thread_2D_pos.y + i) >= numRows || (thread_2D_pos.y + i) < 0)
				convolutionSum += 0;
			else
				convolutionSum += (float)inputChannel[(thread_2D_pos.y + i) * numCols + (thread_2D_pos.x + j)] * d_filter[(i + d_halfKernelWidth) * d_kernelWidth + (j + d_halfKernelWidth)];
		}
	}

	convolutionSum = clamp(convolutionSum, 0, 255);

	outputChannel[thread_1D_pos] = convolutionSum;
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel) {
	// TODO:
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la
	// imagen
	const int numRows = d_size[0];
	const int numCols = d_size[1];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	const const uchar4 inputPixel = inputImageRGBA[thread_1D_pos];

	redChannel[thread_1D_pos] = inputPixel.x;
	greenChannel[thread_1D_pos] = inputPixel.y;
	blueChannel[thread_1D_pos] = inputPixel.z;
}

// This kernel takes in three color channels and recombines them
// into one image. The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA) {
	const int numRows = d_size[0];
	const int numCols = d_size[1];

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// make sure we don't try and access memory outside the image
	// by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	// Alpha should be 255 for no transparency
	const uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char* d_red, * d_green, * d_blue;
//float* d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage,
	const size_t numColsImage,
	const float* const h_filter,
	const size_t filterWidth) {
	// allocate memory for the three different channels
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

	// TODO:
	// Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
	// Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)

	//Con memoria de constantes
	const int h_size[2] = { numRowsImage,numColsImage };
	checkCudaErrors(cudaMemcpyToSymbol(d_size, h_size, sizeof(int) * 2, 0, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(d_filter, h_filter, sizeof(float) * kernelWidth * kernelWidth, 0, cudaMemcpyHostToDevice));
}

void create_filter(float** h_filter, int* filterWidth) {
	// create and fill the filter we will convolve with
	*h_filter = new float[kernelWidth * kernelWidth];

	// TODO: crear los filtros segun necesidad
	// NOTA: cuidado al establecer el tamaño del filtro a utilizar
	//3*3
	//makeSharpnessFilter(h_filter, 2);
	//makeBlurFilter( h_filter, 0);
	//makeHorizontalLineFilter(h_filter);

	//5*5
	makeLaplacianFilter(h_filter);

	//9*9
	//makeBlur9_9(h_filter);

	*filterWidth = kernelWidth;
}

void convolution(const uchar4* const h_inputImageRGBA,
	uchar4* const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows,
	const size_t numCols, unsigned char* d_redFiltered,
	unsigned char* d_greenFiltered, unsigned char* d_blueFiltered,
	const int filterWidth) {
	// TODO: Calcular tamaños de bloque
	const dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	const dim3 gridSize(ceil((float)numCols / BLOCK_WIDTH), ceil((float)numRows / BLOCK_HEIGHT), 1);

	// TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
	separateChannels << < gridSize, blockSize >> > (d_inputImageRGBA, d_red, d_green, d_blue);

	// TODO: Ejecutar convolución. Una por canal
	box_filter << < gridSize, blockSize >> > (d_red, d_redFiltered);
	box_filter << < gridSize, blockSize >> > (d_green, d_greenFiltered);
	box_filter << < gridSize, blockSize >> > (d_blue, d_blueFiltered);

	// Recombining the results.
	recombineChannels << <gridSize, blockSize >> > (d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

// Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_filter));
}

void makeLaplacianFilter(float** h_filter)
{
	// Laplaciano 5x5
	(*h_filter)[0] = 0;
	(*h_filter)[1] = 0;
	(*h_filter)[2] = -1.;
	(*h_filter)[3] = 0;
	(*h_filter)[4] = 0;
	(*h_filter)[5] = 1.;
	(*h_filter)[6] = -1.;
	(*h_filter)[7] = -2.;
	(*h_filter)[8] = -1.;
	(*h_filter)[9] = 0;
	(*h_filter)[10] = -1.;
	(*h_filter)[11] = -2.;
	(*h_filter)[12] = 17.;
	(*h_filter)[13] = -2.;
	(*h_filter)[14] = -1.;
	(*h_filter)[15] = 1.;
	(*h_filter)[16] = -1.;
	(*h_filter)[17] = -2.;
	(*h_filter)[18] = -1.;
	(*h_filter)[19] = 0;
	(*h_filter)[20] = 1.;
	(*h_filter)[21] = 0;
	(*h_filter)[22] = -1.;
	(*h_filter)[23] = 0;
	(*h_filter)[24] = 0;
}

void makeHorizontalLineFilter(float** h_filter)
{
	//Detección linea horizontal
	(*h_filter)[0] = -1;
	(*h_filter)[1] = -1;
	(*h_filter)[2] = -1;
	(*h_filter)[3] = 2;
	(*h_filter)[4] = 2;
	(*h_filter)[5] = 2.;
	(*h_filter)[6] = -1.;
	(*h_filter)[7] = -1.;
	(*h_filter)[8] = -1.;
}

void makeSharpnessFilter(float** h_filter, int type)
{
	//Filtro de nitidez de paso alto 3*3
	switch (type)
	{
	default:
	case 0://Aumentar nitidez
		(*h_filter)[0] = 0;
		(*h_filter)[1] = -0.25;
		(*h_filter)[2] = 0;
		(*h_filter)[3] = -0.25;
		(*h_filter)[4] = 2;
		(*h_filter)[5] = -0.25;
		(*h_filter)[6] = 0;
		(*h_filter)[7] = -0.25;
		(*h_filter)[8] = 0;
		break;
	case 1://Aumentar nitidez II
		(*h_filter)[0] = -0.25;
		(*h_filter)[1] = -0.25;
		(*h_filter)[2] = -0.25;
		(*h_filter)[3] = -0.25;
		(*h_filter)[4] = 3;
		(*h_filter)[5] = -0.25;
		(*h_filter)[6] = -0.25;
		(*h_filter)[7] = -0.25;
		(*h_filter)[8] = -0.25;
		break;
	case 2: //Nitidez 3*3 Filtro paso alto
		(*h_filter)[0] = -1;
		(*h_filter)[1] = -1;
		(*h_filter)[2] = -1;
		(*h_filter)[3] = -1;
		(*h_filter)[4] = 9;
		(*h_filter)[5] = -1;
		(*h_filter)[6] = -1;
		(*h_filter)[7] = -1;
		(*h_filter)[8] = -1;
		break;
	}
}

void makeBlurFilter(float** h_filter, int type)
{
	//filtro de suavizado
	switch (type)
	{
	default:
	case 0://Media aritmetica suave
		(*h_filter)[0] = 0.111;
		(*h_filter)[1] = 0.111;
		(*h_filter)[2] = 0.111;
		(*h_filter)[3] = 0.111;
		(*h_filter)[4] = 0.111;
		(*h_filter)[5] = 0.111;
		(*h_filter)[6] = 0.111;
		(*h_filter)[7] = 0.111;
		(*h_filter)[8] = 0.111;
		break;
	case 1://Suavizado 3*3
		(*h_filter)[0] = 1;
		(*h_filter)[1] = 2;
		(*h_filter)[2] = 1;
		(*h_filter)[3] = 2;
		(*h_filter)[4] = 4;
		(*h_filter)[5] = 2;
		(*h_filter)[6] = 1;
		(*h_filter)[7] = 2;
		(*h_filter)[8] = 1;
		break;
	}
}

void makeBlur9_9(float** h_filter)
{
	//Filtro gaussiano: blur

	const float KernelSigma = 2.;

	float filterSum = 0.f; //for normalization

	for (int r = -kernelWidth / 2; r <= kernelWidth / 2; ++r)
	{
		for (int c = -kernelWidth / 2; c <= kernelWidth / 2; ++c)
		{
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
			(*h_filter)[(r + kernelWidth / 2) * kernelWidth + c + kernelWidth / 2] = filterValue; filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -kernelWidth / 2; r <= kernelWidth / 2; ++r)
	{
		for (int c = -kernelWidth / 2; c <= kernelWidth / 2; ++c)
		{
			(*h_filter)[(r + kernelWidth / 2) * kernelWidth + c + kernelWidth / 2] *= normalizationFactor;
		}
	}
}
