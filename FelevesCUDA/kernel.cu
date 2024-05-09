
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <algorithm>
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

#define N 32
#define K 32
dim3 BlockSize(N, K);


struct Pixel
{
	unsigned char r, g, b, a;
};
__device__ int myMin(int a, int b) {
    return a < b ? a : b;
}

__device__ int myMax(int a, int b) {
    return a > b ? a : b;
}

//void ConvertImageToGrayCPU(unsigned char* imageRGBA, int height, int width)
//{
//	for (int i = 0;i < height;i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			Pixel* ptrPixel = (Pixel*)&imageRGBA[i * width * 4 + 4 * j];
//			unsigned char  PixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
//			ptrPixel->b = PixelValue;
//			ptrPixel->g = PixelValue;
//			ptrPixel->r = PixelValue;
//			ptrPixel->a = 255;
//		}
//	}
//}


__global__ void ConvertImageToGrayGPU(unsigned char* input, unsigned char* output, int width, int height) {
    extern __shared__ unsigned char sharedInput[];

    // A globális x és y 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // blokkon belüli idx
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Shared memory-ba már a kiszámított pixelek mennek.

    if (x < width && y < height) {
        int color_idx = (y * width + x) * 4;
        sharedInput[tid] = (unsigned char)(
            0.2126f * input[color_idx] +
            0.7152f * input[color_idx + 1] +
            0.0722f * input[color_idx + 2]
            );
    }
    __syncthreads();

    // Visszaírjuk a globális memóriába
    if (x < width && y < height) {
        output[y * width + x] = sharedInput[tid];
    }
}



__global__ void sobelFilter(unsigned char* input, unsigned char* output, int width, int height) {
    extern __shared__ unsigned char sharedInput[];

    int halo = 1;
    int x = threadIdx.x + blockIdx.x * blockDim.x - halo;
    int y = threadIdx.y + blockIdx.y * blockDim.y - halo;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    
    if (x >= -1 && x < width + 1 && y >= -1 && y < height + 1) {
        int boundedX = myMax(0, myMin(x, width - 1));
        int boundedY = myMax(0, myMin(y, height - 1));
        sharedInput[tid] = input[boundedY * width + boundedX];
    }
    __syncthreads();

    
    if (threadIdx.x >= halo && threadIdx.x < blockDim.x - halo &&
        threadIdx.y >= halo && threadIdx.y < blockDim.y - halo &&
        x >= 0 && x < width && y >= 0 && y < height) {
        int sIdx = threadIdx.y * blockDim.x + threadIdx.x;
        float gx = -1 * sharedInput[sIdx - blockDim.x - 1] - 2 * sharedInput[sIdx - blockDim.x] - 1 * sharedInput[sIdx - blockDim.x + 1]
            + 1 * sharedInput[sIdx + blockDim.x - 1] + 2 * sharedInput[sIdx + blockDim.x] + 1 * sharedInput[sIdx + blockDim.x + 1];
        float gy = -1 * sharedInput[sIdx - blockDim.x - 1] - 2 * sharedInput[sIdx - 1] - 1 * sharedInput[sIdx + blockDim.x - 1]
            + 1 * sharedInput[sIdx - blockDim.x + 1] + 2 * sharedInput[sIdx + 1] + 1 * sharedInput[sIdx + blockDim.x + 1];

      
        float mag = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = (unsigned char)mag;
    }
}







int main(int argc, char** argv) {
	
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <ImageFile>" << std::endl;
        return -1;
    }

    int width, height, componentCount;
    std::cout << "Loading png..." << std::endl;
    unsigned char* data = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!data) {
        std::cout << "File failed to open \"" << argv[1] << "\"" << std::endl;
        return -1;
    }

    unsigned char* InputImage, * GrayImage, * OutputImage;
    cudaMalloc((void**)&InputImage, width * height * 4);
    cudaMalloc((void**)&GrayImage, width * height);
    cudaMalloc((void**)&OutputImage, width * height);
    cudaCheckError();

    cudaMemcpy(InputImage, data, width * height * 4, cudaMemcpyHostToDevice);
    cudaCheckError();

    //GrayScale
    dim3 numBlocks((width + BlockSize.x - 1) / BlockSize.x, (height + BlockSize.y - 1) / BlockSize.y);

    int sharedMemSize = BlockSize.x * BlockSize.y * sizeof(unsigned char);
    ConvertImageToGrayGPU << <numBlocks, BlockSize, sharedMemSize >> > (InputImage, GrayImage, width, height);

    cudaCheckError();

    
    cudaMemcpy(data, GrayImage, width * height, cudaMemcpyDeviceToHost);
    cudaCheckError();

   //Grayscale kép kiirása
    std::string fileNameOut = std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of('.')) + "_gray.png";
    std::cout << "Writing output Gray png..." << std::endl;
    stbi_write_png(fileNameOut.c_str(), width, height, 1, data, width);
    std::cout << "Done!" << std::endl;

    // SobelFilter

    size_t sharedMemSizeSobel = (BlockSize.x + 2) * (BlockSize.y + 2); 
    sobelFilter << <numBlocks, BlockSize, sharedMemSizeSobel * sizeof(unsigned char) >> > (GrayImage, OutputImage, width, height);
    cudaCheckError();

    // SobelFilter kiirása
    cudaMemcpy(data, OutputImage, width * height, cudaMemcpyDeviceToHost);
    cudaCheckError();
     fileNameOut = std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of('.')) + "_Sobel.png";
    std::cout << "Writing output Sobel png..." << std::endl;
    stbi_write_png(fileNameOut.c_str(), width, height, 1, data, width);
    std::cout << "Done!" << std::endl;
    // Memóriát felszabadítjuk.
    cudaFree(InputImage);
    cudaFree(GrayImage);
    cudaFree(OutputImage);
    cudaCheckError();

    stbi_image_free(data);
	
} 