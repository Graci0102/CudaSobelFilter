#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

//#define GRIDVAL 20.0

//__constant__ int sobel_x[3][3] = {
//	{-1, 0, 1},
//	{-2, 0, 2},
//	{-1, 0, 1}
//};
//__constant__ int sobel_y[3][3] = {
//	{1, 2, 1},
//	{0, 0, 0},
//	{-1, -2, -1}
//};

__global__ void cudaSobelFilter(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float pixelX = 0;
	float pixelY = 0;
	if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
	{
		pixelX = (-1 * inputImage[(y - 1) * width + (x - 1)]) + (-2 * inputImage[y * width + (x - 1)]) + (-1 * inputImage[(y + 1) * width + (x - 1)]) +
				 (inputImage[(y - 1) * width + (x + 1)]) + (2 * inputImage[y * width + (x + 1)]) + (inputImage[(y + 1) * width + (x + 1)]);
		pixelY = (inputImage[(y - 1) * width + (x - 1)]) + (2 * inputImage[(y - 1) * width + x]) + (inputImage[(y - 1) * width + (x + 1)]) +
				 (-1 * inputImage[(y + 1) * width + (x - 1)]) + (-2 * inputImage[(y + 1) * width + x]) + (-1 * inputImage[(y + 1) * width + (x + 1)]);
	}
	outputImage[y * width + x] = static_cast<unsigned char>(sqrtf(pixelX * pixelX + pixelY * pixelY));
}

__host__ double FindGridVal(int width) {
	for (int i = 20; i > 10; i--)
	{
		if (width % i == 0)
		{
			return i;
		}
	}
}

__host__ double FindBestDivider(int size,int aspect) {
	int i = 1;
	while (size / i > aspect)
	{
		i++;
	}
	return i;
}

int main() {

	cv::String imageName = "PANO_1.jpg";

	double GRIDVAL = 0;

	// Load the image
	cv::Mat image = cv::imread(imageName, cv::IMREAD_GRAYSCALE);
	cv::Mat OGimage = cv::imread(imageName); //just for OG colors

	int ogWidth = image.cols;
	int ogHeight = image.rows;

	int width = 0, height = 0;

	if (ogWidth > 1920 && ogHeight > 1080)
	{
		if (ogWidth > ogHeight)
		{
			int divider = FindBestDivider(ogWidth,1920);
			width = ogWidth / divider;
			double aspect_ratio = static_cast<double>(width) / ogWidth;
			height = static_cast<int>(ogHeight * aspect_ratio);
		}
		else {
			int divider = FindBestDivider(ogHeight,1080);
			height = ogHeight / divider;
			double ascpectRatio = static_cast<double>(height) / ogHeight;
			width = static_cast<int>(ogWidth * ascpectRatio);
		}
	}

	GRIDVAL = FindGridVal(width);

	cv::Mat resizedImage;
	cv::Mat OGResizedimage;

	cv::resize(image, resizedImage, cv::Size(width, height),0,0, cv::INTER_AREA);
	cv::resize(OGimage, OGResizedimage, cv::Size(width, height),0,0, cv::INTER_AREA);

	size_t imageSize = width * height;

	unsigned char* d_inputImage, * d_outputImage;

	//allocate memory

	cudaMalloc(&d_inputImage, imageSize);
	cudaMalloc(&d_outputImage, imageSize);

	// Copy the image data to the device
	cudaMemcpy(d_inputImage, resizedImage.data, imageSize, cudaMemcpyHostToDevice);

	// Define the blocks and threads
	dim3 threadPerBlock(GRIDVAL, GRIDVAL,1);
	dim3 numBlocks(ceil(width / GRIDVAL), ceil(height / GRIDVAL), 1);

	cudaSobelFilter<<<numBlocks,threadPerBlock>>>(d_inputImage, d_outputImage, width, height);

	// Copy the result back to the host
	cv::Mat outputImage(height, width, CV_8UC1);
	cudaMemcpy(outputImage.data, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

	// Display the result
	cv::imshow("Sobel Filter Output", outputImage);
	cv::imshow("Original GreyScale Image", resizedImage);
	cv::imshow("Original Image", OGResizedimage);
	cv::waitKeyEx(0);

	// Free device memory
	cudaFree(d_inputImage);
	cudaFree(d_outputImage);

	return 0;
}
