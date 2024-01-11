/*
 * Message to the checker of this assignment:
 * Respected sir,
 * This code is the modified version of my assignment of Numerical Methods and Concurrency module Task-4. Every bit of
 * question is same; the only difference is there we did it on C and here we are doing it on CUDA. I have modified my
 * previous assignment code to make it run on CUDA. My previous code that was written in C is also attached in this task
 * directory with name 'past.c'.
 * */
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda.h>

// defining a CUDA kernel function to apply the blur filter to each pixel of the image
__global__ void applyBlurFilter(unsigned char* image, unsigned char* newImage, int width, int height) {
    // I calculate the index of the pixel this thread is responsible for
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // making sure the index is within the image boundaries
    if (idx < width * height) {
        // calculating the starting index of the pixel in the image array
        int pixel = idx * 4;

        // initializing color values and a counter
        double red = 0, green = 0, blue = 0, alpha = 0;
        int count = 0;

        // looping over the 3x3 neighborhood of the pixel
        for(int ky = -1; ky <= 1; ky++) {
            for(int kx = -1; kx <= 1; kx++) {
                // calculating the coordinates of the neighboring pixel
                int pixelX = idx % width + kx;
                int pixelY = idx / width + ky;

                // skipping the neighbor if it's outside the image boundaries
                if(pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height) {
                    continue;
                }

                // calculating the starting index of the neighboring pixel in the image array
                int neighborPixel = (pixelY * width + pixelX) * 4;

                // adding the color values of the neighbor to the sums
                red += image[neighborPixel];
                green += image[neighborPixel + 1];
                blue += image[neighborPixel + 2];
                alpha += image[neighborPixel + 3];

                // incrementing the counter
                count++;
            }
        }

        // calculating the average color values and assign them to the new image
        newImage[pixel] = (unsigned char)(red / count);
        newImage[pixel + 1] = (unsigned char)(green / count);
        newImage[pixel + 2] = (unsigned char)(blue / count);
        newImage[pixel + 3] = (unsigned char)(alpha / count);
    }
}

int main() {
    // pointers for the original and blurred images
    unsigned char* image;
    unsigned char* newImage;

    // pointers for the device copies of the images
    unsigned char* d_image;
    unsigned char* d_newImage;

    // variables for the image dimensions
    unsigned width, height;

    const char* filename = "earth.png";
    const char* newFileName = "blurred_cuda.png";

    // reading the image file into a 1D array
    unsigned error = lodepng_decode32_file(&image, &width, &height, filename);
    if(error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    // calculating the size of the image in bytes
    int imageSize = width * height * 4 * sizeof(unsigned char);

    // allocating memory for the new image on the host
    newImage = (unsigned char*)malloc(imageSize);

    // allocating memory for the images on the device
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_newImage, imageSize);

    // copying the original image from the host to the device
    cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

    // defining the block size and calculate the number of blocks
    int blockSize = 256;
    int numBlocks = (width * height + blockSize - 1) / blockSize;

    // CUDA kernel
    applyBlurFilter<<<numBlocks, blockSize>>>(d_image, d_newImage, width, height);

    // copying the blurred image from the device to the host
    cudaMemcpy(newImage, d_newImage, imageSize, cudaMemcpyDeviceToHost);

    // writing the blurred image to a file
    error = lodepng_encode32_file(newFileName, newImage, width, height);
    if(error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    // free the allocated memory on the host and the device
    free(image);
    free(newImage);
    cudaFree(d_image);
    cudaFree(d_newImage);

    return 0;
}