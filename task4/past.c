#include <stdio.h>  // Including standard, I/O library
#include "lodepng.h"    // Including lodepng library for image processing
#include <stdlib.h> // Including standard library
#include <pthread.h>    // Including Pthread POSIX library

// Declaring global variables
unsigned char* image;
unsigned width, height;
unsigned char* newImage;
int numThreads;

typedef struct {    // Declaring struct for thread data
    int start;
    int end;
} ThreadData;

void* applyBlurFilterThread(void* threadData) {   // Declaring applyBlurFilterThread() function
    ThreadData* data = (ThreadData*)threadData;
    // Applying blur filter to the image
    for(int y = data->start; y < data->end; y++) {  // Looping through the rows of the image assigned to this thread
        for(int x = 0; x < width; x++) {     // Looping through the columns of the image
            double red = 0, green = 0, blue = 0, alpha = 0; // Storing the sum of red, green, blue, and alpha values of surrounding pixels
            int count = 0;  // Variable to store the count of surrounding pixels

            for(int ky = -1; ky <= 1; ky++) {   // Loop through the surrounding pixels
                for(int kx = -1; kx <= 1; kx++) {
                    // Calculating the x and y coordinates of the surrounding pixel
                    int pixelX = x + kx;
                    int pixelY = y + ky;

                    // Checking if the surrounding pixel is outside the image
                    if(pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height) {
                        continue;   // Skipping the surrounding pixel if it is outside the image
                    }

                    // Add the values of the surrounding pixel to the sum
                    red += image[4 * (pixelY * width + pixelX)];
                    green += image[4 * (pixelY * width + pixelX) + 1];
                    blue += image[4 * (pixelY * width + pixelX) + 2];
                    alpha += image[4 * (pixelY * width + pixelX) + 3];
                    count++;
                }
            }

            // Calculate the average of the surrounding pixels
            newImage[4 * (y * width + x)] = (unsigned char)(red / count);
            newImage[4 * (y * width + x) + 1] = (unsigned char)(green / count);
            newImage[4 * (y * width + x) + 2] = (unsigned char)(blue / count);
            newImage[4 * (y * width + x) + 3] = (unsigned char)(alpha / count);
        }
    }

    return NULL;
}

void applyBlurFilter() {    // Declaring applyBlurFilter() function to apply blur filter to the image using threads
    newImage = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));
    ThreadData* threadData = (ThreadData*)malloc(numThreads * sizeof(ThreadData));
    pthread_t* threads = (pthread_t*)malloc(numThreads * sizeof(pthread_t));
    int rowsPerThread = height / numThreads;
    for(int i = 0; i < numThreads; i++) {   // Loop for applying the blur filter to the image
        threadData[i].start = i * rowsPerThread;
        threadData[i].end = (i + 1) * rowsPerThread;
        if(i == numThreads - 1) {
            threadData[i].end = height;
        }
        pthread_create(&threads[i], NULL, applyBlurFilterThread, &threadData[i]);   // Creating threads
    }

    for(int i = 0; i < numThreads; i++) {   // Looping for joining the threads
        pthread_join(threads[i], NULL);
    }

    // Freeing the memory
    free(threadData);
    free(threads);
    free(image);
    image = newImage;   // Assigning the new image to the old image
}

int main() {    // Declaring main() function to read the image and apply the blur filter
    char* fileName = "earth.png";   // Storing the name of the image file to be read and filtered in a variable
    printf("Enter the number of threads you want to use: ");
    scanf("%d", &numThreads);   // Reading the number of threads to be used from the user
    unsigned error = lodepng_decode32_file(&image, &width, &height, fileName);  // Reading the image
    if(error) { // Checking if there is an error in reading the image
        printf("Error reading image: %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    applyBlurFilter();  // Applying the blur filter to the image
    error = lodepng_encode32_file("filtered_image.png", image, width, height);  // Saving the filtered image to a file
    if(error) { // Checking if there is an error in saving the image to a file
        printf("Error saving image: %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    free(image);    // Freeing the memory
    return 0;
}
