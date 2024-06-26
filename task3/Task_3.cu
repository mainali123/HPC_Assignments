#include <stdio.h>
#include <stdlib.h>

// Compile with: nvcc password_cracking_using_cuda.cu -o password_cracking_using_cuda
// Execute with: ./password_cracking_using_cuda
__device__ char *CudaCrypt(char *rawPassword);

__global__ void CudaCryptWrapper(char *rawPassword, char *result) {
    char *temp = CudaCrypt(rawPassword);
    for (int i = 0; i < 11; i++) {
        result[i] = temp[i];
    }
}

__device__ char *copy_strings(char *dest, const char *src) {
    int i = 0;
    do {
        dest[i] = src[i];
    } while (src[i++] != 0);
    return dest;
}

__device__ int compare_strings(const char *str_a, const char *str_b, unsigned len = 256) {
    int match = 0;
    unsigned i = 0;
    unsigned done = 0;
    while ((i < len) && (match == 0) && !done) {
        if ((str_a[i] == 0) || (str_b[i] == 0)) {
            done = 1;
        } else if (str_a[i] != str_b[i]) {
            match = i + 1;
            if (((int) str_a[i] - (int) str_b[i]) < 0) {
                match = 0 - (i + 1);
            }
        }
        i++;
    }
    return match;
}

// Here to
__device__ char *CudaCrypt(char *rawPassword) {
    char *newPassword = (char *) malloc(sizeof(char) * 11);

    newPassword[0] = rawPassword[0] + 2;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 3;
    newPassword[4] = rawPassword[1] - 3;
    newPassword[5] = rawPassword[1] - 1;
    newPassword[6] = rawPassword[2] + 2;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 4;
    newPassword[10] = '\0';

    for (int i = 0; i < 10; i++) {
        if (i >= 0 && i < 6) { //checking all lower case letter limits
            if (newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if (newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else {
            if (newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if (newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
    return newPassword;
}

__global__ void crack(char *alphabet, char *numbers, char *encPassword) {
    char genRawPass[4];

    genRawPass[0] = alphabet[blockIdx.x];
    genRawPass[1] = alphabet[blockIdx.y];

    genRawPass[2] = numbers[threadIdx.x];
    genRawPass[3] = numbers[threadIdx.y];

    // End Here

    // compare encrypted passwords and then after the match, assign the device variable with raw password
    if (compare_strings(CudaCrypt(genRawPass), encPassword) == 0) { // change 1
        copy_strings(encPassword, genRawPass);
    }
}

int main(int argc, char **argv) {
    char cpuAlphabet[26] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                            's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    char cpuNumbers[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    // encrypted password for `pr75` as input
//    char inputEncPass[11] = "rnquoq9591";
    char inputRawPass[5] = "dm19";
    char *inputEncPass;
    cudaMallocManaged(&inputEncPass, sizeof(char) * 11);
    CudaCryptWrapper<<<1, 1>>>(inputRawPass, inputEncPass);
    cudaDeviceSynchronize();

    char *decryptedPass;
    decryptedPass = (char *) malloc(sizeof(char) * 26);

    // allocate memory for device variables (letters) and copy them from host to device
    char *gpuAlphabet;
    cudaMalloc((void **) &gpuAlphabet, sizeof(char) * 26);
    cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

    // allocate memory for the numbers and copy them from host to device
    char *gpuNumbers;
    cudaMalloc((void **) &gpuNumbers, sizeof(char) * 10);
    cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 10, cudaMemcpyHostToDevice);

    // gpu device memory allocation for encrypted input password
    char *gpuPassword;  // changes 2
    cudaMalloc((void **) &gpuPassword, sizeof(char) * 11);
    cudaMemcpy(gpuPassword, inputEncPass, sizeof(char) * 11, cudaMemcpyHostToDevice);

    //Program works with varying numbers of blocks and threads (blocks <= 26, threads <= 26)
    crack<<< dim3(26, 26, 1), dim3(10, 10, 1) >>>(gpuAlphabet, gpuNumbers, gpuPassword);
    cudaDeviceSynchronize();  // cudaDeviceSynchronize() and cudaThreadSynchronize() works the same

    // copy the memory back to host from device
    cudaMemcpy(decryptedPass, gpuPassword, sizeof(char) * 26, cudaMemcpyDeviceToHost);


    printf("\nEncrypted Password : %s,\nRaw Password :  %s\n\n", inputEncPass, decryptedPass);
    //free alocated memory
    free(decryptedPass);
    free(inputEncPass);
    cudaFree(gpuAlphabet);
    cudaFree(gpuNumbers);
    cudaFree(gpuPassword);
    return 0;
}













