#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>

__host__ __device__ char* CudaCrypt(char* rawPassword) {
    char* newPassword = (char*)malloc(sizeof(char) * 11);

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
        if (i >= 0 && i < 6) {
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

__global__ void crack(char* alphabet, char* numbers, char* result) {
    char genRawPass[5];

    genRawPass[0] = alphabet[blockIdx.x];
    genRawPass[1] = alphabet[blockIdx.y];
    genRawPass[2] = numbers[threadIdx.x];
    genRawPass[3] = numbers[threadIdx.y];
    genRawPass[4] = '\0';

    char* encryptedPassword = CudaCrypt(genRawPass);

    int match = 1;
    for (int i = 0; i < 5; i++) { // Compare only the first 5 characters
        if (encryptedPassword[i] != result[i]) {
            match = 0;
            break;
        }
    }

    if (match) {
//        printf("Decrypted Password: %s\n", genRawPass);
    }

    free(encryptedPassword);
}

__global__ void crack(char* alphabet, char* numbers, char* result, char* newPasswords) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    char genRawPass[5];

    genRawPass[0] = alphabet[index / 26];
    genRawPass[1] = alphabet[index % 26];
    genRawPass[2] = numbers[index / 10];
    genRawPass[3] = numbers[index % 10];
    genRawPass[4] = '\0';

    char* encryptedPassword = CudaCrypt(genRawPass, newPasswords + index * 11);

    int match = 1;
    for (int i = 0; i < 5; i++) {
        if (encryptedPassword[i] != result[i]) {
            match = 0;
            break;
        }
    }

    if (match) {
        // printf("Decrypted Password: %s\n", genRawPass);
    }
}

int main(int argc, char** argv) {
    char cpuAlphabet[26] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    char cpuNumbers[10] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    char* gpuAlphabet;
    cudaMalloc((void**)&gpuAlphabet, sizeof(char) * 26);
    cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

    char* gpuNumbers;
    cudaMalloc((void**)&gpuNumbers, sizeof(char) * 10);
    cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 10, cudaMemcpyHostToDevice);

    FILE* file = fopen("passwords.csv", "r");
    if (file == NULL) {
        printf("Could not open file passwords.csv\n");
        return 1;
    }

    char line[1024];
    char decryptedPassword[5]; // Allocate outside the loop
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0';

        char* gpuEncryptedPassword;
        cudaMalloc((void**)&gpuEncryptedPassword, sizeof(char) * 11);
        cudaMemcpy(gpuEncryptedPassword, line, sizeof(char) * 11, cudaMemcpyHostToDevice);

        crack<<<dim3(26, 26, 1), dim3(10, 10, 1)>>>(gpuAlphabet, gpuNumbers, gpuEncryptedPassword);

        cudaDeviceSynchronize();

        cudaMemcpy(decryptedPassword, gpuEncryptedPassword, sizeof(char) * 11, cudaMemcpyDeviceToHost); // Fix size to 11

        cudaFree(gpuEncryptedPassword);
    }
    fclose(file);

    cudaFree(gpuAlphabet);
    cudaFree(gpuNumbers);

    return 0;
}
