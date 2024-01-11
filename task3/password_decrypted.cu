#include <stdio.h>
#include <stdlib.h>

// defining a function to encrypt a raw password
__host__ __device__ char* CudaCrypt(char* rawPassword){
    char * newPassword = (char *) malloc(sizeof(char) * 11);

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

    // adjusting the characters of the new password to make sure they are valid
    for(int i = 0; i < 10; i++){
        if(i >= 0 && i < 6){
            if(newPassword[i] > 122){
                newPassword[i] = (newPassword[i] - 122) + 97;
            }else if(newPassword[i] < 97){
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        }else{
            if(newPassword[i] > 57){
                newPassword[i] = (newPassword[i] - 57) + 48;
            }else if(newPassword[i] < 48){
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
    return newPassword;
}

// defining a CUDA kernel function to crack the encrypted password
__global__ void crack(char * alphabet, char * numbers, char * result){

    char genRawPass[5];

    genRawPass[0] = alphabet[blockIdx.x];
    genRawPass[1] = alphabet[blockIdx.y];
    genRawPass[2] = numbers[threadIdx.x];
    genRawPass[3] = numbers[threadIdx.y];
    genRawPass[4] = '\0';

    // encrypting the generated raw password
    char* encryptedPassword = CudaCrypt(genRawPass);

    // comparing the generated password with the original encrypted password
    int match = 1;
    for (int i = 0; i < 11; i++) {
        if (encryptedPassword[i] != result[i]) {
            match = 0;
            break;
        }
    }

    // If the passwords match, printing the decrypted password
    if (match) {
        printf("Decrypted Password: %s\n", genRawPass);
    }

    // free the allocated memory for the encrypted password
    free(encryptedPassword);
}

int main(int argc, char ** argv){

    // arrays for the alphabet and numbers
    char cpuAlphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
    char cpuNumbers[10] = {'0','1','2','3','4','5','6','7','8','9'};

    // allocating memory on the GPU for the alphabet and numbers and copy the data from the CPU to the GPU
    char * gpuAlphabet;
    cudaMalloc( (void**) &gpuAlphabet, sizeof(char) * 26);
    cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

    char * gpuNumbers;
    cudaMalloc( (void**) &gpuNumbers, sizeof(char) * 10);
    cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 10, cudaMemcpyHostToDevice);

    // declaring a raw password and encrypt it
    char rawPassword[] = "dm19";
    char * encryptedPassword = CudaCrypt(rawPassword);
    printf("Encrypted Password: %s\n", encryptedPassword);

    // allocating memory on the GPU for the encrypted password and copy the data from the CPU to the GPU
    char * gpuEncryptedPassword;
    cudaMalloc( (void**) &gpuEncryptedPassword, sizeof(char) * 11);
    cudaMemcpy(gpuEncryptedPassword, encryptedPassword, sizeof(char) * 11, cudaMemcpyHostToDevice);

    crack<<< dim3(26,26,1), dim3(10,10,1) >>>( gpuAlphabet, gpuNumbers, gpuEncryptedPassword );

    // free the allocated memory on the GPU
    cudaFree(gpuAlphabet);
    cudaFree(gpuNumbers);
    cudaFree(gpuEncryptedPassword);

    // free the allocated memory on the CPU
    free(encryptedPassword);

    return 0;
}