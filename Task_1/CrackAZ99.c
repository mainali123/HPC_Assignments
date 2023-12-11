#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>

// Global variable to keep track of the number of attempts
int count = 0;

// Function to copy a substring from the source string to the destination string
void substr(char *dest, char *src, int start, int length) {
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

// Function to crack the password without multithreading
void crack(char *salt_and_encrypted) {
    count = 0;
    int x, y, z;
    char salt[7];
    char plain[7];
    char *enc;

    substr(salt, salt_and_encrypted, 0, 6);

    // Loop through all possible combinations of two uppercase letters and two digits
    for (x = 'A'; x <= 'Z'; x++) {
        for (y = 'A'; y <= 'Z'; y++) {
            for (z = 0; z <= 99; z++) {
                sprintf(plain, "%c%c%02d", x, y, z);
                enc = (char *) crypt(plain, salt);
                count++;
                if (strcmp(salt_and_encrypted, enc) == 0) {
                    printf("#%-8d%s %s \t without Multithreading \n", count, plain, enc);
                    return;
                }
            }
        }
    }
}

// Function to crack the password with multithreading using OpenMP
void multithreadingCracking(char *saltAndEncrypted) {
    count = 0;
    char salt[7];
    char plain[7];
    char *enc;

    strncpy(salt, saltAndEncrypted, 6);
    salt[6] = '\0';

    // Loop through all possible combinations of two uppercase letters and two digits
    // The loop is parallelized with OpenMP
#pragma omp parallel for collapse(3) shared(count) private(plain, enc)
    for (int i = 'A'; i <= 'Z'; ++i) {
        for (int j = 'A'; j <= 'Z'; ++j) {
            for (int k = 0; k <= 99; ++k) {
                snprintf(plain, sizeof(plain), "%c%c%02d", i, j, k);
                enc = crypt(plain, salt);
                count++;
                if (strcmp(saltAndEncrypted, enc) == 0) {
                    printf("%s %s \t with Multithreading \n", plain, enc);
                    return;
#pragma omp cancel for
                }
            }
        }
    }
}

// Main function
int main(int argc, char *argv[]) {

    // Open the file with the encrypted password and read it
    FILE *fp;
    char encryptedText[93];
    fp = fopen("encrypted.txt", "r");
    fscanf(fp, "%s", encryptedText);
    fclose(fp);

    // Measure the time taken to crack the password without multithreading
    clock_t start, end;
    start = clock();
    crack(encryptedText);
    end = clock();
    double timeForNonMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Measure the time taken to crack the password with multithreading
    start = clock();
    multithreadingCracking(encryptedText);
    end = clock();
    double timeForMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Print the results
    printf("Time taken without multithreading: %f seconds\n", timeForNonMultithreading);
    printf("Time taken with multithreading: %f seconds\n", timeForMultithreading);
    printf("%d solutions explored\n", count);

    return 0;
}