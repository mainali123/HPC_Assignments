#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <omp.h>
#include <stdbool.h>
#include <time.h>



/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer.

  Compile with:
    cc -o CrackAZ99 CrackAZ99.c -lcrypt

  If you want to analyse the output then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:
    ./CrackAZ99 > output.txt

  Dr Kevan Buckley, University of Wolverhampton, 2018 Modified by Dr. Ali Safaa 2019
******************************************************************************/

int count = 0;     // A counter used to track the number of combinations explored so far

/**
 Required by lack of standard function in C.   
*/

void substr(char *dest, char *src, int start, int length) {
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the 
 start of the line. Note that one of the most time consuming operations that 
 it performs is the output of intermediate results, so performance experiments 
 for this kind of program should not include this. i.e. comment out the printfs.
*/

void crack(char *salt_and_encrypted) {
    count = 0;
    int x, y, z;     // Loop counters
    char salt[7];    // String used in hashing the password. Need space for \0 // incase you have modified the salt value, then should modifiy the number accordingly
    char plain[7];   // The combination of letters currently being checked // Please modifiy the number when you enlarge the encrypted password.
    char *enc;       // Pointer to the encrypted password

    substr(salt, salt_and_encrypted, 0, 6);

    for (x = 'A'; x <= 'Z'; x++) {
        for (y = 'A'; y <= 'Z'; y++) {
            for (z = 0; z <= 99; z++) {
                sprintf(plain, "%c%c%02d", x, y, z);
                enc = (char *) crypt(plain, salt);
                count++;
                if (strcmp(salt_and_encrypted, enc) == 0) {
                    printf("#%-8d%s %s \t without Multithreading \n", count, plain, enc);
                    return;    //uncomment this line if you want to speed-up the running time, program will find you the cracked password only without exploring all possibilites
                }
            }
        }
    }
}

// Multithreading Cracking using OpenMP
bool password_found = false;

void multithreadingCracking(char *saltAndEncrypted) {
    count = 0;
    char salt[7];
    char plain[7];
    char *enc;

    substr(salt, saltAndEncrypted, 0, 6);

#pragma omp parallel for collapse(3)
    for (int i = 'A'; i <= 'Z'; ++i) {
        for (int j = 'A'; j <= 'Z'; ++j) {
            for (int k = 0; k <= 99; ++k) {
                if (!password_found) {
                    sprintf(plain, "%c%c%02d", i, j, k);
                    enc = (char *) crypt(plain, salt);
                    count++;
                    if (strcmp(saltAndEncrypted, enc) == 0) {
                        printf("#%-8d%s %s \t with Multithreading \n", count, plain, enc);
                        password_found = true;
                        break;
                    }
                }
            }
            if (password_found) break;
        }
        if (password_found) break;
    }
}

void optimizedMultithreadingCracking(char *saltAndEncrypted) {
    count = 0;
    char salt[7];
    char plain[7];
    char *enc;

    substr(salt, saltAndEncrypted, 0, 6);

    password_found = false;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 'A'; i <= 'Z'; ++i) {
        for (int j = 'A'; j <= 'Z'; ++j) {
#pragma omp parallel for schedule(dynamic)
            for (int k = 0; k <= 99; ++k) {
                if (!password_found) {
                    sprintf(plain, "%c%c%02d", i, j, k);
                    enc = (char *) crypt(plain, salt);
#pragma omp atomic
                    count++;
                    if (strcmp(saltAndEncrypted, enc) == 0) {
#pragma omp critical
                        {
                            if (!password_found) {
                                printf("#%-8d%s %s \t with Optimized Multithreading \n", count, plain, enc);
                                password_found = true;
                            }
                        }
                    }
                }
            }
            if (password_found) break;
        }
        if (password_found) break;
    }
}


int main(int argc, char *argv[]) {

    FILE *fp;
    char encryptedText[93];
    fp = fopen("encrypted.txt", "r");
    fscanf(fp, "%s", encryptedText);
    fclose(fp);


    clock_t start, end;

    start = clock();
    crack(
            encryptedText);
    end = clock();
    double timeForNonMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;


    start = clock();
    multithreadingCracking(
            encryptedText);
    end = clock();
    double timeForMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;


    start = clock();
    optimizedMultithreadingCracking(
            encryptedText);
    end = clock();
    double timeForOptimizedMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;


    printf("Time taken without multithreading: %f seconds\n", timeForNonMultithreading);
    printf("Time taken with multithreading: %f seconds\n", timeForMultithreading);
    printf("Time taken with optimized multithreading: %f seconds\n", timeForOptimizedMultithreading);
    printf("%d solutions explored\n", count);

    return 0;
}

