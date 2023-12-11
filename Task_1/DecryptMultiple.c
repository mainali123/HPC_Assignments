#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <time.h>

#define SALT_LENGTH 6
#define PASSWORD_LENGTH 7 // Increased by 1 to accommodate the null terminator

typedef struct {
    int count;
    char password[PASSWORD_LENGTH];
} Result;

Result crack(char *salt_and_encrypted) {
    Result result = {0, ""};
    char salt[SALT_LENGTH + 1]; // +1 for null terminator
    char plain[PASSWORD_LENGTH]; // +1 for null terminator
    char *enc;

    strncpy(salt, salt_and_encrypted, SALT_LENGTH);
    salt[SALT_LENGTH] = '\0';

    for (int x = 'A'; x <= 'Z'; x++) {
        for (int y = 'A'; y <= 'Z'; y++) {
            for (int z = 0; z <= 99; z++) {
                snprintf(plain, PASSWORD_LENGTH, "%c%c%02d", x, y, z);
                enc = crypt(plain, salt);
                result.count++;
                if (strcmp(salt_and_encrypted, enc) == 0) {
                    strcpy(result.password, plain);
                    return result;
                }
            }
        }
    }
    return result;
}

Result multithreadingCracking(char *saltAndEncrypted) {
    Result result = {0, ""};
    char salt[SALT_LENGTH + 1]; // +1 for null terminator
    char plain[PASSWORD_LENGTH]; // +1 for null terminator
    char *enc;

    strncpy(salt, saltAndEncrypted, SALT_LENGTH);
    salt[SALT_LENGTH] = '\0';

    for (int i = 'A'; i <= 'B'; ++i) {
        for (int j = 'A'; j <= 'B'; ++j) {
            for (int k = 0; k <= 99; ++k) {
                snprintf(plain, PASSWORD_LENGTH, "%c%c%02d", i, j, k);
                enc = crypt(plain, salt);
                result.count++;
                if (strcmp(saltAndEncrypted, enc) == 0) {
                    strcpy(result.password, plain);
                    return result;
                }
            }
        }
    }
    return result;
}

int main() {
    FILE *fp, *output;
    char encryptedText[93]; // Assuming maximum length of encrypted text is 92 characters
    fp = fopen("bunchEncrypted.txt", "r");
    output = fopen("performance.csv", "w+");

    if (fp == NULL || output == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    fprintf(output, "Count,Decrypted Password,Time Without Multithreading,Time With Multithreading\n");

    while (fscanf(fp, "%92s", encryptedText) != EOF) {
        clock_t start, end;
        Result normalResult, threadResult;

        start = clock();
        normalResult = crack(encryptedText);
        end = clock();
        double timeForNonMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;

        start = clock();
        threadResult = multithreadingCracking(encryptedText);
        end = clock();
        double timeForMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;

        fprintf(output, "%d,%s,%f,%f\n", normalResult.count, normalResult.password, timeForNonMultithreading, timeForMultithreading);
    }

    fclose(fp);
    fclose(output);
    return 0;
}
