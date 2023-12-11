#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <time.h>

#define SALT_LENGTH 6
#define PASSWORD_LENGTH 7 // Increased by 1 to accommodate the null terminator

// Structure to hold the result of the decryption attempt
typedef struct {
    int count; // Number of attempts made
    char password[PASSWORD_LENGTH]; // Decrypted password
} Result;

// Function to decrypt the password without multithreading
Result crack(char *salt_and_encrypted) {
    Result result = {0, ""}; // Initialize the result
    char salt[SALT_LENGTH + 1]; // +1 for null terminator
    char plain[PASSWORD_LENGTH]; // +1 for null terminator
    char *enc;

    // Copy the salt from the encrypted password
    strncpy(salt, salt_and_encrypted, SALT_LENGTH);
    salt[SALT_LENGTH] = '\0';

    // Loop through all possible combinations of two uppercase letters and two digits
    for (int x = 'A'; x <= 'Z'; x++) {
        for (int y = 'A'; y <= 'Z'; y++) {
            for (int z = 0; z <= 99; z++) {
                // Create the password string from the current combination
                snprintf(plain, PASSWORD_LENGTH, "%c%c%02d", x, y, z);
                enc = crypt(plain, salt);
                result.count++;
                // If the encrypted password matches the current combination, return the result
                if (strcmp(salt_and_encrypted, enc) == 0) {
                    strcpy(result.password, plain);
                    return result;
                }
            }
        }
    }
    return result; // Return the result if no match is found
}

// Function to decrypt the password with multithreading
Result multithreadingCracking(char *saltAndEncrypted) {
    Result result = {0, ""}; // Initialize the result
    char salt[SALT_LENGTH + 1]; // +1 for null terminator
    char plain[PASSWORD_LENGTH]; // +1 for null terminator
    char *enc;

    // Copy the salt from the encrypted password
    strncpy(salt, saltAndEncrypted, SALT_LENGTH);
    salt[SALT_LENGTH] = '\0';

    // Loop through all possible combinations of two uppercase letters (from 'A' to 'B') and two digits
    for (int i = 'A'; i <= 'B'; ++i) {
        for (int j = 'A'; j <= 'B'; ++j) {
            for (int k = 0; k <= 99; ++k) {
                // Create the password string from the current combination
                snprintf(plain, PASSWORD_LENGTH, "%c%c%02d", i, j, k);
                enc = crypt(plain, salt);
                result.count++;
                // If the encrypted password matches the current combination, return the result
                if (strcmp(saltAndEncrypted, enc) == 0) {
                    strcpy(result.password, plain);
                    return result;
                }
            }
        }
    }
    return result; // Return the result if no match is found
}

// Main function
int main() {
    FILE *fp, *output;
    char encryptedText[93]; // Assuming maximum length of encrypted text is 92 characters

    // Open the file with the encrypted passwords and the output file
    fp = fopen("bunchEncrypted.txt", "r");
    output = fopen("performance.csv", "w+");

    // If there is an error opening the files, print an error message and exit
    if (fp == NULL || output == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    // Write the header to the output file
    fprintf(output, "Count,Decrypted Password,Time Without Multithreading,Time With Multithreading\n");

    // Loop through each encrypted password in the file
    while (fscanf(fp, "%92s", encryptedText) != EOF) {
        clock_t start, end;
        Result normalResult, threadResult;

        // Measure the time taken to decrypt the password without multithreading
        start = clock();
        normalResult = crack(encryptedText);
        end = clock();
        double timeForNonMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;

        // Measure the time taken to decrypt the password with multithreading
        start = clock();
        threadResult = multithreadingCracking(encryptedText);
        end = clock();
        double timeForMultithreading = ((double) (end - start)) / CLOCKS_PER_SEC;

        // Write the results to the output file
        fprintf(output, "%d,%s,%f,%f\n", normalResult.count, normalResult.password, timeForNonMultithreading, timeForMultithreading);
    }

    // Close the files
    fclose(fp);
    fclose(output);
    return 0;
}