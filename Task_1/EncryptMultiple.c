#include <stdio.h>
#include <unistd.h>
#include <crypt.h>

// Define the salt for the crypt function
#define SALT "$6$AS$"

int main(){
    // Open the file "bunchEncrypted.txt" in write mode
    FILE *fp;
    fp = fopen("bunchEncrypted.txt", "w+");

    // Loop through all possible combinations of two uppercase letters (from 'A' to 'B') and two digits (from 00 to 99)
    for (int i = 'A'; i <= 'B'; i++) {
        for (int j = 'A'; j <= 'B'; j++) {
            for (int k = 0; k <= 99; k++) {
                // Create the password string from the current combination
                char password[8]; // Increase the size of the array
                sprintf(password, "%c%c%02d", i, j, k);

                // Encrypt the password using the crypt function and the defined salt
                // Write the encrypted password to the file
                fprintf(fp, "%s\n", crypt(password, SALT));
            }
        }
    }

    // Close the file
    fclose(fp);
    return 0;
}