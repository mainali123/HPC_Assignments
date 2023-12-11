#include <stdio.h>
#include <unistd.h>

// Define the salt for the crypt function
#define SALT "$6$AS$"

int main(int argc, char *argv[]){
    // Use the crypt function to encrypt the password passed as a command line argument
    // The crypt function uses the SHA-512 hashing algorithm as specified by the salt
    printf("%s\n", crypt(argv[1], SALT));

    // Exporting the encrypted password to a file
    // Open the file "encrypted.txt" in write mode
    FILE *fp;
    fp = fopen("encrypted.txt", "w+");

    // Write the encrypted password to the file
    fprintf(fp, "%s\n", crypt(argv[1], SALT));

    // Close the file
    fclose(fp);

    return 0;
}