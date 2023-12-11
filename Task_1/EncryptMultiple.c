//
// Created by diwash on 11/12/23.
//
#include <stdio.h>
#include <unistd.h>
#include <crypt.h> // Include the crypt header for crypt function

#define SALT "$6$AS$"

int main(){
    FILE *fp;
    fp = fopen("bunchEncrypted.txt", "w+");

    for (int i = 'A'; i <= 'B'; i++) {
        for (int j = 'A'; j <= 'B'; j++) {
            for (int k = 0; k <= 99; k++) {
                char password[8]; // Increase the size of the array
                sprintf(password, "%c%c%02d", i, j, k);
                fprintf(fp, "%s\n", crypt(password, SALT));
            }
        }
    }

    fclose(fp);
    return 0;
}
