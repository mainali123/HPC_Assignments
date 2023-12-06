#include <stdio.h>
#include <unistd.h>
/******************************************************************************
  This program is used to set challenges for password cracking programs. 
  Encrypts using SHA-512.
  
  Compile with:
    cc -o EncryptSHA512 EncryptSHA512.c -lcrypt
    
  To encrypt the password "pass":
    ./EncryptSHA512 pass
    
  It doesn't do any checking, just does the job or fails ungracefully.

  Dr Kevan Buckley, University of Wolverhampton, 2017. Modified by Dr. Ali Safaa 2019
******************************************************************************/

#define SALT "$6$AS$"

int main(int argc, char *argv[]){
  printf("%s\n", crypt(argv[1], SALT));
  // Exporting the encrypted password to a file
    FILE *fp;
    fp = fopen("encrypted.txt", "w+");
    fprintf(fp, "%s\n", crypt(argv[1], SALT));
    fclose(fp);
  return 0;
}
