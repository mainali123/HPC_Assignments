// Include necessary libraries
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Function to perform matrix multiplication using multithreading
void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    // Use OpenMP to parallelize the outer two loops
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            // Multiply each element of the ith row of the first matrix with the jth column of the second matrix
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            // Store the result in the C matrix
            C[i * p + j] = sum;
        }
    }
}

int main() {
    // Open the file for reading
    FILE *file = fopen("MatData.txt", "r");
    // Check if the file was opened successfully
    if (file == NULL) {
        printf("Error opening the file.\n");
        return 1;
    }

    // Open the output file for writing
    FILE *output_file = fopen("Output.txt", "w");
    // Check if the file was opened successfully
    if (output_file == NULL) {
        printf("Error opening the output file.\n");
        return 1;
    }

    int matrices_read = 0;
    // Loop until the end of the file
    while (!feof(file)) {
        int m, n, p;
        // Read the dimensions of the first matrix
        if (fscanf(file, "%d,%d", &m, &n) != 2) break;

        // Allocate memory for the first matrix and read its elements
        double *A = (double *)malloc(m * n * sizeof(double));
        for (int i = 0; i < m * n; i++) {
            fscanf(file, "%lf,", &A[i]);
        }

        // Read the dimensions of the second matrix
        if (fscanf(file, "%d,%d", &n, &p) != 2) break;

        // Allocate memory for the second matrix and read its elements
        double *B = (double *)malloc(n * p * sizeof(double));
        for (int i = 0; i < n * p; i++) {
            fscanf(file, "%lf,", &B[i]);
        }

        // Allocate memory for the result matrix
        double *C = (double *)malloc(m * p * sizeof(double));

        // Perform matrix multiplication
        matrix_multiply(A, B, C, m, n, p);

        // Write the result to the output file
        fprintf(output_file, "Result of multiplication %d:\n", matrices_read + 1);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                fprintf(output_file, "%.6f ", C[i * p + j]);
            }
            fprintf(output_file, "\n");
        }
        fprintf(output_file, "\n");

        // Free the allocated memory
        free(A);
        free(B);
        free(C);

        matrices_read++;
    }

    // Close the files
    fclose(file);
    fclose(output_file);
    return 0;
}