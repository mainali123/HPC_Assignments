#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Function to perform matrix multiplication
void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    // Parallelize the outer two loops using OpenMP
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            // Perform the dot product of the i-th row of A and the j-th column of B
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            // Store the result in the i-th row and j-th column of C
            C[i * p + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    // Check if the correct number of command line arguments has been provided
    if (argc != 3) {
        printf("Usage: %s <filename> <num_threads>\n", argv[0]);
        return 1;
    }

    // Assign the filename and the number of threads from the command line arguments
    char *filename = argv[1];
    int num_threads = atoi(argv[2]);

    // Set the number of threads for OpenMP to use
    omp_set_num_threads(num_threads);

    // Open the input file
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return 1;
    }

    // Open the output file
    FILE *output_file = fopen("Output.txt", "w");
    if (output_file == NULL) {
        printf("Error opening the output file.\n");
        return 1;
    }

    // Variable to keep track of the number of matrices read
    int matrices_read = 0;

    // Loop until the end of the file is reached
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

        // Check if matrix multiplication is possible
        if (m != p) {
            printf("Error: The number of columns in the first matrix is not equal to the number of rows in the second matrix.\n");
            return 1;
        }

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

        // Free the memory allocated for the matrices
        free(A);
        free(B);
        free(C);

        // Increment the number of matrices read
        matrices_read++;
    }

    // Close the input and output files
    fclose(file);
    fclose(output_file);

    return 0;
}