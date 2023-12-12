#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

int main() {
    FILE *file = fopen("testdata.txt", "r");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return 1;
    }

    FILE *output_file = fopen("matrixComputation.csv", "w");
    if (output_file == NULL) {
        printf("Error opening the output file.\n");
        return 1;
    }

    int matrices_read = 0;
    clock_t start, end;
    while (!feof(file)) {
        int m, n, p;
        if (fscanf(file, "%d,%d", &m, &n) != 2) break;

        double *A = (double *)malloc(m * n * sizeof(double));
        for (int i = 0; i < m * n; i++) {
            fscanf(file, "%lf,", &A[i]);
        }

        if (fscanf(file, "%d,%d", &n, &p) != 2) break;

        double *B = (double *)malloc(n * p * sizeof(double));
        for (int i = 0; i < n * p; i++) {
            fscanf(file, "%lf,", &B[i]);
        }

        double *C = (double *)malloc(m * p * sizeof(double));

        start = clock(); // Start the timer

        matrix_multiply(A, B, C, m, n, p);

        end = clock(); // Stop the timer

        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

        fprintf(output_file, "%d,%.6f\n", matrices_read + 1, time_taken);

        free(A);
        free(B);
        free(C);

        matrices_read++;
    }

    fclose(file);
    fclose(output_file);
    return 0;
}
