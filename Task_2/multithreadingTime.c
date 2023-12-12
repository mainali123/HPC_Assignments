#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
#pragma omp parallel for collapse(2)
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

//    FILE *output_file = fopen("Output.txt", "w");
//    if (output_file == NULL) {
//        printf("Error opening the output file.\n");
//        return 1;
//    }

    FILE *csv_file = fopen("parallelMatrixComputation.csv", "w");
    if (csv_file == NULL) {
        printf("Error opening the CSV file.\n");
        return 1;
    }
    fprintf(csv_file, "no, time_taken\n");

    int matrices_read = 0;
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

        double start_time = omp_get_wtime(); // Start timer
        matrix_multiply(A, B, C, m, n, p);
        double end_time = omp_get_wtime(); // End timer
        double time_taken = end_time - start_time;

        fprintf(csv_file, "%d, %.6f\n", matrices_read + 1, time_taken);

//        fprintf(output_file, "Result of multiplication %d:\n", matrices_read + 1);
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < p; j++) {
//                fprintf(output_file, "%.6f ", C[i * p + j]);
//            }
//            fprintf(output_file, "\n");
//        }
//        fprintf(output_file, "\n");

        free(A);
        free(B);
        free(C);

        matrices_read++;
    }

    fclose(file);
//    fclose(output_file);
    fclose(csv_file);
    return 0;
}
