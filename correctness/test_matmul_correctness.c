#include "../multiply.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_matrix(struct Matrix mat, float seed)
{
    for(size_t i = 0; i < mat.rows; i++){
        for(size_t j = 0; j < mat.cols; j++){
            float v = (float)(((i + 7) * 19 + (j + 11) * 37) % 101);
            mat.data[i * mat.cols + j] = (v / 101.0f - 0.5f) + seed;
        }
    }
}

static int run_case(size_t m, size_t n, size_t k, int use_blas_reference)
{
    struct Matrix a = {m, k, NULL};
    struct Matrix b = {k, n, NULL};
    struct Matrix expected = {m, n, NULL};
    struct Matrix actual = {m, n, NULL};

    a.data = malloc(a.rows * a.cols * sizeof(float));
    b.data = malloc(b.rows * b.cols * sizeof(float));
    expected.data = calloc(expected.rows * expected.cols, sizeof(float));
    actual.data = calloc(actual.rows * actual.cols, sizeof(float));

    if(a.data == NULL || b.data == NULL || expected.data == NULL || actual.data == NULL){
        fprintf(stderr, "malloc failed for case %zu x %zu x %zu\n", m, n, k);
        free(a.data);
        free(b.data);
        free(expected.data);
        free(actual.data);
        return 1;
    }

    fill_matrix(a, 0.01f);
    fill_matrix(b, -0.02f);

    if(use_blas_reference){
        matmul_v9_OpenBLAS(a, b, expected);
    }else{
        multiply_plain(a, b, expected);
    }
    matmul_v8_avx512_omp_improved(a, b, actual, 132, 8640, 160);

    double max_abs_error = 0.0;
    size_t max_i = 0;
    size_t max_j = 0;
    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            size_t index = i * n + j;
            double diff = fabs((double)expected.data[index] - (double)actual.data[index]);
            if(diff > max_abs_error){
                max_abs_error = diff;
                max_i = i;
                max_j = j;
            }
        }
    }

    double tolerance = use_blas_reference ? 1.0e-2 : 1.0e-3;
    if(max_abs_error > tolerance){
        printf("FAIL v8 m=%zu n=%zu k=%zu ref=%s max_abs_error=%e at (%zu, %zu)\n",
               m, n, k, use_blas_reference ? "blas" : "plain", max_abs_error, max_i, max_j);
        free(a.data);
        free(b.data);
        free(expected.data);
        free(actual.data);
        return 1;
    }

    printf("PASS v8 m=%zu n=%zu k=%zu ref=%s max_abs_error=%e\n",
           m, n, k, use_blas_reference ? "blas" : "plain", max_abs_error);

    free(a.data);
    free(b.data);
    free(expected.data);
    free(actual.data);
    return 0;
}

int main(void)
{
    static const size_t cases[][3] = {
        {1, 7, 11},
        {2, 13, 9},
        {4, 17, 15},
        {5, 18, 5},
        {6, 19, 13},
        {7, 23, 17},
        {8, 31, 29},
        {20, 31, 17},
        {64, 70, 73},
        {127, 131, 137},
        {239, 241, 243},
        {240, 240, 240},
        {241, 241, 241},
        {247, 263, 249},
        {257, 481, 241},
        {481, 257, 241},
    };

    static const size_t large_cases[][3] = {
        {512, 512, 512},
        {1001, 1003, 1009},
        {1024, 1024, 1024},
        {1536, 1536, 1536},
        {1537, 1543, 1549},
        {2048, 2048, 2048},
        {2051, 2039, 2029},
    };

    int failed = 0;
    for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++){
        failed += run_case(cases[i][0], cases[i][1], cases[i][2], 0);
    }
    for(size_t i = 0; i < sizeof(large_cases) / sizeof(large_cases[0]); i++){
        failed += run_case(large_cases[i][0], large_cases[i][1], large_cases[i][2], 1);
    }

    if(failed != 0){
        printf("v8 correctness test failed: %d case(s)\n", failed);
        return 1;
    }

    printf("all v8 correctness tests passed\n");
    return 0;
}
