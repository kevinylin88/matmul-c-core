#include "../multiply.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

#ifndef DEFAULT_DIM
#define DEFAULT_DIM 8000
#endif
#ifndef DEFAULT_REPEAT
#define DEFAULT_REPEAT 20
#endif
#ifndef DEFAULT_WARMUP
#define DEFAULT_WARMUP 1
#endif

#define MATRIX_ALIGNMENT 64

static double time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1.0e3 + (double)ts.tv_nsec / 1.0e6;
}

static size_t env_size_or_default(const char *name, size_t default_value)
{
    const char *value = getenv(name);
    if(value == NULL || value[0] == '\0'){
        return default_value;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if(end == value || parsed == 0){
        return default_value;
    }
    return (size_t)parsed;
}

static size_t arg_size_or_default(int argc, char **argv, int index, size_t default_value)
{
    if(index >= argc){
        return default_value;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(argv[index], &end, 10);
    if(end == argv[index] || parsed == 0){
        return default_value;
    }
    return (size_t)parsed;
}

static void *aligned_malloc_safe(size_t bytes)
{
    size_t aligned_bytes = ((bytes + MATRIX_ALIGNMENT - 1) / MATRIX_ALIGNMENT) * MATRIX_ALIGNMENT;
    return aligned_alloc(MATRIX_ALIGNMENT, aligned_bytes);
}

static void fill_matrix(struct Matrix mat, uint32_t seed)
{
    size_t total = mat.rows * mat.cols;
    uint32_t state = seed;
    for(size_t i = 0; i < total; i++){
        state = state * 1664525u + 1013904223u;
        mat.data[i] = (float)((state >> 8) & 0xffffu) / 65535.0f - 0.5f;
    }
}

static double checksum_sample(struct Matrix mat)
{
    double sum = 0.0;
    size_t total = mat.rows * mat.cols;
    size_t step = total / 1024 + 1;
    for(size_t i = 0; i < total; i += step){
        sum += (double)mat.data[i];
    }
    return sum;
}

static double gflops_for(size_t dim, double ms)
{
    double flops = 2.0 * (double)dim * (double)dim * (double)dim;
    return flops / (ms * 1.0e6);
}

static void tune_matmul(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, size_t ii, size_t jj, size_t kk)
{
    matmul_v8_avx512_omp_improved(mat1, mat2, mat3, ii, jj, kk);
}

int main(int argc, char **argv)
{
    size_t ii = arg_size_or_default(argc, argv, 1, 144);
    size_t jj = arg_size_or_default(argc, argv, 2, 640);
    size_t kk = arg_size_or_default(argc, argv, 3, 192);
    size_t dim = env_size_or_default("DIM", DEFAULT_DIM);
    size_t repeat = env_size_or_default("REPEAT", DEFAULT_REPEAT);
    size_t warmup = env_size_or_default("WARMUP", DEFAULT_WARMUP);

    struct Matrix a = {dim, dim, NULL};
    struct Matrix b = {dim, dim, NULL};
    struct Matrix c = {dim, dim, NULL};

    size_t matrix_bytes = dim * dim * sizeof(float);
    a.data = aligned_malloc_safe(matrix_bytes);
    b.data = aligned_malloc_safe(matrix_bytes);
    c.data = aligned_malloc_safe(matrix_bytes);

    if(a.data == NULL || b.data == NULL || c.data == NULL){
        fprintf(stderr, "malloc failed for DIM=%zu, bytes_per_matrix=%zu\n", dim, matrix_bytes);
        free(a.data);
        free(b.data);
        free(c.data);
        return 1;
    }

    fill_matrix(a, 0x12345678u);
    fill_matrix(b, 0x87654321u);

    for(size_t r = 0; r < warmup; r++){
        memset(c.data, 0, matrix_bytes);
        tune_matmul(a, b, c, ii, jj, kk);
    }

    double *samples = malloc(repeat * sizeof(double));
    if(samples == NULL){
        fprintf(stderr, "malloc failed for samples\n");
        free(a.data);
        free(b.data);
        free(c.data);
        return 1;
    }

    double total_ms = 0.0;
    double best_ms = DBL_MAX;
    double worst_ms = 0.0;

    for(size_t r = 0; r < repeat; r++){
        memset(c.data, 0, matrix_bytes);
        double start = time_ms();
        tune_matmul(a, b, c, ii, jj, kk);
        double end = time_ms();
        double elapsed = end - start;
        samples[r] = elapsed;
        total_ms += elapsed;
        if(elapsed < best_ms){
            best_ms = elapsed;
        }
        if(elapsed > worst_ms){
            worst_ms = elapsed;
        }
        printf("RUN II=%zu JJ=%zu KK=%zu DIM=%zu ITER=%zu MS=%.3f GFLOPS=%.2f\n",
               ii, jj, kk, dim, r + 1, elapsed, gflops_for(dim, elapsed));
        fflush(stdout);
    }

    qsort(samples, repeat, sizeof(double), cmp_double);
    double med_ms = (repeat % 2 == 1)
        ? samples[repeat / 2]
        : (samples[repeat / 2 - 1] + samples[repeat / 2]) / 2.0;

    double avg_ms = total_ms / (double)repeat;
    double checksum = checksum_sample(c);

    printf("RESULT II=%zu JJ=%zu KK=%zu DIM=%zu REPEAT=%zu THREADS=%d AVG_MS=%.3f MED_MS=%.3f BEST_MS=%.3f WORST_MS=%.3f AVG_GFLOPS=%.2f BEST_GFLOPS=%.2f MED_GFLOPS=%.2f CHECKSUM=%.9e\n",
           ii,
           jj,
           kk,
           dim,
           repeat,
           omp_get_max_threads(),
           avg_ms,
           med_ms,
           best_ms,
           worst_ms,
           gflops_for(dim, avg_ms),
           gflops_for(dim, best_ms),
           gflops_for(dim, med_ms),
           checksum);

    free(samples);

    free(a.data);
    free(b.data);
    free(c.data);
    return 0;
}
