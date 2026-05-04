#ifndef MULTIPLY_H
#define MULTIPLY_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __aarch64__
#include <arm_neon.h>
#define NR 16
#else
#include <immintrin.h>
#include <xmmintrin.h>
#define NR 32
#endif
#include <omp.h>
#include <sys/resource.h>//资源控制
#include <threads.h>
#include <cblas.h>
#ifndef BLOCK_LEAP
#define BLOCK_LEAP 240
#endif
#define SIMD_LEAP 8
#define SIMD_LEAP_ROW 6 //寄存器大小是四行
#define MIN(a,b) ((a) <= (b) ? (a) : (b))

struct Matrix{
    size_t rows;
    size_t cols;
    float *data;
};

typedef struct{
    size_t rows;
    size_t cols;
    int8_t* data;
}MatrixINT8;

typedef struct{
    size_t rows;
    size_t cols;   
    int32_t* data;
}MatrixINT32;

typedef struct{
    float scale;
    int zero_point;
}Transform;

// 函数声明
int set_memory_limit(size_t bytes);
void multiply_plain(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void matmul_v2_ikj(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void matmul_v3_block(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
#ifndef __aarch64__
void matmul_v4_avx2(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void matmul_v5_openmp(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void matmul_v6_omp_avx_6x16(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void matmul_v7_avx512_omp(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void matmul_v8_avx512_omp_improved(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, size_t ii, size_t jj, size_t kk);
#endif
#ifdef __aarch64__
void matmul_v8_neon_omp(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3, size_t ii, size_t jj, size_t kk);
#endif
void matmul_v9_OpenBLAS(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void multiply_improved(struct Matrix mat1, struct Matrix mat2, struct Matrix mat3);
void compare_error(struct Matrix expected, struct Matrix actual);

#ifndef __aarch64__
void avx_kernel_1x8(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int i,
    int j,
    size_t i_start,
    size_t j_start,
    size_t valid_k);
    
void avx_kernel_2x8(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);


void avx_kernel_4x8(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx_kernel_6x8(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx_kernel_2x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx_kernel_6x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_1x32(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_2x32(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_4x32(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_6x32(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_1x32_panelb(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_2x32_panelb(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_4x32_panelb(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void avx512_kernel_6x32_panelb(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);
#endif

#ifdef __aarch64__
void neon_kernel_1x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void neon_kernel_2x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void neon_kernel_4x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void neon_kernel_6x16(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k);

void neon_kernel_panelb_tail(
    struct Matrix mat1,
    struct Matrix mat2,
    struct Matrix mat3,
    int id,
    int jd,
    size_t i_start,
    size_t j_start,
    size_t valid_k,
    size_t rows,
    size_t cols);
#endif

Transform get_transform(float max_val, float min_val);
MatrixINT8 transfrom_float_to_int8(struct Matrix mat, Transform t);
struct Matrix transform_int32_to_float(MatrixINT32 mat, Transform tA, Transform tB);
void matmul_int8(MatrixINT8 mat1, MatrixINT8 mat2, MatrixINT32 mat3, Transform tA, Transform tB);

#endif
